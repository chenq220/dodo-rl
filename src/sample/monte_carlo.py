from pathlib import Path
import os
import shutil
import tempfile
import boto3
import time
import msgpack
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Optional, TYPE_CHECKING
from config import MonteCarloMethodConfig
from common.logger import init_logger
from common.process_shutdown import process_close
from common.registry import extensible
from scheduler.artifact import ArtifactCallerSupport, ArtifactType, ensure_output_parent

if TYPE_CHECKING:
    from scheduler.rollout_sample import RolloutSampleWorker

LOG = init_logger('monte_carlo_method')


@dataclass(frozen=True)
class RecordedEpisode:
    """
    已通过格式校验的一局历史采样。

    MonteCarloMethod 只负责解释“样本文件长什么样”；RolloutSampleWorker 只负责把这一局记录
    通过 env.fake_reset/fake_step 还原为 EpisodeTransition。这样样本格式变化时只需要改蒙特卡洛方法。
    """

    # 原始样本中的episode编号，用于日志定位坏样本。
    episode_id: int
    # 本局应有的最大业务步数；离线训练要求一局完整，不完整的episode会被丢弃。
    episode_limit: int
    # 兼容旧样本格式：旧格式每条记录同时带pre_state/state，新格式用reset记录承载初始态势。
    has_pre_state: bool
    # 单局内按step顺序排列的原始记录；这里保留原始dict，业务解码仍交给Environment。
    records: list[dict[str, Any]]


@extensible
class MonteCarloMethod(ArtifactCallerSupport):
    """
    蒙特卡洛样本方法，统一负责随机/评估采样文件的写入、读取、校验和回放池加载。

    写模式用于在线随机采样或评估过程，把 Environment.record_sample 输出按步写入 MsgPack 流文件；
    读模式用于样本学习评估，从本地文件或 S3 协议存储流式恢复历史样本。
    """

    def __init__(
            self,
            task_id: str,
            config: MonteCarloMethodConfig,
    ):
        # 默认文件名只精确到日期，真正区分不同任务依靠task_id；入口应保证task_id有业务意义。
        timestamp = time.strftime("%Y%m%d", time.localtime())
        # 任务 ID，用于默认样本文件名，通常来自 TrainTaskManager.task_id。
        self.task_id = task_id
        # 录制配置，包含存储类型、本地临时目录、S3 参数、样本上限和上传间隔。
        self.config = config
        # 产物 register/put caller；由 Role 接线，未绑定时跳过。
        self._init_artifact_callers()
        # 当前已经写入的 step 记录数，不等于 episode 数。
        self.current_sample_count = 0
        # 分布式采样时多个SampleRole会并发调用receiver，文件写入和周期上传必须串行化。
        self._write_lock = threading.Lock()
        # 是否开启写入；False 时只允许 reader 读取历史文件。
        self.write_enabled = self.config.write_enabled
        # 读模式下指定的完整本地文件路径，优先级高于 tmp_local_dir + load_sample_name。
        self.load_file_path = self.config.load_file_path
        # 写模式下 load_sample_name 为空会生成新文件；读模式下它表示要加载的历史样本名。
        sample_file_name = self.config.load_sample_name or f"sample_{timestamp}_{self.task_id}.bin"
        if not sample_file_name.endswith(".bin"):
            sample_file_name = f"{sample_file_name}.bin"

        # 本地样本文件路径；S3 模式也会先写本地临时文件，再按需上传。
        self.local_file = self.config.load_file_path or os.path.join(self.config.tmp_local_dir, sample_file_name)
        # S3 对象 Key，默认与样本文件名一致。
        self.s3_key = sample_file_name

        # 本地文件句柄；读模式或关闭后为 None。
        self.file = None
        if self.write_enabled:
            # 追加写入而不是覆盖，避免蒙特卡洛采样长时间运行时入口重启把已有样本清掉。
            self._open_sample_writer()
            LOG.info(f"local sample file opened: path={self.local_file}")
        # S3 协议客户端；仅 storage_type=="s3" 时初始化。
        self.s3_client = None
        if self.config.storage_type == "s3":
            # 初始化S3协议客户端。
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.storage.endpoint_url,
                aws_access_key_id=self.config.storage.access_key,
                aws_secret_access_key=self.config.storage.secret_key
            )

        # 自动创建 Bucket
        if self.config.storage_type == "s3":
            self._ensure_bucket_exists()
        LOG.info(f'Monte Carlo method enabled, storage type is {self.config.storage_type}')
        self._closed = False
        # stage_name -> (临时索引文件, [(offset, length), ...])。
        self._offline_index_files: dict[str, str] = {}
        self._offline_indices: dict[str, list[tuple[int, int]]] = {}

    @process_close
    def close(self):
        """关闭写模式文件句柄；读模式下file为None，重复调用应安全。"""
        with getattr(self, "_write_lock", threading.Lock()):
            if getattr(self, "_closed", False):
                return
            self._closed = True
            if getattr(self, "file", None) is not None and not self.file.closed:
                self.file.flush()
                self.file.close()
                LOG.info(f"local sample file closed: path={self.local_file}, samples={self.current_sample_count}")
            self._put_sample_file()
            self.complete_artifact_stage()
            for path in list(getattr(self, "_offline_index_files", {}).values()):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except OSError:
                    LOG.warning(f"failed to clean offline index directory: path={path}")
            self._offline_index_files.clear()
            self._offline_indices.clear()

    def register_stage_intents(self) -> None:
        if self.write_enabled:
            self.register_artifact_intent(ArtifactType.SAMPLE, planned_hint=os.path.dirname(self.local_file) or ".")

    def _open_sample_writer(self) -> None:
        ensure_output_parent(self.local_file)
        self.file = open(self.local_file, "ab", buffering=0)

    def _put_sample_file(self) -> None:
        if (
                self.write_enabled
                and self.current_sample_count > 0
                and os.path.isfile(self.local_file)
        ):
            self.put_artifact_file(
                ArtifactType.SAMPLE,
                self.local_file,
                {"record_count": int(self.current_sample_count)},
            )

    def _ensure_bucket_exists(self):
        """确保S3 bucket存在；这里按框架默认策略自动创建，避免入口额外依赖存储初始化流程。"""
        try:
            self.s3_client.head_bucket(Bucket=self.config.storage.bucket_name)
            LOG.info(f"s3 bucket exists: bucket={self.config.storage.bucket_name}")
        except self.s3_client.exceptions.ClientError:
            self.s3_client.create_bucket(Bucket=self.config.storage.bucket_name)
            LOG.info(f"s3 bucket created: bucket={self.config.storage.bucket_name}")

    def configure(self, config: MonteCarloMethodConfig) -> None:
        """在保持 RPC 对象身份不变的前提下切换阶段配置。"""
        with self._write_lock:
            if self.file is not None and not self.file.closed:
                self.file.flush()
                self.file.close()
            for path in self._offline_index_files.values():
                shutil.rmtree(path, ignore_errors=True)
            self._offline_index_files.clear()
            self._offline_indices.clear()
            self.config = config
            self.write_enabled = config.write_enabled
            sample_file_name = config.load_sample_name or f"sample_{time.strftime('%Y%m%d')}_{self.task_id}.bin"
            if not sample_file_name.endswith(".bin"):
                sample_file_name = f"{sample_file_name}.bin"
            self.load_file_path = config.load_file_path
            self.local_file = config.load_file_path or os.path.join(config.tmp_local_dir, sample_file_name)
            self.s3_key = sample_file_name
            self.file = None
            if config.write_enabled:
                self._open_sample_writer()
            self.current_sample_count = 0
            self._closed = False

    def upload_to_oss(self) -> None:
        """把当前本地样本文件同步到S3协议存储。"""
        assert self.s3_client is not None
        with self._write_lock:
            self._upload_to_oss_unlocked()

    def _upload_to_oss_unlocked(self) -> None:
        """在调用方已经持有写锁时上传样本文件。"""
        # S3上传读取的是磁盘文件，上传前必须先把本地缓冲刷盘。
        if self.file is not None:
            self.file.flush()
        assert os.path.exists(self.local_file), f"local sample file not found: {self.local_file}"
        try:
            self.s3_client.upload_file(
                Filename=self.local_file,
                Bucket=self.config.storage.bucket_name,
                Key=self.s3_key
            )
            LOG.info(
                f"s3 object uploaded: bucket={self.config.storage.bucket_name}, "
                f"key={self.s3_key}, samples={self.current_sample_count}"
            )
        except Exception as e:
            LOG.exception(f"s3 object upload failed: key={self.s3_key}, error={e}")

    def receiver(self, payload: dict[str, Any]) -> None:
        """
        接收Environment.record_sample输出的一条记录。

        payload 是框架内约定的原始样本格式，不在这里解析业务字段；蒙特卡洛方法只负责顺序写入。
        后续离线学习再由 episode_reader / normalize_episode_records 解释文件结构。
        """
        assert self.file is not None, "MonteCarloMethod is opened in read-only mode"
        with self._write_lock:
            # sample_size限制的是记录条数，不是episode条数；reset记录也会计入。
            if self.current_sample_count >= self.config.sample_size:
                return

            # MsgPack按流式对象连续写入，一个文件中没有额外数组外壳，方便边写边读大文件。
            packed_data = msgpack.packb(payload)
            self.file.write(packed_data)
            self.current_sample_count += 1
            # 长时间采样时周期性上传，避免进程异常退出导致本地已有样本完全丢失。
            if (
                    self.config.storage_type == "s3"
                    and self.current_sample_count > 0
                    and self.current_sample_count % self.config.upload_interval == 0
            ):
                LOG.info(f"s3 periodic upload triggered: samples={self.current_sample_count}")
                self._upload_to_oss_unlocked()

    def local_file_reader(self, file_path: Optional[str] = None) -> Iterator:
        """从本地MsgPack流文件逐条读取样本记录。"""
        if self.file is not None and not self.file.closed:
            # 同一个方法实例边写边读时，先刷盘保证reader能看到最新记录。
            self.file.flush()
        target_file = file_path or self.local_file
        if not os.path.exists(target_file):
            raise FileNotFoundError(target_file)
        LOG.info(f"local sample file reading: path={target_file}")
        with open(target_file, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for item in unpacker:
                yield item

    def oss_file_reader(self, s3_key: Optional[str] = None) -> Iterator:
        """从S3协议对象流式读取样本记录，不把大文件整体下载到内存。"""
        assert self.s3_client is not None
        target_key = s3_key or self.s3_key
        try:
            # 从S3协议获取对象。
            response = self.s3_client.get_object(
                Bucket=self.config.storage.bucket_name,
                Key=target_key
            )
            # response['Body'] 是一个 StreamingBody 网络流对象
            # MsgPack 的 Unpacker 可以直接接管它，边下载边解包，零本地磁盘占用
            unpacker = msgpack.Unpacker(response['Body'], raw=False)
            for item in unpacker:
                yield item
        except self.s3_client.exceptions.NoSuchKey:
            LOG.error(f"s3 object not found: key={target_key}")
            return
        except Exception as e:
            LOG.exception(f"s3 object stream read failed: key={target_key}, error={e}")
            return

    def reader(self) -> Iterator:
        """按配置从后端存储读取样本。"""
        # 统一读入口：上层不需要关心样本来自本地还是S3。
        source = self.oss_file_reader() if self.config.storage_type == "s3" else self.local_file_reader()

        def adapted():
            for raw in source:
                yield self.adapt_record(raw)

        return adapted()

    def adapt_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        """把后端读出的原始记录适配为框架标准 payload；默认恒等。"""
        return raw

    def episode_reader(self) -> Iterator[list[dict[str, Any]]]:
        """
        按episode解析历史样本文件。

        MonteCarloMethod是样本文件生产者，也负责解释文件整体结构；RolloutSampleWorker只接收
        单局记录，完成业务状态向量和EpisodeTransition恢复。
        """
        current_episode = None
        episode_records: list[dict[str, Any]] = []
        for record in self.reader():
            # 样本文件必须按采样顺序写入；episode编号变化就是上一局结束的边界。
            episode_id = record.get("episode", 0)
            if current_episode is None:
                current_episode = episode_id
            if episode_id != current_episode:
                # 不在这里做完整性校验，因为校验需要知道整局步数和动作完整性。
                if episode_records:
                    yield episode_records
                episode_records = []
                current_episode = episode_id
            episode_records.append(record)
        if episode_records:
            yield episode_records

    def normalized_episode_reader(self, expected_agents: Optional[list[str]] = None) -> Iterator[RecordedEpisode]:
        """按episode读取并校验历史样本，坏局直接丢弃。"""
        for records in self.episode_reader():
            segments = self._iter_contiguous_segments(records)
            for segment in segments:
                episode = self.normalize_episode_records(segment, expected_agents=expected_agents)
                if episode is not None:
                    yield episode

    def _iter_contiguous_segments(self, records: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        """按步号缺口切出候选连续片段；默认模式仍把整局作为一个候选。"""
        if not records:
            return
        if not self.config.split_on_step_gap:
            yield records
            return
        segment: list[dict[str, Any]] = [records[0]]
        previous_step = records[0].get("step")
        for record in records[1:]:
            try:
                current_step = int(record["step"])
                previous = int(previous_step)
            except (KeyError, TypeError, ValueError):
                segment.append(record)
                previous_step = record.get("step")
                continue
            if current_step != previous + 1:
                if len(segment) > 1:
                    yield self._segment_anchor(segment)
                segment = [record]
            else:
                segment.append(record)
            previous_step = current_step
        if len(segment) > 1:
            yield self._segment_anchor(segment)

    @staticmethod
    def _segment_anchor(segment: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """新格式片段首条重新作为 step=0 reset 锚点。"""
        has_pre_state = any(item.get("pre_state") is not None for item in segment)
        anchored: list[dict[str, Any]] = []
        for index, item in enumerate(segment):
            record = dict(item)
            record["step"] = index + 1 if has_pre_state else index
            if index == 0 and not has_pre_state:
                record["actions"] = {}
                record["is_reset"] = True
            anchored.append(record)
        return anchored

    def normalize_episode_records(
            self,
            records: list[dict[str, Any]],
            expected_agents: Optional[list[str]] = None,
    ) -> Optional[RecordedEpisode]:
        """
        把原始step记录校验为一局可恢复的样本。

        MonteCarloMethod是样本文件生产者，因此样本格式、步号连续性、动作完整性由这里解释；
        RolloutSampleWorker只负责把已经校验过的一局记录恢复为EpisodeTransition。
        """
        if not records:
            return None
        episode_id = records[0].get("episode", 0)
        # episode_step_limit是新字段；max_step保留给旧样本兼容。
        episode_limit = records[-1].get("episode_step_limit", records[-1].get("max_step"))
        try:
            episode_limit = int(episode_limit)
        except (TypeError, ValueError):
            LOG.warning(f"Drop recorded episode: invalid episode limit, episode={episode_id}")
            return None

        # 旧格式：每个step都有pre_state和state，所以记录数等于真实步数。
        # 新格式：reset单独记录初始state，之后每个step记录执行后的state，所以记录数=步数+1。
        has_pre_state = any(record.get("pre_state") is not None for record in records)
        if self.config.accept_truncated_episodes:
            min_records = 1 if has_pre_state else 2
            if len(records) < min_records:
                return None
        else:
            expected_records = episode_limit if has_pre_state else episode_limit + 1
            if len(records) != expected_records:
                LOG.warning(
                    f"Drop recorded episode: records count not match episode format, episode={episode_id}, "
                    f"records={len(records)}, expected_records={expected_records}, episode_limit={episode_limit}"
                )
                return None

        # expected_agents来自当前环境，用来防止历史样本和当前场景智能体数量/名称不匹配。
        expected_agents_set = set(expected_agents or [])
        expected_first_step = 1 if has_pre_state else 0
        previous_step = None
        for idx, record in enumerate(records):
            # 一局内所有记录必须属于同一个episode；否则说明样本流被截断或拼接错乱。
            if record.get("episode", 0) != episode_id:
                LOG.warning(f"Drop recorded episode: mixed episode id in episode={episode_id}")
                return None
            # state是离线恢复的最低要求；没有state就无法fake_reset/fake_step。
            if record.get("state") is None:
                LOG.warning(f"Drop recorded episode: missing state, episode={episode_id}, index={idx}")
                return None
            # 同一局不能混用新旧格式，否则无法判断S_t来自pre_state还是上一条state。
            if has_pre_state and record.get("pre_state") is None:
                LOG.warning(f"Drop recorded episode: mixed pre_state format, episode={episode_id}, index={idx}")
                return None

            actions = record.get("actions")
            is_initial_state_record = not has_pre_state and idx == 0
            if is_initial_state_record:
                # 新格式第0条是reset初始态势，不代表任何动作。
                if actions not in (None, {}):
                    LOG.warning(f"Drop recorded episode: reset record has actions, episode={episode_id}, index={idx}")
                    return None
            elif not isinstance(actions, dict):
                LOG.warning(f"Drop recorded episode: actions is not dict, episode={episode_id}, index={idx}")
                return None
            elif expected_agents_set and not expected_agents_set.issubset(actions.keys()):
                # 缺少任意一个agent动作都不能训练同构多智能体序列，整局丢弃。
                LOG.warning(f"Drop recorded episode: incomplete actions, episode={episode_id}, index={idx}")
                return None

            try:
                current_step = int(record["step"])
            except (KeyError, TypeError, ValueError):
                LOG.warning(f"Drop recorded episode: invalid step, episode={episode_id}, index={idx}")
                return None
            # reset格式必须从step=0开始；旧pre_state格式没有reset记录，直接从step=1开始。
            if idx == 0 and current_step != expected_first_step:
                LOG.warning(
                    f"Drop recorded episode: first step invalid, episode={episode_id}, "
                    f"first_step={current_step}, expected_first_step={expected_first_step}"
                )
                return None
            # step必须严格连续；缺一帧就无法可靠构造(S_t, A_t, R_t, S_t+1)。
            if previous_step is not None and current_step != previous_step + 1:
                LOG.warning(
                    f"Drop recorded episode: discontinuous step, episode={episode_id}, "
                    f"previous_step={previous_step}, current_step={current_step}"
                )
                return None
            previous_step = current_step

        if previous_step != episode_limit and not (
                self.config.accept_truncated_episodes
                and previous_step is not None
                and previous_step < episode_limit
        ):
            LOG.warning(
                f"Drop recorded episode: last step not equal max step, episode={episode_id}, "
                f"last_step={previous_step}, episode_limit={episode_limit}"
            )
            return None

        # 到这里才认为样本是一局完整、连续、动作齐全的episode。
        return RecordedEpisode(
            episode_id=int(episode_id),
            episode_limit=episode_limit,
            has_pre_state=has_pre_state,
            records=records,
        )

    def prepare_offline_index(
            self,
            *,
            stage_name: str,
            expected_agents: list[str],
    ) -> int:
        """完整扫描并建立当前阶段的本地固定偏移索引，返回有效轨迹数。"""
        # 当前实例只激活一个阶段；切换时使旧 token 的 stage_name 请求立即失效。
        for old_dir in self._offline_index_files.values():
            shutil.rmtree(old_dir, ignore_errors=True)
        self._offline_index_files.clear()
        self._offline_indices.clear()
        Path(self.config.tmp_local_dir).expanduser().mkdir(parents=True, exist_ok=True)
        index_dir = tempfile.mkdtemp(prefix=f"offline_{stage_name}_", dir=self.config.tmp_local_dir)
        index_path = os.path.join(index_dir, "episodes.bin")
        offsets: list[tuple[int, int]] = []
        with open(index_path, "wb") as out:
            for episode in self.normalized_episode_reader(expected_agents=expected_agents):
                payload = {
                    "episode_id": episode.episode_id,
                    "episode_limit": episode.episode_limit,
                    "has_pre_state": episode.has_pre_state,
                    "records": episode.records,
                }
                packed = msgpack.packb(payload, use_bin_type=True)
                offset = out.tell()
                out.write(packed)
                offsets.append((offset, len(packed)))
        if not offsets:
            shutil.rmtree(index_dir, ignore_errors=True)
            raise ValueError(f"No valid offline recorded episodes for stage={stage_name!r}")
        self._offline_index_files[stage_name] = index_dir
        self._offline_indices[stage_name] = offsets
        LOG.info(f"Offline sample index prepared: stage={stage_name}, episodes={len(offsets)}")
        return len(offsets)

    def get_recorded_episode(
            self,
            *,
            stage_name: str,
            episode_num: int,
    ) -> dict[str, Any]:
        """按1-based任务编号读取索引中的固定轨迹。"""
        with self._write_lock:
            try:
                offsets = self._offline_indices[stage_name]
            except KeyError as exc:
                raise KeyError(f"offline index is not prepared for stage={stage_name!r}") from exc
            index = int(episode_num) - 1
            if not 0 <= index < len(offsets):
                raise IndexError(f"offline episode_num out of range: stage={stage_name!r}, episode_num={episode_num}")
            offset, length = offsets[index]
            path = os.path.join(self._offline_index_files[stage_name], "episodes.bin")
            with open(path, "rb") as source:
                source.seek(offset)
                payload = msgpack.unpackb(source.read(length), raw=False)
        return payload

    def load_replay_buffer(self, rollout_worker: 'RolloutSampleWorker', max_episodes: Optional[int] = None) -> int:
        """
        驱动RolloutSampleWorker把历史样本逐局恢复进ReplayBuffer。

        :param rollout_worker: 负责把单局记录转换成EpisodeTransition的局滚动执行器。
        :param max_episodes: 最多成功加载多少局；None表示不限制。
        :return: 成功写入ReplayBuffer的episode数量。
        """
        if max_episodes is not None:
            assert max_episodes > 0, f"max_episodes must be positive, got={max_episodes}"
        expected_agents = None
        if hasattr(rollout_worker, "env"):
            # 用当前环境的possible_agents校验样本动作完整性，避免旧样本加载到不匹配场景。
            expected_agents = getattr(rollout_worker.env, "possible_agents", None)
        loaded_count = 0
        for episode in self.normalized_episode_reader(expected_agents=expected_agents):
            # RolloutSampleWorker负责把单局样本恢复进ReplayBuffer；蒙特卡洛方法只负责驱动“逐局加载”。
            if rollout_worker.offline_rollout_one_episode(episode):
                loaded_count += 1
                if max_episodes is not None and loaded_count >= max_episodes:
                    break
        LOG.info(
            f"Loaded replay buffer from recorded samples, episodes={loaded_count}, "
            f"max_episodes={max_episodes}"
        )
        return loaded_count
