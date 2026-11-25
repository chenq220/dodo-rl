from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from algos.learners import Learner, algo_config_to_updates
from common.logger import init_logger
from config import AgentConfig, LearnerConfig, StageConfig, StageTaskConfig
from sample.replay.replay_persist import (
    default_replay_snapshot_name,
    resolve_replay_snapshot_path,
    samples_dir_from_config,
)

if TYPE_CHECKING:
    from sample.replay_buffer import ReplayBuffer

LOG = init_logger("stage_utils")


def union_stage_agent_configs(stages: List[StageConfig]) -> List[AgentConfig]:
    """
    按AgentName合并所有阶段的Agent配置，得到跨阶段共享的Agent全集。

    Agent是训练对象，跨阶段不重新初始化，因此各阶段的agents_config通常一致；这里仍按name去重，
    保证即使不同阶段写法略有出入，也只构造一份Agent实例集合。
    """
    union: List[AgentConfig] = []
    seen: set[str] = set()
    for stage in stages:
        for agent_config in stage.agents_config:
            if agent_config.name not in seen:
                seen.add(agent_config.name)
                union.append(agent_config)
    assert union, "StageTaskConfig stages must declare at least one agent in agents_config."
    return union


def union_stage_learner_configs(stages: List[StageConfig]) -> List[LearnerConfig]:
    """
    按AgentName合并所有阶段的Learner配置，得到需要构造的Learner全集。

    某个Learner可能只在部分阶段上场；Role一次性构造全集Learner并跨阶段复用（不重置网络），
    每个阶段实际是否学习由该阶段TrainTaskManager下发的学习令牌决定。
    取首次出现的配置作为构造配置；各阶段超参差异在阶段切换时通过dynamic_modify_config应用。
    """
    union: List[LearnerConfig] = []
    seen: set[str] = set()
    for stage in stages:
        for learner_config in stage.learners_config:
            if learner_config.name not in seen:
                seen.add(learner_config.name)
                union.append(learner_config)
    return union


def apply_stage_replay_policy(
        replay_buffer: "ReplayBuffer",
        stage: StageConfig,
) -> None:
    """
    进入阶段时的回放池策略：调用方应先完成 skip snapshot 灌池；这里清 active，最后写入本阶段混采比例。

    离开阶段的 DRBF 落盘、rehearsal 归档、S3 推送由 ReplayBuffer.notify_stage_complete 负责。
    """
    if stage.clear_replay_buffer:
        replay_buffer.clear(generation=0)
    for learner in stage.learners_config:
        replay_buffer.set_agent_capacity(learner.name, learner.algo_config.capacity)
    replay_buffer.set_rehearsal_sample_ratio(stage.rehearsal_sample_ratio)


def apply_stage_to_resources(
        stage: StageConfig,
        env,
        learners_by_name: Dict[str, Learner],
        replay_buffer=None,
) -> None:
    """
    把一个阶段应用到跨阶段共享的运行期资源上（StageTask之外的阶段切换动作，由Role调用）。

    1. Environment：写入 env.stage_name，一个环境实例据此自行切换各阶段业务逻辑；
       Learner 持有同一个 env 引用，因此 learner.env.stage_name 同步生效。
    2. ReplayBuffer：按 StageConfig.clear_replay_buffer 清理 active，并设置本阶段
       rehearsal_sample_ratio。离开上一阶段的归档/落盘见 ReplayBuffer.notify_stage_complete。
    3. Learner：阶段切换先把所有Learner本地generation归零，再对本阶段上场的Learner用
       该阶段算法超参做一次 dynamic_modify_config；
       未在本阶段 learners_config 中出现的Learner不改动，本阶段对应Agent即固定策略。
    """
    env.stage_name = stage.stage_name
    env.apply_env_metadata(stage.train_config.env_metadata)
    if replay_buffer is not None:
        apply_stage_replay_policy(replay_buffer, stage)
    for learner in learners_by_name.values():
        learner.reset_generation()
    for learner_config in stage.learners_config:
        learner = learners_by_name.get(learner_config.name)
        if learner is None:
            continue
        if learner_config.algo_config is not None:
            learner.dynamic_modify_config(algo_config_to_updates(learner_config.algo_config))
    LOG.info(
        f"Applied stage to resources: stage_name={stage.stage_name}, "
        f"clear_replay_buffer={stage.clear_replay_buffer}, "
        f"rehearsal_sample_ratio={stage.rehearsal_sample_ratio}, "
        f"active_learners={[c.name for c in stage.learners_config]}"
    )


def save_stage_replay_if_needed(
        stage: StageConfig,
        replay_buffer: "ReplayBuffer",
        task_id: str,
) -> Optional[Path]:
    """阶段结束时按 StageConfig.save_replay_buffer 落盘本地 DRBF；不含归档与 S3。

    生产路径请用 ReplayBuffer.notify_stage_complete。
    """
    if not stage.save_replay_buffer:
        return None
    samples_dir = samples_dir_from_config(replay_buffer.config)
    filename = default_replay_snapshot_name(stage.stage_name, task_id)
    path = Path(samples_dir) / filename
    written = replay_buffer.save_snapshot(path)
    LOG.info(
        f"Stage replay snapshot saved: stage_name={stage.stage_name}, path={written}"
    )
    return written


def load_skip_stage_replay_if_needed(
        stage_task_config: StageTaskConfig,
        replay_buffer: "ReplayBuffer",
        *,
        already_loaded: bool = False,
) -> bool:
    """
    仅在跳过前置阶段（start_stage>0）且配置了 skip_stage_replay_path 时灌池一次。

    返回是否实际执行了 load_snapshot。already_loaded=True 时直接跳过（Role 侧一次性标志）。
    """
    if already_loaded:
        return False
    if int(stage_task_config.start_stage) <= 0:
        return False
    raw_path = str(stage_task_config.skip_stage_replay_path or "").strip()
    if not raw_path:
        return False
    samples_dir = samples_dir_from_config(replay_buffer.config)
    path = resolve_replay_snapshot_path(raw_path, samples_dir=samples_dir)
    replay_buffer.load_snapshot(path)
    LOG.info(
        f"Skip-stage replay snapshot loaded: start_stage={stage_task_config.start_stage}, "
        f"path={path}"
    )
    return True
