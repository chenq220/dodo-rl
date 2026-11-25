from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Mapping
import torch
import numpy as np
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Any,
    Literal,
    Tuple,
    Generic,
    TypeVar,
    TYPE_CHECKING,
    Generator,
    Iterator,
)
from common.simulation import simulation_exception_types, simulation_parameter_exception_types
from common.registry import extensible
from scheduler.artifact import Artifact, ArtifactType
from common.exception import (
    AgentActionCoverageException,
    AgentGroupCompatibilityException,
    AgentOwnerDuplicateException,
    AgentOwnerNotFoundException,
    AgentRoutingException,
    AgentSelectionNotFoundException,
    EpisodeInterruptedException,
    EnvironmentException,
    EnvironmentTypeException,
    SampleTaskReturned,
    SimulationException,
)
from common.types import AgentName, AgentID, ActionType, AgentLastAction, ObsType, AgentHiddenOut, StateType
from config import AgentType as AgentTypeConfig
from envs.action.model import Action
from scheduler.model import AgentCompose
from contextlib import contextmanager
from sample.monte_carlo import RecordedEpisode
from sample.replay.replay_payload import BatchEncodingUnsupported
from sample.replay.replay_perf import timed
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from rpc.interfaces import ReplayBufferPort, TrainTaskManagerPort
from envs.base import (
    AECEnvironment,
    BaseEnvironment,
    ParallelAgentOrderContext,
    ParallelEnvironment,
    AECAgentOrderContext,
)
from common.logger import init_logger
from utils.pygame_screen_recorder import RenderGifRecorder
from utils.random_seed import set_global_seed
from scheduler.utils.business_metrics import EpisodeMetricCollector
from scheduler.tasks.training_progress import publish_worker_token

if TYPE_CHECKING:
    from scheduler.model import ActionParam, EpisodeInfo
    from scheduler.train_task import TrainTaskManager
    from sample.replay_buffer import ReplayBuffer
    from common.types import AgentType

LOG = init_logger('sample_worker')

EnvironmentType = TypeVar("EnvironmentType", ParallelEnvironment, AECEnvironment)

# 在线 Rollout 的统一轨迹容器：一个 AgentName 对应本物理 episode 的全部组轨迹。
# 旧隐式单组也是长度为 1 的列表；SB3 单体路径值为一个 SAEpisodeTransition 列表。
AgentEpisodeTransitions = Dict[AgentName, List[Union[EpisodeTransition, SAEpisodeTransition]]]


def _agent_group_slots(agent) -> tuple[tuple[AgentID, ...], ...]:
    """返回 Agent 的无名分组二维视图。

    真实 Agent 在构造期建立 agent_groups；未携带该属性的协议对象（测试替身等）
    回退为唯一隐式组，保持旧单组行为。
    """
    groups = getattr(agent, "agent_groups", None)
    if groups:
        return tuple(tuple(group) for group in groups)
    return (tuple(agent.agent_list),)


@contextmanager
def _agent_group_view(agent, group_ids: Tuple[AgentID, ...]):
    """把 agent.agent_list 临时切换为当前组成员（组内槽位视图）。

    Agent 实现（如 RNNAgent）按 self.agent_list 做网络槽位映射，不感知分组；
    Rollout 在组循环内切换视图即可让同一网络按组宽推理。所有权检查、维度注入
    等全量语义不受影响：退出时恢复原始全量列表。
    """
    original = agent.agent_list
    agent.agent_list = list(group_ids)
    try:
        yield
    finally:
        agent.agent_list = original


@extensible
class RolloutSampleWorker(Generic[EnvironmentType, ActionType, AgentID]):
    """
    马尔科夫采样流程执行器，负责按 episode 驱动 Agent 与 Environment 交互。

    在线模式下它从 TrainTaskManager 获取 EpisodeInfo，执行 reset/step，生成 EpisodeTransition 后写入 ReplayBuffer；
    离线模式下它读取 MonteCarloMethod 历史记录，按相邻状态恢复 transition，坏局会整局丢弃。
    """

    def __init__(
            self,
            env: BaseEnvironment,
            task_manager: TrainTaskManager,
            replay_buffer: ReplayBufferPort,
            agent_compose: AgentCompose,
            policy_agents: list[AgentType],
            monte_carlo_method: Optional[Any] = None,
            render_output_dir: Optional[str] = None,
    ):
        # 环境实例，提供 reset/step、观测、状态向量、奖励、动作掩码和离线 fake_step。
        self.env: EnvironmentType = env
        # 参与采样编排的Agent列表
        self.policy_agents = policy_agents
        # AgentCompose描述“谁在局开始决策、谁在每步和环境交互、按什么编排方式执行”。
        self.agent_compose = agent_compose
        self.episode_agent: Optional[AgentType] = self.agent_compose.episode_agent
        self.step_agent: List[AgentType] = self.agent_compose.step_agent
        assert self.episode_agent is None or self.episode_agent in self.policy_agents, \
            "RolloutSampleWorker episode_agent must be included in agents."
        self.offline_agent = self.step_agent[-1]
        # RNN类Agent跨步缓存：last_*是上一步输入，next_*是一轮决策后准备写回的结果。
        self.last_hidden_state: AgentHiddenOut = {}
        self.last_actions: AgentLastAction = {}
        self.next_hidden_state = {}
        self.next_actions = {}
        self.total_rewards: Dict[AgentName, float] = {}
        # 在线采样局的业务指标收集器；每局分派滚动策略前都会用令牌配置重新创建。
        self._episode_metric_collector = EpisodeMetricCollector({})
        self.episode_metrics = {}
        self._episode_steps = 0
        # 局内产物契约；主路和旁路共用，episode_task_context 收尾时一并上报 TM。
        self._pending_episode_artifacts: list[Artifact] = []
        # Role 接线的 GIF 直送 caller：caller(task_id, stage_name, path, metadata)；
        # 未接线时局末 GIF 只走 report_sample_result 业务回报，不入产物台账。
        self.render_gif_put_caller = None
        # 远程令牌的路径属于中心节点；分布式采样优先使用本机配置的录制目录。
        self.render_output_dir = render_output_dir
        self.sample_role_name = "Serial"
        self.last_episode_info = None
        # 拿到令牌后、开 episode 前的模型同步钩子。节拍闸门会让 worker 等待，
        # 若仍在领令牌前 sync，等待期间发布的新模型会被整局错过。
        self.pre_episode_sync_caller = None
        assert self.agent_compose.step_agent, \
            "RolloutSampleWorker requires AgentCompose.step_agent to drive env interaction."

        managed_agent_ids = [agent_id for agent in self.step_agent for agent_id in agent.agent_list]
        if len(managed_agent_ids) != len(set(managed_agent_ids)):
            duplicate_agent_ids = list(dict.fromkeys(
                agent_id
                for agent_id in managed_agent_ids
                if managed_agent_ids.count(agent_id) > 1
            ))
            raise AgentOwnerDuplicateException(
                agent_id=duplicate_agent_ids,
                owners=[agent.name for agent in self.step_agent],
            )
        # 显式分组第一版不支持 AEC 原子步编排；装配期直接拒绝，不进入首局。
        if isinstance(env, AECEnvironment):
            for agent in self.step_agent:
                if len(_agent_group_slots(agent)) > 1:
                    raise AgentGroupCompatibilityException(
                        agent_name=agent.name,
                        detail="显式分组第一版不支持 AEC 编排，请使用 Parallel 环境或 agent_names 单组入口。",
                    )
        # 任务调度器，负责 episode 分发、进度提交和训练完成判定。
        self.task_manager: TrainTaskManager = task_manager
        # episode 回放池，接收在线采样或离线恢复后的 EpisodeTransition。
        self.replay_buffer: ReplayBuffer = replay_buffer
        # 中心/本地 MC 代理；offline token 必须从令牌模式决定是否使用它。
        self.monte_carlo_method = monte_carlo_method

        # 连续仿真超时超过该阈值后，如果仿真来自SimManager，则释放当前资源并重新申请。
        sim_config = getattr(getattr(env, "sim", None), "config", None)
        self.sim_timeout_reacquire_threshold = getattr(sim_config, "metadata", {}).get("sim_timeout_reacquire_threshold", 10)
        if type(self.sim_timeout_reacquire_threshold) is not int or self.sim_timeout_reacquire_threshold < 1:
            raise ValueError("sim_timeout_reacquire_threshold must be a positive integer")

    def _get_init_hidden_state(self) -> Dict[AgentID, Any]:
        """按Agent分组合并初始隐藏状态；显式分组在组视图下逐组获取。

        RNNAgent 等实现按 self.agent_list（组内槽位）切初始 hidden，因此显式分组
        时必须逐组切换视图再调用；hidden 键仍为真实 AgentID，组间天然隔离。
        """
        hidden_state: Dict[AgentID, Any] = {}
        for agent in self.policy_agents:
            for group_ids in _agent_group_slots(agent):
                agent_hidden = {}
                if hasattr(agent, "get_init_hidden_state"):
                    with _agent_group_view(agent, group_ids):
                        hidden = agent.get_init_hidden_state()
                    if hidden is not None:
                        assert isinstance(hidden, dict), (
                            f"{type(agent).__name__}.get_init_hidden_state() must return dict or None, "
                            f"given={type(hidden).__name__}"
                        )
                        agent_hidden = hidden
                for agent_id in group_ids:
                    hidden_state[agent_id] = agent_hidden.get(agent_id, self._empty_hidden_for_agent(agent))
        return hidden_state

    @staticmethod
    def _empty_hidden_for_agent(agent: Any) -> torch.Tensor:
        """随机采样没有网络前向结果时，按Agent网络维度补一个隐藏状态占位。"""
        return torch.zeros(1, agent.dim_info.hidden_dim, device=agent.device)

    def get_random_actions(self, action_mask=None):
        """Sample through each agent's typed action space, preserving named heads."""
        from envs.action.spaces import space_for_agent
        actions, hidden = {}, {}
        for agent in self.policy_agents:
            space = space_for_agent(agent)
            for agent_id in agent.agent_list:
                mask = action_mask.get(agent_id) if action_mask else None
                actions[agent_id] = space.sample(np.random, mask)
                hidden[agent_id] = self._empty_hidden_for_agent(agent)
        return actions, hidden

    def get_random_agent_actions(self, agent, agent_ids=None):
        from envs.action.spaces import space_for_agent
        space = space_for_agent(agent)
        actions, hidden = {}, {}
        for agent_id in agent_ids or agent.agent_list:
            actions[agent_id] = space.sample(np.random, self.env.get_avail_agent_actions(agent_id))
            hidden[agent_id] = self._empty_hidden_for_agent(agent)
        return actions, hidden

    def get_init_last_actions(self):
        """
        获取初始的last action动作
        与参考实现对齐：返回全零的 onehot 动作占位
        """
        # 第一步没有真实上一动作，保持None；Agent适配层会把None解释成全零one-hot。
        # 这里不能提前调用一次策略网络，否则会把一个未执行动作写成第0步上一动作。
        last_actions: AgentLastAction = {agent_id: None for agent_id in self.env.possible_agents}
        return last_actions

    def get_actions(
            self,
            agent: AgentType,
            agent_list: List[AgentID],
            obs: Dict[AgentID, Union[ObsType, StateType]],
            epsilon: float,
            evaluate: bool,
            temperature: float = 1.0,
            deterministic: bool = False,
            special_values: dict[AgentID, List[int]] = None,
            **kwargs
    ) -> tuple[AgentLastAction, AgentHiddenOut]:
        """ 智能体获取动作编排 """
        action_masks = {agent_id: self.env.get_avail_agent_actions(agent_id)
                        for agent_id in agent_list}
        # 全部的态势和上步动作都传给智能体，智能体自己决定要哪个
        return agent.get_actions(
            obs=obs,
            agent_list=agent_list,
            hidden_in=self.last_hidden_state,
            last_actions=self.last_actions,
            action_mask=action_masks,
            epsilon=epsilon,
            evaluate=evaluate,
            temperature=temperature,
            deterministic=deterministic,
            **kwargs
        )

    @staticmethod
    def _episode_action_param(episode_info: EpisodeInfo, agent: AgentType) -> ActionParam:
        """读取任务总线按AgentName下发的动作选择参数。"""
        actions_param = episode_info.actions_param
        assert isinstance(actions_param, dict), (
            f"EpisodeInfo.actions_param must be dict[AgentName, ActionParam], "
            f"given={type(actions_param).__name__}"
        )
        action_param = actions_param.get(agent.name)
        assert action_param is not None, (
            f"EpisodeInfo.actions_param must contain action param for agent={agent.name}."
        )
        return action_param

    @contextmanager
    def episode_task_context(self, **kwargs) -> Generator[Optional[EpisodeInfo], Any, None]:
        """
        在线采样局上下文。

        上文从TrainTaskManager领取EpisodeInfo，下文在整局成功结束后统一回报。
        采样任务已结束或当前无可用令牌时 yield None，由调用方结束调度循环。
        """
        if self.task_manager.is_sample_completed():
            LOG.info("Sample already completed.")
            self.last_episode_info = None
            yield None
            return
        self.total_rewards: Dict[AgentName, float] = {}
        self.sample_role_name = kwargs.get("sample_role_name", "Serial")
        while True:
            self._wait_sample_gate(sample_role_name=self.sample_role_name)
            episode_info = self.task_manager.get_sample_task_info(**kwargs)
            if episode_info is not None:
                break
            if self.task_manager.is_sample_completed():
                LOG.info("Sample already completed.")
                self.last_episode_info = None
                yield None
                return
            # 与其它 worker 抢同一张令牌失败、或刚被节拍/背压阻塞，都应继续等待，
            # 不能把 None 当成「没有任务」让串行角色提前退出。
            # 阶段切换例外：返回 None 让 role 去提交本阶段产物。
            gate_state = self._sample_gate_state(self.sample_role_name)
            if gate_state.get("blocked") and gate_state.get("reason") != "stage_switching":
                continue
            LOG.info("There is no sample task available.")
            self.last_episode_info = None
            yield None
            return
        self.last_episode_info = episode_info
        pre_episode_sync_caller = getattr(self, "pre_episode_sync_caller", None)
        # 进度信息推送给环境变量，让外部shell获取
        publish_worker_token(
            kind="sample",
            value=(
                f"episode={episode_info.episode_num} "
                f"stage={episode_info.stage_name} "
                f"generation={episode_info.generation}"
            ),
        )
        set_global_seed(
            episode_info.seed,
            context=f"RolloutSampleWorker:{self.sample_role_name}:episode:{episode_info.episode_num}",
        )
        try:
            # 同步期间的可恢复 RPC 错误也必须归还本局令牌。
            self._pending_episode_artifacts = []
            self.episode_metrics = {}
            self._episode_steps = 0
            if pre_episode_sync_caller is not None:
                pre_episode_sync_caller(episode_info)
            LOG.info(f"[Stage {self.env.stage_name} - Episode {episode_info.episode_num}] Rolling sample start!")
            LOG.info(f'Get Sample info: epsilon={episode_info.epsilon} temperature={episode_info.temperature} '
                     f'deterministic={episode_info.deterministic} seed={episode_info.seed} metadata={episode_info.metadata}'
                     f' model_versions={getattr(episode_info, "required_model_versions", {})}')
            yield episode_info
            artifacts = self._consume_episode_artifacts()
            reward_details = self._collect_episode_rewards_details()
            if reward_details:
                components = ", ".join(f"{name}={value:+.8f}" for name, value in reward_details.items())
                LOG.info(f"Episode {episode_info.episode_num} reward components: {components}; "
                         f"sum={sum(reward_details.values()):+.8f}")
            self.task_manager.report_sample_result(
                episode_info.episode_num,
                total_rewards=self.total_rewards,
                episode_metrics=self.episode_metrics,
                episode_rewards_details=reward_details,
                artifacts=artifacts or None,
                lease_id=episode_info.lease_id,
                step_count=getattr(self, "_episode_steps", 0),
            )
        except (EpisodeInterruptedException, *simulation_exception_types()) as e:
            is_simulation_error = isinstance(e, (SimulationException, *simulation_exception_types()))
            is_environment_error = isinstance(e, EnvironmentException)
            failure_kind = (
                "simulation" if is_simulation_error
                else "environment" if is_environment_error
                else "recoverable RPC"
            )
            LOG.error(
                f'Episode {episode_info.episode_num} failed by {failure_kind} exception, return token.'
            )
            LOG.exception(e)
            artifacts = self._consume_episode_artifacts()
            self.task_manager.report_sample_result(
                episode_info.episode_num,
                total_rewards={},
                success=False,
                artifacts=artifacts or None,
                lease_id=episode_info.lease_id,
                step_count=getattr(self, "_episode_steps", 0),
            )
            # 在抛出 SampleTaskReturned 交给角色层之前，先阻塞等待仿真恢复；
            # 否则角色层下一轮可能马上拿回同一个令牌并再次失败。
            if is_simulation_error:
                self._wait_simulation_ready(episode_info.episode_num)
            raise SampleTaskReturned(episode_info.episode_num) from e
        except simulation_parameter_exception_types():
            # Invalid SDK configuration/action schemas are fatal, not a recoverable simulation outage.
            raise
        except Exception as e:
            LOG.error(f'Episode {episode_info.episode_num} failed.')
            LOG.exception(e)
            try:
                self.task_manager.report_sample_result(
                    episode_info.episode_num,
                    total_rewards={},
                    success=False,
                    artifacts=self._consume_episode_artifacts() or None,
                    lease_id=episode_info.lease_id,
                    step_count=getattr(self, "_episode_steps", 0),
                )
            except Exception:
                LOG.exception('Failed to return episode lease after unexpected sample error.')
            raise
        finally:
            publish_worker_token(kind="sample", value="") 
    def _get_post_state_sim_time(self) -> float:
        post_state = getattr(self.env, "post_state", None)
        return float(getattr(post_state, "sim_time", 0.0) or 0.0)

    def _get_post_state_raw_data(self):
        post_state = getattr(self.env, "post_state", None)
        if post_state is None:
            return None
        return getattr(post_state, "raw_data", None)
    
    def _sample_gate_state(self, sample_role_name: str) -> dict:
        """读取统一闸门状态；兼容未实现 sample_gate_state 的旧 TM/RPC 代理。"""
        gate_state = getattr(self.task_manager, "sample_gate_state", None)
        if gate_state is not None:
            return gate_state(sample_role_name=sample_role_name)
        state = self.task_manager.sample_backpressure_state(sample_role_name=sample_role_name)
        return {
            "blocked": bool(state.get("enabled")),
            "reason": state.get("reason"),
            "retry_after_seconds": 0.5,
        }

    def _wait_sample_gate(self, sample_role_name: str) -> None:
        """采样被节拍/背压闸门限流时，在采样滚动流程侧等待并打印状态切换日志。"""
        logged = False
        while True:
            state = self._sample_gate_state(sample_role_name)
            if not state.get("blocked"):
                if logged:
                    LOG.info(f"Sample gate released: sample_role={sample_role_name}.")
                return
            if not logged:
                LOG.info(state.get("reason") or
                         f"Sample gate blocked: sample_role={sample_role_name}.")
                logged = True
            # 阶段切换不是节拍背压：必须让 role 回到主循环提交本阶段产物，
            # 否则控制端的 notify_stage_complete 会等 producer 而 sample 永远卡在闸门。
            if state.get("reason") == "stage_switching":
                return
            retry_after = float(state.get("retry_after_seconds") or 0.5)
            # 轮询间隔上限 0.5s；服务端 next_issue_at 已保证节拍不会被客户端压没。
            time.sleep(min(max(retry_after, 0.05), 0.5))

    def rollout_one_episode(
            self,
            evaluate=False,
            random_sample=False,
            step_limit: Optional[int] = None,
            **kwargs
    ) -> Optional[AgentEpisodeTransitions]:
        """ 与仿真在线交互一局；无可用采样令牌或采样已完成时返回 None。

        返回统一为 dict[AgentName, list[Transition]]；旧隐式单组也是长度为 1 的列表。
        """
        with self.episode_task_context(**kwargs) as episode_info:
            if episode_info is None:
                return None
            random_sample = bool((episode_info.metadata or {}).get("random_sample", random_sample))
            self.last_actions = self.get_init_last_actions()
            self.last_hidden_state = self._get_init_hidden_state()
            if episode_info.sample_type == "offline":
                grouped_agents = [
                    agent.name
                    for agent in self.step_agent
                    if len(_agent_group_slots(agent)) > 1
                ]
                if grouped_agents:
                    raise AgentGroupCompatibilityException(
                        agent_name=grouped_agents[0],
                        detail="显式分组第一版不支持离线 MonteCarlo 恢复训练。",
                    )
                if self.monte_carlo_method is None:
                    raise EnvironmentException(detail="offline sample requires MonteCarloMethod")
                encoded = self.monte_carlo_method.get_recorded_episode(
                    stage_name=episode_info.stage_name,
                    episode_num=episode_info.episode_num,
                )
                recorded = RecordedEpisode(
                    episode_id=int(encoded["episode_id"]),
                    episode_limit=int(encoded["episode_limit"]),
                    has_pre_state=bool(encoded["has_pre_state"]),
                    records=list(encoded["records"]),
                )
                if recorded.records:
                    last_record = recorded.records[-1]
                    last_step = int(last_record.get("step", 0))
                    if last_step < recorded.episode_limit:
                        recorded.records[-1] = dict(last_record, is_truncated=True)
                sample_start_time = time.time()
                transition = self.offline_rollout_one_episode(recorded, write_to_replay=False)
                if not isinstance(transition, EpisodeTransition) or transition.seq_len == 0:
                    raise EnvironmentException(detail="offline recorded episode produced no transitions")
                self.total_rewards = {self.offline_agent.name: transition.episode_reward()}
                self._submit_replay_transitions(
                    {self.offline_agent.name: transition}, episode_info, sample_start_time,
                )
                return {self.offline_agent.name: [transition]}
            # episode_agent在局开始前决策流程参数，例如本局步数或后续Agent执行顺序。
            # 它不直接step环境，只把决策结果交给下面的局滚动流程。
            if self.episode_agent:
                # 从令牌获取动作参数
                action_param: ActionParam = self._episode_action_param(episode_info, self.episode_agent)
                episode_agent_obs = self.env.state()
                episode_action_mask = {
                    agent_id: None
                    for agent_id in self.episode_agent.agent_list
                }
                next_actions, next_hidden_state = self.episode_agent.get_actions(
                    agent_list=self.episode_agent.agent_list,
                    obs={agent_id: episode_agent_obs for agent_id in self.episode_agent.agent_list},
                    hidden_in=self.last_hidden_state,
                    last_actions=self.last_actions,
                    action_mask=episode_action_mask,
                    epsilon=action_param.epsilon,
                    deterministic=action_param.deterministic,
                    temperature=action_param.temperature,
                    evaluate=evaluate
                )
                assert len(self.episode_agent.agent_list) == 1, 'There can only be one commander'
                episode_agent_action = next(iter(next_actions.values()))
                step_limit = episode_agent_action.step_num or step_limit
                self.next_actions.update(next_actions)
                self.next_hidden_state.update(next_hidden_state)
            else:
                episode_agent_action = None

            # 无Episode Agent时环境走透明默认顺序；有局级动作时由环境解释完整Action。
            rolling_params = dict(
                episode_info=episode_info,
                evaluate=evaluate,
                random_sample=random_sample,
                step_limit=step_limit,
                episode_agent_action=episode_agent_action,
                recorder=self._build_render_recorder(episode_info, evaluate=evaluate),
            )
            self._episode_steps = 0
            self._episode_metric_collector = EpisodeMetricCollector(episode_info.metric_outputs)
            if isinstance(self.env, ParallelEnvironment):
                ep_trans = self.rolling_parallel_episode(**rolling_params)
            elif isinstance(self.env, AECEnvironment):
                ep_trans = self.rolling_aec_episode(**rolling_params)
            else:
                raise EnvironmentTypeException(
                    operation="rollout_one_episode",
                    expected="ParallelEnvironment or AECEnvironment",
                    given=type(self.env).__name__,
                )

            # 打印奖励
            # estimate_time = self.env.post_state.sim_time
            estimate_time = self._get_post_state_sim_time()
            for agent_name, transitions in ep_trans.items():
                episode_reward = sum(t.episode_reward() for t in transitions)
                LOG.info(
                    f"Agent {agent_name} rollout {episode_info.episode_num} episode end, "
                    f"Reward: >>> {episode_reward:.4f} <<<, ETA: {estimate_time / 60:.1f}min"
                )
            # steps 是采样基础统计，不依赖用户声明业务指标。
            # 消费标准 infos 与环境 metadata 用于AutoML指标评估与建议
            self.episode_metrics = self._episode_metric_collector.finish(
                self.env.config.metadata or {}
            )
            self.episode_metrics["steps"] = self._episode_steps

        return ep_trans

    def _iter_parallel_decision_agent_names(
            self,
            episode_agent_action: Optional[Action],
    ) -> Iterator[AgentName]:
        """逐个产生本物理步的Policy/Step AgentName。

        没有Episode Agent动作时直接使用Step Agent声明顺序，不进入环境解析接口；
        有动作时环境按完整Action和已发生的选择历史解释下一行动者。
        """
        if not isinstance(self.env, ParallelEnvironment):
            raise EnvironmentTypeException(
                operation="_iter_parallel_decision_agent_names",
                expected="ParallelEnvironment",
                given=type(self.env).__name__,
            )

        declared_agent_names = tuple(agent.name for agent in self.step_agent)
        # 无Episode Agent的默认策略
        if episode_agent_action is None:
            yield from declared_agent_names
            return

        # Context只记录真实选择历史；候选、环境状态和动作缓存由环境自行管理或读取。
        resolved_agent_names: list[AgentName] = []
        for _ in declared_agent_names:
            context = ParallelAgentOrderContext(
                resolved_agent_names=tuple(resolved_agent_names),
            )
            agent_name = self.env.episode_agent_order_resolve(
                episode_agent_action,
                context,
            )
            # 真正路由Policy Agent时才检查名称是否存在；重复选择由动作覆盖检查自然暴露。
            if agent_name not in declared_agent_names:
                raise AgentSelectionNotFoundException(
                    scope="parallel_step_agent",
                    selected=agent_name,
                    available=list(declared_agent_names),
                )
            yield agent_name
            resolved_agent_names.append(agent_name)

    def _group_frame_inputs(
            self,
            agent_name: AgentName,
            group_ids: Tuple[AgentID, ...],
            obs: Dict[AgentID, ObsType],
    ) -> Tuple[Dict[AgentID, ObsType], StateType, Dict[AgentID, List[int]]]:
        """冻结一组本物理步的决策输入：深拷贝观测 + 组集中状态 + 当前帧动作掩码。"""
        group_obs = {
            agent_id: copy.deepcopy(obs[agent_id])
            for agent_id in group_ids
        }
        state_value = copy.deepcopy(self.env.state_for_agent_ids(
            agent_name=agent_name,
            agent_ids=group_ids,
        ))
        group_avail = {}
        for agent_id in group_ids:
            try:
                group_avail[agent_id] = copy.deepcopy(self.env.get_avail_agent_actions(agent_id))
            except (KeyError, ValueError):
                action_space = self.env.action_space(agent_id)
                group_avail[agent_id] = [0] * int(getattr(action_space, "n", 0))
        return group_obs, state_value, group_avail

    def rolling_parallel_episode(
            self,
            episode_info: EpisodeInfo,
            evaluate=False,
            random_sample=False,
            step_limit: Optional[int] = None,
            episode_agent_action: Optional[Action] = None,
            recorder: Optional[RenderGifRecorder] = None,
    ) -> AgentEpisodeTransitions:
        """
        一整局同构智能体马尔科夫交互过程（无名分组逐组推理版）。

        旧单组流程不变；显式分组只是把步内迭代铺平为 (AgentName, 无名组) 序列：
        轮到某组决策时，前序组的本步动作已发布进 multi_agent_action_buffer 并刷新
        观测，该组决策帧天然携带前序上下文，推理后立即发布本组动作；全部决策
        完成后只执行一次联合 env.step。组 g 上一物理步转移的 next_* 帧就用下一次
        轮到 g 决策时的这份输入补写（铺平迭代天然对齐前序组的 t+1 动作）；组末/
        局末的最后一条转移按旧语义用 env.step 后观测立即收口。组间 hidden 与
        轨迹互不串扰。

        仿真异常由 episode_task_context 统一归还采样令牌并等待仿真恢复；
        本函数只负责执行一次完整 episode，不做重试。
        """
        sample_start_time = time.time()
        step_limit = self.env.set_episode_step_limit(step_limit)
        self.env.episode_num = episode_info.episode_num
        # 采样令牌携带当前阶段名；据此写入采样侧环境的 stage_name，使分阶段训练在分布式
        # 多进程下也能让每个采样进程的环境实例切到正确阶段（一个环境实例描述所有阶段）。
        self.env.stage_name = episode_info.stage_name
        # TaskConfig.env_metadata 由 TaskManager 随令牌下发；每局 reset 前应用，支持分布式
        # 采样任务切换环境参数，也为后续环境参数 AutoML 留出统一入口。
        self.env.apply_env_metadata(episode_info.env_metadata)
        refresh_sim = getattr(getattr(self.env, "sim", None), "refresh_simulation", None)
        if callable(refresh_sim):
            refresh_sim()
        # reset后拿到第一个观测；首帧在各组第一次真正决策时冻结，纳入前序组动作上下文。
        current_obs, _ = self.env.reset(seed=episode_info.seed, options={"step_limit": step_limit, "render": recorder is not None})
        self.last_actions = self.get_init_last_actions()
        self.last_hidden_state = self._get_init_hidden_state()
        episode_hidden_in = self.last_hidden_state.copy()
        self.next_actions = {}
        self.next_hidden_state = {}
        eval_states = []
        eval_actions = []

        # 每个非 fixed_mode 学习 Agent 的每个无名组一条轨迹；固定策略只出动作不入池。
        group_transitions: Dict[Tuple[AgentName, int], Union[EpisodeTransition, SAEpisodeTransition]] = {}
        group_members: Dict[Tuple[AgentName, int], Tuple[AgentID, ...]] = {}
        for agent in self.step_agent:
            if not self._records_replay_transition(agent):
                continue
            for group_index, group_ids in enumerate(_agent_group_slots(agent)):
                key = (agent.name, group_index)
                group_transitions[key] = self._create_online_episode_transition(
                    agent, slot_agent_ids=group_ids, group_index=group_index,
                )
                group_members[key] = group_ids
        # 上一物理步已执行 env.step、待下一次轮到该组时补 next 帧的半条转移（至多一条/组）。
        executed_steps: Dict[Tuple[AgentName, int], Dict[str, Any]] = {}

        step_agents_by_name = {agent.name: agent for agent in self.step_agent}
        for step in range(step_limit):
            # 决策阶段：清空上一物理步动作上下文后重新读取观测，按铺平的组序列逐组推理。
            step_on_policy_stats = {}
            step_rollout_fields = {}
            decision_frames: Dict[Tuple[AgentName, int], Dict[str, Any]] = {}
            self.env.multi_agent_action_buffer = {}
            physical_step_active_ids = set(self.env.agents)
            current_obs = self.env.get_obs()
            visited_agent_names: set[AgentName] = set()

            for agent_name in self._iter_parallel_decision_agent_names(episode_agent_action):
                if agent_name in visited_agent_names:
                    raise AgentRoutingException(
                        detail=(
                            f"同一物理步内 AgentName={agent_name} 被重复路由；"
                            f"已访问={sorted(visited_agent_names)}。"
                        )
                    )
                visited_agent_names.add(agent_name)
                agent = step_agents_by_name[agent_name]
                for group_index, group_ids in enumerate(_agent_group_slots(agent)):
                    key = (agent_name, group_index)
                    active_agent_ids = [
                        agent_id for agent_id in group_ids
                        if agent_id in physical_step_active_ids
                    ]
                    if not active_agent_ids and key not in executed_steps:
                        continue

                    # 冻结本组决策输入：本组动作尚未发布，前序组本步动作已入上下文。
                    group_obs, group_state, group_avail = self._group_frame_inputs(
                        agent_name, group_ids, current_obs,
                    )
                    # 先用本帧补写该组上一物理步转移：next_* 对齐下一次真实决策输入，
                    # 已含前序组在 t+1 步发布的新动作。
                    executed = executed_steps.pop(key, None)
                    if executed is not None and key in group_transitions:
                        self._add_online_transition_step(
                            episode_transition=group_transitions[key],
                            agent_ids=list(group_ids),
                            obs=executed["obs"],
                            actions=executed["actions"],
                            last_actions=executed["last_actions"],
                            rewards=executed["rewards"],
                            next_obs=group_obs,
                            terminations=executed["terminations"],
                            truncations=executed["truncations"],
                            state=executed["state"],
                            next_state=group_state,
                            avail_actions_next=group_avail,
                            avail_actions=executed["avail_actions"],
                            log_prob=executed.get("log_prob"),
                            value=executed.get("value"),
                            active_agent_ids=set(executed["active_agent_ids"]),
                            next_active_agent_ids=set(active_agent_ids),
                            metadata=executed["metadata"],
                        )
                    if not active_agent_ids:
                        continue

                    if key in group_transitions:
                        decision_frames[key] = {
                            "obs": group_obs,
                            "state": group_state,
                            "avail_actions": group_avail,
                            "active_agent_ids": active_agent_ids,
                        }
                    # 组内推理：临时把 agent_list 切成组内成员，Agent 实现按组宽槽位执行，
                    # 不感知分组；try/finally 退出后恢复全量视图。
                    with _agent_group_view(agent, group_ids):
                        if random_sample:
                            next_action, next_hidden_state = self.get_random_agent_actions(
                                agent, agent_ids=active_agent_ids,
                            )
                        else:
                            action_param = self._episode_action_param(episode_info, agent)
                            next_action, next_hidden_state = self.get_actions(
                                agent=agent,
                                agent_list=active_agent_ids,
                                obs=group_obs,
                                epsilon=action_param.epsilon,
                                evaluate=evaluate,
                                temperature=action_param.temperature,
                                deterministic=action_param.deterministic,
                            )
                    if hasattr(agent, "consume_on_policy_stats"):
                        step_on_policy_stats[key] = agent.consume_on_policy_stats()
                    if key in group_transitions:
                        step_rollout_fields[key] = self._collect_rollout_fields(
                            agent, episode_info, active_agent_ids,
                        )
                    duplicated_ids = set(next_action) & set(self.next_actions)
                    if duplicated_ids:
                        raise AgentActionCoverageException(
                            expected=list(set(self.next_actions) - duplicated_ids),
                            actual=sorted(duplicated_ids),
                        )
                    # 汇总本step所有agent动作；循环结束后统一交给env.step。
                    self.next_actions.update(next_action)
                    self.next_hidden_state.update(next_hidden_state)
                    # 每组结束立即发布动作；后序组/后序AgentName可读取本步上下文。
                    self.env.multi_agent_action_buffer.update(next_action)
                    current_obs = self.env.get_obs()

            # 校验所有活动实体每物理步恰好发过一次动作。
            if set(self.next_actions) != set(self.env.agents):
                raise AgentActionCoverageException(
                    expected=self.env.agents,
                    actual=list(self.next_actions),
                )

            # 环境步进阶段：收齐全部活动AgentID动作后，只执行一次联合env.step。
            _, rewards, terminations, truncations, infos = self.env.step(self.next_actions)
            next_obs = self.env.get_obs()
            self._episode_steps += 1
            self._episode_metric_collector.add_infos(infos)
            if recorder is not None:
                # render只产出当前状态的一帧，GIF编码延迟到整局结束，避免step热路径做文件IO。
                recorder.capture(self.env.render)
            eval_actions.append(self.next_actions.copy())
            eval_states.append(self.env.post_state.raw_data)

            # 样本暂存阶段：为每个本步决策过的组记录执行数据，等下一次轮到该组时补帧；
            # 组在本次 env.step 后已无存活成员的，直接用 step 后观测收口最后一条转移。
            post_step_active_ids = set(self.env.agents)
            for key, frame in decision_frames.items():
                group_ids = group_members[key]
                log_prob, value = step_on_policy_stats.get(key, (None, None))
                executed = {
                    "obs": frame["obs"],
                    "state": frame["state"],
                    "avail_actions": frame["avail_actions"],
                    "actions": {agent_id: self.next_actions.get(agent_id) for agent_id in group_ids},
                    "last_actions": {agent_id: self.last_actions.get(agent_id) for agent_id in group_ids},
                    "rewards": {agent_id: float(rewards[agent_id]) for agent_id in group_ids},
                    "terminations": {
                        agent_id: bool(terminations.get(agent_id, False))
                        for agent_id in group_ids
                    },
                    "truncations": {
                        agent_id: bool(truncations.get(agent_id, False))
                        for agent_id in group_ids
                    },
                    "active_agent_ids": frame["active_agent_ids"],
                    "metadata": {
                        "episode_num": episode_info.episode_num,
                        "step": step + 1,
                        "group_index": key[1],
                        "episode_agent_action": episode_agent_action,
                        **({"batch_fields": step_rollout_fields[key]} if step_rollout_fields.get(key) else {}),
                    },
                    "log_prob": log_prob,
                    "value": value,
                }
                remaining_active = [
                    agent_id for agent_id in group_ids if agent_id in post_step_active_ids
                ]
                if remaining_active:
                    executed_steps[key] = executed
                    continue
                # 组末收口（旧语义）：终帧取 env.step 后观测；真实终止不再 bootstrap，
                # 单纯截断按终帧掩码保留可继续槽位。
                terminal_obs, terminal_state, terminal_avail = self._group_frame_inputs(
                    key[0], group_ids, next_obs,
                )
                self._add_online_transition_step(
                    episode_transition=group_transitions[key],
                    agent_ids=list(group_ids),
                    obs=executed["obs"],
                    actions=executed["actions"],
                    last_actions=executed["last_actions"],
                    rewards=executed["rewards"],
                    next_obs=terminal_obs,
                    terminations=executed["terminations"],
                    truncations=executed["truncations"],
                    state=executed["state"],
                    next_state=terminal_state,
                    avail_actions_next=terminal_avail,
                    avail_actions=executed["avail_actions"],
                    log_prob=log_prob,
                    value=value,
                    active_agent_ids=set(executed["active_agent_ids"]),
                    next_active_agent_ids={
                        agent_id
                        for agent_id in group_ids
                        if not terminations.get(agent_id, False)
                        and _has_available_action(terminal_avail.get(agent_id, []))
                    },
                    metadata=executed["metadata"],
                )

            # 状态滚动阶段：把next_*推进为下一步的last_*，清空本步临时输出。
            self.last_actions = self.next_actions.copy()
            self.last_hidden_state.update(self.next_hidden_state)
            self.next_actions = {}
            self.next_hidden_state = {}

            if not self.env.agents:
                break

        # 局末收口：仍待补帧的组属于继续运行但被 step_limit/局结束截断的组，
        # 按旧语义用最后一次 env.step 后的观测补齐最后一条转移。
        if executed_steps:
            final_obs = self.env.get_obs()
            for key, executed in executed_steps.items():
                group_ids = group_members[key]
                final_group_obs, final_state, final_avail = self._group_frame_inputs(
                    key[0], group_ids, final_obs,
                )
                self._add_online_transition_step(
                    episode_transition=group_transitions[key],
                    agent_ids=list(group_ids),
                    obs=executed["obs"],
                    actions=executed["actions"],
                    last_actions=executed["last_actions"],
                    rewards=executed["rewards"],
                    next_obs=final_group_obs,
                    terminations=executed["terminations"],
                    truncations=executed["truncations"],
                    state=executed["state"],
                    next_state=final_state,
                    avail_actions_next=final_avail,
                    avail_actions=executed["avail_actions"],
                    log_prob=executed.get("log_prob"),
                    value=executed.get("value"),
                    active_agent_ids=set(executed["active_agent_ids"]),
                    next_active_agent_ids={
                        agent_id
                        for agent_id in group_ids
                        if not executed["terminations"].get(agent_id, False)
                        and _has_available_action(final_avail.get(agent_id, []))
                    },
                    metadata=executed["metadata"],
                )
            executed_steps.clear()

        # 组装统一返回：AgentName -> 本局全部组轨迹；空轨迹丢弃并记日志，不入池。
        agent_ep_trans: AgentEpisodeTransitions = {}
        for key, transition in group_transitions.items():
            if transition.seq_len == 0:
                LOG.warning(
                    f"Group trajectory is empty and skipped: agent={key[0]}, "
                    f"group_index={key[1]}, episode={episode_info.episode_num}"
                )
                continue
            # RNN序列训练需要整局初始hidden和最后hidden，按组内真实AgentID拆分保存。
            if isinstance(transition, EpisodeTransition):
                transition.episode_num = episode_info.episode_num
                transition.set_hidden(
                    {
                        agent_id: episode_hidden_in[agent_id]
                        for agent_id in group_members[key]
                    },
                    {
                        agent_id: self.last_hidden_state[agent_id]
                        for agent_id in group_members[key]
                    },
                )
            agent_ep_trans.setdefault(key[0], []).append(transition)

        # 在线采样完成后写入ReplayBuffer；ReplayBuffer再决定是否通知任务总线释放学习令牌。
        self.total_rewards = {
            agent_name: sum(t.episode_reward() for t in transitions)
            for agent_name, transitions in agent_ep_trans.items()
        }
        if self._replay_write_enabled(episode_info) and agent_ep_trans:
            self._submit_replay_transitions(agent_ep_trans, episode_info, sample_start_time)
        else:
            LOG.info(
                f"Episode skipped ReplayBuffer push: "
                f"episode={episode_info.episode_num}, rewards={self.total_rewards}"
            )
        if recorder is not None:
            record_path = recorder.finish(self._render_record_filename(episode_info))
            if record_path is not None:
                LOG.info(f"Render GIF saved: {record_path}")
                self._pending_episode_artifacts.append(Artifact(ArtifactType.RENDER_GIF, str(record_path)))
                self._put_render_gif(episode_info, record_path)
        self.post_episode(eval_states=eval_states, eval_actions=eval_actions, ep_trans=agent_ep_trans)
        return agent_ep_trans

    def _put_render_gif(self, episode_info: EpisodeInfo, record_path) -> None:
        """局末把 GIF 经 Role 接线的 caller 直送产物台账；未接线跳过，失败不阻断采样。"""
        caller = self.render_gif_put_caller
        if caller is None:
            return
        try:
            caller(
                episode_info.task_id,
                episode_info.stage_name,
                str(record_path),
                {"role_id": self.sample_role_name},
            )
        except Exception as exc:
            LOG.error(
                f"Render GIF artifact transport failed, keep local file: "
                f"episode={episode_info.episode_num}, path={record_path}"
            )
            LOG.exception(exc)

    def _ensure_sample_task_result_accepted(self, episode_info: EpisodeInfo) -> None:
        """阻止已超时或已被重新分发的旧采样结果进入ReplayBuffer。

        生产令牌由 TaskManager 签发、必带租约；租约查表不中（超时回收、
        换代清场）时拒绝准入，由调用方按 EnvironmentException 归还令牌。
        """
        if not self.task_manager.accept_sample_task_result(
                episode_info.episode_num, episode_info.lease_id,
        ):
            raise EnvironmentException(
                detail=(
                    f"sample task lease expired before replay-buffer submission: "
                    f"episode={episode_info.episode_num}, lease_id={episode_info.lease_id}"
                )
            )

    def _submit_replay_transitions(
            self,
            agent_ep_trans,
            episode_info: EpisodeInfo,
            sample_start_time: float,
    ) -> None:
        """保持 lease 校验紧邻大载荷发送；仅 Torch RPC 使用发送前 permit。"""
        push_kwargs = {
            "sample_role_name": self.sample_role_name,
            "episode_num": episode_info.episode_num,
            "sample_cost_secs": round(time.time() - sample_start_time, 4),
            "generation": episode_info.generation,
        }
        versions = getattr(episode_info, "required_model_versions", {})
        if versions:
            push_kwargs["sample_version"] = versions
        if not getattr(self.replay_buffer, "requires_replay_push_permit", False):
            self._ensure_sample_task_result_accepted(episode_info)
            self._prepare_replay_transitions(agent_ep_trans)
            self.replay_buffer.push(agent_ep_trans, **push_kwargs)
            return

        permit = self.replay_buffer.acquire_replay_push_permit(
            sample_role_name=self.sample_role_name,
            episode_num=episode_info.episode_num,
            generation=episode_info.generation,
        )
        push_started = False
        try:
            # 等待 permit 后重新确认 task lease，避免排队期间过期的旧 episode 进入回放池。
            self._ensure_sample_task_result_accepted(episode_info)
            self._prepare_replay_transitions(agent_ep_trans)
            push_started = True
            self.replay_buffer.push_with_permit(permit, agent_ep_trans, **push_kwargs)
        finally:
            if not push_started:
                self.replay_buffer.release_replay_push_permit(
                    permit,
                    sample_role_name=self.sample_role_name,
                    episode_num=episode_info.episode_num,
                    generation=episode_info.generation,
                )

    def _collect_episode_rewards_details(self) -> Optional[Dict[str, float]]:
        """读取并校验环境的整局奖励构成；观测能力异常不影响训练主流程。"""
        getter = getattr(self.env, "get_episode_rewards_details", None)
        if not callable(getter):
            return None
        try:
            raw_details = getter()
        except Exception:
            LOG.exception(
                f"{type(self.env).__name__}.get_episode_rewards_details() failed; "
                "reward details drawing will be disabled."
            )
            return None
        if raw_details is None:
            return None
        if not isinstance(raw_details, Mapping):
            LOG.warning(
                f"{type(self.env).__name__}.get_episode_rewards_details() must return "
                f"dict[str, float] or None, given={type(raw_details).__name__}; "
                "reward details drawing will be disabled."
            )
            return None
        normalized: Dict[str, float] = {}
        for component_name, value in raw_details.items():
            if not isinstance(component_name, str) or isinstance(value, bool):
                LOG.warning(
                    f"Invalid episode reward detail: key={component_name!r}, value={value!r}; "
                    "reward details drawing will be disabled."
                )
                return None
            try:
                normalized_value = float(value)
            except (TypeError, ValueError):
                LOG.warning(
                    f"Invalid episode reward detail: key={component_name!r}, value={value!r}; "
                    "reward details drawing will be disabled."
                )
                return None
            if not math.isfinite(normalized_value):
                LOG.warning(
                    f"Non-finite episode reward detail: key={component_name!r}, value={value!r}; "
                    "reward details drawing will be disabled."
                )
                return None
            normalized[component_name] = normalized_value
        return normalized

    def _consume_episode_artifacts(self) -> list[Artifact]:
        artifacts = self._pending_episode_artifacts
        self._pending_episode_artifacts = []
        return artifacts

    @staticmethod
    def _replay_write_enabled(episode_info: EpisodeInfo) -> bool:
        return bool((episode_info.metadata or {}).get("replay_write", True))

    def _build_render_recorder(self, episode_info: EpisodeInfo, evaluate: bool = False) -> Optional[RenderGifRecorder]:
        metadata = episode_info.metadata or {}
        if not bool(metadata.get("record_render", False)):
            return None
        output_dir = self.render_output_dir or metadata["record_output_dir"]
        recorder = RenderGifRecorder(
            output_dir=output_dir,
            fps=int(metadata.get("record_fps", 15)),
            frame_stride=int(metadata.get("record_frame_stride", 1)),
            max_frames=metadata.get("record_max_frames"),
        )
        recorder.start()
        return recorder

    def _render_record_filename(self, episode_info: EpisodeInfo) -> str:
        metadata = episode_info.metadata or {}
        record_name = metadata.get("record_name")
        if record_name:
            return str(record_name)
        return f"record_{episode_info.task_id}_{int(episode_info.episode_num)}.gif"

    @staticmethod
    def _records_replay_transition(agent) -> bool:
        """固定策略不更新网络，不组装、不推送 ReplayBuffer 轨迹。"""
        return not bool(getattr(getattr(agent, "config", None), "fixed_mode", False))

    @staticmethod
    def _create_online_episode_transition(
            agent,
            slot_agent_ids: Tuple[AgentID, ...] = (),
            group_index: int = 0,
    ):
        """SB3单智能体算法使用扁平样本，其余算法保持多智能体EpisodeTransition。"""
        if agent.agent_type == AgentTypeConfig.SB3.value:
            assert len(agent.agent_list) == 1, (
                f"SB3 agent must manage exactly one entity, agent={agent.name}, "
                f"agent_list={agent.agent_list}"
            )
            return SAEpisodeTransition(action_space=getattr(agent, "action_space", None))
        return EpisodeTransition(
            action_spaces={agent_id: getattr(agent, "action_space", None)
                           for agent_id in (slot_agent_ids or agent.agent_list)},
            agent_ids=tuple(slot_agent_ids) if slot_agent_ids else None,
            group_index=group_index,
        )

    @staticmethod
    def _add_online_transition_step(
            episode_transition,
            agent_ids,
            obs,
            actions,
            last_actions,
            rewards,
            next_obs,
            terminations,
            truncations,
            state,
            next_state,
            avail_actions_next,
            metadata,
            log_prob: Optional[float] = None,
            value: Optional[float] = None,
            active_agent_ids: Optional[set[AgentID]] = None,
            next_active_agent_ids: Optional[set[AgentID]] = None,
            avail_actions: Optional[Dict[AgentID, List[int]]] = None,
    ) -> None:
        """按样本类型写入一次环境转移，隔离单智能体和多智能体数据布局。"""
        if isinstance(episode_transition, SAEpisodeTransition):
            assert len(agent_ids) == 1
            agent_id = agent_ids[0]
            episode_transition.add_step(
                obs=obs[agent_id],
                action=actions[agent_id],
                last_action=last_actions.get(agent_id),
                reward=rewards[agent_id],
                next_obs=next_obs[agent_id],
                terminated=terminations[agent_id],
                truncated=truncations[agent_id],
                avail_actions=avail_actions_next.get(agent_id),
                log_prob=log_prob,
                value=value,
                metadata=metadata,
            )
            return

        if isinstance(log_prob, dict):
            metadata = {**(metadata or {}), "behavior_log_probs": log_prob}
        episode_transition.add_step(
            obs={agent_id: obs[agent_id] for agent_id in agent_ids},
            action={agent_id: actions.get(agent_id) for agent_id in agent_ids},
            last_action={agent_id: last_actions.get(agent_id) for agent_id in agent_ids},
            reward={agent_id: rewards[agent_id] for agent_id in agent_ids},
            next_obs={agent_id: next_obs[agent_id] for agent_id in agent_ids},
            termination={agent_id: terminations[agent_id] for agent_id in agent_ids},
            truncation={agent_id: truncations[agent_id] for agent_id in agent_ids},
            state=state,
            next_state=next_state,
            avail_actions_next={agent_id: avail_actions_next[agent_id] for agent_id in agent_ids},
            avail_actions=(
                {agent_id: avail_actions[agent_id] for agent_id in agent_ids}
                if avail_actions is not None else None
            ),
            metadata=metadata,
            active_agent_ids=active_agent_ids,
            next_active_agent_ids=next_active_agent_ids,
        )

    @staticmethod
    def _collect_rollout_fields(agent, episode_info, active_agent_ids):
        requests = (episode_info.metadata or {}).get("batch_sample_sources", {}).get(agent.name, [])
        if not requests:
            return {}
        required = (episode_info.metadata or {}).get("batch_required_sources", {}).get(agent.name, requests)
        producer = getattr(agent, "consume_rollout_fields", None)
        if not callable(producer):
            if required:
                raise ValueError(f"Agent {agent.name} does not provide required rollout fields {required}")
            return {}
        fields = producer(requests)
        if not isinstance(fields, dict):
            raise TypeError(f"Agent {agent.name} rollout fields must be a dictionary")
        for name in requests:
            if name not in fields and name not in required:
                continue
            values = fields.get(name)
            if not isinstance(values, dict):
                raise ValueError(f"Agent {agent.name} is missing rollout field {name!r}")
            missing = set(active_agent_ids) - values.keys()
            if missing:
                raise ValueError(
                    f"Agent {agent.name} episode={episode_info.episode_num} "
                    f"field={name!r} missing active agents {sorted(missing)}"
                )
        return copy.deepcopy({name: fields[name] for name in requests if name in fields})

    def _wait_simulation_ready(self, episode_num: int) -> None:
        """
        仿真异常归还令牌后阻塞等待仿真恢复。

        探活使用 env.reset()，因为 reset 覆盖了仿真重启、时钟倍率设置、态势接收和 pause；
        这比单独 ping 中间件更接近真实采样入口。探活成功后才允许角色层继续下一轮调度。
        """
        consecutive_failures = 0
        while True:
            try:
                LOG.info(f"Testing simulation health before next rollout, episode={episode_num}.")
                # 健康检查不是有效采样数据。reset 会触发 Environment.record_sample(reset=True)，
                # 因此探活期间临时关闭录制回调，避免把探活初始态势写入蒙特卡洛样本文件。
                record_caller = getattr(self.env, "record_caller", None)
                self.env.record_caller = None
                try:
                    self.env.reset()
                finally:
                    self.env.record_caller = record_caller
                LOG.info(f"Simulation health check passed, episode={episode_num}.")
                return
            except (SimulationException, *simulation_exception_types()) as exc:
                if self.task_manager.is_sample_completed():
                    return
                consecutive_failures += 1
                LOG.warning(
                    f"Simulation health check failed, episode={episode_num}, "
                    f"consecutive_failures={consecutive_failures}."
                )
                LOG.exception(exc)
                if consecutive_failures >= self.sim_timeout_reacquire_threshold:
                    self._try_reacquire_simulation(episode_num, consecutive_failures)
                    consecutive_failures = 0
                time.sleep(1.0)

    def _try_reacquire_simulation(self, episode_num: int, consecutive_failures: int) -> None:
        """连续健康检查失败后尝试通过SimManager换一个仿真资源。"""
        try:
            reacquired = self.env.sim is not None and self.env.sim.reacquire_simulation()
        except Exception as exc:
            LOG.exception(
                f"Simulation reacquire failed during health check, episode={episode_num}."
            )
            return
        if reacquired:
            LOG.warning(
                f"Simulation reacquired during health check, episode={episode_num}, "
                f"failures={consecutive_failures}."
            )
        else:
            LOG.warning(
                f"Simulation cannot be reacquired during health check, episode={episode_num}; "
                f"current simulation is not managed by SimManager."
            )

    def rolling_aec_episode(
            self,
            episode_info: EpisodeInfo,
            evaluate: bool = False,
            random_sample: bool = False,
            step_limit: Optional[int] = None,
            episode_agent_action: Optional[Action] = None,
            recorder: Optional[RenderGifRecorder] = None,
    ) -> AgentEpisodeTransitions:
        """执行标准AEC原子步采样；每次实体动作都立即产生并记录一次状态转移。"""
        if not isinstance(self.env, AECEnvironment):
            raise EnvironmentTypeException(
                operation="rolling_aec_episode",
                expected="AECEnvironment",
                given=type(self.env).__name__,
            )
        sample_start_time = time.time()
        agent_ep_trans = {
            agent.name: self._create_online_episode_transition(agent)
            for agent in self.step_agent
            if self._records_replay_transition(agent)
        }
        step_limit = self.env.set_episode_step_limit(step_limit)
        self.env.episode_num = episode_info.episode_num
        self.env.stage_name = episode_info.stage_name
        self.env.apply_env_metadata(episode_info.env_metadata)
        refresh_sim = getattr(getattr(self.env, "sim", None), "refresh_simulation", None)
        if callable(refresh_sim):
            refresh_sim()
        self.env.reset(seed=episode_info.seed, options={"step_limit": step_limit, "render": recorder is not None})
        self.last_actions = self.get_init_last_actions()
        self.last_hidden_state = self._get_init_hidden_state()
        episode_hidden_in = self.last_hidden_state.copy()
        self.next_actions = {}
        self.next_hidden_state = {}
        eval_states: list[Any] = []
        eval_actions: list[dict] = []
        resolved_agent_ids: list[AgentID] = []

        def apply_episode_agent_order() -> None:
            """在进入下一次agent_iter前，用Episode Action可选覆盖环境原生选择。"""
            if episode_agent_action is None or not self.env.agents:
                return
            native_next_agent_id = self.env.agent_selection
            # PettingZoo要求已经结束的Agent优先通过step(None)出队，局级调度不能覆盖该游标。
            if (
                    self.env.terminations.get(native_next_agent_id, False)
                    or self.env.truncations.get(native_next_agent_id, False)
            ):
                return
            self.env.agent_selection = self.env.episode_agent_order_resolve(
                episode_agent_action,
                AECAgentOrderContext(
                    resolved_agent_ids=tuple(resolved_agent_ids),
                ),
            )

        # AEC迭代单位是环境AgentID，需先建立到Policy/Step Agent组的唯一路由。
        owner_by_agent_id = {}
        for step_agent in self.step_agent:
            for agent_id in step_agent.agent_list:
                if agent_id in owner_by_agent_id:
                    raise AgentOwnerDuplicateException(
                        agent_id=agent_id,
                        owners=[owner_by_agent_id[agent_id].name, step_agent.name],
                    )
                owner_by_agent_id[agent_id] = step_agent

        # reset先给出环境原生首个行动者；Episode Agent存在时在首次迭代前覆盖一次。
        apply_episode_agent_order()
        current_state = self.env.state()
        atomic_step = 0

        # 使用PettingZoo原生迭代器；env.agents清空后自动结束，物理步上限由环境truncation维护。
        for selected_agent_id in self.env.agent_iter():
            # AgentID真正进入策略路由时才检索owner；检索不到即说明当前顺序解析结果无法执行。
            step_agent = owner_by_agent_id.get(selected_agent_id)
            if step_agent is None:
                raise AgentOwnerNotFoundException(
                    agent_id=selected_agent_id,
                    owned_agent_ids=list(owner_by_agent_id),
                )

            # last(False)只读取当前agent_selection的AEC协议状态，不重复生成观测。
            _, cumulative_reward, terminated, truncated, info = self.env.last(observe=False)
            # 原子Transition的即时奖励在step(action)后从env.rewards结算；这里保留AEC累计奖励语义但不混用。
            # PettingZoo规定已终止Agent必须step(None)才能从AEC队列中清理。
            if terminated or truncated:
                self.env.step(None)
                # 死亡清理不是策略决策，不写入历史；清理后再为下一次迭代准备选择。
                apply_episode_agent_order()
                continue

            # get_obs()是本原子步唯一观测快照，同时用于网络决策和Transition组装。
            current_obs = self.env.get_obs()
            observation = current_obs[selected_agent_id]
            step_on_policy_stats = (None, None)
            # 策略决策：只对本原子步选中的AgentID运行网络，不为其他槽位生成伪动作。
            if random_sample:
                next_action, next_hidden_state = self.get_random_agent_actions(
                    step_agent, [selected_agent_id]
                )
            else:
                action_param = self._episode_action_param(episode_info, step_agent)
                next_action, next_hidden_state = self.get_actions(
                    agent=step_agent,
                    agent_list=[selected_agent_id],
                    obs={selected_agent_id: observation},
                    epsilon=action_param.epsilon,
                    evaluate=evaluate,
                    temperature=action_param.temperature,
                    deterministic=action_param.deterministic,
                )
            if hasattr(step_agent, "consume_on_policy_stats"):
                step_on_policy_stats = step_agent.consume_on_policy_stats()
            step_rollout_fields = self._collect_rollout_fields(step_agent, episode_info, [selected_agent_id])

            decision_masks = self._get_avail_actions_next()
            action = next_action[selected_agent_id]
            # 原子步进：单Agent动作立即产生X_k -> X_{k+1}，AEC路径不使用Parallel动作缓存。
            self.env.step(action)
            # Context只记录真正执行过策略动作的AgentID；随后覆盖下一次agent_iter的选择。
            resolved_agent_ids.append(selected_agent_id)
            apply_episode_agent_order()
            # 采样元数据记录最终将被agent_iter读取的选择，而不是env.step产生的原生候选。
            next_active_agent_id = (
                self.env.agent_selection
                if self.env.agents else None
            )
            next_state = self.env.state()
            next_obs = self.env.get_obs()
            rewards = {agent_id: float(self.env.rewards.get(agent_id, 0.0)) for agent_id in self.env.possible_agents}
            terminations = {
                agent_id: bool(self.env.terminations.get(agent_id, False))
                for agent_id in self.env.possible_agents
            }
            truncations = {
                agent_id: bool(self.env.truncations.get(agent_id, False))
                for agent_id in self.env.possible_agents
            }
            infos = {agent_id: self.env.infos.get(agent_id, {}) for agent_id in self.env.possible_agents}
            self._episode_steps += 1
            self._episode_metric_collector.add_infos(infos)
            avail_actions_next = self._get_avail_actions_next()
            if recorder is not None:
                recorder.capture(self.env.render)
            eval_actions.append({selected_agent_id: action})
            eval_states.append(self._get_post_state_raw_data())

            # 组装样本：保留全量Agent维度，用active_agent_ids标记本步真正行动者。
            for policy_agent in self.step_agent:
                transition = agent_ep_trans.get(policy_agent.name)
                if transition is None:
                    continue
                policy_agent_ids = list(policy_agent.agent_list)
                if isinstance(transition, SAEpisodeTransition):
                    if selected_agent_id not in policy_agent_ids:
                        continue
                    log_prob, value = step_on_policy_stats
                    self._add_online_transition_step(
                        episode_transition=transition,
                        agent_ids=[selected_agent_id],
                        obs=current_obs,
                        actions={selected_agent_id: action},
                        last_actions=self.last_actions,
                        rewards=rewards,
                        next_obs=next_obs,
                        terminations=terminations,
                        truncations=truncations,
                        state=current_state,
                        next_state=next_state,
                        avail_actions_next=avail_actions_next,
                        log_prob=log_prob,
                        value=value,
                        metadata={
                            "episode_num": episode_info.episode_num,
                            "atomic_step": atomic_step + 1,
                            "active_agent_id": selected_agent_id,
                            "next_active_agent_id": next_active_agent_id,
                            "episode_agent_action": episode_agent_action,
                            **({"batch_fields": step_rollout_fields} if step_rollout_fields else {}),
                        },
                    )
                    continue

                transition.add_step(
                    obs={agent_id: current_obs[agent_id] for agent_id in policy_agent_ids},
                    action=(
                        {selected_agent_id: action}
                        if selected_agent_id in policy_agent_ids else {}
                    ),
                    last_action={agent_id: self.last_actions.get(agent_id) for agent_id in policy_agent_ids},
                    reward={agent_id: rewards[agent_id] for agent_id in policy_agent_ids},
                    next_obs={agent_id: next_obs[agent_id] for agent_id in policy_agent_ids},
                    termination={agent_id: terminations[agent_id] for agent_id in policy_agent_ids},
                    truncation={agent_id: truncations[agent_id] for agent_id in policy_agent_ids},
                    state=current_state,
                    next_state=next_state,
                    avail_actions_next={agent_id: avail_actions_next[agent_id] for agent_id in policy_agent_ids},
                    active_agent_ids={selected_agent_id} & set(policy_agent_ids),
                    next_active_agent_ids=(
                        {next_active_agent_id} & set(policy_agent_ids)
                        if next_active_agent_id is not None
                        and not all(terminations[agent_id] for agent_id in policy_agent_ids)
                        else set()
                    ),
                    avail_actions={agent_id: decision_masks[agent_id] for agent_id in policy_agent_ids},
                    metadata={
                        "behavior_log_probs": (step_on_policy_stats[0] if selected_agent_id in policy_agent_ids
                                               and isinstance(step_on_policy_stats[0], dict) else {}),
                        "episode_num": episode_info.episode_num,
                        "atomic_step": atomic_step + 1,
                        "active_agent_id": selected_agent_id,
                        "next_active_agent_id": next_active_agent_id,
                        "episode_agent_action": episode_agent_action,
                        **({"batch_fields": step_rollout_fields}
                           if step_rollout_fields and selected_agent_id in policy_agent_ids else {}),
                    },
                )

            current_state = next_state
            # 原子状态滚动：只更新当前行动Agent的last_action和hidden state。
            self.last_actions[selected_agent_id] = action
            self.last_hidden_state.update(next_hidden_state)
            self.next_actions = {}
            self.next_hidden_state = {}
            atomic_step += 1

        for step_agent in self.step_agent:
            transition = agent_ep_trans.get(step_agent.name)
            if isinstance(transition, EpisodeTransition):
                transition.set_hidden(
                    {agent_id: episode_hidden_in[agent_id] for agent_id in step_agent.agent_list},
                    {agent_id: self.last_hidden_state[agent_id] for agent_id in step_agent.agent_list},
                )

        self.total_rewards = {
            agent_name: transition.episode_reward()
            for agent_name, transition in agent_ep_trans.items()
        }
        if self._replay_write_enabled(episode_info) and agent_ep_trans:
            self._submit_replay_transitions(agent_ep_trans, episode_info, sample_start_time)
        if recorder is not None:
            record_path = recorder.finish(self._render_record_filename(episode_info))
            if record_path is not None:
                LOG.info(f"Render GIF saved: {record_path}")
                self._pending_episode_artifacts.append(Artifact(ArtifactType.RENDER_GIF, str(record_path)))
                self._put_render_gif(episode_info, record_path)
        self.post_episode(eval_states=eval_states, eval_actions=eval_actions, ep_trans=agent_ep_trans)
        # AEC 不支持显式分组；统一列表返回，旧隐式单组包装为长度 1 的列表。
        return {
            agent_name: [transition]
            for agent_name, transition in agent_ep_trans.items()
        }

    @staticmethod
    @timed("rollout.prepare_for_replay")
    def _prepare_replay_transitions(agent_ep_trans) -> None:
        """
        在采样进程局末完成列式张量化。

        push/sample公开契约保持不变；不规则第三方Transition由ReplayBuffer兼容编码兜底。
        值支持 dict[AgentName, Transition | list[Transition]]，逐条预封装。
        """
        for transitions in agent_ep_trans.values():
            items = transitions if isinstance(transitions, list) else [transitions]
            for transition in items:
                prepare = getattr(transition, "prepare_for_replay", None)
                if not callable(prepare):
                    continue
                try:
                    prepare()
                except BatchEncodingUnsupported as exc:
                    LOG.warning(
                        f"Transition replay prepacking fallback: "
                        f"type={type(transition).__name__}, reason={exc}"
                    )

    def offline_rollout_one_episode(self, episode: RecordedEpisode, *, write_to_replay: bool = True):
        """
        从单局历史采样记录构造EpisodeTransition并写入ReplayBuffer。
    
        历史样本的文件格式和完整性由MonteCarloMethod校验；这里只借用环境业务接口
        fake_reset/fake_step，把一局记录恢复成EpisodeTransition并写入ReplayBuffer。
        离线动作来自历史记录，不重新调用Agent；这里的Agent只负责提供Learner序列批处理需要的
        初始hidden占位。
        """
        ep_trans = EpisodeTransition()
        hidden_in = self._get_init_hidden_state()
        hidden_out = hidden_in
        last_actions: AgentLastAction = {agent_id: None for agent_id in self.env.possible_agents}
        records = episode.records

        if episode.has_pre_state:
            # 新格式每条record自带pre_state，可以逐条fake_reset后直接恢复该步transition。
            for idx, record in enumerate(records):
                current_obs, _ = self.env.fake_reset(record)
                current_state = self.env.state()
                current_obs, current_state, last_actions, _ = self._next_offline_iteration(
                    ep_trans=ep_trans,
                    obs=current_obs,
                    state=current_state,
                    last_actions=last_actions,
                    record=record,
                    record_index=idx,
                    fallback_without_pre_state=False,
                )
        else:
            # 旧格式只有当前state，第一条作为初始态势，从第二条开始恢复相邻状态转移。
            current_obs, _ = self.env.fake_reset(records[0])
            current_state = self.env.state()
            for idx, record in enumerate(records[1:], start=1):
                current_obs, current_state, last_actions, episode_done = self._next_offline_iteration(
                    ep_trans=ep_trans,
                    obs=current_obs,
                    state=current_state,
                    last_actions=last_actions,
                    record=record,
                    record_index=idx,
                    fallback_without_pre_state=True,
                )
                if episode_done:
                    break

        ep_trans.set_hidden(hidden_in, hidden_out)
        if ep_trans.seq_len == 0:
            return False
        LOG.info(f"Pushing episode {episode.episode_id} rewards: {ep_trans.episode_reward()}")
        offline_agent_name = self.offline_agent.name
        if not write_to_replay:
            return ep_trans
        # 兼容旧 SampleLearningEvaluationRole 的批量灌池入口。
        self.replay_buffer.load_replay_buffer(
            iter([ep_trans]),
            notify_learning=False,
            agent_name=offline_agent_name,
        )
        return True

    def _next_offline_iteration(
            self,
            ep_trans: EpisodeTransition,
            obs: Dict[AgentID, ObsType],
            state: StateType,
            last_actions: AgentLastAction,
            record: dict[str, Any],
            record_index: int,
            fallback_without_pre_state: bool,
    ) -> Tuple[Dict[AgentID, ObsType], StateType, AgentLastAction, bool]:
        """推进离线样本恢复主循环的下一次迭代，并把本轮transition写入EpisodeTransition。"""
        # fake_step只把历史record解析成环境标准输出，不和真实仿真交互。
        next_obs, rewards, terminations, truncations, _ = self.env.fake_step(record)
        next_state = self.env.state()
        metadata = {
            "offline_record": True,
            "record_step": record.get("step", record_index + 1),
            "episode_step_limit": record.get("episode_step_limit", record.get("max_step")),
        }
        if fallback_without_pre_state:
            metadata["fallback_without_pre_state"] = True
        ep_trans.add_step(
            obs=obs,
            action=self.env.current_actions,
            last_action=last_actions,
            reward=rewards,
            next_obs=next_obs,
            termination=terminations,
            truncation=truncations,
            state=state,
            next_state=next_state,
            avail_actions_next=self._get_avail_actions_next(),
            metadata=metadata,
            active_agent_ids=(
                set(self.env.current_actions)
                if isinstance(self.env, AECEnvironment)
                else None
            ),
        )
        episode_done = any(terminations.values()) or any(truncations.values())
        return next_obs, next_state, self.env.current_actions.copy(), episode_done

    def _get_avail_actions_next(self):
        # 样本结构按固定 possible_agents 槽位写入。KAZ 这类环境中 agent 死亡后
        # env.agents 会缩小，但训练序列仍需要为死亡实体保留下一步动作掩码占位。
        masks = {}
        for agent_id in self.env.possible_agents:
            try:
                masks[agent_id] = self.env.get_avail_agent_actions(agent_id)
            except (KeyError, ValueError):
                action_space = self.env.action_space(agent_id)
                masks[agent_id] = [0] * int(getattr(action_space, "n", 0))
        return masks

    def post_episode(self, *args, **kwargs):
        """ 一局完成后的动作，项目定制化接口 """
        pass

    def _get_agent_actions_mask(self, init=False):
        action_mask = {}
        for agent_id in self.env.possible_agents:
            avail_action = self.env.get_avail_agent_actions(agent_id)
            action_mask[agent_id] = avail_action
        return action_mask


def _has_available_action(mask):
    """None means bounded continuous/unconstrained; categorical heads must each be usable."""
    if mask is None:
        return True
    if isinstance(mask, dict):
        return bool(mask) and all(_has_available_action(head) for head in mask.values())
    return any(mask)
