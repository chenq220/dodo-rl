from __future__ import annotations

import copy
import torch
import numpy as np
from typing import Dict, List, Optional, Generic, Union
from common.registry import extensible
from sample.model import EpisodeBatch, SABatchTensorData, SampledEpisodes
from sample.replay.replay_payload import PreparedReplayEpisode, prepare_replay_episode
from sample.replay.replay_perf import timed
from common.types import (
    ObsType, StateType, AgentID, ActionType,
    AgentHiddenOut, AgentHiddenIn, AgentLastAction,
)
from common.logger import init_logger

LOG = init_logger('sample_transfer')


@extensible
class EpisodeTransition(Generic[AgentID, ActionType, StateType, ObsType]):
    """
    多智能体单个 episode 的完整序列转移数据，按 agent 维度保存 obs/action/reward 等时序。

    它是框架内部最基础的训练样本单元：RolloutSampleWorker 逐步写入，ReplayBuffer 按 episode 存储，
    Learner 再把多个 EpisodeTransition 转换成算法需要的张量批次。
    """

    def __init__(
            self,
            agent_ids: Optional[tuple[AgentID, ...]] = None,
            group_index: int = 0,
            action_spaces: Optional[Dict] = None,
    ):
        # 本条组轨迹的固定成员顺序；就是 QMIX 槽位顺序。None/空表示旧单组样本，
        # 首个 add_step 依据 obs 字典插入顺序固定一次。
        self.agent_ids: tuple[AgentID, ...] = tuple(agent_ids) if agent_ids else ()
        # 诊断用组序号；仅用于日志与传输条目定位，不是业务身份。
        self.group_index: int = group_index
        self.action_space_specs = {
            key: space.to_dict() for key, space in (action_spaces or {}).items() if space is not None
        }
        # 局末由采样进程生成的列式回放载荷；不改变公开Transition字段。
        self._prepared_replay_episode: Optional[PreparedReplayEpisode] = None
        # 兼容旧结构的隐藏状态序列缓存，键为 agent id。
        self.hidden_states: Dict[AgentID, List[torch.Tensor]] = {}
        # 各 agent 观测序列 O_t，键为 agent id，值为按时间排列的观测向量列表。
        self.obs: Dict[AgentID, List[ObsType]] = {}
        # 当前动作：旧离散保留整数；新空间保存包含物理值和布局签名的动作记录。
        self.actions: Dict[AgentID, List[int | dict | None]] = {}
        # 上一动作：旧离散首步使用 0；新空间首步使用 None。
        self.last_actions: Dict[AgentID, List[int | dict | None]] = {}
        # 各 agent 奖励序列 R_t，保留明细奖励以兼容非公共奖励算法或调试。
        self.rewards: Dict[AgentID, List[float]] = {}
        # 各 agent 下一观测序列 O_{t+1}。
        self.next_obs: Dict[AgentID, List[ObsType]] = {}
        # 各 agent 环境真实终止标志，terminated=True 表示环境逻辑完成。
        self.terminations: Dict[AgentID, List[bool]] = {}
        # 各 agent 截断标志，truncated=True 表示达到步数上限等外部停止条件。
        self.truncations: Dict[AgentID, List[bool]] = {}
        # episode 开始或 batch 前向使用的 RNN 输入隐藏状态。
        self.hidden_in: AgentHiddenIn = {}
        # episode 结束或 batch 前向产生的 RNN 输出隐藏状态。
        self.hidden_out: AgentHiddenOut = {}
        # episode 编号，当前主要用于调试和离线样本回放。
        self.episode_num: int = 0
        # 全局状态 S_t 序列，元素通常是 flatten 后的一维向量。
        self.states: List[StateType] = []
        # 下一全局状态 S_{t+1} 序列，元素通常是 flatten 后的一维向量。
        self.next_states: List[StateType] = []
        # 每一步附加元数据，例如 episode_info、离线记录步号、episode_step_limit。
        self.metadata: List[Dict] = []
        # episode 有效时间步长度；所有 agent 的序列长度必须与该值一致。
        self.seq_len = 0
        # 公共奖励序列，当前按每步所有 agent 奖励求和，QMIX/VDN 可直接使用。
        self.step_rewards: List[float] = []
        # 下一步可用动作掩码序列；每步保存 agent_id -> list[int]，1=可用，0=不可用。
        self.avail_actions_next: List[Dict[AgentID, List[int]]] = []
        # 每个时间步哪些固定Agent槽位真正执行了动作。Parallel默认全1；AEC通常只有一个槽位为1。
        self.active_agent_mask: Dict[AgentID, List[float]] = {}
        # 执行动作后的下一时刻仍可继续行动的固定Agent槽位；用于局部死亡和终局bootstrap。
        self.next_active_agent_mask: Dict[AgentID, List[float]] = {}
        # 每步当前决策帧的可用动作掩码序列；与 avail_actions_next 一起构成前后帧约束。
        self.avail_actions: List[Dict[AgentID, List[int]]] = []
        # 该轨迹是否从真实局首开始；回放切片会显式设为 False。
        self.starts_with_episode: bool = True
        # 中途切片所需的同组同局历史，完整轨迹没有前缀。
        self.burn_in: Optional[EpisodeTransition] = None

    def __setstate__(self, state):
        """读取历史轨迹时补齐当时尚未声明的可选字段。"""
        restored = dict(state)
        restored.setdefault("agent_ids", tuple(restored.get("obs", {})))
        restored.setdefault("action_space_specs", {})
        restored.setdefault("active_agent_mask", {})
        restored.setdefault("next_active_agent_mask", {})
        restored.setdefault("hidden_in", {})
        restored.setdefault("hidden_out", {})
        restored.setdefault("avail_actions", [])
        restored.setdefault("avail_actions_next", [])
        restored.setdefault("group_index", 0)
        restored.setdefault("starts_with_episode", True)
        restored.setdefault("burn_in", None)
        restored.setdefault("_prepared_replay_episode", None)
        self.__dict__.update(restored)

    def validate_activity_masks(self, agent_ids):
        """空字典表示未记录；已记录的掩码必须覆盖全部槽位和有效步。"""
        for name, masks in (("active_agent_mask", self.active_agent_mask),
                            ("next_active_agent_mask", self.next_active_agent_mask)):
            if not masks:
                continue
            if set(masks) != set(agent_ids):
                raise ValueError(f"trajectory {self.episode_num} {name} must cover all agent slots")
            for aid, values in masks.items():
                if len(values) != self.seq_len:
                    raise ValueError(
                        f"trajectory {self.episode_num} {name}[{aid!r}] has {len(values)} steps, "
                        f"expected {self.seq_len}"
                    )

    def add_step(
            self,
            obs: Dict[AgentID, ObsType],
            action: Dict[AgentID, ActionType],
            last_action: AgentLastAction,
            reward: Dict[AgentID, float],
            next_obs: Dict[AgentID, ObsType],
            termination: Dict[AgentID, bool],
            truncation: Dict[AgentID, bool],
            state: StateType,
            next_state: StateType,
            avail_actions_next: Optional[Dict[AgentID, List[int]]],
            metadata: Optional[Dict] = None,
            active_agent_ids: Optional[set[AgentID]] = None,
            next_active_agent_ids: Optional[set[AgentID]] = None,
            avail_actions: Optional[Dict[AgentID, List[int]]] = None,
    ):
        """
        添加单步数据；输入成员必须全部属于本条轨迹的固定成员集合。
        """
        self._prepared_replay_episode = None
        # 初始化智能体列表（仅第一步）；旧样本依据 obs 插入顺序固定一次成员顺序。
        if self.seq_len == 0:
            if not self.agent_ids:
                self.agent_ids = tuple(obs.keys())
            unknown_members = set(obs) - set(self.agent_ids)
            if unknown_members:
                from common.exception import ReplayBatchShapeException
                raise ReplayBatchShapeException(
                    agent_name="",
                    detail=(
                        f"add_step 收到了不属于本组成员的 AgentID={sorted(unknown_members)}，"
                        f"成员={list(self.agent_ids)}。"
                    ),
                )
            for agent in obs.keys():
                self.obs[agent] = []
                self.actions[agent] = []
                self.last_actions[agent] = []
                self.rewards[agent] = []
                self.next_obs[agent] = []
                self.terminations[agent] = []
                self.truncations[agent] = []
                self.active_agent_mask[agent] = []
                self.next_active_agent_mask[agent] = []

        active_agent_ids = set(obs) if active_agent_ids is None else set(active_agent_ids)
        next_active_agent_ids = (
            set(active_agent_ids)
            if next_active_agent_ids is None
            else set(next_active_agent_ids)
        )

        # 添加单步数据到各智能体的序列
        for agent in obs.keys():
            action_value = action.get(agent)
            spec = self.action_space_specs.get(agent)
            action_idx = _record_action(action_value, spec)
            previous_action = last_action.get(agent)
            # 第一步没有真实上一动作。记录0保证后续任何one-hot路径都有合法索引；
            # 循环控制器在 t=0 会自行使用全零上一动作输入，不依赖这个占位值。
            last_action_idx = _record_action(previous_action, spec)
            self.obs[agent].append(obs[agent])
            self.actions[agent].append(action_idx)
            self.last_actions[agent].append(last_action_idx)
            self.rewards[agent].append(float(reward.get(agent, 0.0)))
            self.next_obs[agent].append(next_obs[agent])
            self.terminations[agent].append(bool(termination.get(agent, False)))
            self.truncations[agent].append(bool(truncation.get(agent, False)))
            self.active_agent_mask[agent].append(1.0 if agent in active_agent_ids else 0.0)
            self.next_active_agent_mask[agent].append(
                1.0 if agent in next_active_agent_ids else 0.0
            )
        # 添加全局状态
        self.states.append(state.flatten())
        self.next_states.append(next_state.flatten())
        self.metadata.append(metadata)
        step_reward = sum(reward.values())
        self.step_rewards.append(step_reward)
        if avail_actions_next is not None:
            self.avail_actions_next.append(avail_actions_next)
        if avail_actions is not None:
            self.avail_actions.append(avail_actions)
        # 更新序列长度
        self.seq_len += 1

    def set_hidden(
            self, hidden_in: AgentHiddenIn, hidden_out: AgentHiddenOut
    ):
        self._prepared_replay_episode = None
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out

    def episode_agent_reward(self) -> Dict[AgentID, float]:
        """返回每个 agent 的累计奖励（用于兼容/调试）"""
        return {agent_id: sum(v) for agent_id, v in self.rewards.items()}

    def episode_reward(self) -> float:
        # return sum([v for v in self.episode_agent_reward().values()])
        """
        返回 episode 总奖励（QMIX 格式）
        = sum(每步所有 agent 的奖励之和)
        与 QMIX/env.py 的逻辑对齐：每步 r=sum(rew)，episode 总奖励=sum(r)
        """
        return sum(self.step_rewards)

    @staticmethod
    def _assemble_batch(episodes, *, learner_name, schema, context):
        from algos.learners.batch_schema import assemble_episodes
        strategy = episodes.sample_strategy if isinstance(episodes, SampledEpisodes) else None
        return assemble_episodes(episodes, schema, agent=context.agent, device=context.device,
                                 sample_strategy=strategy)

    def to_replay_step_payloads(self):
        """
        按整局导出ReplayBuffer需要的逐Step值。

        该接口只组织字段，不创建逐Step小Tensor；prepare_for_replay会在采样进程把相同字段
        一次性堆叠成第一维为Step的列式张量。
        """
        dict_fields = (
            "hidden_states",
            "obs",
            "actions",
            "last_actions",
            "rewards",
            "next_obs",
            "terminations",
            "truncations",
            "active_agent_mask",
            "next_active_agent_mask",
        )
        list_fields = (
            "states",
            "next_states",
            "metadata",
            "step_rewards",
            "avail_actions_next",
            "avail_actions",
        )
        step_payloads = [{"fields": {}} for _ in range(self.seq_len)]
        step_field_layout = {}

        for field_name in dict_fields:
            values_by_agent = getattr(self, field_name)
            agent_ids = tuple(values_by_agent)
            step_field_layout[field_name] = ("dict_list", agent_ids)
            for step in range(self.seq_len):
                step_payloads[step]["fields"][field_name] = {
                    agent_id: values[step]
                    for agent_id, values in values_by_agent.items()
                    if step < len(values)
                }

        for field_name in list_fields:
            values = getattr(self, field_name)
            step_field_layout[field_name] = ("list", ())
            for step in range(min(self.seq_len, len(values))):
                step_payloads[step]["fields"][field_name] = values[step]

        if self.seq_len and self.hidden_in:
            for step in range(self.seq_len):
                step_payloads[step]["hidden_in"] = self.hidden_in if step == 0 else None
        if self.seq_len and self.hidden_out:
            for step in range(self.seq_len):
                step_payloads[step]["hidden_out"] = (
                    self.hidden_out if step == self.seq_len - 1 else None
                )

        known_fields = {
            *dict_fields,
            *list_fields,
            "hidden_in",
            "hidden_out",
            "seq_len",
            "episode_num",
            "starts_with_episode",
            "_prepared_replay_episode",
        }
        static_fields = {
            field_name: copy.deepcopy(value)
            for field_name, value in vars(self).items()
            if field_name not in known_fields
        }
        return step_payloads, step_field_layout, static_fields

    @timed("transition.prepare_for_replay")
    def prepare_for_replay(self) -> PreparedReplayEpisode:
        """在采样进程完成整局列式张量化，ReplayBuffer只负责存储和建立索引。"""
        if self._prepared_replay_episode is None:
            self._prepared_replay_episode = prepare_replay_episode(
                *self.to_replay_step_payloads()
            )
        return self._prepared_replay_episode

    @property
    def prepared_replay_episode(self) -> Optional[PreparedReplayEpisode]:
        return self._prepared_replay_episode

    def to_bc_json(self):
        obs = np.array([v for v in self.next_obs.values()]).transpose(1, 0, 2)
        state = np.array([v for v in self.next_states])
        actions = np.array([v for v in self.actions.values()]).T
        filled = np.ones((obs.shape[0], 1), dtype=np.float32)
        json_data = {
            "obs": obs.tolist(),
            "state": state.tolist(),
            "actions": actions.tolist(),
            "filled": filled.tolist(),
        }
        return json_data

    @staticmethod
    def _get_batch_dims(
            agent_ids: List[AgentID], ep: EpisodeTransition
    ) -> tuple[int, int]:
        """返回兼容离散批次需要的 state_dim 和 obs_dim。"""
        first_agent = agent_ids[0]
        obs_sample = ep.obs[first_agent]
        state_dim: int = np.prod(ep.states[0].shape).item()
        obs_dim: int = np.prod(obs_sample[0].shape).item()
        return state_dim, obs_dim

    @staticmethod
    def _infer_action_dim(
            agent_ids: List[AgentID],
            episode_trans: List[EpisodeTransition],
            action_dim: Optional[int] = None,
    ) -> int:
        """推断动作维度；优先使用外部动作表维度，其次从可用动作掩码和历史动作索引推断。"""
        for ep in episode_trans:
            if any(spec["kind"] != "discrete" for spec in ep.action_space_specs.values()):
                raise ValueError("Discrete replay adapter supports only discrete action spaces")
        if action_dim is not None:
            return int(action_dim)

        inferred_action_dim = 0
        for ep in episode_trans:
            for step_avail_actions in ep.avail_actions_next:
                for agent_id in agent_ids:
                    avail = step_avail_actions.get(agent_id)
                    if avail is not None:
                        inferred_action_dim = max(inferred_action_dim, len(avail))
            for agent_id in agent_ids:
                if ep.actions.get(agent_id):
                    inferred_action_dim = max(inferred_action_dim, max(ep.actions[agent_id]) + 1)
        return max(inferred_action_dim, 1)

    @staticmethod
    @timed("transition.to_batch")
    def to_batch(
            agent_ids: List[AgentID],
            episode_trans: List[EpisodeTransition],
            max_seq_len: Optional[int] = None,
            action_dim: Optional[int] = None,
            device=torch.device("cpu"),
            expected_n_agents: Optional[int] = None,
    ) -> EpisodeBatch:
        """
        Compatibility helper for callers of the previous discrete batch layout.
        New training paths use the Learner's batch_schema via _assemble_batch.

        每条轨迹读取自己的 agent_ids 作为固定槽位顺序；未显式声明成员的旧样本
        回退到入参 agent_ids。expected_n_agents 给定时校验各条轨迹组宽一致，
        不同轨迹的真实 AgentID 可以不同。
        """
        # IDEA: 后续可以评估在采样或回放池阶段生成更接近EpisodeBatch的缓存结构，
        # 减少学习前二次组装；但职责边界还需明确，不能把工程格式转换负担压给Learner。
        if not episode_trans:
            raise ValueError("episode trans should not be empty")

        default_agent_ids = list(agent_ids)
        n_agents = int(expected_n_agents) if expected_n_agents is not None else len(default_agent_ids)
        if n_agents <= 0:
            raise ValueError("to_batch requires a positive n_agents")

        batch_member_ids: List[Tuple[AgentID, ...]] = []
        for ep in episode_trans:
            member_ids = tuple(ep.agent_ids) or tuple(default_agent_ids)
            if len(member_ids) != n_agents:
                raise ValueError(
                    f"episode transition member width {len(member_ids)} does not match "
                    f"expected n_agents={n_agents}, members={list(member_ids)}"
                )
            ep.validate_activity_masks(member_ids)
            batch_member_ids.append(member_ids)

        batch_size = len(episode_trans)
        state_dim, obs_dim = EpisodeTransition._get_batch_dims(
            list(batch_member_ids[0]), episode_trans[0]
        )
        transition_seq_len = max_seq_len or max(ep.seq_len for ep in episode_trans)
        frame_count = transition_seq_len + 1
        action_dim = EpisodeTransition._infer_action_dim(
            list(batch_member_ids[0]), episode_trans, action_dim,
        )

        has_burn_in = any(ep.burn_in is not None for ep in episode_trans)
        frames = (batch_size, frame_count)
        batch = EpisodeBatch({
            "obs": torch.zeros(*frames, n_agents, obs_dim, device=device),
            "state": torch.zeros(*frames, state_dim, device=device),
            "actions": torch.zeros(*frames, n_agents, 1, dtype=torch.long, device=device),
            "actions_onehot": torch.zeros(*frames, n_agents, action_dim, device=device),
            "avail_actions": torch.zeros(*frames, n_agents, action_dim, device=device),
            "reward": torch.zeros(*frames, 1, device=device),
            "terminated": torch.zeros(*frames, 1, device=device),
            "filled": torch.zeros(*frames, 1, dtype=torch.long, device=device),
            "active_agents": torch.zeros(*frames, n_agents, 1, device=device),
        })
        if has_burn_in:
            batch.data["initial_actions_onehot"] = torch.zeros(batch_size, n_agents, action_dim, device=device)
        obs_np = np.zeros((batch_size, frame_count, n_agents, obs_dim), dtype=np.float32)
        state_np = np.zeros((batch_size, frame_count, state_dim), dtype=np.float32)
        actions_np = np.zeros((batch_size, frame_count, n_agents, 1), dtype=np.int64)
        actions_onehot_np = np.zeros((batch_size, frame_count, n_agents, action_dim), dtype=np.float32)
        # 未显式记录掩码的位置默认所有动作可用，保持旧转换逻辑。
        avail_actions_np = np.ones((batch_size, frame_count, n_agents, action_dim), dtype=np.float32)
        reward_np = np.zeros((batch_size, frame_count, 1), dtype=np.float32)
        terminated_np = np.zeros((batch_size, frame_count, 1), dtype=np.float32)
        filled_np = np.zeros((batch_size, frame_count, 1), dtype=np.int64)
        active_agents_np = np.zeros((batch_size, frame_count, n_agents, 1), dtype=np.float32)
        initial_actions_np = np.zeros((batch_size, n_agents, action_dim), dtype=np.float32)

        for episode_idx, ep in enumerate(episode_trans):
            seq_len = min(ep.seq_len, transition_seq_len)
            if seq_len == 0:
                continue
            member_ids = batch_member_ids[episode_idx]
            prefix = ep.burn_in
            if prefix is not None:
                prefix_members = tuple(prefix.agent_ids) or tuple(default_agent_ids)
                if prefix_members != member_ids or prefix.seq_len <= 0:
                    raise ValueError("burn-in history must have the same members and nonempty steps")
                for slot, agent_id in enumerate(member_ids):
                    active = prefix.active_agent_mask.get(agent_id)
                    if active is None or active[-1]:
                        previous_action = prefix.actions[agent_id][-1]
                        if not 0 <= previous_action < action_dim:
                            raise ValueError("burn-in previous action is outside the action space")
                        initial_actions_np[episode_idx, slot, previous_action] = 1.0

            state_np[episode_idx, :seq_len] = np.asarray(
                ep.states[:seq_len], dtype=np.float32
            ).reshape(seq_len, state_dim)
            state_np[episode_idx, seq_len] = np.asarray(
                ep.next_states[seq_len - 1], dtype=np.float32
            ).reshape(state_dim)

            # filled只标记真实transition，最后一帧只用于bootstrap，不参与transition loss。
            filled_np[episode_idx, :seq_len, 0] = 1
            reward_np[episode_idx, :seq_len, 0] = np.asarray(ep.step_rewards[:seq_len], dtype=np.float32)

            next_active_masks = ep.next_active_agent_mask
            for t in range(seq_len):
                next_is_active = []
                for agent_id in member_ids:
                    masks = next_active_masks.get(agent_id)
                    if masks is not None and t < len(masks):
                        next_is_active.append(bool(masks[t]))
                        continue
                    # 旧样本兼容：优先根据下一动作掩码判断Parallel槽位是否仍存活。
                    avail = (
                        ep.avail_actions_next[t].get(agent_id)
                        if t < len(ep.avail_actions_next) else None
                    )
                    next_is_active.append(bool(avail and any(avail)))
                all_agents_terminated = all(
                    bool(ep.terminations[agent_id][t]) for agent_id in member_ids
                )
                # 单机死亡不是QMIX团队终止；成功或下一时刻已无任何存活槽位才终止bootstrap。
                terminated_np[episode_idx, t, 0] = float(
                    all_agents_terminated or not any(next_is_active)
                )

            for agent_idx, agent_id in enumerate(member_ids):
                obs_np[episode_idx, :seq_len, agent_idx] = np.asarray(
                    ep.obs[agent_id][:seq_len], dtype=np.float32
                ).reshape(seq_len, obs_dim)
                obs_np[episode_idx, seq_len, agent_idx] = np.asarray(
                    ep.next_obs[agent_id][seq_len - 1], dtype=np.float32
                ).reshape(obs_dim)

                action_index = np.clip(
                    np.asarray(ep.actions[agent_id][:seq_len], dtype=np.int64),
                    0,
                    action_dim - 1,
                )
                actions_np[episode_idx, :seq_len, agent_idx, 0] = action_index
                actions_onehot_np[episode_idx, np.arange(seq_len), agent_idx, action_index] = 1.0
                masks = ep.active_agent_mask.get(agent_id)
                if masks is None:
                    # 旧样本未记录active mask时按Parallel语义处理，所有真实时间步均活跃。
                    active_agents_np[episode_idx, :seq_len, agent_idx, 0] = 1.0
                else:
                    # 真实transition使用采样时记录的固定槽位活跃掩码。
                    active_agents_np[episode_idx, :seq_len, agent_idx, 0] = np.asarray(
                        masks[:seq_len], dtype=np.float32
                    )
                    next_masks = ep.next_active_agent_mask.get(agent_id)
                    if next_masks is not None and len(next_masks) >= seq_len:
                        active_agents_np[episode_idx, seq_len, agent_idx, 0] = float(
                            next_masks[seq_len - 1]
                        )
                    else:
                        next_active_agent_id = (ep.metadata[seq_len - 1] or {}).get(
                            "next_active_agent_id"
                        )
                        if next_active_agent_id is None:
                            avail = (
                                ep.avail_actions_next[seq_len - 1].get(agent_id)
                                if len(ep.avail_actions_next) >= seq_len else None
                            )
                            active_agents_np[episode_idx, seq_len, agent_idx, 0] = float(
                                bool(avail and any(avail))
                            )
                        else:
                            active_agents_np[episode_idx, seq_len, agent_idx, 0] = float(
                                agent_id == next_active_agent_id
                            )

                # 新样本：帧 t 掩码来自决策时记录的 avail_actions[t]；
                # 旧样本未记录当前帧掩码时保持首帧默认全1，仅由 next 掩码覆盖后续帧。
                ep_cur_avails = ep.avail_actions
                for t in range(seq_len):
                    if t < len(ep_cur_avails):
                        cur_avail = ep_cur_avails[t].get(agent_id)
                        if cur_avail is not None:
                            usable_dim = min(len(cur_avail), action_dim)
                            avail_actions_np[episode_idx, t, agent_idx, :usable_dim] = np.asarray(
                                cur_avail[:usable_dim], dtype=np.float32
                            )
                    if t >= len(ep.avail_actions_next):
                        continue
                    avail = ep.avail_actions_next[t].get(agent_id)
                    if avail is None:
                        continue
                    usable_dim = min(len(avail), action_dim)
                    avail_actions_np[episode_idx, t + 1, agent_idx, :usable_dim] = np.asarray(
                        avail[:usable_dim], dtype=np.float32
                    )

        EpisodeTransition._copy_numpy_to_batch_tensor(batch["obs"], obs_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["state"], state_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["actions"], actions_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["actions_onehot"], actions_onehot_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["avail_actions"], avail_actions_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["reward"], reward_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["terminated"], terminated_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["filled"], filled_np)
        EpisodeTransition._copy_numpy_to_batch_tensor(batch["active_agents"], active_agents_np)
        if has_burn_in:
            EpisodeTransition._copy_numpy_to_batch_tensor(
                batch["initial_actions_onehot"], initial_actions_np,
            )

        return batch

    @staticmethod
    def _copy_numpy_to_batch_tensor(target: torch.Tensor, value: np.ndarray) -> None:
        """把CPU侧NumPy批量缓存一次性拷贝进EpisodeBatch目标张量，避免逐step创建小Tensor。"""
        source = torch.from_numpy(value)
        target.copy_(source.to(device=target.device, dtype=target.dtype))


@extensible
class SAEpisodeTransition(Generic[AgentID, ActionType, StateType, ObsType]):
    """
    单智能体单个 episode 的完整序列转移数据（single-agent episode）。

    它是单智能体算法(DQN/PPO/A2C 等)最基础的训练样本单元：采样侧逐步 add_step 写入，
    ReplayBuffer 以 episode 为单位存储，Learner 再用 to_batch_tensor 把多个 episode 拍平成
    SABatchTensorData 喂给具体算法。

    和多智能体的 EpisodeTransition 不同，这里不按 agent 维度组织，也不维护全局 state；
    所有字段都是单条时序列表。算法特有但非必需的量(可用动作掩码、行为策略 log_prob / value)
    采用可选列表存储，整段 episode 要么都有要么都没有，转批次时据此决定是否输出对应张量。
    """

    def __init__(self, action_space=None):
        # 局末由采样进程生成的列式回放载荷；不改变公开Transition字段。
        self._prepared_replay_episode: Optional[PreparedReplayEpisode] = None
        # 观测序列 O_t。
        self.action_space_spec = action_space.to_dict() if action_space is not None else None
        self.obs: List[ObsType] = []
        # 当前动作索引序列 A_t（保存 Action.action_idx 或原始离散索引）。
        self.actions: List[int | dict | None] = []
        # 上一动作索引序列 A_{t-1}；第一步没有真实上一动作时用 0 占位。
        self.last_actions: List[int | dict | None] = []
        # 即时奖励序列 R_t。
        self.rewards: List[float] = []
        # 下一观测序列 O_{t+1}。
        self.next_obs: List[ObsType] = []
        # 环境真实终止标志序列；terminated=True 表示环境逻辑完成。
        self.terminations: List[bool] = []
        # 截断标志序列；truncated=True 表示达到步数上限等外部停止条件。
        self.truncations: List[bool] = []
        # 可用动作掩码序列；每步保存 list[int](1=可用,0=不可用)或 None。
        self.avail_actions: List[Optional[List[int]]] = []
        # 行为策略动作对数概率序列 log π_old(a|s)；on-policy 算法需要，缺省 None。
        self.old_log_probs: List[Optional[float]] = []
        # 行为策略状态价值序列 V_old(s)；on-policy 算法需要，缺省 None。
        self.old_values: List[Optional[float]] = []
        # 每步附加元数据。
        self.metadata: List[Optional[Dict]] = []
        # episode 编号，主要用于调试和离线样本回放。
        self.episode_num: int = 0
        # 该片段第一步是否是真实 episode 第一步；按步切片时，中间片段应为 False。
        self.starts_with_episode: bool = True
        # episode 有效时间步长度。
        self.seq_len: int = 0

    def __setstate__(self, state):
        """历史轨迹的可选字段只在读取时补齐。"""
        restored = dict(state)
        restored.setdefault("action_space_spec", None)
        restored.setdefault("avail_actions", [])
        restored.setdefault("old_log_probs", [])
        restored.setdefault("old_values", [])
        restored.setdefault("starts_with_episode", True)
        restored.setdefault("_prepared_replay_episode", None)
        self.__dict__.update(restored)

    def add_step(
            self,
            obs: ObsType,
            action: Union[ActionType, int],
            reward: float,
            next_obs: ObsType,
            terminated: bool,
            truncated: bool,
            last_action: Optional[Union[ActionType, int]] = None,
            avail_actions: Optional[List[int]] = None,
            log_prob: Optional[float] = None,
            value: Optional[float] = None,
            metadata: Optional[Dict] = None,
    ):
        """添加单步数据；action / last_action 支持业务 Action 实体或离散索引。"""
        self._prepared_replay_episode = None
        self.obs.append(obs)
        self.actions.append(_record_action(action, self.action_space_spec))
        # 第一步没有真实上一动作时记录 0 占位。
        self.last_actions.append(_record_action(last_action, self.action_space_spec))
        self.rewards.append(float(reward))
        self.next_obs.append(next_obs)
        self.terminations.append(bool(terminated))
        self.truncations.append(bool(truncated))
        self.avail_actions.append(avail_actions)
        self.old_log_probs.append(None if log_prob is None else float(log_prob))
        self.old_values.append(None if value is None else float(value))
        self.metadata.append(metadata)
        self.seq_len += 1

    @staticmethod
    def _to_action_idx(action: Union[ActionType, int]) -> int:
        """把业务 Action 实体或原始离散值统一转成动作索引。"""
        if hasattr(action, "action_idx"):
            return int(action.action_idx)
        return int(action)

    def episode_reward(self) -> float:
        """返回 episode 累计奖励。"""
        return float(sum(self.rewards))

    def to_replay_step_payloads(self):
        """
        按整局导出ReplayBuffer需要的逐Step值。

        保持现有公开List字段不变，但让ReplayBuffer可以在一次push内按字段批量张量化，
        避免逐Step反射Transition对象。
        """
        list_fields = (
            "obs",
            "actions",
            "last_actions",
            "rewards",
            "next_obs",
            "terminations",
            "truncations",
            "avail_actions",
            "old_log_probs",
            "old_values",
            "metadata",
        )
        step_payloads = [{"fields": {}} for _ in range(self.seq_len)]
        step_field_layout = {}
        for field_name in list_fields:
            values = getattr(self, field_name)
            step_field_layout[field_name] = ("list", ())
            for step in range(min(self.seq_len, len(values))):
                step_payloads[step]["fields"][field_name] = values[step]

        known_fields = {
            *list_fields,
            "seq_len",
            "episode_num",
            "starts_with_episode",
            "_prepared_replay_episode",
        }
        static_fields = {
            field_name: copy.deepcopy(value)
            for field_name, value in vars(self).items()
            if field_name not in known_fields
        }
        return step_payloads, step_field_layout, static_fields

    @timed("transition.prepare_for_replay")
    def prepare_for_replay(self) -> PreparedReplayEpisode:
        """在采样进程完成整局列式张量化，ReplayBuffer只负责存储和建立索引。"""
        if self._prepared_replay_episode is None:
            self._prepared_replay_episode = prepare_replay_episode(
                *self.to_replay_step_payloads()
            )
        return self._prepared_replay_episode

    @property
    def prepared_replay_episode(self) -> Optional[PreparedReplayEpisode]:
        return self._prepared_replay_episode

    @staticmethod
    def _assemble_batch(episodes, *, learner_name, schema, context):
        from algos.learners.batch_schema import assemble_single_agent
        strategy = episodes.sample_strategy if isinstance(episodes, SampledEpisodes) else None
        return assemble_single_agent(episodes, schema, agent=context.agent, device=context.device,
                                     sample_strategy=strategy)

    @staticmethod
    @timed("transition.to_batch")
    def to_batch_tensor(
            episode_trans: List["SAEpisodeTransition"],
            device=torch.device("cpu"),
    ) -> SABatchTensorData:
        """
        将 List[SAEpisodeTransition] 按 transition 拍平为 SABatchTensorData。

        输出第一维 N = 所有 episode 的有效步之和(无时间维、无 agent 维)。可选字段
        (avail_actions / old_log_probs / old_values)只有当所有 episode 的所有步都完整提供时
        才输出对应张量，否则置 None，交由 Learner 决定如何补算。
        :param episode_trans: 待转换的单智能体 episode 列表。
        :param device: 目标设备。
        :return: 拍平后的 SABatchTensorData。
        """
        if any(ep.action_space_spec and ep.action_space_spec["kind"] != "discrete"
               for ep in episode_trans):
            raise ValueError("SB3 replay adapter supports only discrete action spaces")
        if not episode_trans:
            raise ValueError("episode trans should not be empty")

        valid_eps = [ep for ep in episode_trans if ep.seq_len > 0]
        if not valid_eps:
            raise ValueError("all episodes are empty")

        obs_list: List[np.ndarray] = []
        next_obs_list: List[np.ndarray] = []
        actions: List[int] = []
        last_actions: List[int] = []
        rewards: List[float] = []
        terminated: List[float] = []
        truncated: List[float] = []
        episode_starts: List[float] = []

        avail_list: List[List[int]] = []
        log_prob_list: List[float] = []
        value_list: List[float] = []
        has_avail = True
        has_log_prob = True
        has_value = True

        for ep in valid_eps:
            for step in range(ep.seq_len):
                obs_list.append(np.asarray(ep.obs[step], dtype=np.float32).flatten())
                next_obs_list.append(np.asarray(ep.next_obs[step], dtype=np.float32).flatten())
                actions.append(int(ep.actions[step]))
                last_actions.append(int(ep.last_actions[step]))
                rewards.append(float(ep.rewards[step]))
                terminated.append(1.0 if ep.terminations[step] else 0.0)
                truncated.append(1.0 if ep.truncations[step] else 0.0)
                starts_with_episode = ep.starts_with_episode
                episode_starts.append(1.0 if step == 0 and starts_with_episode else 0.0)

                avail = ep.avail_actions[step] if step < len(ep.avail_actions) else None
                if avail is None:
                    has_avail = False
                elif has_avail:
                    avail_list.append(list(avail))

                log_prob = ep.old_log_probs[step] if step < len(ep.old_log_probs) else None
                if log_prob is None:
                    has_log_prob = False
                elif has_log_prob:
                    log_prob_list.append(float(log_prob))

                value = ep.old_values[step] if step < len(ep.old_values) else None
                if value is None:
                    has_value = False
                elif has_value:
                    value_list.append(float(value))

        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(-1)
        last_actions_t = torch.as_tensor(last_actions, dtype=torch.long, device=device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        terminated_t = torch.as_tensor(terminated, dtype=torch.float32, device=device).unsqueeze(-1)
        truncated_t = torch.as_tensor(truncated, dtype=torch.float32, device=device).unsqueeze(-1)
        episode_starts_t = torch.as_tensor(episode_starts, dtype=torch.float32, device=device).unsqueeze(-1)

        avail_t = (
            torch.as_tensor(np.stack(avail_list), dtype=torch.float32, device=device)
            if has_avail and avail_list else None
        )
        log_probs_t = (
            torch.as_tensor(log_prob_list, dtype=torch.float32, device=device).unsqueeze(-1)
            if has_log_prob and log_prob_list else None
        )
        values_t = (
            torch.as_tensor(value_list, dtype=torch.float32, device=device).unsqueeze(-1)
            if has_value and value_list else None
        )

        return EpisodeBatch(dict(
            obs=obs,
            actions=actions_t,
            last_actions=last_actions_t,
            rewards=rewards_t,
            next_obs=next_obs,
            terminated=terminated_t,
            truncated=truncated_t,
            episode_starts=episode_starts_t,
            avail_actions=avail_t,
            old_log_probs=log_probs_t,
            old_values=values_t,
            returns=None,
            advantages=None,
        ))


def episode_transitions_reward(transitions) -> float:
    """汇总一个 AgentName 的全部组轨迹回报。"""
    return sum(transition.episode_reward() for transition in transitions)


def summarize_agent_episode_rewards(
        agent_episode_transitions: dict,
) -> dict:
    """把 dict[AgentName, list[Transition]] 汇总成各 AgentName 的物理 episode 回报。"""
    return {
        agent_name: episode_transitions_reward(transitions)
        for agent_name, transitions in agent_episode_transitions.items()
    }


def _record_action(action, spec=None):
    """Legacy scalar indices or a versioned lossless typed action record."""
    from envs.action.spaces import ActionSpace
    if spec is not None and spec["kind"] != "discrete":
        return None if action is None else ActionSpace.from_dict(spec).record(action)
    if action is None:
        return 0
    if hasattr(action, "action_idx"):
        return int(action.action_idx)
    if isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_)):
        return int(action)
    raise ValueError("non-discrete transition actions require an action-space specification")
