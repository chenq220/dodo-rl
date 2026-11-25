from __future__ import annotations

from algos.modules.agents.rnn_agent import RNNAgent
from algos.modules.agents.recurrent_controller import IndependentRecurrentController
from common.plugins import agent_plugin
from config import RnnNsAgentConfig


# 非共享网络RNN Agent：每个业务智能体槽位各自持有一套循环网络参数。
# 它适用于同一批智能体观测维度和动作空间一致、但角色需要分化或不希望共享策略参数的场景。
# 注意它不是完整异构动作空间方案；agent顺序必须稳定，因为第i个agent会绑定第i套网络。
@agent_plugin(RnnNsAgentConfig, default_obs_agent_id=False)
class RNNNSAgent(RNNAgent):
    mac_cls = IndependentRecurrentController
    epymarl_sharing_scope = "none"
    force_agent_id_input = False
