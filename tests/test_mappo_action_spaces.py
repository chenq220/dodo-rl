from __future__ import annotations
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, Independent, TanhTransform
from algos.modules.agents.policy_agent import PolicyRNNAgent
from algos.training.mappo_learner import MAPPOActorCriticLearner
from algos.modules.agents.policy import ActionDistribution
from algos.training.mappo_batch import build_batch, gae
from algos.utils.policy_batch import actor_outputs
from config import AgentConfig, LearnerConfig, ReplayBufferConfig
from envs.sense.action_space_examples import ActionSpaceCounterParallelEnvironment, ActionSpaceCounterAECEnvironment
from sample.replay_buffer import ReplayBuffer
from scheduler.model import AgentCompose, EpisodeInfo, ActionParam
from scheduler.rollout_sample import RolloutSampleWorker
from utils.module_utils import inject_env_dims
from test_action_spaces import spaces


def setup(space, env_cls=ActionSpaceCounterParallelEnvironment):
    config = AgentConfig(name='policy',agent_type='policy_rnn',agent_names=['a','b'],obs_dim=2,
                         obs_last_action=True,hidden_dim=16,action_space_spec=space.to_dict())
    assert config.agent_output_type == 'pi_logits'
    agent = PolicyRNNAgent('policy_rnn',config,'blanco','cpu')
    assert agent.consume_on_policy_stats() == (None, None)
    env = env_cls({'a':space,'b':space},max_steps=4)
    inject_env_dims(env,[agent])
    cfg = LearnerConfig(name='policy',algo_type='mappo',standardise_rewards=False)
    cfg.algo_config.epochs = 2
    cfg.algo_config.num_mini_batch = 2
    learner = MAPPOActorCriticLearner('mappo',cfg,agent,env,'cpu')
    return agent,env,learner


def rollout(agent,env,step_limit=3):
    worker = RolloutSampleWorker(env,SimpleNamespace(accept_sample_task_result=lambda *a,**kw:True),
                                SimpleNamespace(push=lambda *a,**kw:None),AgentCompose(step_agent=[agent]),[agent])
    info = EpisodeInfo(episode_num=1,task_id='mappo-test',epsilon=0,deterministic=False,temperature=1,
                       actions_param={'policy':ActionParam(epsilon=0,deterministic=False,temperature=1)},
                       seed=3,start_datetime=None,end_datetime=None,metadata={})
    run = worker.rolling_aec_episode if isinstance(env,ActionSpaceCounterAECEnvironment) else worker.rolling_parallel_episode
    return run(info,step_limit=step_limit)['policy'][0]


@pytest.mark.parametrize('space',spaces(),ids=lambda s:s.kind)
@pytest.mark.parametrize('env_cls',[ActionSpaceCounterParallelEnvironment,ActionSpaceCounterAECEnvironment])
def test_rollout_replay_recurrent_ppo_and_snapshot(space,env_cls,tmp_path):
    torch.manual_seed(4)
    agent,env,learner = setup(space,env_cls)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={'policy':32},batch_size=2))
    try:
        episodes = [rollout(agent,env,2),rollout(agent,env,3)]
        for ep in episodes:
            ep.prepare_for_replay()
        replay.push({'policy':episodes},notify_learning=False)
        recovered = replay.sample('policy',batch_size=2)
        batch = build_batch(agent,recovered,'cpu')
        with torch.no_grad():
            output = actor_outputs(learner.actor,batch)[:-1]
            dist = learner.actor.distribution(output,batch.get('action_masks'),batch['temperature'])
            logs = dist.log_prob(batch['actions'])
            active = (batch['valid']*batch['active'][:-1]).bool()
            torch.testing.assert_close(logs[active],batch['old_log_probs'][active],atol=1e-5,rtol=1e-5)
        before = copy.deepcopy(learner.snapshot_state())
        metrics = learner.update(recovered)
        assert all(np.isfinite(v) for v in metrics.values())
        assert learner.update_cnt == 1
        assert any(not torch.equal(before['agent_net'][k],v) for k,v in learner.actor.state_dict().items())
        learner.restore_state(before)
        assert learner.update_cnt == 0
        for k,v in learner.actor.state_dict().items():
            torch.testing.assert_close(v,before['agent_net'][k])
        path = tmp_path/'actor.pth'
        torch.save(learner.actor.state_dict(),path)
        restored = PolicyRNNAgent('policy_rnn',copy.deepcopy(agent.config),'file','cpu',model_file_path=str(path))
        for k,v in restored.agent_net.state_dict().items():
            torch.testing.assert_close(v,learner.actor.state_dict()[k])
        learner.dynamic_modify_config({'lr':1e-5,'entropy_coef':0.01})
        assert learner.actor_optimizer.param_groups[0]['lr'] == 1e-5
    finally:
        replay.close()
        env.close()


def test_bounded_gaussian_log_probability_matches_pytorch():
    layout = spaces()[2].layout
    means = torch.tensor([[0.2,-0.6,1.1]],requires_grad=True)
    log_std = torch.tensor([-0.2,0.3,0.1],requires_grad=True)
    dist = ActionDistribution(means,layout,log_std)
    actions = torch.tensor([[0.5,-0.7,0.999]])
    reference = Independent(TransformedDistribution(Normal(means,log_std.exp()),[TanhTransform()]),1)
    torch.testing.assert_close(dist.log_prob(actions).squeeze(-1),reference.log_prob(actions))
    assert torch.all(dist.sample().abs() < 1)
    (-dist.log_prob(actions).mean()-0.01*dist.entropy().mean()).backward()
    assert torch.isfinite(means.grad).all() and torch.isfinite(log_std.grad).all()


def test_multi_discrete_joint_probability_and_per_head_masks():
    space = spaces()[1]
    dist = ActionDistribution(torch.zeros(2,5),space.layout,masks=torch.tensor([[1,0,1,0,1]]*2))
    values = dist.sample()
    assert torch.all(values[:,1] == 1)
    torch.testing.assert_close(dist.log_prob(values),torch.full((2,1),-np.log(2),dtype=torch.float32))


def test_gae_distinguishes_terminal_and_time_limit_and_padding():
    reward = torch.tensor([1.,2.,999.]).reshape(3,1,1,1)
    values = torch.tensor([10.,20.,30.,999.]).reshape(4,1,1,1)
    valid = torch.tensor([1.,1.,0.]).reshape_as(reward)
    terminated = torch.zeros_like(reward)
    truncated = torch.zeros_like(reward); truncated[1] = 1
    _,returns = gae(reward,values,terminated,truncated,valid,1.,1.)
    torch.testing.assert_close(returns[:2].flatten(),torch.tensor([33.,32.]))
    terminated[1] = 1
    _,returns = gae(reward,values,terminated,truncated,valid,1.,1.)
    torch.testing.assert_close(returns[:2].flatten(),torch.tensor([3.,2.]))


def test_snapshot_restores_optimizer_and_normalizers_before_next_update():
    torch.manual_seed(7)
    agent,env,learner = setup(spaces()[3])
    try:
        learner.config.standardise_rewards = True
        learner.config.standardise_returns = True
        episodes = [rollout(agent,env,2),rollout(agent,env,3)]
        learner.update(episodes)
        checkpoint = copy.deepcopy(learner.snapshot_state())
        torch.manual_seed(71)
        expected_stats = learner.update(episodes)
        expected = copy.deepcopy(learner.snapshot_state())
        learner.restore_state(checkpoint)
        torch.manual_seed(71)
        actual_stats = learner.update(episodes)
        for name in expected_stats:
            assert actual_stats[name] == pytest.approx(expected_stats[name],abs=1e-7)
        for name in ('agent_net','critic_net','reward_moments','return_moments'):
            for key,value in learner.snapshot_state()[name].items():
                torch.testing.assert_close(value,expected[name][key])
    finally:
        env.close()


def test_missing_behavior_statistics_are_rejected():
    agent,env,learner = setup(spaces()[0])
    try:
        ep = rollout(agent,env,2)
        ep.metadata[0].pop('behavior_log_probs')
        with pytest.raises(ValueError,match='behavior log probabilities'):
            learner.update([ep])
    finally:
        env.close()


def test_partial_recurrent_episode_is_rejected_without_a_burn_in_prefix():
    agent,env,learner = setup(spaces()[0])
    try:
        ep = rollout(agent,env,2)
        ep.starts_with_episode = False
        with pytest.raises(ValueError,match='complete episodes'):
            learner.update([ep])
    finally:
        env.close()
