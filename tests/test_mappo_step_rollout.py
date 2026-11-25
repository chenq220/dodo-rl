"""原生MAPPO直接消费样本池现有按步游标窗口。"""
import copy
import numpy as np
import pytest
import torch
from config import ReplayBufferConfig
from sample.replay_buffer import ReplayBuffer
from algos.training.mappo_batch import build_batch, gae
from algos.utils.policy_batch import actor_outputs, burn_in_hidden
from envs.sense.action_space_examples import ActionSpaceCounterParallelEnvironment, ActionSpaceCounterAECEnvironment
from test_mappo_action_spaces import setup, rollout
from test_action_spaces import spaces


def pool(agent):
    return ReplayBuffer(ReplayBufferConfig(agent_capacities={'policy':32},batch_size=3))


@pytest.mark.parametrize('space',spaces(),ids=lambda s:s.kind)
@pytest.mark.parametrize('env_cls',[ActionSpaceCounterParallelEnvironment,ActionSpaceCounterAECEnvironment])
def test_cursor_middle_window_reconstructs_current_policy_hidden(space,env_cls):
    torch.set_num_threads(1)
    agent,env,learner=setup(space,env_cls)
    replay=pool(agent)
    try:
        ep=rollout(agent,env,4)
        replay.push({'policy':[ep]},notify_learning=False)
        first=replay.sample('policy',batch_size=2,strategy='on_policy_cursor_steps')
        middle=replay.sample('policy',batch_size=2,strategy='on_policy_cursor_steps')
        assert sum(e.seq_len for e in first)==2
        assert sum(e.seq_len for e in middle)==2
        assert not middle[0].starts_with_episode
        assert middle[0].burn_in.seq_len==2
        batch=build_batch(agent,middle,'cpu')
        with torch.no_grad():
            outputs=actor_outputs(agent.agent_net,batch)[:-1]
            dist=agent.agent_net.distribution(outputs,batch.get('action_masks'),batch['temperature'])
            mask=(batch['valid']*batch['active'][:-1]).bool()
            torch.testing.assert_close(dist.log_prob(batch['actions'])[mask],batch['old_log_probs'][mask],atol=1e-5,rtol=1e-5)
        # 先更新前半段，再用当前参数重建后半段，不能复用旧权重下的hidden。
        old_hidden=burn_in_hidden(learner.actor,batch['burn_in'][0]).clone()
        learner.update(first)
        new_hidden=burn_in_hidden(learner.actor,batch['burn_in'][0])
        assert not torch.equal(old_hidden,new_hidden)
        weights=copy.deepcopy(learner.snapshot_state())
        metrics=learner.update(middle)
        assert all(np.isfinite(v) for v in metrics.values())
        assert any(not torch.equal(weights['agent_net'][k],v) for k,v in learner.actor.state_dict().items())
        assert replay.sample('policy',batch_size=1,strategy='on_policy_cursor_steps')==[]
        invalid=copy.deepcopy(middle[0]);invalid.burn_in.episode_num=-1
        with pytest.raises(ValueError,match='same-episode'):build_batch(agent,[invalid],'cpu')
        invalid=copy.deepcopy(middle[0]);invalid.burn_in.next_states[-1]=np.full_like(invalid.states[0],999)
        with pytest.raises(ValueError,match='connect directly'):build_batch(agent,[invalid],'cpu')
    finally:env.close()


def test_exact_1000_new_steps_across_episode_boundaries():
    torch.set_num_threads(1)
    agent,env,learner=setup(next(s for s in spaces() if s.kind=='continuous'))
    env.config.max_step_per_episode=600
    env.max_step_per_episode=600
    replay=pool(agent)
    try:
        for i in range(4):
            ep=rollout(agent,env,600);ep.episode_num=i+1
            replay.push({'policy':[ep]},notify_learning=False)
        windows=[]
        for _ in range(2):
            episodes=replay.sample('policy',batch_size=1000,strategy='on_policy_cursor_steps')
            assert sum(e.seq_len for e in episodes)==1000
            windows.extend((e.episode_num,m['step']) for e in episodes for m in e.metadata)
            batch=build_batch(agent,episodes,'cpu')
            assert batch['valid'].sum().item()==2000 # 1000环境步、两个设备
            metrics=learner.update(episodes)
            assert all(np.isfinite(v) for v in metrics.values())
        assert len(windows)==len(set(windows))==2000
        assert replay.sample('policy',batch_size=1000,strategy='on_policy_cursor_steps')==[]
    finally:env.close()


def test_segment_gae_bootstraps_without_false_termination():
    rewards=torch.tensor([1.]).reshape(1,1,1,1);values=torch.tensor([2.,10.]).reshape(2,1,1,1)
    valid=torch.ones_like(rewards)
    for terminal,expected in [(0,8.),(1,-1.)]:
        advantages,_=gae(rewards,values,valid*terminal,valid*0,valid,.9,.95)
        assert advantages.item()==pytest.approx(expected)
