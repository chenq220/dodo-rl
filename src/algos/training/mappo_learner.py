"""DodoRL adapter for recurrent MAPPO (marlbenchmark/on-policy design)."""
from __future__ import annotations
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from algos.learners import Learner
from algos.map import AlogsMap
from algos.modules.critics.mappo import CentralValue
from algos.training.mappo_batch import build_batch, gae
from algos.utils.normalization import RunningMoments
from algos.utils.policy_batch import actor_outputs, batch_from_schema
from algos.learners.batch_schema import DefaultEpisodeFields
from sample.model import EpisodeBatch
from common.plugins import learner_plugin
from config import MappoConfig
from utils.checkpoint import extract_state_dict
from utils.dim_tools import state_shape


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def optimization_settings(cfg):
    """Resolve optional overrides without changing legacy lr/epochs semantics."""
    rates = [cfg.lr if value is None else value
             for value in (cfg.actor_lr, cfg.critic_lr)]
    epochs = [cfg.epochs if value is None else value
              for value in (cfg.actor_epochs, cfg.critic_epochs)]
    if any(not np.isfinite(value) or value <= 0 for value in rates):
        raise ValueError('MAPPO Actor/Critic learning rates must be finite and positive')
    if (cfg.epochs < 1 or int(cfg.epochs) != cfg.epochs
            or any(value < 1 or int(value) != value for value in epochs)):
        raise ValueError('MAPPO Actor/Critic epochs must be positive integers')
    warmup = cfg.critic_warmup_updates
    if warmup < 0 or int(warmup) != warmup:
        raise ValueError('MAPPO critic_warmup_updates must be a nonnegative integer')
    if not isinstance(cfg.warmup_use_mc_returns, bool):
        raise ValueError('MAPPO warmup_use_mc_returns must be a boolean')
    return float(rates[0]), float(rates[1]), int(epochs[0]), int(epochs[1]), int(warmup)


def value_diagnostics(prediction, target, mask):
    """Same-batch diagnostics; a pre-fit score is not held-out evaluation."""
    prediction, target = prediction[mask.bool()], target[mask.bool()]
    if not target.numel():
        return dict(explained_variance=0., explained_variance_valid=0., mse=0.)
    variance = target.var(unbiased=False)
    error = target - prediction
    valid = bool(variance > 1e-12)
    return dict(explained_variance=float(1-error.var(unbiased=False)/variance) if valid else 0.,
                explained_variance_valid=float(valid), mse=float(error.square().mean()))


@learner_plugin(MappoConfig)
class MAPPOActorCriticLearner(Learner[DefaultEpisodeFields]):
    supported_action_spaces = ('discrete', 'multi_discrete', 'continuous', 'multi_continuous')

    def __init__(self, algo_type, config, agent, env, device, net_source='blanco',
                 model_file_path=None, caller=None, uploader=None, **kwargs):
        supported = (*AlogsMap.compatible_agent_types(AlogsMap.mappo), "mappo_rnn")
        assert agent.agent_type in supported, f'MAPPOActorCriticLearner supports agent types {supported}, given={agent.agent_type}'
        self.update_cnt = 0
        super().__init__(algo_type, config, agent, env, device, net_source,
                         model_file_path, caller, uploader, **kwargs)

    @property
    def agent_net(self):
        return self.actor

    @property
    def models(self):
        return {'agent_net': self.actor, 'critic_net': self.critic}

    def build_empty_network(self, learner_config, **kwargs):
        cfg = learner_config.algo_config
        actor_lr, critic_lr, _, _, _ = optimization_settings(cfg)
        self.actor = copy.deepcopy(self.agent.agent_net).to(self.device)
        group_shape = learner_config.state_shape if self.agent.config.agent_groups else state_shape(self.env)
        self.state_dim = int(np.prod(group_shape))
        critic_dim = self.state_dim
        if cfg.obs_individual_obs:
            critic_dim += self.agent.config.obs_dim
        if cfg.obs_last_action:
            critic_dim += self.agent.n_agents * self.agent.action_layout['previous_action_dim']
        if cfg.obs_agent_id:
            critic_dim += self.agent.n_agents
        if cfg.critic_type != 'cv_critic':
            raise ValueError('MAPPO requires centralized critic_type=cv_critic')
        self.critic = CentralValue(critic_dim, self.agent.config.hidden_dim).to(self.device)
        self.reward_moments = RunningMoments().to(self.device)
        self.return_moments = RunningMoments().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        return self.actor, self.critic

    def build_network_from_checkpoint(self, checkpoint, learner_config, **kwargs):
        self.build_empty_network(learner_config, **kwargs)
        # Different policy architecture: fail clearly on old EPyMARL checkpoints.
        self.actor.load_state_dict(extract_state_dict(checkpoint['agent_net']), strict=True)
        self.critic.load_state_dict(extract_state_dict(checkpoint['critic_net']), strict=True)
        return self.actor, self.critic

    def _offline_net_weights(self):
        return {'critic_net': self.critic.state_dict()}

    def snapshot_state(self):
        return {**{k:v.state_dict() for k,v in self.models.items()},
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'reward_moments': self.reward_moments.state_dict(),
                'return_moments': self.return_moments.state_dict(),
                'update_cnt': self.update_cnt}

    def restore_state(self, state):
        for k,v in self.models.items():
            v.load_state_dict(state[k])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.reward_moments.load_state_dict(state['reward_moments'])
        self.return_moments.load_state_dict(state['return_moments'])
        self.update_cnt = state['update_cnt']

    def dynamic_modify_config(self, updates):
        # Validate before mutating the live configuration or optimizer groups.
        candidate = copy.copy(self.config.algo_config)
        if updates:
            if not isinstance(updates, dict):
                raise TypeError('dynamic_modify_config expects dict updates')
            for key, value in updates.items():
                if not hasattr(candidate, key):
                    raise AttributeError(f'{type(candidate).__name__} has no algo config field {key!r}')
                setattr(candidate, key, value)
        actor_lr, critic_lr, _, _, _ = optimization_settings(candidate)
        super().dynamic_modify_config(updates)
        for opt, lr in ((self.actor_optimizer, actor_lr), (self.critic_optimizer, critic_lr)):
            for group in opt.param_groups:
                group['lr'] = lr

    def critic_inputs(self, batch):
        cfg = self.config.algo_config
        parts = [batch['states']]
        if cfg.obs_individual_obs:
            parts.append(batch['obs'])
        if cfg.obs_last_action:
            t,b,n,_ = batch['previous'].shape
            parts.append(batch['previous'].reshape(t,b,1,-1).expand(-1,-1,n,-1))
        if cfg.obs_agent_id:
            t,b,n,_ = batch['states'].shape
            parts.append(torch.eye(n, device=self.device).expand(t,b,n,n))
        return torch.cat(parts, -1)

    @torch.no_grad()
    def policy_diagnostics(self, batch, policy_mask):
        outputs = actor_outputs(self.actor, batch)[:-1]
        dist = self.actor.distribution(outputs, batch.get('action_masks'), batch['temperature'])
        log_ratio = dist.log_prob(batch['actions']) - batch['old_log_probs']
        selected = log_ratio[policy_mask.bool()]
        ratio = selected.exp()
        return dict(approx_kl=float(((ratio-1)-selected).mean()),
                    clip_fraction=float(((ratio-1).abs()>self.config.algo_config.eps_clip).float().mean()),
                    log_ratio_mean=float(selected.mean()),
                    log_ratio_std=float(selected.std(unbiased=False)),
                    log_ratio_abs_max=float(selected.abs().max()))

    def update(self, batch_data: EpisodeBatch[DefaultEpisodeFields]):
        cfg = self.config.algo_config
        _, _, actor_epochs, critic_epochs, warmup_updates = optimization_settings(cfg)
        if cfg.num_mini_batch < 1 or not 0 <= cfg.gae_lambda <= 1:
            raise ValueError('MAPPO requires positive minibatches and gae_lambda in [0,1]')
        warmup = self.update_cnt < warmup_updates
        use_mc = warmup and cfg.warmup_use_mc_returns
        if isinstance(batch_data, EpisodeBatch):
            fields = batch_data.data
            episodes = [
                SimpleNamespace(
                    seq_len=int(fields["valid"][row].sum().item()),
                    starts_with_episode=bool(fields["starts_with_episode"][row]),
                    burn_in=fields["burn_in"][row],
                )
                for row in range(fields["valid"].shape[0])
            ]
            batch = batch_from_schema(self.agent, fields)
        else:
            episodes = [ep for ep in batch_data if ep.seq_len]
            batch = build_batch(self.agent, episodes, self.device)
        critic_inputs = self.critic_inputs(batch)
        policy_mask = batch['valid'] * batch['active'][:-1]
        if not policy_mask.any():
            raise ValueError('MAPPO batch has no active decisions')
        with torch.no_grad():
            rewards = batch['rewards']
            if self.config.common_reward:
                rewards = rewards.sum(-2, keepdim=True).expand_as(rewards)
            # Keep raw reward-space targets for independent MC diagnostics.
            raw_rewards = rewards
            if self.config.standardise_rewards:
                self.reward_moments.update(rewards[batch['valid'].bool()])
                rewards = self.reward_moments.normalize(rewards)
            old_values = self.critic(critic_inputs)
            if self.config.standardise_returns:
                old_values = self.return_moments.denormalize(old_values)
            advantages, returns = gae(rewards, old_values, batch['terminated'], batch['truncated'],
                                      batch['valid'], self.config.gamma, 1. if use_mc else cfg.gae_lambda)
            selected = advantages[policy_mask.bool()]
            metrics = dict(raw_advantage_mean=float(selected.mean()),
                           raw_advantage_std=float(selected.std(unbiased=False)),
                           return_mean=float(returns[batch['valid'].bool()].mean()),
                           return_std=float(returns[batch['valid'].bool()].std(unbiased=False)),
                           value_mean=float(old_values[:-1][batch['valid'].bool()].mean()),
                           value_std=float(old_values[:-1][batch['valid'].bool()].std(unbiased=False)))
            before_value = value_diagnostics(old_values[:-1], returns, batch['valid'])
            metrics.update({f'batch_value_{key}_pre': value for key, value in before_value.items()
                            if key != 'explained_variance_valid'})
            metrics['batch_value_explained_variance_valid'] = before_value['explained_variance_valid']
            # MC is eligible only for complete, reset-to-terminal trajectories, never
            # cursor slices/truncations. Reward normalization has no single affine
            # raw-value conversion, so do not compare incompatible units in that mode.
            mc_mask = torch.zeros_like(policy_mask)
            if not self.config.standardise_rewards:
                for j, ep in enumerate(episodes):
                    if (ep.starts_with_episode
                            and ep.burn_in is None):
                        for i in range(policy_mask.shape[2]):
                            if (batch['terminated'][ep.seq_len-1,j,i,0] > 0
                                    and not batch['truncated'][:ep.seq_len,j,i].any()):
                                mc_mask[:ep.seq_len,j,i] = policy_mask[:ep.seq_len,j,i]
            _, mc_returns = gae(raw_rewards, old_values, batch['terminated'], batch['truncated'],
                                batch['valid'], self.config.gamma, 1.)
            mc_stats = value_diagnostics(old_values[:-1], mc_returns, mc_mask)
            mc_selected = mc_returns[mc_mask.bool()]
            metrics.update(mc_value_explained_variance_pre=mc_stats['explained_variance'],
                           mc_value_explained_variance_valid=mc_stats['explained_variance_valid'],
                           mc_value_mse_pre=mc_stats['mse'],
                           mc_return_mean=float(mc_selected.mean()) if mc_selected.numel() else 0.,
                           mc_return_std=float(mc_selected.std(unbiased=False)) if mc_selected.numel() else 0.,
                           mc_complete_samples=float(mc_mask.sum()))
            advantages = (advantages-selected.mean()) / selected.std(unbiased=False).clamp_min(1e-5)
            if self.config.standardise_returns:
                self.return_moments.update(returns[batch['valid'].bool()])
                targets = self.return_moments.normalize(returns)
                value_reference = self.return_moments.normalize(old_values[:-1])
            else:
                targets, value_reference = returns, old_values[:-1]
        # A warmup does not change Actor mode, weights, gradients or Adam state.
        if not warmup:
            self.actor.train()
        self.critic.train()
        metrics.update({f'pre_{key}': value
                        for key, value in self.policy_diagnostics(batch, policy_mask).items()})
        actor_metrics, critic_metrics = [], []
        actor_steps = critic_steps = 0
        b = batch['inputs'].shape[1]
        # Keeping the shared round/permutation and Actor-then-Critic ordering makes
        # the default (equal epoch counts, no warmup) match the previous updates.
        for epoch in range(max(0 if warmup else actor_epochs, critic_epochs)):
            for index in torch.randperm(b, device=self.device).tensor_split(min(b,cfg.num_mini_batch)):
                if not warmup and epoch < actor_epochs:
                    outputs = actor_outputs(self.actor, batch, index)[:-1]
                    masks = batch.get('action_masks')
                    if masks is not None:
                        masks = masks[:,index]
                    dist = self.actor.distribution(outputs, masks, batch['temperature'][:,index])
                    logs = dist.log_prob(batch['actions'][:,index])
                    old_logs = batch['old_log_probs'][:,index]
                    mask = policy_mask[:,index]
                    ratio = (logs-old_logs).exp()
                    advantage = advantages[:,index]
                    surrogate = torch.minimum(ratio*advantage,
                                              ratio.clamp(1-cfg.eps_clip,1+cfg.eps_clip)*advantage)
                    entropy = masked_mean(dist.entropy(), mask)
                    actor_loss = -masked_mean(surrogate,mask) - cfg.entropy_coef*entropy
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(),self.config.grad_norm_clip)
                    self.actor_optimizer.step()
                    actor_steps += 1
                    with torch.no_grad():
                        kl = masked_mean((ratio-1)-(logs-old_logs),mask)
                        clip_fraction = masked_mean(((ratio-1).abs()>cfg.eps_clip).float(),mask)
                    actor_metrics.append([actor_loss.item(),entropy.item(),kl.item(),
                                          clip_fraction.item(),float(actor_grad)])
                if epoch < critic_epochs:
                    values = self.critic(critic_inputs[:-1,index])
                    target = targets[:,index]
                    loss = torch.nn.functional.huber_loss(values,target,reduction='none',delta=cfg.huber_delta)
                    if cfg.use_clipped_value_loss:
                        reference = value_reference[:,index]
                        clipped = reference + (values-reference).clamp(-cfg.eps_clip,cfg.eps_clip)
                        loss = torch.maximum(loss,torch.nn.functional.huber_loss(
                            clipped,target,reduction='none',delta=cfg.huber_delta))
                    critic_loss = masked_mean(loss,batch['valid'][:,index]) * cfg.value_loss_coef
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(),self.config.grad_norm_clip)
                    self.critic_optimizer.step()
                    critic_steps += 1
                    critic_metrics.append([critic_loss.item(),float(critic_grad)])
        if actor_metrics:
            metrics.update(zip(['policy_loss','entropy','approx_kl','clip_fraction','agent_grad_norm'],
                               np.mean(actor_metrics,axis=0).tolist()))
        else:
            with torch.no_grad():
                output = actor_outputs(self.actor, batch)[:-1]
                dist = self.actor.distribution(output,batch.get('action_masks'),batch['temperature'])
                entropy = masked_mean(dist.entropy(),policy_mask)
            metrics.update(policy_loss=0., entropy=float(entropy), approx_kl=metrics['pre_approx_kl'],
                           clip_fraction=metrics['pre_clip_fraction'], agent_grad_norm=0.)
        metrics.update(zip(['critic_loss','critic_grad_norm'],np.mean(critic_metrics,axis=0).tolist()))
        metrics.update({f'post_{key}': value
                        for key, value in self.policy_diagnostics(batch, policy_mask).items()})
        with torch.no_grad():
            post_values = self.critic(critic_inputs[:-1])
            if self.config.standardise_returns:
                post_values = self.return_moments.denormalize(post_values)
            after_value = value_diagnostics(post_values,returns,batch['valid'])
        metrics.update({f'batch_value_{key}_post': value for key, value in after_value.items()
                        if key != 'explained_variance_valid'})
        metrics.update(actor_updates=float(actor_steps), critic_updates=float(critic_steps),
                       critic_warmup_active=float(warmup), mc_returns_used=float(use_mc))
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f'Nonfinite MAPPO update: {metrics}')
        self.update_cnt += 1
        core = ('policy_loss', 'critic_loss', 'entropy', 'approx_kl',
                'clip_fraction', 'agent_grad_norm', 'critic_grad_norm')
        return {**{key: metrics[key] for key in core},
                **{key: value for key, value in metrics.items() if key not in core}}
