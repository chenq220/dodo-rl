"""Typed descriptions of the training data assembled from replay trajectories."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType, UnionType
from typing import Annotated, Any, Generic, Sequence, TYPE_CHECKING, TypedDict, TypeVar, Union, get_args, get_origin, get_type_hints, is_typeddict

import numpy as np
import torch
from torch import Tensor

from algos.utils.action_masks import action_mask_vector
from envs.action.spaces import ActionSpace
from sample.model import EpisodeBatch

if TYPE_CHECKING:
    from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition


@dataclass(frozen=True)
class TensorField:
    source: str
    axes: tuple[str | int, ...]
    dtype: torch.dtype | None
    padding_value: int | float | bool | None = None


@dataclass(frozen=True)
class ListField:
    source: str
    axis: str


@dataclass(frozen=True)
class DictField:
    source: str


class EpisodeTensorFields(TypedDict):
    obs: Annotated[Tensor, TensorField("obs_with_bootstrap", ("B", "T+1", "N", "O"), torch.float32)]
    state: Annotated[Tensor, TensorField("state_with_bootstrap", ("B", "T+1", "S"), torch.float32)]
    actions: Annotated[Tensor, TensorField("actions", ("B", "T", "N", "D"), None)]
    last_actions: Annotated[Tensor, TensorField("last_actions", ("B", "T", "N", "D"), None)]
    rewards: Annotated[Tensor, TensorField("rewards", ("B", "T", "N"), torch.float32)]
    team_rewards: Annotated[Tensor, TensorField("team_rewards", ("B", "T"), torch.float32)]
    terminated: Annotated[Tensor, TensorField("terminated", ("B", "T", "N"), torch.bool)]
    team_terminated: Annotated[Tensor, TensorField("team_terminated", ("B", "T"), torch.bool)]
    truncated: Annotated[Tensor, TensorField("truncated", ("B", "T", "N"), torch.bool)]
    valid: Annotated[Tensor, TensorField("valid", ("B", "T"), torch.bool)]
    active: Annotated[Tensor, TensorField("active_with_bootstrap", ("B", "T+1", "N"), torch.bool)]
    starts_with_episode: Annotated[Tensor, TensorField("starts_with_episode", ("B",), torch.bool)]
    action_masks: Annotated[
        Tensor | None, TensorField("action_masks_with_bootstrap", ("B", "T+1", "N", "A"), torch.bool)
    ]
    old_log_probs: Annotated[
        Tensor | None, TensorField("behavior_log_probs", ("B", "T", "N"), torch.float32)
    ]
    temperature: Annotated[
        Tensor | None, TensorField("behavior_temperature", ("B", "T", "N"), torch.float32)
    ]
    hidden_in: Annotated[Tensor | None, TensorField("hidden_in", ("B", "N", "L", "H"), torch.float32)]


class DefaultEpisodeFields(EpisodeTensorFields):
    burn_in: Annotated[list[EpisodeTensorFields | None], ListField("burn_in", "B")]


class MaddpgEpisodeFields(TypedDict):
    """ MADDPG 算法定制化 batch schema，默认 schema 在 MADDPG 的应用中性能较差。 """
    obs: Annotated[Tensor, TensorField("obs_with_bootstrap", ("B", "T+1", "N", "O"), torch.float32)]
    actions: Annotated[Tensor, TensorField("actions", ("B", "T", "N", "D"), None)]
    last_actions: Annotated[Tensor, TensorField("last_actions", ("B", "T", "N", "D"), None)]
    rewards: Annotated[Tensor, TensorField("rewards", ("B", "T", "N"), torch.float32)]
    team_rewards: Annotated[Tensor, TensorField("team_rewards", ("B", "T"), torch.float32)]
    terminated: Annotated[Tensor, TensorField("terminated", ("B", "T", "N"), torch.bool)]
    truncated: Annotated[Tensor, TensorField("truncated", ("B", "T", "N"), torch.bool)]
    valid: Annotated[Tensor, TensorField("valid", ("B", "T"), torch.bool)]
    active: Annotated[Tensor, TensorField("active_with_bootstrap", ("B", "T+1", "N"), torch.bool)]


class SingleAgentFields(TypedDict):
    obs: Annotated[Tensor, TensorField("single_obs", ("M", "O"), torch.float32)]
    actions: Annotated[Tensor, TensorField("single_actions", ("M", "D"), torch.long)]
    last_actions: Annotated[Tensor, TensorField("single_last_actions", ("M", "D"), torch.long)]
    rewards: Annotated[Tensor, TensorField("single_rewards", ("M", 1), torch.float32)]
    next_obs: Annotated[Tensor, TensorField("single_next_obs", ("M", "O"), torch.float32)]
    terminated: Annotated[Tensor, TensorField("single_terminated", ("M", 1), torch.float32)]
    truncated: Annotated[Tensor, TensorField("single_truncated", ("M", 1), torch.float32)]
    episode_starts: Annotated[Tensor, TensorField("single_episode_starts", ("M", 1), torch.float32)]
    avail_actions: Annotated[Tensor | None, TensorField("single_avail_actions", ("M", "A"), torch.float32)]
    old_log_probs: Annotated[Tensor | None, TensorField("single_old_log_probs", ("M", 1), torch.float32)]
    old_values: Annotated[Tensor | None, TensorField("single_old_values", ("M", 1), torch.float32)]
    returns: Annotated[Tensor | None, TensorField("single_returns", ("M", 1), torch.float32)]
    advantages: Annotated[Tensor | None, TensorField("single_advantages", ("M", 1), torch.float32)]
    metadata: Annotated[dict, DictField("single_metadata")]


FieldsT = TypeVar("FieldsT")


@dataclass(frozen=True)
class _Field:
    descriptor: TensorField | ListField | DictField | None
    field_type: object
    optional: bool
    children: object = None


def _optional(field_type):
    origin = get_origin(field_type)
    return origin in (Union, UnionType) and type(None) in get_args(field_type)


def _matches_type(value, annotation):
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _matches_type(value, args[0])
    if origin in (Union, UnionType):
        return any(_matches_type(value, item) for item in args)
    if is_typeddict(annotation):
        return isinstance(value, dict)
    if origin is list:
        return isinstance(value, list) and (not args or all(_matches_type(item, args[0]) for item in value))
    if origin is dict:
        return isinstance(value, dict) and (not args or all(
            _matches_type(key, args[0]) and _matches_type(item, args[1]) for key, item in value.items()
        ))
    return isinstance(value, annotation)


def _validate_descriptor(name, field_type, descriptor):
    if not isinstance(descriptor.source, str) or not descriptor.source:
        raise TypeError(f"Batch field {name!r} requires a nonempty source")
    base_types = tuple(item for item in get_args(field_type) if item is not type(None)) if _optional(field_type) else (field_type,)
    if len(base_types) != 1:
        raise TypeError(f"Batch field {name!r} must declare one value type, optionally None")
    base_type = base_types[0]
    if isinstance(descriptor, TensorField):
        if base_type is not Tensor:
            raise TypeError(f"Batch field {name!r} with TensorField must have Tensor type")
        if descriptor.dtype is not None and not isinstance(descriptor.dtype, torch.dtype):
            raise TypeError(f"Batch field {name!r} dtype must be torch.dtype or None")
        if (not isinstance(descriptor.axes, tuple) or not descriptor.axes or
                any(not ((isinstance(axis, str) and axis) or
                         (type(axis) is int and axis > 0)) for axis in descriptor.axes)):
            raise TypeError(f"Batch field {name!r} axes must be a nonempty tuple of symbols or positive sizes")
        symbols = [axis for axis in descriptor.axes if isinstance(axis, str)]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"Batch field {name!r} has repeated axis symbols")
    elif isinstance(descriptor, ListField):
        if get_origin(base_type) is not list and base_type is not list:
            raise TypeError(f"Batch field {name!r} with ListField must have list type")
        if not isinstance(descriptor.axis, str) or not descriptor.axis:
            raise TypeError(f"Batch field {name!r} list axis must be a nonempty symbol")
    elif get_origin(base_type) is not dict and base_type is not dict and not is_typeddict(base_type):
        raise TypeError(f"Batch field {name!r} with DictField must have dict type")


def _read_fields(fields_type):
    if not is_typeddict(fields_type):
        raise TypeError("BatchSchema fields must be declared as a TypedDict")
    result = {}
    for name, annotation in get_type_hints(fields_type, include_extras=True).items():
        if get_origin(annotation) is Annotated:
            field_type, *metadata = get_args(annotation)
            descriptors = [item for item in metadata if isinstance(item, (TensorField, ListField, DictField))]
            if len(descriptors) != 1:
                raise TypeError(f"Batch field {name!r} requires exactly one field descriptor")
            _validate_descriptor(name, field_type, descriptors[0])
            result[name] = _Field(descriptors[0], field_type, _optional(field_type))
        elif is_typeddict(annotation):
            result[name] = _Field(None, annotation, False, _read_fields(annotation))
        else:
            raise TypeError(f"Batch field {name!r} requires a field descriptor or nested TypedDict")
    return MappingProxyType(result)


@dataclass(frozen=True)
class BatchSchema(Generic[FieldsT]):
    fields_type: type
    fields: object

    @classmethod
    def from_type(cls, fields_type):
        return cls(fields_type=fields_type, fields=_read_fields(fields_type))

    def sample_sources(self, *, required_only=False):
        built_in = {
            "obs_with_bootstrap", "state_with_bootstrap", "actions", "last_actions",
            "rewards", "team_rewards", "terminated", "team_terminated", "truncated", "valid", "active_with_bootstrap",
            "active_steps", "starts_with_episode", "action_masks_with_bootstrap",
            "hidden_in", "burn_in", "seq_len",
        }
        sources = set()

        def visit(fields):
            for field in fields.values():
                if field.children is not None:
                    visit(field.children)
                else:
                    source = field.descriptor.source
                    if (not required_only or not field.optional) and source not in built_in and not source.startswith("single_"):
                        sources.add(source)

        visit(self.fields)
        return sorted(sources)

    def uses_source(self, source):
        """判断 schema 是否声明了某个数据来源，供采样侧裁剪物化结果。"""
        def visit(fields):
            for field in fields.values():
                if field.children is not None:
                    if visit(field.children):
                        return True
                elif field.descriptor is not None and field.descriptor.source == source:
                    return True
            return False

        return visit(self.fields)


DEFAULT_EPISODE_SCHEMA = BatchSchema.from_type(DefaultEpisodeFields)
MADDPG_EPISODE_SCHEMA = BatchSchema.from_type(MaddpgEpisodeFields)
SINGLE_AGENT_SCHEMA = BatchSchema.from_type(SingleAgentFields)


def _recorded_action(space, value):
    if value is None:
        return None
    if isinstance(value, dict):
        return ActionSpace.restore(value, expected_space=space)
    if space.kind == "discrete" and not space.strict_masks:
        value = int(np.clip(value, 0, space.layout["category_counts"][0] - 1))
    return space.decode(value)


def _recorded_mask(space, value):
    if space.kind == "discrete" and not space.strict_masks:
        # Legacy environments sometimes returned fewer or more entries than the
        # action table. The old trajectory converter retained the available prefix
        # and left any missing entries enabled.
        vector = np.asarray(value)
        if vector.ndim == 1:
            count = space.layout["category_counts"][0]
            result = np.ones(count, dtype=bool)
            result[:min(count, len(vector))] = vector[:count]
            return result
    return action_mask_vector(space, value)


def _episode_sources(episodes: Sequence[EpisodeTransition], agent, device, max_seq_len=None):
    """Build canonical B-first sources without changing replay's selected windows."""
    if not episodes or any(ep.seq_len <= 0 for ep in episodes):
        raise ValueError("Batch assembly requires nonempty trajectories")
    space = agent.action_space
    layout = space.layout
    batch_size = len(episodes)
    steps = max(ep.seq_len for ep in episodes)
    if max_seq_len is not None:
        if max_seq_len < steps:
            raise ValueError("max_seq_len is shorter than a sampled trajectory")
        steps = max_seq_len
    n_agents = agent.n_agents
    obs_dim = agent.config.obs_dim
    state_dim = np.asarray(episodes[0].states[0]).size
    action_dim = layout["value_dim"]
    action_dtype = np.int64 if layout["kind"] in ("discrete", "multi_discrete") else np.float32
    masks_dim = sum(layout["category_counts"])
    layers = agent.config.num_layers
    hidden_dim = agent.config.hidden_dim
    dims = {
        "B": batch_size, "T": steps, "T+1": steps + 1, "N": n_agents,
        "O": obs_dim, "S": state_dim, "D": action_dim, "A": masks_dim,
        "L": layers, "H": hidden_dim,
    }
    shape = (batch_size, steps, n_agents)
    arrays = {
        "obs_with_bootstrap": np.zeros((batch_size, steps + 1, n_agents, obs_dim), np.float32),
        "state_with_bootstrap": np.zeros((batch_size, steps + 1, state_dim), np.float32),
        "actions": np.zeros((*shape, action_dim), action_dtype),
        "last_actions": np.zeros((*shape, action_dim), action_dtype),
        "rewards": np.zeros(shape, np.float32),
        "team_rewards": np.zeros((batch_size, steps), np.float32),
        "terminated": np.zeros(shape, bool),
        "team_terminated": np.zeros((batch_size, steps), bool),
        "truncated": np.zeros(shape, bool),
        "valid": np.zeros((batch_size, steps), bool),
        "active_with_bootstrap": np.zeros((batch_size, steps + 1, n_agents), bool),
        "starts_with_episode": np.zeros(batch_size, bool),
        "action_masks_with_bootstrap": (
            np.ones((batch_size, steps + 1, n_agents, masks_dim), bool) if masks_dim else None
        ),
        "behavior_log_probs": np.zeros(shape, np.float32),
        "behavior_temperature": np.ones(shape, np.float32),
        "hidden_in": np.zeros((batch_size, n_agents, layers, hidden_dim), np.float32),
    }
    complete_logs = True
    has_hidden = False
    extra = {}
    for b, episode in enumerate(episodes):
        ids = tuple(episode.agent_ids) or tuple(agent.agent_list)
        if len(ids) != n_agents:
            raise ValueError(f"trajectory {episode.episode_num} has {len(ids)} slots, expected {n_agents}")
        episode.validate_activity_masks(ids)
        for aid, spec in episode.action_space_specs.items():
            if aid in ids and ActionSpace.from_dict(spec).signature != space.signature:
                raise ValueError("replay action layout does not match learner action space")
        length = episode.seq_len
        arrays["valid"][b, :length] = True
        arrays["team_rewards"][b, :length] = episode.step_rewards[:length]
        arrays["starts_with_episode"][b] = episode.starts_with_episode
        arrays["state_with_bootstrap"][b, :length] = np.asarray(episode.states).reshape(length, state_dim)
        arrays["state_with_bootstrap"][b, length] = np.asarray(episode.next_states[-1]).reshape(state_dim)
        for n, aid in enumerate(ids):
            arrays["obs_with_bootstrap"][b, :length, n] = np.asarray(episode.obs[aid]).reshape(length, obs_dim)
            arrays["obs_with_bootstrap"][b, length, n] = np.asarray(episode.next_obs[aid][-1]).reshape(obs_dim)
            arrays["rewards"][b, :length, n] = episode.rewards[aid][:length]
            arrays["terminated"][b, :length, n] = episode.terminations[aid][:length]
            arrays["truncated"][b, :length, n] = episode.truncations[aid][:length]
            active = episode.active_agent_mask.get(aid)
            arrays["active_with_bootstrap"][b, :length, n] = active[:length] if active is not None else True
            next_active = episode.next_active_agent_mask.get(aid)
            if active is None:
                pass
            elif next_active is not None and len(next_active) >= length:
                arrays["active_with_bootstrap"][b, length, n] = next_active[length - 1]
            else:
                metadata = episode.metadata[length - 1] or {} if length <= len(episode.metadata) else {}
                next_id = metadata.get("next_active_agent_id")
                if next_id is not None:
                    arrays["active_with_bootstrap"][b, length, n] = aid == next_id
                else:
                    available = (
                        episode.avail_actions_next[length - 1].get(aid)
                        if length <= len(episode.avail_actions_next) else None
                    )
                    arrays["active_with_bootstrap"][b, length, n] = (
                        available is not None and bool(any(available))
                    )
            hidden = (
                episode.hidden_in.get(aid)
                if agent.agent_type not in {"rnn", "rnn_ns"} else None
            )
            if hidden is not None:
                arrays["hidden_in"][b, n] = np.asarray(torch.as_tensor(hidden).cpu()).reshape(layers, hidden_dim)
                has_hidden = True
            for t in range(length):
                for source, field in (("actions", episode.actions), ("last_actions", episode.last_actions)):
                    value = _recorded_action(space, field[aid][t])
                    if value is not None:
                        arrays[source][b, t, n] = np.asarray(space.encode(value)).reshape(action_dim)
                if masks_dim:
                    current = episode.avail_actions[t].get(aid) if t < len(episode.avail_actions) else None
                    following = (
                        episode.avail_actions_next[t].get(aid)
                        if t < len(episode.avail_actions_next) else None
                    )
                    if current is not None:
                        arrays["action_masks_with_bootstrap"][b, t, n] = _recorded_mask(space, current)
                    if following is not None:
                        arrays["action_masks_with_bootstrap"][b, t + 1, n] = _recorded_mask(space, following)
                metadata = episode.metadata[t] or {} if t < len(episode.metadata) else {}
                logs = metadata.get("behavior_log_probs", {})
                if arrays["active_with_bootstrap"][b, t, n]:
                    if aid in logs:
                        arrays["behavior_log_probs"][b, t, n] = logs[aid]
                        arrays["behavior_temperature"][b, t, n] = logs.get("__temperature__", 1.0)
                    else:
                        complete_logs = False
                for name, values in metadata.get("batch_fields", {}).items():
                    if aid in values:
                        extra.setdefault(name, {})[(b, t, n)] = values[aid]
        for t in range(length):
            next_active = []
            for aid in ids:
                recorded = episode.next_active_agent_mask.get(aid)
                if recorded is not None and t < len(recorded):
                    next_active.append(bool(recorded[t]))
                else:
                    available = episode.avail_actions_next[t].get(aid) if t < len(episode.avail_actions_next) else None
                    next_active.append(available is not None and bool(any(available)))
            arrays["team_terminated"][b, t] = (
                all(episode.terminations[aid][t] for aid in ids) or not any(next_active)
            )
    if not complete_logs:
        arrays["behavior_log_probs"] = None
        arrays["behavior_temperature"] = None
    if not has_hidden:
        arrays["hidden_in"] = None
    sources = {
        key: torch.as_tensor(value, device=device) if value is not None else None
        for key, value in arrays.items()
    }
    sources["active_steps"] = sources["active_with_bootstrap"][:, :-1]
    sources["seq_len"] = [episode.seq_len for episode in episodes]
    sources["_extra"] = extra
    return sources, dims


_SOURCE_AXES = {
    "obs_with_bootstrap": ("B", "T+1", "N", "O"),
    "state_with_bootstrap": ("B", "T+1", "S"),
    "actions": ("B", "T", "N", "D"),
    "last_actions": ("B", "T", "N", "D"),
    "rewards": ("B", "T", "N"),
    "team_rewards": ("B", "T"),
    "terminated": ("B", "T", "N"),
    "team_terminated": ("B", "T"),
    "truncated": ("B", "T", "N"),
    "valid": ("B", "T"),
    "active_with_bootstrap": ("B", "T+1", "N"),
    "active_steps": ("B", "T", "N"),
    "starts_with_episode": ("B",),
    "action_masks_with_bootstrap": ("B", "T+1", "N", "A"),
    "behavior_log_probs": ("B", "T", "N"),
    "behavior_temperature": ("B", "T", "N"),
    "hidden_in": ("B", "N", "L", "H"),
    "single_obs": ("M", "O"),
    "single_actions": ("M", "D"),
    "single_last_actions": ("M", "D"),
    "single_rewards": ("M", 1),
    "single_next_obs": ("M", "O"),
    "single_terminated": ("M", 1),
    "single_truncated": ("M", 1),
    "single_episode_starts": ("M", 1),
    "single_avail_actions": ("M", "A"),
    "single_old_log_probs": ("M", 1),
    "single_old_values": ("M", 1),
    "single_returns": ("M", 1),
    "single_advantages": ("M", 1),
}


def _assemble_fields(fields, sources, dims, episodes, agent, device):
    result = {}
    for name, field in fields.items():
        if field.children is not None:
            result[name] = _assemble_fields(field.children, sources, dims, episodes, agent, device)
            continue
        descriptor = field.descriptor
        if isinstance(descriptor, DictField):
            value = sources.get(descriptor.source)
            if value is None and field.optional:
                result[name] = None
                continue
            if not isinstance(value, dict):
                raise ValueError(f"Missing dictionary source {descriptor.source!r} for {name!r}")
            if not _matches_type(value, field.field_type):
                raise TypeError(f"Batch field {name!r} contains values inconsistent with its dictionary type")
            result[name] = deepcopy(value)
            continue
        if isinstance(descriptor, ListField):
            if descriptor.source == "burn_in":
                values = []
                prefix_schema = BatchSchema.from_type(EpisodeTensorFields)
                for episode in episodes:
                    prefix = episode.burn_in
                    values.append(None if prefix is None else assemble_episodes(
                        [prefix], prefix_schema, agent=agent, device=device,
                    ).data)
            elif descriptor.source in sources["_extra"] and descriptor.axis == "B":
                entries = sources["_extra"][descriptor.source]
                values = []
                row = 0
                for b, episode in enumerate(episodes):
                    trajectory = []
                    ids = (tuple(agent.agent_list) if "M" in dims else
                           tuple(episode.agent_ids) or tuple(agent.agent_list))
                    for step in range(episode.seq_len):
                        by_agent = {}
                        for n, aid in enumerate(ids):
                            index = (row,) if "M" in dims else (b, step, n)
                            active = "M" in dims or bool(sources["active_steps"][b, step, n])
                            if active and index not in entries:
                                raise ValueError(
                                    f"Learner {agent.name} episode={episode.episode_num} step={step} "
                                    f"agent={aid} missing field {descriptor.source!r}"
                                )
                            if index in entries:
                                by_agent[aid] = deepcopy(entries[index])
                        trajectory.append(by_agent)
                        row += 1
                    values.append(trajectory)
            else:
                values = sources.get(descriptor.source)
            if values is None:
                if field.optional:
                    result[name] = None
                    continue
                raise ValueError(f"Missing batch source {descriptor.source!r} for {name!r}")
            if descriptor.axis not in dims:
                raise ValueError(f"Unknown list axis {descriptor.axis!r} for batch field {name!r}")
            if len(values) != dims[descriptor.axis]:
                raise ValueError(f"Invalid list size for batch field {name!r}")
            if not _matches_type(values, field.field_type):
                raise TypeError(f"Batch field {name!r} contains values inconsistent with its list type")
            result[name] = values
            continue
        value = sources.get(descriptor.source)
        source_axes = _SOURCE_AXES.get(descriptor.source)
        if value is None and descriptor.source in sources["_extra"]:
            entries = sources["_extra"][descriptor.source]
            base_axes = ("M",) if "M" in dims else ("B", "T", "N")
            feature_axes = tuple(axis for axis in descriptor.axes if axis not in base_axes)
            first = torch.as_tensor(next(iter(entries.values())))
            if first.ndim != len(feature_axes):
                raise ValueError(f"Batch field {name!r} has incompatible per-agent feature dimensions")
            for axis, size in zip(feature_axes, first.shape):
                if isinstance(axis, int):
                    if axis != size:
                        raise ValueError(f"Batch field {name!r} expected feature size {axis}, got {size}")
                elif axis in dims and dims[axis] != size:
                    raise ValueError(f"Batch field {name!r} expected {axis}={dims[axis]}, got {size}")
                else:
                    dims[axis] = size
            source_axes = (*base_axes, *feature_axes)
            shape = tuple(dims[axis] if isinstance(axis, str) else axis for axis in source_axes)
            value = torch.zeros(shape, dtype=descriptor.dtype or first.dtype, device=device)
            if base_axes == ("M",):
                required = [(index,) for index in range(dims["M"])]
            else:
                mask = sources["active_steps"] & sources["valid"].unsqueeze(-1)
                required = [tuple(index) for index in mask.nonzero().tolist()]
            for index in required:
                if index not in entries:
                    if base_axes == ("M",):
                        b, t = sources["_rows"][index[0]]
                        aid = agent.agent_list[0]
                    else:
                        b, t, n = index
                        ids = tuple(episodes[b].agent_ids) or tuple(agent.agent_list)
                        aid = ids[n]
                    raise ValueError(
                        f"Learner {agent.name} episode={episodes[b].episode_num} step={t} "
                        f"agent={aid} missing field {descriptor.source!r}"
                    )
            for index, item in entries.items():
                item = torch.as_tensor(item, dtype=value.dtype, device=device)
                if tuple(item.shape) != tuple(first.shape):
                    raise ValueError(f"Batch field {name!r} has inconsistent per-agent feature shapes")
                value[index] = item
        if value is None:
            if field.optional:
                result[name] = None
                continue
            if descriptor.source not in _SOURCE_AXES:
                if "M" in dims and sources["_rows"]:
                    b, t = sources["_rows"][0]
                    aid = agent.agent_list[0]
                else:
                    mask = sources["active_steps"] & sources["valid"].unsqueeze(-1)
                    positions = mask.nonzero().tolist()
                    b, t, n = positions[0] if positions else (0, 0, 0)
                    ids = tuple(episodes[b].agent_ids) or tuple(agent.agent_list)
                    aid = ids[n]
                raise ValueError(
                    f"Learner {agent.name} episode={episodes[b].episode_num} step={t} "
                    f"agent={aid} missing field {descriptor.source!r}"
                )
            raise ValueError(f"Missing batch source {descriptor.source!r} for {name!r}")
        if source_axes is None or Counter(source_axes) != Counter(descriptor.axes):
            raise ValueError(f"Invalid axes {descriptor.axes!r} for source {descriptor.source!r}")
        remaining = list(enumerate(source_axes))
        permutation = []
        for axis in descriptor.axes:
            offset = next(index for index, (_, source_axis) in enumerate(remaining) if source_axis == axis)
            permutation.append(remaining.pop(offset)[0])
        value = value.permute(*permutation)
        if descriptor.dtype is not None:
            value = value.to(dtype=descriptor.dtype)
        time_axis = next((axis for axis in ("T", "T+1") if axis in descriptor.axes), None)
        if descriptor.padding_value is not None and time_axis is not None and "B" in descriptor.axes:
            lengths = torch.tensor(sources["seq_len"], device=value.device)
            positions = torch.arange(dims[time_axis], device=value.device)
            valid_positions = positions.unsqueeze(0) < (
                lengths.unsqueeze(1) + (1 if time_axis == "T+1" else 0)
            )
            if descriptor.axes.index(time_axis) < descriptor.axes.index("B"):
                valid_positions = valid_positions.transpose(0, 1)
            mask_shape = [1] * len(descriptor.axes)
            mask_shape[descriptor.axes.index("B")] = dims["B"]
            mask_shape[descriptor.axes.index(time_axis)] = dims[time_axis]
            value = value.masked_fill(~valid_positions.reshape(mask_shape), descriptor.padding_value)
        expected = tuple(dims[axis] if isinstance(axis, str) else axis for axis in descriptor.axes)
        expected_device = torch.device(device)
        device_matches = (value.device.type == expected_device.type and
                          (expected_device.index is None or value.device.index == expected_device.index))
        if tuple(value.shape) != expected or not device_matches:
            raise ValueError(f"Batch field {name!r} has shape/device {tuple(value.shape)}/{value.device}, "
                             f"expected {expected}/{device}")
        result[name] = value
    return result


def assemble_episodes(episodes: Sequence[EpisodeTransition], schema: BatchSchema[FieldsT], *,
                      agent, device, max_seq_len=None,
                      sample_strategy: str | None = None) -> EpisodeBatch[FieldsT]:
    """按 schema 组批，抽样策略由调用方显式传入。"""
    if "burn_in" in schema.fields:
        for episode in episodes:
            prefix = episode.burn_in
            if prefix is None:
                if not episode.starts_with_episode:
                    raise ValueError(f"trajectory {episode.episode_num} is missing burn-in from reset")
                continue
            if (prefix.seq_len <= 0 or episode.seq_len <= 0
                    or tuple(prefix.agent_ids) != tuple(episode.agent_ids)
                    or prefix.episode_num != episode.episode_num
                    or not prefix.starts_with_episode
                    or prefix.burn_in is not None):
                raise ValueError(f"trajectory {episode.episode_num} has invalid burn-in history")
            if (not np.array_equal(prefix.next_states[-1], episode.states[0])
                    or any(not np.array_equal(prefix.next_obs[aid][-1], episode.obs[aid][0])
                           for aid in episode.agent_ids)):
                raise ValueError(f"trajectory {episode.episode_num} burn-in is not contiguous")
    sources, dims = _episode_sources(episodes, agent, device, max_seq_len=max_seq_len)
    return EpisodeBatch(
        _assemble_fields(schema.fields, sources, dims, episodes, agent, device),
        sample_strategy=sample_strategy,
    )


def assemble_single_agent(episodes: Sequence[SAEpisodeTransition], schema: BatchSchema[FieldsT], *,
                          agent, device, sample_strategy: str | None = None) -> EpisodeBatch[FieldsT]:
    """组装单智能体批，抽样策略由调用方显式传入。"""
    from utils.transition_hash import single_transition_key, transition_batch_hash
    from sample.sample_transfer import SAEpisodeTransition

    original = SAEpisodeTransition.to_batch_tensor(episodes, device=device)
    sources = {f"single_{name}": value for name, value in original.data.items()}
    sources["_extra"] = {}
    sources["_rows"] = []
    sources["seq_len"] = [episode.seq_len for episode in episodes]
    aid = agent.agent_list[0]
    for b, episode in enumerate(episodes):
        for step in range(episode.seq_len):
            row = len(sources["_rows"])
            sources["_rows"].append((b, step))
            metadata = episode.metadata[step] or {} if step < len(episode.metadata) else {}
            for name, values in metadata.get("batch_fields", {}).items():
                if aid in values:
                    sources["_extra"].setdefault(name, {})[(row,)] = values[aid]
    sources["single_metadata"] = {
        "batch_hash": transition_batch_hash(episodes),
        "transition_keys": [
            single_transition_key(episode, step)
            for episode in episodes for step in range(episode.seq_len)
        ],
    }
    dims = {
        "B": len(episodes), "M": original.obs.shape[0], "O": original.obs.shape[1],
        "D": original.actions.shape[1],
        "A": 0 if original.avail_actions is None else original.avail_actions.shape[1],
    }
    return EpisodeBatch(
        _assemble_fields(schema.fields, sources, dims, episodes, agent, device),
        sample_strategy=sample_strategy,
    )
