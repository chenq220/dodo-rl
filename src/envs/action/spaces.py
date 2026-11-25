"""Typed action spaces shared by SDK, policies and rollout workers.

Business actions hold physical values. encode/decode use policy coordinates:
integer indices for categorical heads and float32 [-1, 1] for bounded heads.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from .base import ActionTable
from .model import Action, ActionDefined, CompositeAction, StepActionDefined

ACTION_SPACE_KINDS = frozenset({"discrete", "multi_discrete", "continuous", "multi_continuous"})


def _name(value, label="name"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _real(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"expected a finite real number, got {value!r}")
    return float(value)


def _vector(value, size, *, integer=False):
    if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("action vectors cannot contain boolean values")
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in ("iu" if integer else "iuf"):
        raise ValueError(f"expected {'integer' if integer else 'numeric'} vector shape ({size},)")
    if not np.isfinite(array).all():
        raise ValueError("action vector must be finite")
    return array


@dataclass(frozen=True)
class ContinuousParameter:
    """A named physical parameter with inclusive, finite bounds; never enumerated."""
    key: str
    low: float
    high: float
    unit: str | None = None

    def __post_init__(self):
        _name(self.key, "parameter key")
        low, high = _real(self.low), _real(self.high)
        if low >= high or not np.isfinite(high - low):
            raise ValueError("continuous bounds require finite low < high and finite width")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass
class ContinuousAction:
    """One business command, with physical (not normalized) parameter values."""
    action_key: str
    parameters: dict[str, float]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        _name(self.action_key, "action_key")
        self.parameters = {_name(k): _real(v) for k, v in self.parameters.items()}
        self.metadata = deepcopy(self.metadata or {})

    def to_dict(self):
        return {"format_version": 1, "kind": "continuous", "action_key": self.action_key,
                "parameters": dict(self.parameters), "metadata": deepcopy(self.metadata)}

    @classmethod
    def from_dict(cls, data):
        _version(data)
        if data.get("kind") != "continuous":
            raise ValueError("expected continuous action")
        return cls(data["action_key"], data["parameters"], data.get("metadata"))


@dataclass
class MultiHeadAction:
    """Simultaneous named commands; has no global Cartesian action index."""
    heads: dict[str, Action | CompositeAction | ContinuousAction]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.heads:
            raise ValueError("multi-head action cannot be empty")
        for key in self.heads:
            _name(key, "head name")
        self.heads = dict(self.heads)
        self.metadata = deepcopy(self.metadata or {})

    def __getitem__(self, name):
        return self.heads[name]

    def to_dict(self):
        return {"format_version": 1, "kind": "multi_head", "heads": [
            {"name": name, "action": action.to_dict()} for name, action in self.heads.items()
        ], "metadata": deepcopy(self.metadata)}

    @classmethod
    def from_dict(cls, data):
        _version(data)
        if data.get("kind") != "multi_head":
            raise ValueError("expected multi_head action")
        heads = {}
        for item in data["heads"]:
            if item["name"] in heads:
                raise ValueError("duplicate head name")
            value = item["action"]
            decoder = ContinuousAction if value.get("kind") == "continuous" else (
                CompositeAction if "actions" in value else Action)
            heads[item["name"]] = decoder.from_dict(value)
        return cls(heads, data.get("metadata"))


def _version(data):
    if type(data.get("format_version")) is not int or data["format_version"] != 1:
        raise ValueError("unsupported action format_version")


class ActionSpace(ABC):
    """Common business/policy codec. Specifications are JSON-compatible and ordered."""
    kind: str

    @property
    def signature(self):
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()

    @property
    @abstractmethod
    def layout(self): ...

    @abstractmethod
    def sample(self, rng=None, mask=None): ...

    @abstractmethod
    def validate(self, action): ...

    @abstractmethod
    def encode(self, action): ...

    @abstractmethod
    def decode(self, value): ...

    @abstractmethod
    def encode_previous_action(self, action): ...

    @abstractmethod
    def to_gym_space(self): ...

    @abstractmethod
    def to_dict(self): ...

    @classmethod
    def from_dict(cls, data):
        _version(data)
        kind = data.get("kind")
        if cls is not ActionSpace and kind != cls.kind:
            raise ValueError(f"expected {cls.kind}, got {kind}")
        if kind == "discrete":
            entries = data["actions"]
            table = ActionTable([], multi_action_mode="free")
            for i, item in enumerate(entries):
                value = (CompositeAction if "actions" in item else Action).from_dict(item)
                if type(value.action_idx) is not int or value.action_idx != i:
                    raise ValueError("serialized action indices must be contiguous from zero")
                table.action_map[i] = value
            return DiscreteActionSpace(table, strict_masks=data.get("strict_masks", True))
        if kind == "continuous":
            return ContinuousActionSpace(data["action_key"], [ContinuousParameter(**p) for p in data["parameters"]])
        if kind in {"multi_discrete", "multi_continuous"}:
            multi = MultiDiscreteActionSpace if kind == "multi_discrete" else MultiContinuousActionSpace
            return multi([(item["name"], ActionSpace.from_dict(item["space"])) for item in data["heads"]])
        raise ValueError(f"unknown action space kind: {kind!r}")

    def record(self, action):
        """Portable record; embeds the ordered space so physical actions are recoverable."""
        value = self.encode(action)
        return {"format_version": 1, "space": self.to_dict(), "space_signature": self.signature,
                "value": value.tolist() if isinstance(value, np.ndarray) else int(value),
                "action": action.to_dict()}

    @staticmethod
    def restore(record, expected_space=None):
        _version(record)
        space = ActionSpace.from_dict(record["space"])
        if space.signature != record["space_signature"] or (
            expected_space is not None and expected_space.signature != space.signature
        ):
            raise ValueError("action record space layout/signature mismatch")
        # Validate both policy value and exact physical business value; retain metadata.
        space.decode(record["value"])
        raw = record["action"]
        decoder = MultiHeadAction if space.kind.startswith("multi_") else (
            ContinuousAction if space.kind == "continuous" else (
                CompositeAction if "actions" in raw else Action))
        action = decoder.from_dict(raw)
        space.validate(action)
        if not np.array_equal(space.encode(action), record["value"]):
            raise ValueError("action record value does not match business action")
        return action


class DiscreteActionSpace(ActionSpace):
    """Live adapter over an existing ActionTable, including legacy free mode."""
    kind = "discrete"

    def __init__(self, table: ActionTable, *, strict_masks: bool = True):
        if not isinstance(table, ActionTable):
            raise TypeError("DiscreteActionSpace requires ActionTable")
        self.table = table
        self.strict_masks = strict_masks

    @property
    def layout(self):
        return {"kind": self.kind, "head_count": 1, "category_counts": [self.table.n_actions],
                "continuous_dim": 0, "value_dim": 1, "previous_action_dim": self.table.n_actions}

    def validate(self, action):
        if not isinstance(action, (Action, CompositeAction)):
            raise ValueError("expected a discrete business action")
        index = action.action_idx
        if type(index) is not int or index not in self.table.action_map:
            raise ValueError(f"invalid action index {index!r}")
        expected = self.table.action_map[index].to_dict()
        actual = action.to_dict()
        # Metadata may be annotated by business callers without changing the action.
        expected.pop("metadata", None)
        actual.pop("metadata", None)
        if actual != expected:
            raise ValueError("business action does not match its discrete index")

    def encode(self, action):
        self.validate(action)
        return int(action.action_idx)

    def decode(self, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("discrete action requires an integer index")
        if int(value) not in self.table.action_map:
            raise ValueError(f"invalid action index {value}")
        return deepcopy(self.table.action_map[int(value)])

    def sample(self, rng=None, mask=None):
        rng = rng if rng is not None else np.random.default_rng()
        count = self.table.n_actions
        choices = np.arange(count)
        if mask is not None:
            available = np.asarray(mask)
            if not self.strict_masks and available.ndim == 1:
                # Historical environments can expose a superset mask for their action table.
                available = np.pad(available[:count], (0, max(0, count - len(available))))
            if available.shape != (count,) or available.dtype.kind not in "biuf" or not np.isin(available, [0, 1]).all():
                raise ValueError(f"discrete mask must contain {count} binary entries")
            choices = choices[available == 1]
        if not len(choices):
            raise ValueError("no available discrete actions")
        return self.decode(int(rng.choice(choices)))

    def encode_previous_action(self, action):
        vector = np.zeros(self.table.n_actions, dtype=np.float32)
        if action is not None:
            vector[self.encode(action)] = 1
        return vector

    def to_gym_space(self):
        return Discrete(self.table.n_actions)

    def to_dict(self):
        return {"format_version": 1, "kind": self.kind, "strict_masks": self.strict_masks,
                "actions": [self.table.action_map[i].to_dict() for i in range(self.table.n_actions)]}


class ContinuousActionSpace(ActionSpace):
    kind = "continuous"

    def __init__(self, action_key, parameters):
        self.action_key = _name(action_key, "action_key")
        self.parameters = tuple(parameters)
        if not self.parameters or not all(isinstance(p, ContinuousParameter) for p in self.parameters):
            raise ValueError("expected nonempty ContinuousParameter sequence")
        if len({p.key for p in self.parameters}) != len(self.parameters):
            raise ValueError("duplicate continuous parameter key")
        self._low = np.array([p.low for p in self.parameters], dtype=np.float64)
        self._high = np.array([p.high for p in self.parameters], dtype=np.float64)

    @property
    def layout(self):
        dim = len(self.parameters)
        return {"kind": self.kind, "head_count": 1, "category_counts": [],
                "continuous_dim": dim, "value_dim": dim, "previous_action_dim": dim}

    def validate(self, action):
        if not isinstance(action, ContinuousAction) or action.action_key != self.action_key:
            raise ValueError("continuous business action has incorrect type/action_key")
        if set(action.parameters) != {p.key for p in self.parameters}:
            raise ValueError("continuous action parameter keys do not match space")
        values = np.array([_real(action.parameters[p.key]) for p in self.parameters])
        if np.any(values < self._low) or np.any(values > self._high):
            raise ValueError("continuous physical action is out of bounds")

    def encode(self, action):
        self.validate(action)
        physical = np.array([action.parameters[p.key] for p in self.parameters], dtype=np.float64)
        return (2 * ((physical - self._low) / (self._high - self._low)) - 1).astype(np.float32)

    def decode(self, value):
        vector = _vector(value, len(self.parameters)).astype(np.float64)
        if np.any(vector < -1) or np.any(vector > 1):
            raise ValueError("normalized continuous action must lie in [-1, 1]")
        # Convex interpolation preserves inclusive physical endpoints.
        fraction = (vector + 1) / 2
        physical = (1 - fraction) * self._low + fraction * self._high
        return ContinuousAction(self.action_key, {p.key: float(v) for p, v in zip(self.parameters, physical)})

    def sample(self, rng=None, mask=None):
        if mask is not None:
            raise ValueError("continuous spaces do not accept discrete action masks")
        rng = rng if rng is not None else np.random.default_rng()
        return self.decode(rng.uniform(-1, 1, len(self.parameters)).astype(np.float32))

    def encode_previous_action(self, action):
        return np.zeros(len(self.parameters), dtype=np.float32) if action is None else self.encode(action)

    def to_gym_space(self):
        return Box(-1.0, 1.0, shape=(len(self.parameters),), dtype=np.float32)

    def to_dict(self):
        return {"format_version": 1, "kind": self.kind, "action_key": self.action_key,
                "parameters": [asdict(p) for p in self.parameters]}


class _MultiActionSpace(ActionSpace):
    head_type: type

    def __init__(self, heads):
        items = list(heads.items()) if isinstance(heads, dict) else list(heads)
        if len(items) < 2:
            raise ValueError("multi-head space requires at least two heads")
        self.heads = {}
        for name, space in items:
            _name(name, "head name")
            if name in self.heads:
                raise ValueError(f"duplicate head name: {name}")
            if not isinstance(space, self.head_type):
                raise ValueError(f"{self.kind} heads must all be {self.head_type.__name__}")
            self.heads[name] = space

    @property
    def layout(self):
        offset = 0
        heads = []
        categories = []
        previous_dim = 0
        for name, space in self.heads.items():
            layout = space.layout
            dim = layout["value_dim"]
            heads.append({"name": name, "slice": [offset, offset + dim], **layout})
            offset += dim
            categories.extend(layout["category_counts"])
            previous_dim += layout["previous_action_dim"]
        return {"kind": self.kind, "head_count": len(self.heads), "heads": heads,
                "category_counts": categories, "continuous_dim": 0 if categories else offset,
                "value_dim": offset, "previous_action_dim": previous_dim}

    def validate(self, action):
        if not isinstance(action, MultiHeadAction) or set(action.heads) != set(self.heads):
            raise ValueError("multi-head business action head names do not match space")
        for name, space in self.heads.items():
            space.validate(action.heads[name])

    def sample(self, rng=None, mask=None):
        if self.kind == "multi_continuous" and mask is not None:
            raise ValueError("continuous spaces do not accept discrete action masks")
        if mask is not None and (not isinstance(mask, dict) or set(mask) != set(self.heads)):
            raise ValueError("multi-discrete mask must provide each named head")
        rng = rng if rng is not None else np.random.default_rng()
        return MultiHeadAction({name: space.sample(rng, None if mask is None else mask[name])
                                for name, space in self.heads.items()})

    def encode(self, action):
        self.validate(action)
        if self.kind == "multi_discrete":
            return np.array([space.encode(action.heads[name]) for name, space in self.heads.items()], dtype=np.int64)
        return np.concatenate([space.encode(action.heads[name]) for name, space in self.heads.items()])

    def decode(self, value):
        vector = _vector(value, self.layout["value_dim"], integer=self.kind == "multi_discrete")
        offset = 0
        heads = {}
        for name, space in self.heads.items():
            width = space.layout["value_dim"]
            piece = int(vector[offset]) if self.kind == "multi_discrete" else vector[offset:offset + width]
            heads[name] = space.decode(piece)
            offset += width
        return MultiHeadAction(heads)

    def encode_previous_action(self, action):
        if action is not None:
            self.validate(action)
        return np.concatenate([space.encode_previous_action(None if action is None else action.heads[name])
                               for name, space in self.heads.items()])

    def to_gym_space(self):
        if self.kind == "multi_discrete":
            return MultiDiscrete(self.layout["category_counts"], dtype=np.int64)
        return Box(-1.0, 1.0, shape=(self.layout["value_dim"],), dtype=np.float32)

    def to_dict(self):
        return {"format_version": 1, "kind": self.kind, "heads": [
            {"name": name, "space": space.to_dict()} for name, space in self.heads.items()]}


class MultiDiscreteActionSpace(_MultiActionSpace):
    kind = "multi_discrete"
    head_type = DiscreteActionSpace


class MultiContinuousActionSpace(_MultiActionSpace):
    kind = "multi_continuous"
    head_type = ContinuousActionSpace


def _space_from_head(action_type, actions, targets, multi_action_mode, strict_masks):
    if action_type == "discrete":
        return DiscreteActionSpace(
            ActionTable(actions, targets, multi_action_mode=multi_action_mode),
            strict_masks=strict_masks,
        )
    if action_type != "continuous":
        raise ValueError(f"unknown action_type: {action_type!r}")
    if len(actions) != 1:
        raise ValueError("a continuous action head must define exactly one action")
    action = actions[0]
    if action.add_target or action.allow_null_targets or action.allow_empty_action or targets:
        raise ValueError("continuous action heads do not support targets or empty actions")
    if multi_action_mode != "single":
        raise ValueError("continuous action heads require multi_action_mode='single'")
    parameters = []
    for parameter in action.parameters:
        if parameter.values is not None or parameter.step is not None:
            raise ValueError(
                f"continuous parameter {parameter.key!r} uses min/max bounds and cannot define values/step"
            )
        parameters.append(ContinuousParameter(
            parameter.key, parameter.min, parameter.max, parameter.unit,
        ))
    return ContinuousActionSpace(action.action_key, parameters)


def action_space_from_definition(definition):
    """把一个智能体的统一业务动作定义装配为算法动作空间。"""
    if not isinstance(definition, ActionDefined):
        raise TypeError("action_defined must be an ActionDefined instance")
    if definition.head_type not in {"single", "multi"}:
        raise ValueError("head_type must be 'single' or 'multi'")
    if definition.action_type not in {"discrete", "continuous"}:
        raise ValueError("action_type must be 'discrete' or 'continuous'")
    count = len(definition.action_key)
    if definition.head_type == "single" and count != 1:
        raise ValueError(f"single-head ActionDefined requires exactly one action_key, got {count}")
    if definition.head_type == "multi" and count < 2:
        raise ValueError(f"multi-head ActionDefined requires at least two action_key entries, got {count}")
    spaces = []
    for key_definition in definition.action_key:
        action_key = key_definition.key
        action = StepActionDefined(
            action_key=action_key,
            parameters=list(key_definition.parameters),
            add_target=key_definition.add_target,
            allow_null_targets=key_definition.allow_null_targets,
            role=key_definition.role,
            allow_empty_action=key_definition.allow_empty_action,
        )
        if definition.action_type == "continuous" and (
                key_definition.add_target or key_definition.allow_null_targets or key_definition.targets):
            raise ValueError(f"continuous action {action_key!r} cannot define targets")
        if definition.action_type == "discrete":
            if key_definition.targets and not key_definition.add_target:
                raise ValueError(
                    f"discrete action {action_key!r} defines targets but add_target is false"
                )
            if key_definition.allow_null_targets and not key_definition.add_target:
                raise ValueError(
                    f"discrete action {action_key!r} allows a null target but add_target is false"
                )
            if (key_definition.add_target and not key_definition.targets
                    and not key_definition.allow_null_targets):
                raise ValueError(
                    f"discrete action {action_key!r} enables targets but has no selectable target"
                )
        spaces.append((action_key, _space_from_head(
            definition.action_type,
            [action],
            key_definition.targets,
            key_definition.multi_action_mode,
            key_definition.strict_masks,
        )))
    if definition.head_type == "single":
        return spaces[0][1]
    multi = MultiDiscreteActionSpace if definition.action_type == "discrete" else MultiContinuousActionSpace
    return multi(spaces)


def resolve_action_space(config):
    """优先装配统一业务定义，并兼容旧空间规格和旧离散动作列表。"""
    definition = getattr(config, "action_defined", None)
    spec = getattr(config, "action_space_spec", None)
    if definition is not None:
        if spec is not None or config.define_action:
            raise ValueError("action_defined conflicts with action_space_spec/nonempty define_action")
        if config.multi_action_mode != "single" or config.agent_target_names:
            raise ValueError("action_defined owns action modes and targets; legacy fields must be empty/default")
        return action_space_from_definition(definition)
    if spec is not None:
        if config.define_action:
            raise ValueError("action_space_spec conflicts with nonempty define_action")
        if config.multi_action_mode != "single" or config.agent_target_names:
            raise ValueError("action_space_spec conflicts with legacy multi_action_mode/agent_target_names")
        return ActionSpace.from_dict(spec)
    return DiscreteActionSpace(ActionTable(config.define_action, config.agent_target_names,
                                          multi_action_mode=config.multi_action_mode), strict_masks=False)


def require_action_space_support(implementation, space, *, label=None):
    supported = getattr(implementation, "supported_action_spaces", ("discrete",))
    if space.kind not in supported:
        name = label or type(implementation).__name__
        raise ValueError(f"{name}: requested action space {space.kind!r}; supported={tuple(supported)}")


def space_for_agent(agent):
    space = getattr(agent, "action_space", None)
    return space if isinstance(space, ActionSpace) else DiscreteActionSpace(agent.action_table, strict_masks=False)


def action_from_dict(data, expected_space=None):
    """Restore a new typed record/value or a historical business action dictionary."""
    if "space_signature" in data:
        return ActionSpace.restore(data, expected_space)
    if data.get("kind") == "continuous":
        action = ContinuousAction.from_dict(data)
    elif data.get("kind") == "multi_head":
        action = MultiHeadAction.from_dict(data)
    else:
        action = (CompositeAction if "actions" in data else Action).from_dict(data)
    if expected_space is not None:
        expected_space.validate(action)
    return action
