from itertools import product
from typing import Any, Dict, Generic, List, TypeVar

from envs.action.model import (
    Action,
    ActionDefined,
    ActionParameterDefined,
    CompositeAction,
    EpisodeActionDefined,
    StepActionDefined,
)

from .utils import generate_range

ActionDefine = TypeVar("ActionDefine", ActionDefined, StepActionDefined, EpisodeActionDefined)
ActionEntry = Action | CompositeAction


class ActionTable(Generic[ActionDefine]):
    """把业务动作定义展开为带连续整数索引的离散动作表。"""

    def __init__(
        self,
        action_config: List[ActionDefine],
        targets=None,
        multi_action_mode: str = "single",
    ):
        if multi_action_mode not in {"single", "combination", "free"}:
            raise ValueError(f"multi_action_mode input error: {multi_action_mode}")
        self.action_config = []
        for action in action_config:
            if isinstance(action, ActionDefined):
                if action.head_type != "single" or action.action_type != "discrete":
                    raise ValueError("ActionTable only accepts single-head discrete ActionDefined")
                if len(action.action_key) != 1:
                    raise ValueError("single-head ActionDefined requires exactly one action_key")
                definition = action.action_key[0]
                self.action_config.append(StepActionDefined(
                    action_key=definition.key,
                    parameters=list(definition.parameters),
                    add_target=definition.add_target,
                    allow_null_targets=definition.allow_null_targets,
                    role=definition.role,
                    allow_empty_action=definition.allow_empty_action,
                ))
            else:
                self.action_config.append(action)
        self.targets_name_list = targets
        self.targets_num = len(targets) if targets else 0
        self.multi_action_mode = multi_action_mode
        self.action_map: Dict[int, ActionEntry] = self._analyze_action()

    @property
    def n_actions(self) -> int:
        """Return the live table size, including entries added in free mode."""
        expected = set(range(len(self.action_map)))
        if set(self.action_map) != expected:
            raise ValueError(
                "action_map keys must be contiguous indices starting at 0: "
                f"actual={sorted(self.action_map)}"
            )
        return len(self.action_map)

    def _analyze_action(self) -> Dict[int, ActionEntry]:
        empty_action_enabled = any(
            isinstance(action, StepActionDefined) and action.allow_empty_action
            for action in self.action_config
        )
        empty_action = Action(
            action_key="",
            parameters={},
            target=-2,
            action_idx=0,
            is_empty_action=True,
        )
        if self.multi_action_mode == "free":
            return {0: empty_action} if empty_action_enabled else {}
        expanded = [self._expand_action(action) for action in self.action_config]
        if self.multi_action_mode == "single":
            flat = [action for choices in expanded for action in choices]
            offset = int(empty_action_enabled)
            self._validate_size(len(flat) + offset)
            action_map: Dict[int, ActionEntry] = {}
            if empty_action_enabled:
                action_map[0] = empty_action
            action_map.update({
                index + offset: self._with_index(action, index + offset)
                for index, action in enumerate(flat)
            })
            return action_map

        if not expanded:
            return {0: empty_action} if empty_action_enabled else {}
        total = 1
        for choices in expanded:
            total *= len(choices)
        offset = int(empty_action_enabled)
        self._validate_size(total + offset)
        action_map: Dict[int, ActionEntry] = {}
        if empty_action_enabled:
            action_map[0] = empty_action
        for raw_index, choices in enumerate(product(*expanded)):
            index = raw_index + offset
            action_map[index] = CompositeAction(
                actions=[self._with_index(action, index) for action in choices],
                action_idx=index,
            )
        return action_map

    def _expand_action(self, action: ActionDefine) -> List[Action]:
        params_values = [self._expand_parameter(parameter) for parameter in action.parameters]
        keys = [parameter.key for parameter in action.parameters]
        combos = list(product(*params_values)) if params_values else [()]
        if isinstance(action, StepActionDefined):
            add_target = action.add_target
            allow_null_targets = action.allow_null_targets
            step_num = None
            order_rules = [None]
        else:
            add_target = False
            allow_null_targets = False
            step_num = action.step_num
            order_rules = action.agent_order_rules or [None]

        target_values = [-2]
        if add_target:
            target_values = list(range(-1, self.targets_num))
            if not allow_null_targets:
                target_values = [target for target in target_values if target != -1]

        expanded: List[Action] = []
        for order_rule, combo, target in product(order_rules, combos, target_values):
            expanded.append(Action(
                action_key=action.action_key,
                parameters=dict(zip(keys, combo)),
                target=target,
                action_idx=-1,
                step_num=step_num,
                agent_order_rule=order_rule,
            ))
        self._validate_size(len(expanded))
        return expanded

    @staticmethod
    def _with_index(action: Action, index: int) -> Action:
        return Action(
            action_key=action.action_key,
            parameters=dict(action.parameters),
            target=action.target,
            action_idx=index,
            is_empty_action=action.is_empty_action,
            step_num=action.step_num,
            agent_order_rule=action.agent_order_rule,
            metadata=dict(action.metadata),
        )

    @staticmethod
    def _validate_size(size: int) -> None:
        if size > 1_000_000:
            raise MemoryError(
                f"The action space is too large: ({size})! please narrow down the action parameters."
            )

    @staticmethod
    def _expand_parameter(parameter: ActionParameterDefined) -> List[Any]:
        if parameter.values is not None:
            return list(parameter.values)
        return list(generate_range(parameter.min, parameter.max, parameter.step))
