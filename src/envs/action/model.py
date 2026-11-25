from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Any, Dict, List, Optional


@dataclass
class Action:
    """
    离散动作表中的单个业务动作，是算法动作索引和仿真业务指令之间的桥接对象。

    算法层只需要 action_idx；环境和仿真层读取 action_key、parameters 和 target 生成具体控制命令。
    """

    # 动作名，对应动作配置中的 action_key，也是仿真动作转换时的业务指令类型。
    action_key: str
    # 动作参数字典，key 为参数名，value 为动作表展开后的具体枚举值或步进值。
    parameters: Dict[str, Any]
    # 目标序号：0~n 表示 agent_target_names 下标，-1 表示空目标，-2 表示该动作不支持目标。
    target: int
    # 该动作在 ActionTable.action_map 中的离散索引，也是算法网络输出维度的下标。
    action_idx: int
    # episode级动作可以决策本局或后续阶段执行多少步；普通step动作为空。
    step_num: Optional[int] = None
    # Episode Action可选的顺序语义字段；环境收到完整Action后可选择是否解释该字段。
    agent_order_rule: Optional[str] = None
    # 动作补充说明的元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 是否为空动作；为True时表示当前步骤不执行任何业务指令。
    is_empty_action: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        将前端传来的字典转换为 Action 实例
        """
        return cls(
            action_key=data.get('action_key'),
            parameters=data.get('parameters', {}),
            target=data.get('target', -1),
            action_idx=data.get('action_idx', -1),
            is_empty_action=data.get('is_empty_action', False),
            step_num=data.get('step_num', None),
            agent_order_rule=data.get('agent_order_rule', None),
            metadata=data.get('metadata', {}),
        )

    def to_dict(self):
        """
        将实例转回字典格式
        """
        data = {
            'action_key': self.action_key,
            'parameters': self.parameters,
            'target': self.target,
            'action_idx': self.action_idx,
            'is_empty_action': self.is_empty_action,
            'step_num': self.step_num,
            'agent_order_rule': self.agent_order_rule,
            'metadata': self.metadata,
        }
        return data


@dataclass
class CompositeAction:
    """A single discrete choice composed of several business actions."""

    actions: List[Action]
    action_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = [action.action_key for action in self.actions]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate action_key in CompositeAction: {keys}")
        for action in self.actions:
            action.action_idx = self.action_idx

    def __getitem__(self, action_key: str) -> Action:
        for action in self.actions:
            if action.action_key == action_key:
                return action
        raise KeyError(action_key)

    def get(self, action_key: str, default=None):
        try:
            return self[action_key]
        except KeyError:
            return default

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            actions=[Action.from_dict(item) for item in data.get("actions", [])],
            action_idx=int(data.get("action_idx", -1)),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "action_idx": self.action_idx,
            "metadata": self.metadata,
        }


@dataclass
class ActionParameterDefined(DataClassJsonMixin):
    """
    离散化的动作参数定义。

    values表示显式枚举；min/max/step表示按步长展开。二者选一种即可。
    """

    # 参数名，最终作为Action.parameters的字典key。
    key: str
    # 显式可选值集合；不为None时优先于min/max/step。
    values: Optional[List[Any]] = None
    # 数值离散化的起点，仅在values=None时使用。
    min: Optional[float] = None
    # 数值离散化的终点，仅在values=None时使用。
    max: Optional[float] = None
    # 数值离散化的步长，仅在values=None时使用。
    step: Optional[float] = None
    # 参数的业务单位，只用于描述和界面展示。
    unit: Optional[str] = None


@dataclass
class _LegacyStepActionDefined(DataClassJsonMixin):
    """
    step级动作定义，最终会被环境/仿真执行。

    未给默认值的字段必须由业务方明确配置；allow_null_targets和
    allow_empty_action为可选项。
    """
    # 动作名称
    action_key: str
    # 动作参数
    parameters: List[ActionParameterDefined]
    # 是否支持目标
    add_target: bool = False
    # 是否允许目标，add_target为False时没有意义
    allow_null_targets: bool = False
    # 动作角色
    role: Optional[str] = None
    # 是否在整张动作表的0号位置增加一个“什么都不做”的空动作
    allow_empty_action: bool = False


# 旧科目兼容别名；新业务代码只使用 ActionDefined。
StepActionDefined = _LegacyStepActionDefined


@dataclass
class EpisodeActionDefined(DataClassJsonMixin):
    """
    episode级动作定义，用于让上层agent决策流程控制信息。

    step_num和agent_order_rules为可选项；为空时表示该动作不控制对应流程变量。
    """
    # 动作名称
    action_key: str
    # 动作参数
    parameters: List[ActionParameterDefined]
    # 决策步进轮数
    step_num: Optional[int] = None
    # 可供Episode Agent选择的顺序规则。每个字符串独立展开为一个离散动作，
    # 字符串语义由具体ParallelEnvironment/AECEnvironment解释。
    agent_order_rules: Optional[List[str]] = None
    # 动作角色
    role: Optional[str] = None


@dataclass
class ActionKeyDefined(DataClassJsonMixin):
    """action_key 列表中一个业务动作的名称、参数、目标和离散表配置。"""

    key: str
    parameters: List[ActionParameterDefined] = field(default_factory=list)
    add_target: bool = False
    allow_null_targets: bool = False
    role: Optional[str] = None
    allow_empty_action: bool = False
    targets: List[str] = field(default_factory=list)
    multi_action_mode: str = "single"
    strict_masks: bool = True

    def to_dict(self, encode_json: bool = False) -> Dict[str, Any]:
        return {
            "key": self.key,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "add_target": self.add_target,
            "allow_null_targets": self.allow_null_targets,
            "role": self.role,
            "allow_empty_action": self.allow_empty_action,
            "targets": list(self.targets),
            "multi_action_mode": self.multi_action_mode,
            "strict_masks": self.strict_masks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], infer_missing: bool = False) -> "ActionKeyDefined":
        parameters = data.get("parameters", [])
        if isinstance(parameters, dict):
            parameters = [value for key, value in sorted(
                parameters.items(), key=lambda item: int(item[0])
            )]
        return cls(
            key=data["key"],
            parameters=[item if isinstance(item, ActionParameterDefined)
                        else ActionParameterDefined.from_dict(item) for item in parameters],
            add_target=data.get("add_target", False),
            allow_null_targets=data.get("allow_null_targets", False),
            role=data.get("role"),
            allow_empty_action=data.get("allow_empty_action", False),
            targets=list(data.get("targets", [])),
            multi_action_mode=data.get("multi_action_mode", "single"),
            strict_masks=data.get("strict_masks", True),
        )


@dataclass
class ActionDefined(DataClassJsonMixin):
    """
    一个智能体的完整业务动作定义。

    head_type 和 action_type 两个正交字段组成四种动作类型。action_key 是列表，
    每项的 key 同时作为业务指令名和动作头名，列表顺序就是策略向量切片顺序。
    """

    action_key: List[ActionKeyDefined] = field(default_factory=list)
    # single 表示单头，multi 表示多个同时输出的命名动作头。
    head_type: str = "single"
    # discrete 表示离散索引，continuous 表示归一化连续向量。
    action_type: str = "discrete"

    def __post_init__(self) -> None:
        if isinstance(self.action_key, dict):
            try:
                self.action_key = [value for _, value in sorted(
                    self.action_key.items(), key=lambda item: int(item[0])
                )]
            except (TypeError, ValueError) as exc:
                raise TypeError("action_key must be a numbered list, not a named dictionary") from exc
        normalized = []
        for value in self.action_key:
            if isinstance(value, ActionKeyDefined):
                item = value
            elif isinstance(value, dict):
                item = ActionKeyDefined.from_dict(value)
            else:
                raise TypeError("action_key entries must be ActionKeyDefined instances or tables")
            if not isinstance(item.key, str) or not item.key.strip():
                raise ValueError("action_key entry key must be a nonempty string")
            normalized.append(item)
        keys = [item.key for item in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate action_key entries: {keys}")
        self.action_key = normalized

    @property
    def kind(self) -> str:
        prefix = "multi_" if self.head_type == "multi" else ""
        return f"{prefix}{self.action_type}"

    def to_dict(self, encode_json: bool = False) -> Dict[str, Any]:
        return {
            "action_key": [value.to_dict() for value in self.action_key],
            "head_type": self.head_type,
            "action_type": self.action_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], infer_missing: bool = False) -> "ActionDefined":
        action_key = data.get("action_key", [])
        if isinstance(action_key, dict):
            action_key = [value for _, value in sorted(
                action_key.items(), key=lambda item: int(item[0])
            )]
        return cls(
            action_key=action_key,
            head_type=data.get("head_type", "single"),
            action_type=data.get("action_type", "discrete"),
        )
