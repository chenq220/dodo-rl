from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union

from dataclasses_json import DataClassJsonMixin, config

from common.exception import CoordinateFrameException, StateEntityNotFoundException
from envs.state.spatial import CGCS2000Frame, LocalFrame, WGS84Frame


class CoordinateSystem(str, Enum):
    WGS84 = "wgs84"
    CGCS2000 = "cgcs2000"
    LOCAL = "local"


@dataclass(frozen=True)
class StateDescribe(DataClassJsonMixin):
    coordinate_system: CoordinateSystem = CoordinateSystem.LOCAL


Position = Union[WGS84Frame, CGCS2000Frame, LocalFrame]

_FRAME_BY_SYSTEM = {
    CoordinateSystem.WGS84: WGS84Frame,
    CoordinateSystem.CGCS2000: CGCS2000Frame,
    CoordinateSystem.LOCAL: LocalFrame,
}


def _encode_position(position: Position) -> dict[str, float]:
    return position.to_dict()


def _decode_position(value: Any) -> Position:
    if isinstance(value, (WGS84Frame, CGCS2000Frame, LocalFrame)):
        return value
    raise CoordinateFrameException(
        detail=(
            "Entity.position dict must be restored by State.from_dict "
            "using describe.coordinate_system"
        ),
    )


def _position_from_mapping(value: Any, coordinate_system: CoordinateSystem) -> Position:
    if isinstance(value, (WGS84Frame, CGCS2000Frame, LocalFrame)):
        return value
    if not isinstance(value, dict):
        raise CoordinateFrameException(
            detail=f"Entity.position must be a mapping or Frame, got {type(value).__name__}",
        )
    return _FRAME_BY_SYSTEM[coordinate_system].from_dict(value)


def _describe_from_mapping(value: Any, default: StateDescribe) -> StateDescribe:
    if value is None:
        return default
    if isinstance(value, StateDescribe):
        return value
    if not isinstance(value, dict):
        raise CoordinateFrameException(
            detail=f"State.describe must be a mapping, got {type(value).__name__}",
        )
    return StateDescribe.from_dict(value)


@dataclass(frozen=True)
class Entity(DataClassJsonMixin):
    """
    态势实体基类，描述单个可被环境检索、观测或作为动作目标的对象。

    实体对象只表达态势数据，不包含距离、角度、筛选等计算逻辑；这些算子应放在空间或环境算子模块中。
    """
    # 实体唯一名称，通常与仿真想定中的对象名一致，也是 find_entity 的检索键。
    name: str
    # 实体细分类别，例如 aircraft、radar、jammer 等业务类型。
    type: str
    # 实体位置，由所属 State.describe.coordinate_system 决定 Frame 类型。
    position: Position = field(
        metadata=config(encoder=_encode_position, decoder=_decode_position),
    )
    # 实体速度，单位 m/s。
    speed: float


@dataclass(frozen=True)
class EnvState(DataClassJsonMixin):
    """
    自然环境状态基类，用于承载天气、电磁、地形等非实体态势。

    当前基类没有内置字段，具体场景可以继承后增加风场、云层、干扰背景等环境属性。
    """
    pass


@dataclass
class State(DataClassJsonMixin):
    """
    仿真推演全量态势基类，统一承载实体列表、自然环境状态和仿真时间。

    子类负责把不同仿真协议的原始数据转换成统一属性树；算法和环境应依赖该基类接口，
    避免直接耦合具体中间件消息结构。
    """
    # 当前态势中的实体列表，实体顺序由具体仿真适配器决定。
    entities: List[Entity]
    # 整帧态势的坐标系描述；缺失时默认为 local。
    describe: StateDescribe = field(default_factory=StateDescribe)
    # 当前自然环境状态；未建模时使用空 EnvState。
    env: EnvState = field(default_factory=EnvState)
    # 当前态势对应的仿真时间，单位由适配器定义，也可表示逻辑步号。
    sim_time: float = 0.0

    def __post_init__(self):
        """
        后置初始化，声明实例级别的私有原始数据缓存域
        """
        # 原始未解析态势字典缓存；from_dict 反序列化时赋值，便于调试或录制原始数据。
        self._raw_data: Optional[dict] = None
        self._assert_positions_match_describe()

    @classmethod
    def _default_describe(cls) -> StateDescribe:
        return StateDescribe()

    def _assert_positions_match_describe(self) -> None:
        expected = _FRAME_BY_SYSTEM[self.describe.coordinate_system]
        for entity in self.entities:
            if type(entity.position) is not expected:
                raise CoordinateFrameException(
                    detail=(
                        f"Entity {entity.name!r} position is "
                        f"{type(entity.position).__name__}, expected {expected.__name__} "
                        f"for coordinate_system={self.describe.coordinate_system.value}"
                    ),
                )

    def find_entity(self, name: str) -> Entity:
        """
        根据实体名称查找对应实体对象
        :param name: 实体名称
        :return: 实体对象实例
        """
        for entity in self.entities:
            if entity.name == name:
                return entity
        raise StateEntityNotFoundException(
            name=name,
            available=[entity.name for entity in self.entities],
        )

    def find_entities(self, names: List[str]) -> List[Entity]:
        """
        批量查找指定名称列表的实体对象
        :param names: 实体名称列表
        :return: 找到的实体对象列表
        """
        return [self.find_entity(name) for name in names]

    @classmethod
    def from_dict(cls, kvs: dict, *, infer_missing: bool = False):
        """
        安全拦截反序列化切面，将原始字典动态绑定到实例域，防止多线程或多局之间数据污染
        :param kvs: 原始态势字典
        :param infer_missing: 是否推断缺失字段
        :return: 状态对象具体实例
        """
        data = dict(kvs)
        # "platforms" 是历史录制和旧适配器字段，进入框架模型时统一迁移到 entities。
        if "entities" not in data and "platforms" in data:
            data["entities"] = data["platforms"]
        describe = _describe_from_mapping(data.get("describe"), cls._default_describe())
        data["describe"] = {
            "coordinate_system": describe.coordinate_system.value,
        }
        restored = []
        for item in data.get("entities") or []:
            if isinstance(item, Entity):
                restored.append(item)
                continue
            item = dict(item)
            item["position"] = _position_from_mapping(
                item.get("position"),
                describe.coordinate_system,
            )
            restored.append(item)
        data["entities"] = restored
        obj = super().from_dict(data, infer_missing=infer_missing)
        obj._raw_data = kvs
        return obj

    @property
    def raw_data(self) -> Optional[dict]:
        """
        获取当前态势的原始未解析字典
        :return: 原始数据字典
        """
        return self._raw_data
