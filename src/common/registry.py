# 插件注册器
import functools
from common.logger import init_logger

LOG = init_logger('registry')


class ExplicitExtensionRegistry:
    def __init__(self):
        # 记录所有标记为可扩展的基线类 { "ClassName": BaseClass }
        self.base_classes = {}
        # 记录项目显式指定的重写映射 { BaseClass: CustomClass }
        self.overrides = {}

    def register_extensible(self, cls):
        """基线类装饰器核心逻辑"""
        self.base_classes[cls.__name__] = cls

        # 备份原本的 __new__
        orig_new = cls.__new__

        @functools.wraps(orig_new)
        def redirected_new(target_cls, *args, **kwargs):
            # 关键拦截：只有当明确实例化基类本身，且项目显式配置了重写类时，才进行替换
            if target_cls is cls and cls in self.overrides:
                custom_cls = self.overrides[cls]
                # 使用 object.__new__ 创建定制类的实例
                return object.__new__(custom_cls)

            # 否则（比如没有配置重写，或者子类自己调用__new__），走正常流程
            if orig_new is object.__new__:
                return object.__new__(target_cls)
            return orig_new(target_cls, *args, **kwargs)

        cls.__new__ = redirected_new
        return cls

    def set_override(self, base_cls_name: str, custom_cls):
        """供具体项目在启动时显式调用，绑定重写关系"""
        base_cls = self.base_classes.get(base_cls_name)
        if not base_cls:
            raise ValueError(f"There is no baseline class named '{base_cls_name}', "
                             f"or baseline class does not set @extensible")

        if not issubclass(custom_cls, base_cls):
            raise TypeError(f"Custom class {custom_cls.__name__} Must inherit from the baseline class {base_cls.__name__}")

        self.overrides[base_cls] = custom_cls
        LOG.info(f"Register rewriting: {base_cls_name} -> {custom_cls.__name__}")

    def clear_overrides(self):
        """清空当前运行期的覆盖关系，供独立插件会话重新激活。"""
        self.overrides.clear()


# 全局单例
_REGISTRY = ExplicitExtensionRegistry()
extensible = _REGISTRY.register_extensible


def configure_project_overrides(mappings: dict):
    """项目层面的统一配置入口"""
    for origin_cls, custom_cls in mappings.items():
        base_name = origin_cls.__name__
        _REGISTRY.set_override(base_name, custom_cls)


def clear_project_overrides() -> None:
    """清空项目覆盖关系；不会移除已由 @extensible 登记的基线类。"""
    _REGISTRY.clear_overrides()


def find_extensible_base(custom_cls):
    """返回定制类继承链中最近的 @extensible 基线类；不存在时返回 None。"""
    for base_cls in custom_cls.__mro__[1:]:
        if base_cls in _REGISTRY.base_classes.values():
            return base_cls
    return None
