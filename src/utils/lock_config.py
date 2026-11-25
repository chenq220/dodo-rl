import types
from typing import Any
from config import EnvConfig


class LockedConfig:
    """
    优雅的只读配置代理（Read-Only Proxy）。
    """
    def __init__(self, obj: object):
        object.__setattr__(self, "_obj", obj)

    @property
    def __class__(self):
        return self._obj.__class__

    def __getattr__(self, name):
        attr = getattr(self._obj, name)

        if isinstance(attr, list):
            return tuple(attr)

        if isinstance(attr, dict):
            return types.MappingProxyType(attr)

        if hasattr(attr, '__dataclass_fields__'):
            return LockedConfig(attr)

        return attr

    def __setattr__(self, name, value):
        raise PermissionError(
            f"[Architecture Limit] Configuration is globally locked. Modifying '{name}' is strictly prohibited.")

    def __delattr__(self, name):
        raise PermissionError(
            f"[Architecture Limit] Configuration is globally locked. Deleting '{name}' is strictly prohibited.")
