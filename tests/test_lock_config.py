from dataclasses import dataclass, field

import pytest

from utils.lock_config import LockedConfig


@dataclass
class Inner:
    value: int = 1


@dataclass
class Outer:
    name: str = "x"
    items: list[int] = field(default_factory=lambda: [1, 2])
    nested: Inner = field(default_factory=Inner)


def test_locked_config_blocks_mutation():
    locked = LockedConfig(Outer())
    assert locked.name == "x"
    assert locked.items == (1, 2)
    assert locked.nested.value == 1
    with pytest.raises(PermissionError):
        locked.name = "y"
    with pytest.raises(PermissionError):
        del locked.name
