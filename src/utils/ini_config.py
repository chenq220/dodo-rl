from __future__ import annotations

import configparser
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

import torch

ConfigT = TypeVar("ConfigT")


def load_ini_config(config_file: str | Path) -> configparser.ConfigParser:
    """读取INI配置文件，保留普通字符串键值并交给后续解析函数做类型转换。"""
    parser = configparser.ConfigParser()
    read_files = parser.read(config_file, encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"INI config file not found or unreadable: {config_file}")
    return parser


def section_to_dict(parser: configparser.ConfigParser, section: str) -> dict[str, Any]:
    """把一个section解析为普通dict；逗号分隔值会转成list，none/bool/数字会转成Python类型。"""
    if not parser.has_section(section):
        return {}
    return {
        key: parse_ini_field(key, value)
        for key, value in parser.items(section)
    }


def build_config_from_section(
        config_cls: type[ConfigT],
        parser: configparser.ConfigParser,
        section: str,
        *,
        constructor_keys: Iterable[str] = (),
) -> ConfigT:
    """
    从INI section构造配置对象。

    constructor_keys用于AgentConfig.agent_type、LearnerConfig.algo_type这类必须在__post_init__
    前确定的字段；其余字段在实例创建后再写入。
    """
    values = section_to_dict(parser, section)
    constructor_key_set = set(constructor_keys)
    constructor_kwargs = {
        key: values[key]
        for key in constructor_key_set
        if key in values
    }
    config = config_cls(**constructor_kwargs)
    apply_section_to_object(config, values, skip_keys=constructor_key_set)
    return config


def apply_ini_section(
        obj: Any,
        parser: configparser.ConfigParser,
        section: str,
        *,
        skip_keys: Iterable[str] = (),
) -> Any:
    """把INI section中的字段写入已有对象。"""
    return apply_section_to_object(obj, section_to_dict(parser, section), skip_keys=skip_keys)


def apply_section_to_object(
        obj: Any,
        values: dict[str, Any],
        *,
        skip_keys: Iterable[str] = (),
) -> Any:
    """把dict字段写入对象；字段必须是dataclass字段或对象已有属性。"""
    skip_key_set = set(skip_keys)
    allowed_keys = _allowed_object_keys(obj)
    for key, value in values.items():
        if key in skip_key_set:
            continue
        if key not in allowed_keys and not hasattr(obj, key):
            raise AttributeError(f"{type(obj).__name__} has no config field '{key}'")
        setattr(obj, key, value)
    return obj


def parse_ini_value(raw_value: str) -> Any:
    """把INI字符串转换成常用Python类型。"""
    value = raw_value.strip()
    lower_value = value.lower()
    if lower_value == "none":
        return None
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False
    if "," in value:
        return [
            parse_ini_value(item)
            for item in value.split(",")
            if item.strip() != ""
        ]
    if value == "":
        return []
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_ini_field(key: str, raw_value: str) -> Any:
    """按字段名解析INI值；device=auto在这里统一选择运行设备。"""
    value = parse_ini_value(raw_value)
    if key == "device":
        return select_device(value)
    if key.endswith("_names") and not isinstance(value, list):
        return [value] if value is not None else []
    return value


def select_device(device_value: Any = "auto") -> torch.device | None:
    """
    解析配置中的device字段。

    device=auto时按项目常用顺序选择：MPS -> CUDA -> CPU。
    None保留为None，显式字符串如cpu/cuda/mps直接转换为torch.device。
    """
    if device_value is None:
        return None
    if isinstance(device_value, torch.device):
        return device_value
    if str(device_value).lower() != "auto":
        return torch.device(str(device_value))
    if hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def indexed_sections(parser: configparser.ConfigParser, prefix: str) -> list[str]:
    """返回形如 agent.0 / agent.1 的section，按编号升序排列，不包含 agent.0.subject 等子section。"""
    matched_sections = []
    prefix_dot = f"{prefix}."
    for section in parser.sections():
        if not section.startswith(prefix_dot):
            continue
        suffix = section[len(prefix_dot):]
        if suffix.isdigit():
            matched_sections.append(section)
    return sorted(matched_sections, key=lambda section: int(section.rsplit(".", 1)[1]))


def _allowed_object_keys(obj: Any) -> set[str]:
    if is_dataclass(obj):
        return {field.name for field in fields(obj)}
    return set(vars(obj).keys())
