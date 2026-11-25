from dataclasses import dataclass

import torch

from utils.ini_config import (
    indexed_sections,
    load_ini_config,
    parse_ini_value,
    section_to_dict,
    select_device,
)


def test_parse_ini_value_types():
    assert parse_ini_value("true") is True
    assert parse_ini_value("none") is None
    assert parse_ini_value("3") == 3
    assert parse_ini_value("1.5") == 1.5
    assert parse_ini_value("a,b") == ["a", "b"]


def test_load_ini_and_section(tmp_path):
    path = tmp_path / "demo.ini"
    path.write_text("[train]\nlr = 0.1\nenabled = true\n\n[agent.0]\nname = a\n\n[agent.1]\nname = b\n", encoding="utf-8")
    parser = load_ini_config(path)
    values = section_to_dict(parser, "train")
    assert values["lr"] == 0.1
    assert values["enabled"] is True
    assert indexed_sections(parser, "agent") == ["agent.0", "agent.1"]


def test_select_device_explicit_and_none():
    assert select_device(None) is None
    assert select_device("cpu") == torch.device("cpu")
    auto = select_device("auto")
    assert auto.type in {"cpu", "cuda", "mps"}
