from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_SOURCE_DIR = PROJECT_ROOT / "sdk" / "src"
if str(SDK_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE_DIR))


PORTAL_ENTRIES = {
    "control.py": ("control", "ControlRole", True),
    "eval.py": ("eval", "evaluation_class", None),
    "eval_control.py": ("eval_control", "evaluation_class", True),
    "eval_sample.py": ("sample", "SampleRole", False),
    "learn.py": ("learn", "LearnRole", False),
    "learn_control.py": ("learn_control", "LearnControlRole", True),
    "model.py": ("model", "ModelRole", False),
    "random_sampling.py": ("random_sampling", "RandomSamplingRole", None),
    "sample.py": ("sample", "SampleRole", False),
    "serial_stage_train.py": ("serial_stage_train", "SerialStageTrainRole", None),
    "serial_train.py": ("serial_train", "SerialTrainRole", None),
    "stage_control.py": ("stage_control", "StageControlRole", True),
    "stage_learn.py": ("stage_learn", "StageLearnRole", True),
    "train.py": ("train", "TrainRole", None),
}


def test_every_portal_role_uses_shared_runtime_init_before_role_creation() -> None:
    portal_dir = PROJECT_ROOT / "src" / "portal"
    entry_source = (PROJECT_ROOT / "src" / "common" / "portal_entry.py").read_text(encoding="utf-8")
    freeze_call = entry_source.index("task_id.freeze()")
    log_call = entry_source.index("log_startup_process(")
    manager_call = entry_source.index("build_artifacts_manager(")
    assert freeze_call < log_call < manager_call
    assert entry_source.count("task_id.freeze()") == 1
    assert entry_source.count("log_startup_process(") == 1
    assert "register_config_artifacts(" not in entry_source

    for file_name, (role_name, role_factory, is_master) in PORTAL_ENTRIES.items():
        source = (portal_dir / file_name).read_text(encoding="utf-8")
        init_call = source.index("initialize_role_runtime(")
        role_creation = source.rindex(f"{role_factory}(")
        assert source.count("initialize_role_runtime(") == 1, file_name
        assert "task_id.freeze()" not in source, file_name
        assert "log_startup_process(" not in source, file_name
        assert "register_config_artifacts(" not in source, file_name
        assert f'role="{role_name}"' in source, file_name
        assert f"is_master={is_master}" in source, file_name
        assert init_call < role_creation, file_name


def test_role_modules_do_not_repeat_generic_runtime_initialization_log() -> None:
    roles_dir = PROJECT_ROOT / "src" / "roles"

    for path in roles_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "Startup process" not in source, path.name
        assert "Role initializing:" not in source, path.name


def test_sdk_role_names_are_stable_and_startup_process_is_centralized() -> None:
    from dodorl_sdk.portal import (
        ControlSDK,
        EvalControlSDK,
        EvaluationSDK,
        LearnControlSDK,
        LearnSDK,
        ModelSDK,
        SampleSDK,
        SerialStageTrainSDK,
        SerialTrainSDK,
        StageControlSDK,
        StageLearnSDK,
    )

    expected = {
        ControlSDK: ("control", True),
        StageControlSDK: ("stage_control", True),
        LearnSDK: ("learn", False),
        ModelSDK: ("model", False),
        LearnControlSDK: ("learn_control", True),
        StageLearnSDK: ("stage_learn", True),
        SampleSDK: ("sample", False),
        SerialStageTrainSDK: ("serial_stage_train", None),
        SerialTrainSDK: ("serial_train", None),
        EvaluationSDK: ("eval", None),
        EvalControlSDK: ("eval_control", True),
    }
    assert {
        sdk_class: (sdk_class.role_name, sdk_class.is_master) for sdk_class in expected
    } == expected

    source = (
        PROJECT_ROOT / "sdk" / "src" / "dodorl_sdk" / "portal.py"
    ).read_text(encoding="utf-8")
    assert source.count("initialize_role_runtime(") == 1
    assert "task_id.freeze()" not in source
    assert "log_startup_process(" not in source
    assert "register_config_artifacts(" not in source
