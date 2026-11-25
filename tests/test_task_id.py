import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_task_id_process(source: str, environment_task_id: str | None) -> dict:
    env = os.environ.copy()
    python_path = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    if environment_task_id is None:
        env.pop("DODO_TASK_ID", None)
    else:
        env["DODO_TASK_ID"] = environment_task_id
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_task_id_singleton_uses_environment_and_is_the_only_export() -> None:
    result = _run_task_id_process(
        """
        import json
        import os
        import utils.task_id as module
        from utils.task_id import task_id

        print(json.dumps({
            "value": str(task_id),
            "environment": os.environ["DODO_TASK_ID"],
            "exports": module.__all__,
            "has_getter": hasattr(module, "get_task_id"),
            "has_public_value": hasattr(task_id, "TASK_ID"),
        }))
        """,
        " task-001 ",
    )

    assert result == {
        "value": "task-001",
        "environment": "task-001",
        "exports": ["task_id"],
        "has_getter": False,
        "has_public_value": False,
    }


def test_task_id_singleton_generates_eight_hex_digits_and_reuses_them() -> None:
    result = _run_task_id_process(
        """
        import json
        import os
        from utils.task_id import task_id

        first = str(task_id)
        print(json.dumps({
            "first": first,
            "second": str(task_id),
            "environment": os.environ["DODO_TASK_ID"],
        }))
        """,
        None,
    )

    assert re.fullmatch(r"[0-9a-f]{8}", result["first"])
    assert result["second"] == result["first"]
    assert result["environment"] == result["first"]


@pytest.mark.parametrize("environment_task_id", ["none", "null", "NONE", "Null"])
def test_non_empty_environment_task_id_is_always_literal(
        environment_task_id: str,
) -> None:
    result = _run_task_id_process(
        """
        import json
        import os
        from utils.task_id import task_id

        print(json.dumps({
            "value": str(task_id),
            "environment": os.environ["DODO_TASK_ID"],
        }))
        """,
        environment_task_id,
    )

    assert result == {
        "value": environment_task_id,
        "environment": environment_task_id,
    }


@pytest.mark.parametrize("empty_value", [None, "", "   ", "none", "NONE", "null", "Null"])
def test_task_id_empty_configuration_preserves_current_value(empty_value: str | None) -> None:
    result = _run_task_id_process(
        f"""
        import json
        import os
        from utils.task_id import task_id

        configured = task_id.configure({empty_value!r})
        print(json.dumps({{
            "configured": configured,
            "value": str(task_id),
            "environment": os.environ["DODO_TASK_ID"],
        }}))
        """,
        "environment-id",
    )

    assert result == {
        "configured": "environment-id",
        "value": "environment-id",
        "environment": "environment-id",
    }


def test_task_id_configure_overrides_environment_and_is_inherited_by_child() -> None:
    result = _run_task_id_process(
        """
        import json
        import os
        import subprocess
        import sys
        from utils.task_id import task_id

        configured = task_id.configure("toml-id")
        child = subprocess.check_output(
            [sys.executable, "-c", "import os; print(os.environ['DODO_TASK_ID'])"],
            text=True,
        ).strip()
        print(json.dumps({
            "configured": configured,
            "value": str(task_id),
            "environment": os.environ["DODO_TASK_ID"],
            "child": child,
        }))
        """,
        "environment-id",
    )

    assert set(result.values()) == {"toml-id"}


def test_task_id_freeze_is_idempotent_and_rejects_a_different_value() -> None:
    result = _run_task_id_process(
        """
        import json
        import os
        from common.exception import TaskIdFrozenException
        from utils.task_id import task_id

        task_id.configure("fixed-id")
        first = task_id.freeze()
        second = task_id.freeze()
        try:
            task_id.configure("other-id")
        except TaskIdFrozenException as exc:
            error_type = type(exc).__name__
        else:
            error_type = ""
        print(json.dumps({
            "first": first,
            "second": second,
            "same": task_id.configure("fixed-id"),
            "empty": task_id.configure(None),
            "value": str(task_id),
            "environment": os.environ["DODO_TASK_ID"],
            "error_type": error_type,
        }))
        """,
        None,
    )

    assert result == {
        "first": "fixed-id",
        "second": "fixed-id",
        "same": "fixed-id",
        "empty": "fixed-id",
        "value": "fixed-id",
        "environment": "fixed-id",
        "error_type": "TaskIdFrozenException",
    }


def test_task_id_rejects_non_string_configuration() -> None:
    result = _run_task_id_process(
        """
        import json
        from utils.task_id import task_id

        errors = []
        for value in (1, True, [], {}):
            try:
                task_id.configure(value)
            except TypeError:
                errors.append(type(value).__name__)
        print(json.dumps({"errors": errors}))
        """,
        None,
    )

    assert result == {"errors": ["int", "bool", "list", "dict"]}


def test_toml_task_id_precedence_default_and_plugin_probe_behavior() -> None:
    result = _run_task_id_process(
        """
        import json
        from pathlib import Path
        import tempfile
        from utils.task_id import task_id
        from utils.toml_config import TomlTrainingConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured_path = root / "configured.toml"
            configured_path.write_text('[design]\\ntask_id = "toml-id"\\n', encoding="utf-8")
            TomlTrainingConfig.read_plugins_config(configured_path, project_root=root)
            after_probe = str(task_id)
            configured = TomlTrainingConfig(configured_path, project_root=root)

            default_path = root / "default.toml"
            default_path.write_text('[design]\\ntask_name = "demo"\\n', encoding="utf-8")
            default = TomlTrainingConfig(default_path, project_root=root)

            print(json.dumps({
                "after_probe": after_probe,
                "configured_value": str(task_id),
                "configured_design": configured.design["task_id"],
                "default_design": default.design["task_id"],
            }))
        """,
        "environment-id",
    )

    assert result == {
        "after_probe": "environment-id",
        "configured_value": "toml-id",
        "configured_design": "toml-id",
        "default_design": "",
    }


def test_toml_task_id_rejects_non_string_value() -> None:
    result = _run_task_id_process(
        """
        import json
        from pathlib import Path
        import tempfile
        from utils.toml_config import TomlTrainingConfig

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text("[design]\\ntask_id = 123\\n", encoding="utf-8")
            try:
                TomlTrainingConfig(path, project_root=directory)
            except TypeError as exc:
                error_type = type(exc).__name__
            else:
                error_type = ""
        print(json.dumps({"error_type": error_type}))
        """,
        None,
    )

    assert result == {"error_type": "TypeError"}


def test_portal_entry_logs_and_freezes_final_toml_task_id_before_role_creation() -> None:
    result = _run_task_id_process(
        """
        import json
        from pathlib import Path
        import runpy
        import sys
        import tempfile

        import common.logger as logger_module
        from common.exception import TaskIdFrozenException
        import roles.serial_train as serial_train_role_module
        from utils.task_id import task_id

        events = []

        class Recorder:
            def info(self, message):
                events.append(["startup_process", message])

        class FakeSerialTrainRole:
            def __init__(self, **kwargs):
                events.append(["role_created", str(task_id)])

            def run(self):
                events.append(["role_run", str(task_id)])

        logger_module._STARTUP_PROCESS_LOG = Recorder()
        serial_train_role_module.SerialTrainRole = FakeSerialTrainRole

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "portal.toml"
            config_path.write_text(
                '[env_config]\\nuse_sim = false\\n[design]\\ntask_id = "portal-task-id"\\n',
                encoding="utf-8",
            )
            sys.argv = ["portal.serial_train", "-f", str(config_path)]
            runpy.run_module("portal.serial_train", run_name="__main__")

            try:
                task_id.configure("other-task-id")
            except TaskIdFrozenException:
                frozen = True
            else:
                frozen = False

            print(json.dumps({
                "events": events,
                "task_id": str(task_id),
                "frozen": frozen,
                "config_file": str(config_path.resolve()),
            }))
        """,
        "environment-id",
    )

    expected_log = (
        "Startup process: role=serial_train "
        f"config_file={result['config_file']} task_id=portal-task-id"
    )
    assert result["events"] == [
        ["startup_process", expected_log],
        ["role_created", "portal-task-id"],
        ["role_run", "portal-task-id"],
    ]
    assert result["task_id"] == "portal-task-id"
    assert result["frozen"] is True


def test_derive_generation_run_seed_stable() -> None:
    from utils.random_seed import derive_generation_run_seed

    a = derive_generation_run_seed("abc123", "combat", 0)
    b = derive_generation_run_seed("abc123", "combat", 0)
    c = derive_generation_run_seed("abc123", "combat", 1)
    assert a == b
    assert a != c
    assert 0 < a < 2 ** 32


def test_apply_train_seed_prefers_model_initialization_seed(monkeypatch) -> None:
    from config import TrainConfig
    from utils import random_seed

    applied = []
    monkeypatch.setattr(random_seed, "set_global_seed", lambda seed, **_: applied.append(seed) or seed)

    config = TrainConfig(seed=607729106, initialization_seed=1472572158)
    result = random_seed.apply_train_seed(config, context="archive-replay")

    assert result == 1472572158
    assert applied == [1472572158]
