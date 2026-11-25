from pathlib import Path
import sys

import pytest

from utils.toml_config import TomlTrainingConfig


SDK_SOURCE_DIR = Path(__file__).resolve().parents[1] / "sdk" / "src"
if str(SDK_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE_DIR))


def test_rl_env_metadata_keeps_nested_values_from_text(tmp_path: Path) -> None:
    config = TomlTrainingConfig(config_text='''
[env_config]
use_sim = false
[env_config.metadata]
business_name = "demo"
[env_config.metadata.sensor]
channels = [1, 2, 3]
''', project_root=tmp_path)
    assert config.env_config.metadata == {"business_name": "demo", "sensor": {"channels": [1, 2, 3]}}
    assert config.sim_config is None and config.sim_manager_config is None


def test_evaluation_builder_must_return_eval_config(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "custom_eval.py"
    module.write_text("def bad_builder(config): return {}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    config_file = tmp_path / "eval.toml"
    config_file.write_text(
        '[entry]\neval_config_builder = "custom_eval:bad_builder"\n',
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    from common.portal_eval import resolve_eval_config

    with pytest.raises(TypeError, match="must return EvalConfig"):
        resolve_eval_config(config)


def test_custom_evaluation_class_uses_public_base(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "project_evaluation.py"
    module.write_text(
        "from dodorl_sdk import Evaluation\n"
        "class ProjectEvaluation(Evaluation):\n"
        "    def finished_report(self): return {'ok': True}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_file = tmp_path / "evaluation.toml"
    config_file.write_text(
        '[entry]\nevaluation_class = "project_evaluation:ProjectEvaluation"\n',
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    from common.portal_eval import resolve_evaluation_class
    from dodorl_sdk import Evaluation

    resolved = resolve_evaluation_class(config)
    assert issubclass(resolved, Evaluation)
    assert resolved.finished_report(None) == {"ok": True}


def test_custom_distributed_evaluation_class_uses_public_base(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "project_eval_control.py"
    module.write_text(
        "from dodorl_sdk import EvalControlRole\n"
        "class ProjectEvalControl(EvalControlRole):\n"
        "    def finished_report(self): return {'ok': True}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_file = tmp_path / "evaluation.toml"
    config_file.write_text(
        '[entry]\ndistributed_evaluation_class = "project_eval_control:ProjectEvalControl"\n',
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    from common.portal_eval import resolve_distributed_evaluation_class
    from dodorl_sdk import EvalControlRole

    resolved = resolve_distributed_evaluation_class(config)
    assert issubclass(resolved, EvalControlRole)
    assert resolved.finished_report(None) == {"ok": True}
