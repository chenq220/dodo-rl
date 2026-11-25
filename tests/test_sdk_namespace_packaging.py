from __future__ import annotations

import importlib.util
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_rewriter():
    path = PROJECT_ROOT / "sdk" / "rewrite_sdk_imports.py"
    spec = importlib.util.spec_from_file_location("rewrite_sdk_imports", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sdk_build_imports_are_qualified_under_dodorl_sdk() -> None:
    rewriter = _load_rewriter()
    source = """\
from envs.base import ParallelEnvironment
from common.exception import SimulationTimeoutException
from external_sdk import Resource
from sample.model import EpisodeBatch
PLUGIN = "algos.modules.agents.rnn_agent:RNNAgent"
"""

    rewritten, replacements = rewriter.rewrite_text(source)

    assert replacements == 4
    assert "from dodorl_sdk.dodorl.envs.base import ParallelEnvironment" in rewritten
    assert "from dodorl_sdk.dodorl.common.exception import SimulationTimeoutException" in rewritten
    assert "from external_sdk import Resource" in rewritten
    assert "from dodorl_sdk.dodorl.sample.model import EpisodeBatch" in rewritten
    assert '"dodorl_sdk.dodorl.algos.modules.agents.rnn_agent:RNNAgent"' in rewritten


def test_sdk_build_only_discovers_dodorl_sdk_top_level_package() -> None:
    build_script = (PROJECT_ROOT / "sdk" / "build_sdk.sh").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "sdk" / "pyproject.toml").read_text(encoding="utf-8")
    runtime = (PROJECT_ROOT / "sdk" / "src" / "dodorl_sdk" / "_runtime.py").read_text(encoding="utf-8")

    assert '$BUILD_DIR/src/dodorl_sdk/dodorl/$package' in build_script
    assert '$BUILD_DIR/src/dodorl_sdk/dodorl/config.py' in build_script
    assert 'include = ["dodorl_sdk*"]' in pyproject
    assert "py-modules" not in pyproject
    assert "sys.path.insert" not in runtime


def test_sdk_training_is_thin_portal_compatibility_module() -> None:
    package_dir = PROJECT_ROOT / "sdk" / "src" / "dodorl_sdk"
    training_source = (package_dir / "training.py").read_text(encoding="utf-8")

    assert training_source.splitlines() == [
        '"""遗留兼容入口；新代码请从 ``dodorl_sdk.portal`` 导入角色 SDK。"""',
        "",
        "from .portal import *",
        "from .portal import __all__",
    ]
    assert (package_dir / "portal.py").is_file()


def test_sdk_role_documentation_uses_portal_entry() -> None:
    role_names = {
        "ControlSDK",
        "StageControlSDK",
        "LearnSDK",
        "ModelSDK",
        "LearnControlSDK",
        "StageLearnSDK",
        "SampleSDK",
        "SerialStageTrainSDK",
        "SerialTrainSDK",
        "EvaluationSDK",
        "EvalControlSDK",
    }
    documentation = [PROJECT_ROOT / "sdk" / "README.md"]
    documentation.extend((PROJECT_ROOT / "sdk" / "examples").glob("*.py"))

    for path in documentation:
        source = path.read_text(encoding="utf-8")
        imports = re.findall(r"from dodorl_sdk import (\([^)]*\)|[^\n]+)", source)
        for imported_names in imports:
            assert role_names.isdisjoint(set(re.findall(r"\b\w+\b", imported_names))), path


def test_sdk_build_qualifies_root_module_lazy_paths():
    rewriter = _load_rewriter()
    source = "CONFIG = 'config:TrainConfig'\n"
    rewritten, count = rewriter.rewrite_text(source)
    assert count == 1
    assert "dodorl_sdk.dodorl.config:TrainConfig" in rewritten
