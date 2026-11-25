import sys

import pytest
from pathlib import Path

SDK_SOURCE_DIR = Path(__file__).resolve().parents[1] / "sdk" / "src"
if str(SDK_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE_DIR))


def test_sdk_extensions_exports_runtime_classes():
    from dodorl_sdk.extensions import AECEnvironment, EnvConfig, ParallelEnvironment
    from config import EnvConfig as RuntimeEnvConfig
    from envs.base import AECEnvironment as RuntimeAEC, ParallelEnvironment as RuntimeParallel

    assert EnvConfig is RuntimeEnvConfig
    assert AECEnvironment is RuntimeAEC
    assert ParallelEnvironment is RuntimeParallel
