from pathlib import Path
import sys


SDK_SOURCE_DIR = Path(__file__).resolve().parents[1] / "sdk" / "src"
if str(SDK_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE_DIR))


def test_sdk_public_extension_api_uses_runtime_class_identity() -> None:
    from dodorl_sdk import (
        AECEnvironment,
        AgentConfig,
        AgentID,
        CGCS2000Frame,
        CoordinateSystem,
        EnvConfig,
        EvalControlRole,
        Evaluation,
        EvaluationConfigBuilder,
        EvaluationProtocol,
        GeoPoint,
        LocalFrame,
        ObsType,
        ParallelEnvironment,
        StateDescribe,
        StateType,
        WGS84Frame,
    )
    from config import EnvConfig as RuntimeEnvConfig
    from config import AgentConfig as RuntimeAgentConfig
    from envs.base import (
        AgentID as RuntimeAgentID,
        AECEnvironment as RuntimeAECEnvironment,
        ObsType as RuntimeObsType,
        ParallelEnvironment as RuntimeParallelEnvironment,
        StateType as RuntimeStateType,
    )
    from envs.state.spatial import (
        CGCS2000Frame as RuntimeCGCS2000Frame,
        GeoPoint as RuntimeGeoPoint,
        LocalFrame as RuntimeLocalFrame,
        WGS84Frame as RuntimeWGS84Frame,
    )
    from envs.state.base import (
        CoordinateSystem as RuntimeCoordinateSystem,
        StateDescribe as RuntimeStateDescribe,
    )
    from roles.evaluation import EvaluationRole as RuntimeEvaluation
    from roles.eval_control import EvalControlRole as RuntimeEvalControl

    assert Evaluation is RuntimeEvaluation
    assert EvalControlRole is RuntimeEvalControl
    assert AECEnvironment is RuntimeAECEnvironment
    assert AgentConfig is RuntimeAgentConfig
    assert EvaluationConfigBuilder is not None
    assert EvaluationProtocol is not None
    assert ParallelEnvironment is RuntimeParallelEnvironment
    assert EnvConfig is RuntimeEnvConfig
    assert AgentID is RuntimeAgentID
    assert ObsType is RuntimeObsType
    assert StateType is RuntimeStateType
    assert GeoPoint is RuntimeGeoPoint
    assert WGS84Frame is RuntimeWGS84Frame
    assert CGCS2000Frame is RuntimeCGCS2000Frame
    assert LocalFrame is RuntimeLocalFrame
    assert CoordinateSystem is RuntimeCoordinateSystem
    assert StateDescribe is RuntimeStateDescribe
    assert GeoPoint is WGS84Frame
    from dodorl_sdk.state import (
        CGCS2000Frame as StateModuleCGCS2000Frame,
        CoordinateSystem as StateModuleCoordinateSystem,
        LocalFrame as StateModuleLocalFrame,
        StateDescribe as StateModuleStateDescribe,
        WGS84Frame as StateModuleWGS84Frame,
    )
    assert WGS84Frame is StateModuleWGS84Frame
    assert CGCS2000Frame is StateModuleCGCS2000Frame
    assert LocalFrame is StateModuleLocalFrame
    assert CoordinateSystem is StateModuleCoordinateSystem
    assert StateDescribe is StateModuleStateDescribe
    import dodorl_sdk
    assert not hasattr(dodorl_sdk, "WGS84LocalFrame")
    assert "WGS84LocalFrame" not in dodorl_sdk.__all__


def test_sdk_public_extension_api_is_in_top_level_all() -> None:
    import dodorl_sdk

    expected = {
        "AgentID", "EnvConfig", "GeoPoint", "ObsType", "ParallelEnvironment",
        "StateType",
        "WGS84Frame", "CGCS2000Frame", "LocalFrame", "CoordinateSystem", "StateDescribe",
    }
    assert expected <= set(dodorl_sdk.__all__)


def test_sdk_training_roles_and_training_config_are_public() -> None:
    from dodorl_sdk import (
        ControlSDK,
        EvalControlSDK,
        LearnControlSDK,
        LearnSDK,
        ModelSDK,
        SampleSDK,
        SerialTrainSDK,
        StageControlSDK,
    )
    from dodorl_sdk.config import EntryConfig, EnvConfig, GrpcConfig, TorchRpcConfig


    assert all(value is not None for value in (
        ControlSDK, StageControlSDK, LearnSDK, ModelSDK, LearnControlSDK,
        EvalControlSDK,
    ))
    assert SampleSDK is not None
    assert SerialTrainSDK is not None
    assert all(value is not None for value in (EntryConfig, EnvConfig, GrpcConfig, TorchRpcConfig))


def test_sdk_portal_is_canonical_role_entry_and_training_is_compatible() -> None:
    import importlib
    import warnings

    import dodorl_sdk
    import dodorl_sdk.training as direct_training
    from dodorl_sdk import portal, training
    from dodorl_sdk.training import SerialTrainSDK as LegacySerialTrainSDK

    expected = [
        "SDKPluginRegistry",
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
    ]

    assert portal.__all__ == expected
    assert training.__all__ == expected
    assert direct_training is training
    assert LegacySerialTrainSDK is portal.SerialTrainSDK
    assert training.SDKPluginRegistry is portal.SDKPluginRegistry

    for name in expected[1:]:
        portal_class = getattr(portal, name)
        assert getattr(training, name) is portal_class
        assert getattr(dodorl_sdk, name) is portal_class
        assert portal_class.__module__ == "dodorl_sdk.portal"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        importlib.reload(training)
    assert captured == []


def test_sdk_advanced_modules_use_runtime_class_identity() -> None:
    from dodorl_sdk.algorithms import Agent, BatchSchema, DEFAULT_EPISODE_SCHEMA, Learner
    from dodorl_sdk.contracts import EpisodeInfo, ModelInfo
    from algos.agents import Agent as RuntimeAgent
    from algos.learners import Learner as RuntimeLearner
    from algos.learners.batch_schema import BatchSchema as RuntimeBatchSchema
    from models.model import ModelInfo as RuntimeModelInfo
    from scheduler.model import EpisodeInfo as RuntimeEpisodeInfo

    assert Agent is RuntimeAgent
    assert Learner is RuntimeLearner
    assert BatchSchema is RuntimeBatchSchema
    assert Learner.batch_schema is DEFAULT_EPISODE_SCHEMA
    assert EpisodeInfo is RuntimeEpisodeInfo
    assert ModelInfo is RuntimeModelInfo
