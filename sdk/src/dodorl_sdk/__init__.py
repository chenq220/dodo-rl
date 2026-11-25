"""公开 SDK 接口按需加载，允许在计算库导入前设置进程环境。"""
from importlib import import_module
from .version import __version__
from ._runtime import bootstrap_runtime

bootstrap_runtime()

_EXPORTS = {'AgentConfig': ('.config', 'AgentConfig'),
 'EnvConfig': ('.config', 'EnvConfig'),
 'EvalConfig': ('.config', 'EvalConfig'),
 'GrpcConfig': ('.config', 'GrpcConfig'),
 'LearnerConfig': ('.config', 'LearnerConfig'),
 'StageConfig': ('.config', 'StageConfig'),
 'StageTaskConfig': ('.config', 'StageTaskConfig'),
 'SampleType': ('.config', 'SampleType'),
 'TaskConfig': ('.config', 'TaskConfig'),
 'MonteCarloMethodConfig': ('.config', 'MonteCarloMethodConfig'),
 'TrainConfig': ('.config', 'TrainConfig'),
 'TorchRpcConfig': ('.config', 'TorchRpcConfig'),
 'configure_sdk': ('.config', 'configure_sdk'),
 'get_project_root': ('.config', 'get_project_root'),
 'Action': ('.action', 'Action'),
 'ActionDefined': ('.action', 'ActionDefined'),
 'ActionParameterDefined': ('.action', 'ActionParameterDefined'),
 'ActionTable': ('.action', 'ActionTable'),
 'action_space_from_definition': ('.action', 'action_space_from_definition'),
 'action_from_dict': ('.action', 'action_from_dict'),
 'ACTION_SPACE_KINDS': ('.action', 'ACTION_SPACE_KINDS'),
 'ActionSpace': ('.action', 'ActionSpace'),
 'ContinuousParameter': ('.action', 'ContinuousParameter'),
 'ContinuousAction': ('.action', 'ContinuousAction'),
 'MultiHeadAction': ('.action', 'MultiHeadAction'),
 'DiscreteActionSpace': ('.action', 'DiscreteActionSpace'),
 'ContinuousActionSpace': ('.action', 'ContinuousActionSpace'),
 'MultiDiscreteActionSpace': ('.action', 'MultiDiscreteActionSpace'),
 'MultiContinuousActionSpace': ('.action', 'MultiContinuousActionSpace'),
 'CompositeAction': ('.action', 'CompositeAction'),
 'EpisodeActionDefined': ('.action', 'EpisodeActionDefined'),
 'StepActionDefined': ('.action', 'StepActionDefined'),
 'ShapingOutputRange': ('.reward', 'ShapingOutputRange'),
 'region_activity_score': ('.reward', 'region_activity_score'),
 'region_activity_shaping': ('.reward', 'region_activity_shaping'),
 'region_activity_strip_score': ('.reward', 'region_activity_strip_score'),
 'region_activity_strip_shaping': ('.reward', 'region_activity_strip_shaping'),
 'region_threat_score': ('.reward', 'region_threat_score'),
 'region_threat_shaping': ('.reward', 'region_threat_shaping'),
 'region_threat_transition_score': ('.reward', 'region_threat_transition_score'),
 'region_visible_target_seek_score': ('.reward', 'region_visible_target_seek_score'),
 'region_visible_target_seek_shaping': ('.reward', 'region_visible_target_seek_shaping'),
 'target_selection_shaping': ('.reward', 'target_selection_shaping'),
 'CGCS2000Frame': ('.state', 'CGCS2000Frame'),
 'CoordinateSystem': ('.state', 'CoordinateSystem'),
 'Entity': ('.state', 'Entity'),
 'EnvState': ('.state', 'EnvState'),
 'GeoPoint': ('.state', 'GeoPoint'),
 'LocalFrame': ('.state', 'LocalFrame'),
 'State': ('.state', 'State'),
 'StateDescribe': ('.state', 'StateDescribe'),
 'WGS84Frame': ('.state', 'WGS84Frame'),
 'AECEnvironment': ('.extensions', 'AECEnvironment'),
 'AgentID': ('.extensions', 'AgentID'),
 'ObsType': ('.extensions', 'ObsType'),
 'ParallelEnvironment': ('.extensions', 'ParallelEnvironment'),
 'ValueErrorException': ('.extensions', 'ValueErrorException'),
 'StateType': ('.extensions', 'StateType'),
 'Evaluation': ('.evaluation', 'Evaluation'),
 'EvalControlRole': ('.evaluation', 'EvalControlRole'),
 'EvaluationConfigBuilder': ('.evaluation', 'EvaluationConfigBuilder'),
 'EvaluationProtocol': ('.evaluation', 'EvaluationProtocol'),
 'ControlSDK': ('.portal', 'ControlSDK'),
 'EvalControlSDK': ('.portal', 'EvalControlSDK'),
 'EvaluationSDK': ('.portal', 'EvaluationSDK'),
 'LearnSDK': ('.portal', 'LearnSDK'),
 'LearnControlSDK': ('.portal', 'LearnControlSDK'),
 'ModelSDK': ('.portal', 'ModelSDK'),
 'SampleSDK': ('.portal', 'SampleSDK'),
 'SerialStageTrainSDK': ('.portal', 'SerialStageTrainSDK'),
 'SerialTrainSDK': ('.portal', 'SerialTrainSDK'),
 'StageControlSDK': ('.portal', 'StageControlSDK'),
 'StageLearnSDK': ('.portal', 'StageLearnSDK'),
 'read_globe_env_vars': ('.environment', 'read_globe_env_vars'),
 'apply_globe_env_vars': ('.environment', 'apply_globe_env_vars'),
 'bootstrap_environment': ('.environment', 'bootstrap_environment')}
__all__ = ['__version__', 'AECEnvironment', 'ValueErrorException', 'ParallelEnvironment', 'EnvConfig', 'AgentConfig', 'LearnerConfig', 'TrainConfig', 'SampleType', 'TaskConfig', 'MonteCarloMethodConfig', 'EvalConfig', 'GrpcConfig', 'TorchRpcConfig', 'StageConfig', 'StageTaskConfig', 'AgentID', 'ObsType', 'StateType', 'GeoPoint', 'WGS84Frame', 'CGCS2000Frame', 'LocalFrame', 'CoordinateSystem', 'StateDescribe', 'State', 'Entity', 'EnvState', 'Action', 'ActionDefined', 'ActionParameterDefined', 'ActionTable', 'StepActionDefined', 'EpisodeActionDefined', 'ShapingOutputRange', 'target_selection_shaping', 'region_visible_target_seek_score', 'region_visible_target_seek_shaping', 'region_activity_score', 'region_activity_shaping', 'region_activity_strip_score', 'region_activity_strip_shaping', 'region_threat_score', 'region_threat_transition_score', 'region_threat_shaping', 'configure_sdk', 'get_project_root', 'ControlSDK', 'StageControlSDK', 'LearnSDK', 'ModelSDK', 'LearnControlSDK', 'StageLearnSDK', 'SampleSDK', 'EvaluationSDK', 'EvalControlSDK', 'Evaluation', 'EvalControlRole', 'EvaluationConfigBuilder', 'EvaluationProtocol', 'SerialStageTrainSDK', 'SerialTrainSDK', 'ActionSpace', 'ContinuousParameter', 'ContinuousAction', 'MultiHeadAction', 'DiscreteActionSpace', 'ContinuousActionSpace', 'MultiDiscreteActionSpace', 'MultiContinuousActionSpace', 'CompositeAction', 'action_from_dict', 'action_space_from_definition', 'ACTION_SPACE_KINDS', 'read_globe_env_vars', 'apply_globe_env_vars', 'bootstrap_environment']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
