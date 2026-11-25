from importlib import import_module

ENV_MAPPING = {
    'track_seek': 'envs.sense.track_seek_env:TrackSeekEnv',
    'region_activity': 'envs.sense.region_activity_env:RegionActivityEnv',
    'threat_region': 'envs.sense.threat_region_env:ThreatRegionEnv',
    'target_selection': 'envs.sense.target_selection_env:TargetSelectionEnv',
}


def get_env_class(scene_name: str):
    env_path = ENV_MAPPING.get(scene_name)
    if env_path is None:
        return None
    module_name, class_name = env_path.split(':')
    return getattr(import_module(module_name), class_name)

__all__ = [
    'ENV_MAPPING',
    'get_env_class',
]
