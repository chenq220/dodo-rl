"""项目内置插件清单。

Portal 入口默认加载本清单；训练 TOML 只需要在 [plugins_config] 中声明
启用/禁用策略，或追加本项目之外的插件。这里不导入插件类，避免入口启动时
产生环境、仿真或扩展模块的副作用。
"""
from __future__ import annotations

from algos.builtin_plugins import BUILTIN_AGENT_PLUGINS, BUILTIN_LEARNER_PLUGINS
from common.plugins import GlobalPluginRegistry


PROJECT_PLUGIN_REGISTRY = GlobalPluginRegistry(
    simulation={
        "jsql": "dodo_sim.simulations.jsql:JsqlSim",
        "afsim": "dodo_sim.simulations.afsim:AFSim",
    },
    environment={
        "cooperative_search": "envs.sense.cooperative_search_env:CooperativeSearchEnv",
        "cooperative_search_partition": (
            "envs.sense.cooperative_search_partition_env:CooperativeSearchPartitionEnv"
        ),
        "cooperative_search_threat_training": (
            "envs.sense.cooperative_search_threat_env:CooperativeSearchThreatEnv"
        ),
        "cooperative_search_partition_threat_training": (
            "envs.sense.cooperative_search_partition_threat_env:"
            "CooperativeSearchPartitionThreatEnv"
        ),
        "region_activity": "envs.sense.region_activity_env:RegionActivityEnv",
        "threat_region": "envs.sense.threat_region_env:ThreatRegionEnv",
        "track_seek": "envs.sense.track_seek_env:TrackSeekEnv",
        "target_selection": "envs.sense.target_selection_env:TargetSelectionEnv",
    },
    agent=BUILTIN_AGENT_PLUGINS,
    learner=BUILTIN_LEARNER_PLUGINS,
    module={
        # 默认不内置 demo module；需要流程改写时在 TOML [plugins_config.module] 追加。
        # "alpha_stage_task": "portal.plugins.custom_stage_task:AlphaStageTask",
    },
)


def project_plugin_registry() -> GlobalPluginRegistry:
    """取得项目全局插件定义。定义对象不可变，可安全复用。"""
    return PROJECT_PLUGIN_REGISTRY


__all__ = ["PROJECT_PLUGIN_REGISTRY", "project_plugin_registry"]
