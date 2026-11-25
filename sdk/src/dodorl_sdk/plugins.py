"""Dodo SDK 自带的算法插件清单。"""
from __future__ import annotations

from ._runtime import bootstrap_runtime

bootstrap_runtime()

from algos.builtin_plugins import BUILTIN_AGENT_PLUGINS, BUILTIN_LEARNER_PLUGINS
from common.plugins import GlobalPluginRegistry


SDK_DEFAULT_PLUGIN_REGISTRY = GlobalPluginRegistry(
    agent=BUILTIN_AGENT_PLUGINS,
    learner=BUILTIN_LEARNER_PLUGINS,
)


__all__ = ["SDK_DEFAULT_PLUGIN_REGISTRY"]
