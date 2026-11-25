"""将 AutoML 归档 advice.yaml 中的超参建议写回训练 TOML。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

ParamSource = Literal["best", "current", "advice"]


@dataclass
class HandoffParams:
    """从 advice.yaml 解析出的固定训练超参。"""

    stage_name: str
    task_id: str
    generation: int
    reason: str
    algo_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    skeleton_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    env_metadata_overrides: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


def load_advice_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"advice.yaml must be a mapping, given={type(payload).__name__}")
    return payload


def _set_nested(root: dict[str, Any], path: str, value: Any) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _split_optuna_params(params: dict[str, Any]):
    algo_overrides: dict[str, dict[str, Any]] = {}
    skeleton_overrides: dict[str, dict[str, Any]] = {}
    env_overrides: dict[str, Any] = {}
    seed: Optional[int] = None
    for key, value in params.items():
        if key == "seed":
            seed = int(value)
            continue
        key = str(key)
        if key.startswith("env_metadata."):
            _set_nested(env_overrides, key[len("env_metadata."):], value)
            continue
        if key.startswith("learner."):
            learner_name, config_path = key[len("learner."):].split(".", 1)
        elif "." in key:
            # 历史 advice.yaml 仍可回填；仅 TOML 搜索空间旧格式被一次性废弃。
            learner_name, old_path = key.split(".", 1)
            config_path = old_path if old_path == "gamma" else f"algo_config.{old_path}"
        else:
            continue
        if config_path.startswith("algo_config."):
            algo_overrides.setdefault(learner_name, {})[
                config_path[len("algo_config."):]
            ] = value
        else:
            skeleton_overrides.setdefault(learner_name, {})[config_path] = value
    return algo_overrides, skeleton_overrides, env_overrides, seed


def extract_handoff_params(
        advice: dict[str, Any],
        *,
        source: ParamSource = "best",
) -> HandoffParams:
    """从 advice 提取要写回 TOML 的超参与种子。"""
    stage_name = str(advice.get("stage_name") or "")
    if not stage_name:
        raise ValueError("advice.yaml missing stage_name")

    optuna = advice.get("optuna") or {}
    if not isinstance(optuna, dict):
        optuna = {}

    algo_overrides: dict[str, dict[str, Any]] = {}
    skeleton_overrides = copy.deepcopy(advice.get("skeleton_overrides") or {})
    env_overrides = copy.deepcopy(advice.get("env_metadata_overrides") or {})
    seed: Optional[int] = None

    if source == "best":
        best_params = optuna.get("best_params") or {}
        if not isinstance(best_params, dict) or not best_params:
            raise ValueError("advice.yaml missing optuna.best_params (required for --source best)")
        algo_overrides, skeleton_overrides, env_overrides, seed = _split_optuna_params(best_params)
    elif source in ("current", "advice"):
        raw_algo = advice.get("algo_config_overrides") or {}
        if not isinstance(raw_algo, dict):
            raise ValueError("advice.yaml algo_config_overrides must be a mapping")
        algo_overrides = copy.deepcopy(raw_algo)
        if source == "current":
            seed_raw = advice.get("next_trial_seed", advice.get("seed"))
        else:
            seed_raw = advice.get("seed", advice.get("next_trial_seed"))
        if seed_raw is not None:
            seed = int(seed_raw)
    else:
        raise ValueError(f"unsupported param source: {source!r}")

    if seed is None:
        best_params = optuna.get("best_params") or {}
        if isinstance(best_params, dict) and "seed" in best_params:
            seed = int(best_params["seed"])
        elif advice.get("seed") is not None:
            seed = int(advice["seed"])

    algo_overrides = {k: v for k, v in algo_overrides.items() if v}
    skeleton_overrides = {
        k: v for k, v in skeleton_overrides.items() if isinstance(v, dict) and v
    }
    return HandoffParams(
        stage_name=stage_name,
        task_id=str(advice.get("task_id") or ""),
        generation=int(advice.get("generation") or 0),
        reason=str(advice.get("reason") or ""),
        algo_config_overrides=algo_overrides,
        skeleton_overrides=skeleton_overrides,
        env_metadata_overrides=env_overrides,
        seed=seed,
    )


def _indexed_items(table: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = [(key, value) for key, value in table.items() if str(key).isdigit()]
    return sorted(items, key=lambda item: int(item[0]))


def _find_stage_key(stages: dict[str, Any], stage_name: str) -> str:
    for key, values in _indexed_items(stages):
        if isinstance(values, dict) and str(values.get("name")) == stage_name:
            return key
    known = [
        str(values.get("name"))
        for _, values in _indexed_items(stages)
        if isinstance(values, dict)
    ]
    raise ValueError(
        f"stage {stage_name!r} not found in TOML [stages.*]; known={known}"
    )


def _find_learner_key(learners: dict[str, Any], learner_name: str) -> Optional[str]:
    for key, values in _indexed_items(learners):
        if isinstance(values, dict) and str(values.get("name")) == learner_name:
            return key
    return None


def _ensure_mapping(root: dict[str, Any], *keys: str) -> dict[str, Any]:
    current = root
    for key in keys:
        value = current.get(key)
        if not isinstance(value, dict):
            value = {}
            current[key] = value
        current = value
    return current


def apply_handoff_to_toml_dict(
        toml_data: dict[str, Any],
        handoff: HandoffParams,
        *,
        disable_stage_automl: bool = True,
) -> list[str]:
    """就地修改 TOML 字典并返回人类可读变更摘要。"""
    changes: list[str] = []
    stages = toml_data.setdefault("stages", {})
    stage_key = _find_stage_key(stages, handoff.stage_name)
    stage_table = stages[stage_key]

    learners = toml_data.get("learners") or {}

    for learner_name, overrides in handoff.algo_config_overrides.items():
        algo_table = _ensure_mapping(
            stage_table,
            "learner_overrides",
            learner_name,
            "algo_config",
        )
        for param_key, param_value in overrides.items():
            old = algo_table.get(param_key)
            algo_table[param_key] = param_value
            changes.append(
                f"[stages.{stage_key}.learner_overrides.{learner_name}.algo_config] "
                f"{param_key}: {old!r} -> {param_value!r}"
            )

        learner_key = _find_learner_key(learners, learner_name)
        if learner_key is not None:
            global_algo = _ensure_mapping(learners, learner_key, "algo_config")
            for param_key, param_value in overrides.items():
                old = global_algo.get(param_key)
                global_algo[param_key] = param_value
                changes.append(
                    f"[learners.{learner_key}.algo_config] "
                    f"{param_key}: {old!r} -> {param_value!r}"
                )

    for learner_name, overrides in handoff.skeleton_overrides.items():
        skeleton_table = _ensure_mapping(stage_table, "learner_overrides", learner_name)
        for param_key, param_value in overrides.items():
            if param_key == "algo_config":
                continue
            old = skeleton_table.get(param_key)
            skeleton_table[param_key] = param_value
            changes.append(
                f"[stages.{stage_key}.learner_overrides.{learner_name}] "
                f"{param_key}: {old!r} -> {param_value!r}"
            )

    if handoff.env_metadata_overrides:
        env_metadata = _ensure_mapping(stage_table, "train_overrides", "env_metadata")
        def merge(target: dict[str, Any], source: dict[str, Any], prefix: str = "") -> None:
            for key, value in source.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    merge(_ensure_mapping(target, key), value, path)
                else:
                    old = target.get(key)
                    target[key] = value
                    changes.append(
                        f"[stages.{stage_key}.train_overrides.env_metadata] "
                        f"{path}: {old!r} -> {value!r}"
                    )
        merge(env_metadata, handoff.env_metadata_overrides)

    if handoff.seed is not None:
        train_config = toml_data.setdefault("train_config", {})
        old = train_config.get("seed")
        train_config["seed"] = handoff.seed
        changes.append(f"[train_config] seed: {old!r} -> {handoff.seed!r}")

        design = toml_data.get("design")
        if isinstance(design, dict):
            old_design = design.get("seed")
            design["seed"] = handoff.seed
            changes.append(f"[design] seed: {old_design!r} -> {handoff.seed!r}")

    if disable_stage_automl:
        auto_ml = _ensure_mapping(stage_table, "auto_ml_overrides")
        old = auto_ml.get("enabled")
        auto_ml["enabled"] = False
        changes.append(
            f"[stages.{stage_key}.auto_ml_overrides] enabled: {old!r} -> False"
        )

    return changes
