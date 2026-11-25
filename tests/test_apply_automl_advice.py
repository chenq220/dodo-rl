from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import toml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.utils.automl_advice_apply import (
    apply_handoff_to_toml_dict,
    extract_handoff_params,
)


def _sample_toml() -> dict:
    return {
        "design": {"task_name": "demo"},
        "train_config": {"max_episode": 500},
        "stages": {
            "0": {
                "name": "foundation",
                "learners": ["knight"],
                "learner_overrides": {
                    "knight": {
                        "algo_config": {
                            "lr": 0.00015,
                            "epochs": 4,
                            "entropy_coef": 0.04,
                        }
                    }
                },
                "auto_ml_overrides": {
                    "metric_primary": "reward_min_agent",
                    "learner_search_space": {
                        "0": {"agent_name": "knight", "path": "algo_config.lr"},
                    },
                    "seed_search": {"enabled": True},
                },
            }
        },
        "learners": {
            "0": {
                "name": "knight",
                "algo_type": "mappo",
                "algo_config": {
                    "lr": 0.00015,
                    "epochs": 4,
                    "entropy_coef": 0.04,
                },
            }
        },
    }


def test_extract_best_params_from_example_advice() -> None:
    advice = {
        "stage_name": "foundation",
        "task_id": "6b61f49b",
        "generation": 7,
        "reason": "extra_complete_round",
        "seed": 43985942,
        "algo_config_overrides": {
            "knight": {
                "lr": 0.0003101615275654457,
                "epochs": 4,
                "entropy_coef": 0.010001255840247122,
            }
        },
        "optuna": {
            "best_params": {
                "learner.knight.algo_config.lr": 0.0002817850364926987,
                "learner.knight.algo_config.epochs": 2,
                "learner.knight.algo_config.entropy_coef": 0.026607134430179168,
                "seed": 43985942,
                "env_metadata.reward_weights.navigation": 0.25,
            }
        },
    }
    handoff = extract_handoff_params(advice, source="best")
    assert handoff.stage_name == "foundation"
    assert handoff.seed == 43985942
    assert handoff.algo_config_overrides["knight"]["epochs"] == 2
    assert handoff.algo_config_overrides["knight"]["lr"] == 0.0002817850364926987
    assert handoff.env_metadata_overrides == {"reward_weights": {"navigation": 0.25}}


def test_apply_handoff_updates_toml_and_disables_automl() -> None:
    advice = {
        "stage_name": "foundation",
        "seed": 43985942,
        "algo_config_overrides": {"knight": {"lr": 1e-4, "epochs": 2}},
        "optuna": {"best_params": {
            "learner.knight.algo_config.lr": 1e-4,
            "learner.knight.algo_config.epochs": 2,
            "seed": 43985942,
        }},
    }

    handoff = extract_handoff_params(advice, source="best")
    toml_data = _sample_toml()
    changes = apply_handoff_to_toml_dict(toml_data, handoff)
    assert changes
    assert toml_data["stages"]["0"]["auto_ml_overrides"]["enabled"] is False
    assert toml_data["train_config"]["seed"] == handoff.seed
    assert toml_data["stages"]["0"]["learner_overrides"]["knight"]["algo_config"]["epochs"] == (
        handoff.algo_config_overrides["knight"]["epochs"]
    )
    if handoff.env_metadata_overrides:
        assert toml_data["stages"]["0"]["train_overrides"]["env_metadata"] == (
            handoff.env_metadata_overrides
        )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.toml"
        out.write_text(toml.dumps(toml_data), encoding="utf-8")
        loaded = toml.load(out)
        assert loaded["stages"]["0"]["auto_ml_overrides"]["enabled"] is False
