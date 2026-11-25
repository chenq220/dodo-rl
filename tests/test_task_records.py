from types import SimpleNamespace

from scheduler.tasks.task_records import build_episode_rows, build_metric_output_rows


def test_build_episode_rows_joins_epsilon_and_metrics():
    seq = [SimpleNamespace(epsilon=0.1), SimpleNamespace(epsilon=0.05)]
    learn = {
        "archer": {
            "episode_num": [1, 2],
            "loss": [0.5, 0.3],
            "skip_update": [True, True],
            "skip_reason": ["warmup", "warmup"],
        },
        "ignored": {"episode_num": [1], "loss": [9.0]},
    }
    rows = build_episode_rows(
        sample_completed=2,
        generation=1,
        sample_task_seq=seq,
        learn_results=learn,
        learner_names=["archer"],
    )
    assert rows[0] == {"episode": 1, "generation": 1, "epsilon": 0.1, "archer_loss": 0.5}
    assert "archer_skip_update" not in rows[0]
    assert "ignored_loss" not in rows[0]
    assert rows[1]["archer_loss"] == 0.3


def test_build_metric_output_rows_skips_none():
    out = build_metric_output_rows(
        sample_completed=3,
        generation=0,
        sample_metrics={"success": [1.0, None, 0.0]},
        output_names=["success", "missing"],
    )
    assert out["success"] == [
        {"episode": 1, "generation": 0, "value": 1.0},
        {"episode": 3, "generation": 0, "value": 0.0},
    ]
    assert out["missing"] == []
