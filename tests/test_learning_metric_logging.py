import json
from types import SimpleNamespace

from scheduler import rollout_learn


def test_learning_diagnostics_keep_precision_and_all_metrics(monkeypatch):
    messages = []
    monkeypatch.setattr(rollout_learn, "LOG", SimpleNamespace(info=messages.append))
    # 小 KL 在原来四位小数的表格中会显示为零，完整诊断仍须可复现。
    metrics = {f"diagnostic_{i}": i / 1000000 for i in range(16)}
    metrics["post_approx_kl"] = 0.0000123456789
    rollout_learn.RolloutLearnWorker._log_learning_batch_table(
        SimpleNamespace(name="policy", algo_type="mappo", generation=2),
        episode_num=27, sample_strategy="on_policy_cursor", batch_size=9,
        learning_rows=[dict(batch="1/1", time_cost=0.123456789, result=metrics)],
    )
    records = [json.loads(m.split("Learner metrics: ", 1)[1])
               for m in messages if m.startswith("Learner metrics: ")]
    assert len(records) == 1
    assert records[0]["metrics"] == metrics
    assert records[0]["time_cost"] == 0.123456789
    assert records[0]["episode_num"] == 27
    assert records[0]["generation"] == 2
