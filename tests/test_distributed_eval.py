from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import AgentConfig, AgentType, EvalConfig
from scheduler.artifact import Artifact, ArtifactType
from common.portal_eval import resolve_distributed_evaluation_class
from roles.eval_control import EvalControlRole
from roles.common.eval_reports import EvalEpisodeReportMixin
from roles.evaluation import EvaluationRole
from scheduler.train_task import TrainTaskManager
from subjects.cooperative_search.evaluation import (
    CooperativeSearchEvalControl,
    CooperativeSearchEvaluation,
)
from utils.toml_config import TomlTrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "start_distributed_eval.sh"

_EPISODE_REPORT_KWARGS = (
    "total_rewards",
    "episode_metrics",
    "episode_rewards_details",
    "artifacts",
    "episode_info",
)


def _reported_eval_task_manager(tmp_path: Path):
    config = EvalConfig(
        max_episode=1,
        result_save_dir=str(tmp_path / "results"),
        render_video_dir=str(tmp_path / "recordings"),
        artifacts_archive_save_dir=str(tmp_path / "archives"),
    )
    task_manager = TrainTaskManager(
        config=config,
        policy_agent_configs=[
            AgentConfig(name="uav", agent_type=AgentType.SB3.value, agent_names=["uav"]),
        ],
        learner_configs=[],
        stage_name="eval",
    )
    token = task_manager.get_sample_task_info(sample_role_name="sample-0")
    assert token is not None
    task_manager.report_sample_result(
        token.episode_num,
        total_rewards={"uav": 1.5},
        episode_metrics={"steps": 10.0, "success": 1.0},
        episode_rewards_details={"detect": 0.25},
        artifacts=[Artifact(ArtifactType.RENDER_GIF, "/tmp/eval.gif")],
        lease_id=token.lease_id,
    )
    return task_manager, token


def _capture_episode_report_calls(role) -> list[dict]:
    calls: list[dict] = []

    def episode_report(episode_index, **kwargs):
        calls.append({"episode_index": episode_index, **kwargs})
        return EvalEpisodeReportMixin.episode_report(role, episode_index, **kwargs)

    role.episode_report = episode_report
    return calls


def _assert_episode_report_kwargs_match_task_manager(calls: list[dict], task_manager) -> None:
    records = task_manager.get_eval_episode_records()
    assert len(calls) == 1
    assert len(records) == 1
    call = calls[0]
    record = records[0]
    assert set(call) == {"episode_index", *_EPISODE_REPORT_KWARGS}
    assert call["episode_index"] == record["episode"]
    for key in _EPISODE_REPORT_KWARGS:
        assert call[key] == record[key]
    assert call["total_rewards"] == {"uav": 1.5}
    assert call["episode_metrics"] == {"steps": 10.0, "success": 1.0}
    assert call["episode_rewards_details"] == {"detect": 0.25}
    assert call["artifacts"] is None
    assert call["episode_info"]["seed"] == record["episode_info"]["seed"]
    assert call["episode_info"]["stage_name"] == "eval"
    assert call["episode_info"]["task_id"] == record["episode_info"]["task_id"]
    assert call["episode_info"]["start_datetime"] is not None
    assert call["episode_info"]["end_datetime"] is not None


def test_resolve_distributed_evaluation_class_defaults_to_eval_control(tmp_path: Path) -> None:
    config_file = tmp_path / "eval.toml"
    config_file.write_text("[entry]\n", encoding="utf-8")
    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    assert resolve_distributed_evaluation_class(config) is EvalControlRole


def test_resolve_distributed_evaluation_class_rejects_non_eval_control(
        tmp_path: Path,
        monkeypatch,
) -> None:
    module = tmp_path / "bad_eval.py"
    module.write_text(
        "from roles.evaluation import EvaluationRole\n"
        "class NotAControl(EvaluationRole):\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_file = tmp_path / "eval.toml"
    config_file.write_text(
        '[entry]\ndistributed_evaluation_class = "bad_eval:NotAControl"\n',
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    with pytest.raises(TypeError, match="must be a subclass of EvalControlRole"):
        resolve_distributed_evaluation_class(config)


def test_eval_control_role_reports_from_task_manager() -> None:
    finished = []
    closed = []
    completed = []
    role = EvalControlRole.__new__(EvalControlRole)
    role.role_id = "eval-control-0"
    role.episode_results = []
    role.task_manager = SimpleNamespace(
        is_sample_completed=lambda: True,
        task_id="task-1",
        stage_name="eval",
        get_eval_episode_records=lambda: [{
            "episode": 3,
            "total_rewards": {"uav": 1.5},
            "episode_metrics": {"steps": 10.0, "success": 1.0},
            "episode_rewards_details": {"detect": 0.25},
            "artifacts": {"gif": "/tmp/eval.gif"},
            "episode_info": {"seed": 11, "stage_name": "eval", "task_id": "task-1"},
        }],
        notify_task_complete=lambda: (completed.append("completed"), assert_rpc_open()),
        put_report_outputs=lambda outputs: None,
    )
    role.monte_carlo_method = None
    role.artifacts_manager = SimpleNamespace(
        put_log=lambda *args, **kwargs: None,
        complete=lambda *args, **kwargs: None,
    )
    role._rpc_host = SimpleNamespace(start=lambda: None)
    def assert_rpc_open():
        assert not closed, "archive must complete while RPC is still serving workers"

    role.close = lambda: closed.append("closed")
    role.finished_report = lambda: finished.append("done")

    role.run()

    assert len(role.episode_results) == 1
    assert role.episode_results[0]["episode"] == 3
    assert role.episode_results[0]["total_reward"] == 1.5
    assert role.episode_results[0]["group_rewards"] == {"uav": 1.5}
    assert role.episode_results[0]["episode_rewards_details"] == {"detect": 0.25}
    assert role.episode_results[0]["artifacts"] == {"gif": "/tmp/eval.gif"}
    assert role.episode_results[0]["episode_info"]["seed"] == 11
    assert finished == ["done"]
    assert closed == ["closed"]
    assert completed == ["completed"]


def test_evaluation_role_reports_from_task_manager(monkeypatch) -> None:
    monkeypatch.setattr("roles.evaluation.log_simulation_middleware", lambda *args, **kwargs: None)
    finished = []
    closed = []
    completed = []
    role = EvaluationRole.__new__(EvaluationRole)
    role.sim_manager = None
    role.episode_results = []
    role.monte_carlo_method = None
    role.env = SimpleNamespace(close=lambda: closed.append("env"))
    role.task_manager = SimpleNamespace(
        is_sample_completed=lambda: True,
        task_id="task-1",
        get_eval_episode_records=lambda: [{
            "episode": 3,
            "total_rewards": {"uav": 1.5},
            "episode_metrics": {"steps": 10.0, "success": 1.0},
            "episode_rewards_details": {"detect": 0.25},
            "artifacts": {"gif": "/tmp/eval.gif"},
            "episode_info": {"seed": 11, "stage_name": "eval", "task_id": "task-1"},
        }],
        notify_task_complete=lambda: completed.append("completed"),
        put_report_outputs=lambda outputs: None,
    )
    role.finished_report = lambda: finished.append("done")

    role.run()

    assert len(role.episode_results) == 1
    assert role.episode_results[0]["episode"] == 3
    assert role.episode_results[0]["total_reward"] == 1.5
    assert role.episode_results[0]["group_rewards"] == {"uav": 1.5}
    assert role.episode_results[0]["episode_rewards_details"] == {"detect": 0.25}
    assert role.episode_results[0]["artifacts"] == {"gif": "/tmp/eval.gif"}
    assert role.episode_results[0]["episode_info"]["seed"] == 11
    assert finished == ["done"]
    assert closed == ["env"]
    assert completed == ["completed"]


def test_eval_control_episode_report_receives_task_manager_payload(tmp_path: Path) -> None:
    task_manager, token = _reported_eval_task_manager(tmp_path)
    try:
        role = EvalControlRole.__new__(EvalControlRole)
        role.role_id = "eval-control-0"
        role.episode_results = []
        role.task_manager = task_manager
        role.monte_carlo_method = None
        role.artifacts_manager = SimpleNamespace(
            put_log=lambda *args, **kwargs: None,
            complete=lambda *args, **kwargs: None,
        )
        role._rpc_host = SimpleNamespace(start=lambda: None)
        role.close = lambda: None
        calls = _capture_episode_report_calls(role)
        role.run()
        _assert_episode_report_kwargs_match_task_manager(calls, task_manager)
        assert calls[0]["episode_index"] == token.episode_num
    finally:
        task_manager.close()


def test_evaluation_episode_report_receives_task_manager_payload(
        tmp_path: Path,
        monkeypatch,
) -> None:
    monkeypatch.setattr("roles.evaluation.log_simulation_middleware", lambda *args, **kwargs: None)
    task_manager, token = _reported_eval_task_manager(tmp_path)
    try:
        role = EvaluationRole.__new__(EvaluationRole)
        role.sim_manager = None
        role.episode_results = []
        role.monte_carlo_method = None
        role.env = SimpleNamespace(close=lambda: None)
        role.task_manager = task_manager
        calls = _capture_episode_report_calls(role)
        role.run()
        _assert_episode_report_kwargs_match_task_manager(calls, task_manager)
        assert calls[0]["episode_index"] == token.episode_num
    finally:
        task_manager.close()


@pytest.mark.parametrize(
    ("role_cls", "csv_prefix"),
    [
        (CooperativeSearchEvalControl, "cooperative_search_qmix_dist_eval_"),
        (CooperativeSearchEvaluation, "cooperative_search_qmix_eval_"),
    ],
)
def test_cooperative_search_eval_writes_csv_from_task_manager_record(
        tmp_path: Path,
        monkeypatch,
        role_cls,
        csv_prefix,
) -> None:
    monkeypatch.chdir(tmp_path)
    role = role_cls.__new__(role_cls)
    record = role.episode_report(
        9,
        total_rewards={"uav": 1.0},
        episode_metrics={
            "steps": 5.0,
            "success": 0.0,
            "target_discovered_count": 2.0,
            "own_loss_count": 1.0,
            "coverage_rate_percent": 50.0,
            "false_target_count": 1.0,
        },
        episode_rewards_details={"detect": 0.2},
        artifacts={"gif": "eval.gif"},
        episode_info={"seed": 9},
    )
    role.episode_results = [record]
    result = role.finished_report()

    metrics = record["episode_metrics"]
    assert metrics["episode"] == 9
    assert metrics["steps"] == 5
    assert metrics["success"] is False
    assert metrics["target_discovered_count"] == 2
    assert metrics["own_loss_count"] == 1
    assert metrics["coverage_rate_percent"] == 50.0
    assert metrics["false_target_count"] == 1
    csv_path = Path(result["episode_metrics_csv"])
    assert csv_path.is_file()
    assert csv_path.name.startswith(csv_prefix)
    with csv_path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["episode"] == "9"
    assert rows[0]["own_loss_count"] == "1"
    assert record["episode_rewards_details"] == {"detect": 0.2}
    assert record["artifacts"] == {"gif": "eval.gif"}


def test_get_eval_episode_records_excludes_internal_artifact_contract() -> None:
    task_manager = TrainTaskManager(
        config=EvalConfig(max_episode=1),
        policy_agent_configs=[
            AgentConfig(name="uav", agent_type=AgentType.SB3.value, agent_names=["uav"]),
        ],
        learner_configs=[],
        stage_name="eval",
    )
    try:
        token = task_manager.get_sample_task_info(sample_role_name="sample-0")
        assert token is not None
        task_manager.report_sample_result(
            token.episode_num,
            total_rewards={"uav": 1.5},
            episode_metrics={"steps": 10.0, "success": 1.0},
            episode_rewards_details={"detect": 0.25},
            artifacts=[Artifact(ArtifactType.RENDER_GIF, "/tmp/eval.gif")],
            lease_id=token.lease_id,
        )
        records = task_manager.get_eval_episode_records()
    finally:
        task_manager.close()

    assert len(records) == 1
    record = records[0]
    assert record["episode"] == 1
    assert record["total_rewards"] == {"uav": 1.5}
    assert record["episode_metrics"]["steps"] == 10.0
    assert record["episode_metrics"]["success"] == 1.0
    assert record["episode_rewards_details"] == {"detect": 0.25}
    assert record["artifacts"] is None
    assert task_manager.episode_artifacts[0] == [
        Artifact(ArtifactType.RENDER_GIF, "/tmp/eval.gif"),
    ]
    assert record["episode_info"]["seed"] == token.seed
    assert record["episode_info"]["stage_name"] == "eval"
    assert record["episode_info"]["task_id"] == token.task_id
    assert record["episode_info"]["start_datetime"] is not None
    assert record["episode_info"]["end_datetime"] is not None


def _fake_python(tmp_path: Path, protocol: str) -> Path:
    path = tmp_path / "fake-python"
    path.write_text(
        f"""#!/usr/bin/env bash
set -eu
if [[ "${{1:-}}" == *"resolve_design_task_id.py" ]]; then
  printf '%s\n' "$DODO_TASK_ID"
  exit 0
fi
if [[ "${{1:-}}" == "-c" ]]; then
  if [[ "{protocol}" == "torch" ]]; then
    printf 'torch\\n2\\n'
  else
    printf 'grpc\\n0\\n'
  fi
  exit 0
fi
module="${{2:-}}"
printf '%s,%s\\n' "${{RANK:-none}}" "$module" >>"$FAKE_RECORD_FILE"
if [[ "$module" == "src.portal.eval_control" ]]; then
  printf 'DODO_TRAINING_STAGE=eval\nDODO_TRAINING_EPISODE=3/50\n' >"${{DODO_PROGRESS_FILE}}"
  if [[ "${{FAKE_CONTROL_MODE:-}}" == "finish" ]]; then
    sleep 0.2
    exit 0
  fi
  exec sleep 30
fi
if [[ "$FAKE_SAMPLE_MODE" == "exit" ]]; then
  exit 0
fi
exec sleep 0.4
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_eval_launcher(
        tmp_path: Path,
        protocol: str,
        *,
        extra_args: list[str] | None = None,
        control_mode: str = "finish",
        sample_mode: str = "wait",
):
    config = tmp_path / "training.toml"
    config.write_text("# parsed by fake python\n", encoding="utf-8")
    record = tmp_path / "roles.csv"
    environment = os.environ.copy()
    environment.update({
        "PYTHON_BIN": str(_fake_python(tmp_path, protocol)),
        "DODO_TASK_ID": f"eval-launcher-{protocol}-{os.getpid()}",
        "FAKE_RECORD_FILE": str(record),
        "FAKE_CONTROL_MODE": control_mode,
        "FAKE_SAMPLE_MODE": sample_mode,
        "DODO_LOG_DIR": str(tmp_path / "distributed-logs"),
    })
    command = ["bash", str(LAUNCHER), str(config)]
    if extra_args:
        command.extend(extra_args)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=8,
    )
    records = record.read_text(encoding="utf-8").splitlines() if record.exists() else []
    return result, records


def test_eval_launcher_starts_torch_rpc_eval_control_and_samples(tmp_path: Path) -> None:
    result, records = _run_eval_launcher(tmp_path, "torch")

    assert result.returncode == 0, result.stdout
    assert "评估进度：局数=3/50" in result.stdout
    assert sorted(records) == sorted([
        "0,src.portal.eval_control",
        "1,src.portal.eval_sample",
    ])


def test_eval_launcher_starts_grpc_eval_control_and_samples_without_rank(tmp_path: Path) -> None:
    result, records = _run_eval_launcher(tmp_path, "grpc", extra_args=["2"])

    assert result.returncode == 0, result.stdout
    assert "评估进度：局数=3/50" in result.stdout
    assert sorted(records) == sorted([
        "none,src.portal.eval_control",
        "none,src.portal.eval_sample",
        "none,src.portal.eval_sample",
    ])
