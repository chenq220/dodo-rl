"""统一租约表与回报路由（v0.14.3）单元测试。

覆盖：EvalConfig 原缺陷回归（RENDER_GIF 产物不参与路由）、旁路 kind 分派、
未知/跨代/缺失租约的分级失败、表驱动超时回收三态、accept 结算后拒绝、
以及簿记死角看门狗。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AgentConfig, AgentType, EvalConfig, TrainConfig
from scheduler.artifact import Artifact, ArtifactType
from common.exception import SampleLeaseException
from scheduler.train_task import (
    LEASE_KIND_END_EVAL,
    LEASE_KIND_MID_EVAL,
    LEASE_KIND_SAMPLE,
    SampleLeaseEntry,
    TrainTaskManager,
)

_TIMEOUT = 60.0
_EXPIRED = timedelta(seconds=_TIMEOUT * 2)


def _agent_config(name: str = "agent_a") -> AgentConfig:
    return AgentConfig(name=name, agent_type=AgentType.SB3.value, agent_names=[name])


def _train_manager(max_episode: int = 2, **kwargs) -> TrainTaskManager:
    config = TrainConfig(
        max_episode=max_episode,
        sample_learn_max_lag=None,
        sample_task_timeout=_TIMEOUT,
        **kwargs,
    )
    return TrainTaskManager(
        config=config,
        policy_agent_configs=[_agent_config()],
        learner_configs=[],
    )


def _eval_manager(max_episode: int = 2, tmp_path: Path | None = None, **kwargs) -> TrainTaskManager:
    config = EvalConfig(
        max_episode=max_episode,
        sample_task_timeout=_TIMEOUT,
        **kwargs,
    )
    return TrainTaskManager(
        config=config,
        policy_agent_configs=[_agent_config()],
        learner_configs=[],
    )


def _claim(manager: TrainTaskManager):
    token = manager.get_sample_task_info(sample_role_name="sample-0")
    assert token is not None
    return token


def _report_main(manager: TrainTaskManager, token, **kwargs):
    manager.report_sample_result(
        token.episode_num, total_rewards={"agent_a": 1.0}, lease_id=token.lease_id, **kwargs,
    )


def _register_bypass_lease(manager: TrainTaskManager, lease_id: str, kind: str, episode_num: int):
    """测试辅助：手工登记一张旁路在途租约（生产由出队签发路径写入）。"""
    manager._lease_table[lease_id] = SampleLeaseEntry(
        kind=kind,
        episode_num=episode_num,
        generation=manager.generation,
        issued_at=datetime.now(),
        dispatch_count=1,
    )


def test_eval_gif_artifact_settles_mainline_and_keeps_records(tmp_path: Path) -> None:
    """原 0.13.1 缺陷回归：EvalConfig 回报携带 GIF 产物必须走主路结算。"""
    manager = _eval_manager(max_episode=1, tmp_path=tmp_path)
    try:
        token = _claim(manager)
        _report_main(
            manager, token,
            artifacts=[Artifact(ArtifactType.RENDER_GIF, "/tmp/eval.gif")],
        )
        assert manager.sample_task_completed == 1
        assert manager.sample_task_seq[0].end_datetime is not None
        assert manager.episode_artifacts[0] == [
            Artifact(ArtifactType.RENDER_GIF, "/tmp/eval.gif"),
        ]
        assert manager._mid_eval_gif_paths == []
        records = manager.get_eval_episode_records()
        assert len(records) == 1
        # 产物走 episode_artifacts 台账；records 字段按现行契约不泄露产物清单。
        assert records[0]["artifacts"] is None
        assert records[0]["episode"] == 1
        assert token.lease_id not in manager._lease_table
    finally:
        manager.close()


def test_mid_eval_bypass_report_does_not_advance_mainline(tmp_path: Path) -> None:
    """TrainConfig 旁路租约按 kind 分派记账，不推进主路进度。"""
    manager = _train_manager(max_episode=2)
    try:
        _register_bypass_lease(manager, "mid-lease-1", LEASE_KIND_MID_EVAL, episode_num=1)
        manager.report_sample_result(
            1,
            {},
            lease_id="mid-lease-1",
            artifacts=[Artifact(ArtifactType.RENDER_GIF, "/tmp/mid.gif")],
        )
        assert manager.sample_task_completed == 0
        assert manager._mid_eval_gif_paths == ["/tmp/mid.gif"]
        assert manager.sample_task_seq[0].end_datetime is None
        assert not manager._lease_table
    finally:
        manager.close()


def test_unknown_lease_report_is_counted_and_discarded() -> None:
    manager = _train_manager()
    try:
        manager.report_sample_result(1, {}, lease_id="no-such-lease")
        assert manager._stale_report_count == 1
        assert manager.sample_task_completed == 0
        assert not manager._lease_table
    finally:
        manager.close()


def test_missing_lease_id_raises_protocol_error() -> None:
    manager = _train_manager()
    try:
        with pytest.raises(SampleLeaseException, match="lease_id is required"):
            manager.report_sample_result(1, {})
    finally:
        manager.close()


def test_episode_mismatch_with_lease_entry_raises_protocol_error() -> None:
    manager = _train_manager(max_episode=2)
    try:
        token = _claim(manager)
        with pytest.raises(SampleLeaseException, match="registered episode"):
            manager.report_sample_result(
                token.episode_num + 1, {}, lease_id=token.lease_id,
            )
        # 协议违规时租约已消费、任务未结算：等超时回收归还重试。
        assert token.lease_id not in manager._lease_table
        assert manager.sample_task_seq[0].end_datetime is None
        assert manager.sample_task_completed == 0
    finally:
        manager.close()


def test_stale_report_after_generation_restart_is_discarded() -> None:
    manager = _train_manager(max_episode=1)
    try:
        token = _claim(manager)
        with manager._lock:
            manager.generation += 1
        manager.report_sample_result(
            token.episode_num, {}, lease_id=token.lease_id,
        )
        assert manager._stale_report_count == 1
        assert manager.sample_task_completed == 0
    finally:
        manager.close()


def test_timeout_reclaim_sample_returns_token_for_retry() -> None:
    manager = _train_manager(max_episode=1)
    try:
        token = _claim(manager)
        with manager._lock:
            manager._lease_table[token.lease_id].issued_at -= _EXPIRED
        assert manager._reclaim_timed_out_leases() == 1
        assert token.lease_id not in manager._lease_table
        assert manager.sample_task_seq[0].start_datetime is None
        assert manager.sample_task_seq[0].failed_count == 1
        retry = _claim(manager)
        assert retry.episode_num == 1 and retry.lease_id != token.lease_id
    finally:
        manager.close()


def test_timeout_reclaim_mid_eval_drops_without_redispatch() -> None:
    manager = _train_manager(max_episode=1)
    try:
        _register_bypass_lease(manager, "mid-lease-1", LEASE_KIND_MID_EVAL, episode_num=1)
        with manager._lock:
            manager._lease_table["mid-lease-1"].issued_at -= _EXPIRED
        assert manager._reclaim_timed_out_leases() == 1
        assert not manager._lease_table
        assert not manager._bypass_sample_queue
    finally:
        manager.close()


def test_timeout_reclaim_end_eval_redispatches_token(tmp_path: Path) -> None:
    manager = _train_manager(max_episode=1)
    try:
        _register_bypass_lease(manager, "end-lease-1", LEASE_KIND_END_EVAL, episode_num=1)
        with manager._lock:
            manager._lease_table["end-lease-1"].issued_at -= _EXPIRED
        assert manager._reclaim_timed_out_leases() == 1
        assert "end-lease-1" not in manager._lease_table
        assert len(manager._bypass_sample_queue) == 1
        task, kind, dispatch_count = manager._bypass_sample_queue[0]
        assert kind == LEASE_KIND_END_EVAL
        assert dispatch_count == manager._end_eval_dispatch_count == 1
    finally:
        manager.close()


def test_accept_rejects_after_lease_settled() -> None:
    manager = _train_manager(max_episode=1)
    try:
        token = _claim(manager)
        assert manager.accept_sample_task_result(1, token.lease_id) is True
        _report_main(manager, token)
        assert manager.accept_sample_task_result(1, token.lease_id) is False
        assert manager.accept_sample_task_result(1, "unknown") is False
        assert manager.accept_sample_task_result(1, "") is False
    finally:
        manager.close()


def test_accept_rejects_bypass_lease() -> None:
    manager = _train_manager(max_episode=1)
    try:
        _register_bypass_lease(manager, "mid-lease-1", LEASE_KIND_MID_EVAL, episode_num=1)
        assert manager.accept_sample_task_result(1, "mid-lease-1") is False
    finally:
        manager.close()


def test_progress_deadlock_watchdog_raises() -> None:
    """无在途、无待签发、无主路可领令牌却未完成：簿记死角必须响亮报错。"""
    manager = _train_manager(max_episode=2)
    try:
        _claim(manager)
        with manager._lock:
            # 人为制造记账分歧：租约全部丢失、令牌全部标记已结束，但进度未推进。
            manager._lease_table.clear()
            for task in manager.sample_task_seq:
                task.start_datetime = None
                task.end_datetime = datetime.now()
            assert manager.sample_task_completed < manager.max_episode
            with pytest.raises(SampleLeaseException, match="deadlocked"):
                manager._raise_sample_progress_deadlock_locked()
    finally:
        manager.close()


def test_watchdog_passes_while_claimable_tokens_remain() -> None:
    manager = _train_manager(max_episode=2)
    try:
        with manager._lock:
            manager._raise_sample_progress_deadlock_locked()
        token = _claim(manager)
        with manager._lock:
            # 一张在途 + 一张可领：不触发死角。
            manager._raise_sample_progress_deadlock_locked()
        _report_main(manager, token)
        assert manager.sample_task_completed == 1
    finally:
        manager.close()


def test_eval_task_with_gif_enabled_completes_all_episodes(tmp_path: Path) -> None:
    """集成：render_gif_enabled 评估跑满后收口，记录与产物齐全。"""
    manager = _eval_manager(
        max_episode=3,
        render_gif_enabled=True,
        result_save_dir=str(tmp_path / "results"),
        render_video_dir=str(tmp_path / "recordings"),
    )
    try:
        assert manager.is_sample_completed() is False
        for _ in range(3):
            token = _claim(manager)
            _report_main(
                manager, token,
                artifacts=[Artifact(ArtifactType.RENDER_GIF, f"/tmp/eval-{token.episode_num}.gif")],
            )
        assert manager.is_sample_completed() is True
        assert manager._stale_report_count == 0
        records = manager.get_eval_episode_records()
        assert [record["episode"] for record in records] == [1, 2, 3]
        # 产物进 episode_artifacts 台账；records 不泄露产物清单（现行契约）。
        assert all(manager.episode_artifacts[idx] for idx in range(3))
        assert all(record["artifacts"] is None for record in records)
        assert manager._mid_eval_gif_paths == []
    finally:
        manager.close()
