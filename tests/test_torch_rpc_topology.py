from __future__ import annotations

import pytest

from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    MODEL_MANAGER,
    MONTE_CARLO_METHOD,
    REPLAY_BUFFER,
    SIM_MANAGER,
    TRAIN_TASK_MANAGER,
)
from rpc.torch_rpc.runtime import resolve_service_ranks


CONTROL_SERVICES = {
    TRAIN_TASK_MANAGER: object(),
    REPLAY_BUFFER: object(),
    SIM_MANAGER: object(),
    MONTE_CARLO_METHOD: object(),
    ARTIFACTS_MANAGER: object(),
}


def test_merged_center_routes_every_service_to_rank_zero() -> None:
    routes = resolve_service_ranks(
        0,
        "stage-learn-0",
        {**CONTROL_SERVICES, MODEL_MANAGER: object()},
    )

    assert set(routes.values()) == {0}


def test_full_topology_hosts_control_at_zero_and_model_at_two() -> None:
    control_routes = resolve_service_ranks(0, "stage-control-0", CONTROL_SERVICES)
    model_routes = resolve_service_ranks(2, "model-0", {MODEL_MANAGER: object()})
    learn_routes = resolve_service_ranks(1, "learn-0")

    for routes in (control_routes, model_routes, learn_routes):
        assert routes[TRAIN_TASK_MANAGER] == 0
        assert routes[REPLAY_BUFFER] == 0
        assert routes[ARTIFACTS_MANAGER] == 0
        assert routes[MODEL_MANAGER] == 2


def test_sample_rank_offset_selects_topology_without_extra_environment_variable() -> None:
    simplified = resolve_service_ranks(4, "sample-3")
    full = resolve_service_ranks(6, "sample-3")

    assert simplified[MODEL_MANAGER] == 0
    assert full[MODEL_MANAGER] == 2
    assert simplified[TRAIN_TASK_MANAGER] == full[TRAIN_TASK_MANAGER] == 0


def test_sample_rank_offset_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="sample_index"):
        resolve_service_ranks(2, "sample-0")
