import copy
from types import SimpleNamespace
import pytest
from config import TrainConfig
from common.simulation import apply_stage_sim_metadata


def test_metadata_uses_baseline_and_does_not_replace_manager(monkeypatch):
    monkeypatch.setattr('importlib.metadata.version', lambda name: '0.4.1')
    applied = []
    driver = SimpleNamespace(config=SimpleNamespace(metadata={'image': 'base', 'nested': {'keep': 1}}),
                             configure=lambda changes: applied.append(copy.deepcopy(changes)))
    manager = SimpleNamespace(instances=driver)
    apply_stage_sim_metadata(manager, TrainConfig(sim_metadata={'nested': {'phase': 0}, 'environment': ['A=1']}))
    apply_stage_sim_metadata(manager, TrainConfig(sim_metadata={'environment': ['A=2']}))
    assert applied[0]['metadata']['nested'] == {'keep': 1, 'phase': 0}
    assert applied[1]['metadata']['nested'] == {'keep': 1}
    assert applied[1]['metadata']['image'] == 'base'
    assert manager.instances is driver
    apply_stage_sim_metadata(None, TrainConfig())


@pytest.mark.parametrize('manager,config', [
    (None, TrainConfig(sim_metadata={'environment': []})),
    (SimpleNamespace(instances=None), TrainConfig(sim_metadata={'image': 'a'})),
    (None, TrainConfig(sample_type='offline', sim_metadata={'image': 'a'})),
])
def test_invalid_resource_modes_fail_before_sampling(manager, config):
    with pytest.raises(ValueError, match='task-owned'):
        apply_stage_sim_metadata(manager, config)
