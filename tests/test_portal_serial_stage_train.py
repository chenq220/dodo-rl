from argparse import Namespace
from pathlib import Path

from common.portal_entry import load_training_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "subjects/cooperative_search/cooperative_search_qmix.toml"


def test_serial_stage_train_portal_maps_stage_task_config(monkeypatch):
    monkeypatch.chdir(ROOT)
    conf = load_training_config(Namespace(config_file=str(CONFIG)), ROOT)
    kwargs = dict(
        stage_task_config=conf.stage_task_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        replay_buffer_config=conf.replay_buffer_config,
        model_manager_config=conf.model_manager_config,
        sim_manager_config=conf.sim_manager_config,
    )
    stages = kwargs["stage_task_config"].stages
    assert [stage.stage_name for stage in stages] == ["search", "evade"]
    assert kwargs["env_config"].use_sim is False
    assert conf.train_config.record_sample is False
