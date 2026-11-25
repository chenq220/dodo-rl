from argparse import Namespace
from pathlib import Path

from common.portal_entry import load_training_config, resolve_role_id
from config import ModelManagerConfig
from portal import learn_control as learn_control_portal


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "subjects/cooperative_search/cooperative_search_qmix.toml"


def test_learn_control_parse_args_and_constructor_mapping(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "-f", str(CONFIG), "--role-id", "custom-learn"],
    )
    args = learn_control_portal._parse_args()
    assert args.role_id == "custom-learn"
    monkeypatch.chdir(ROOT)
    conf = load_training_config(Namespace(config_file=str(CONFIG)), ROOT)
    role_id = args.role_id or resolve_role_id(conf, "learn-control-0")
    assert role_id == "custom-learn"
    model_manager_config = conf.model_manager_config or ModelManagerConfig()
    mapped = {
        "role_id": role_id,
        "train_config": conf.train_config,
        "env_config": conf.env_config,
        "sim_config": conf.sim_config,
        "step_agents_config": conf.step_agents_config,
        "learners_config": conf.learners_config,
        "replay_buffer_config": conf.replay_buffer_config,
        "model_manager_config": model_manager_config,
        "rpc_config": conf.rpc_config,
    }
    assert mapped["env_config"].sense_name == "cooperative_search"
    assert mapped["rpc_config"] is not None
