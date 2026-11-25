def test_random_sampling_portal_maps_config(monkeypatch):
    from argparse import Namespace
    from pathlib import Path

    from common.portal_entry import load_training_config

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    conf = load_training_config(
        Namespace(config_file=str(root / "subjects/cooperative_search/cooperative_search_qmix.toml")),
        root,
    )
    mapped = dict(
        monte_carlo_method_config=conf.monte_carlo_method_config,
        agent_config=conf.agents_config[0],
        env_config=conf.env_config,
        sim_config=conf.sim_config,
    )
    assert mapped["env_config"].sense_name == "cooperative_search"
    assert mapped["agent_config"].name
