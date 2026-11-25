def test_module_catalog_exports_core_types():
    import module_catalog

    names = {item.__name__ for item in module_catalog.__all__}
    assert "Learner" in names
    assert "TrainTaskManager" in names
    assert "ReplayBuffer" in names
    assert "SerialStageTrainRole" in names
