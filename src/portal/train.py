# 该入口使用全量模块做模块测试用
from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from config import ModelManagerConfig, SampleLearningTasksConfig
from roles.train import TrainRole

LOG = init_logger("train")


if __name__ == "__main__":
    conf = load_training_config(parse_config_args("单进程采样学习训练"), PROJECT_ROOT)
    artifacts_manager = initialize_role_runtime(conf, role="train", is_master=None)

    model_manager_config = conf.model_manager_config or ModelManagerConfig()
    TrainRole(
        config=conf.train_config,
        step_agents_config=conf.step_agents_config,
        episode_agent_config=None,
        learners_config=conf.learners_config,
        model_manager_config=model_manager_config,
        sim_manager_config=conf.sim_manager_config,
        replay_buffer_config=conf.replay_buffer_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        sample_learning_config=SampleLearningTasksConfig(),
        monte_carlo_method_config=conf.monte_carlo_method_config,
        artifacts_manager=artifacts_manager,
    ).run()
