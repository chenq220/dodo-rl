from __future__ import annotations

try:
    # Normal project invocation with ``src`` on sys.path.
    from portal._path import bootstrap_sys_path
except ModuleNotFoundError:
    # Also support the command copied from IDEs configured with the repository
    # root as the source root: ``python -m src.portal.learn_control``.
    from ._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args, resolve_role_id
from config import ModelManagerConfig
from roles.learn_control import LearnControlRole

LOG = init_logger("learn_control")


def _parse_args():
    def add_role_id(parser):
        parser.add_argument(
            "--role-id",
            default=None,
            help="进程角色 ID，默认从 TOML 推导。",
        )

    return parse_config_args("分布式中心学习控制", extra_arguments=add_role_id)


if __name__ == "__main__":
    args = _parse_args()
    conf = load_training_config(args, PROJECT_ROOT)
    role_id = args.role_id or resolve_role_id(conf, "learn-control-0")
    artifacts_manager = initialize_role_runtime(
        conf, role="learn_control", is_master=True, role_id=role_id,
    )

    model_manager_config = conf.model_manager_config or ModelManagerConfig()
    LearnControlRole(
        role_id=role_id,
        train_config=conf.train_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        step_agents_config=conf.step_agents_config,
        learners_config=conf.learners_config,
        replay_buffer_config=conf.replay_buffer_config,
        model_manager_config=model_manager_config,
        sim_manager_config=conf.sim_manager_config,
        monte_carlo_method_config=conf.monte_carlo_method_config,
        task_manager_address=getattr(conf.rpc_config, "task_manager_address", ""),
        replay_buffer_address=getattr(conf.rpc_config, "replay_buffer_address", ""),
        model_manager_address=getattr(conf.rpc_config, "model_manager_address", ""),
        sim_manager_address=getattr(conf.rpc_config, "sim_manager_address", ""),
        monte_carlo_method_address=getattr(conf.rpc_config, "monte_carlo_method_address", ""),
        rpc_config=conf.rpc_config,
        artifacts_manager=artifacts_manager,
    ).run()
