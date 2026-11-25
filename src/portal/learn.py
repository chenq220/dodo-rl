from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args, resolve_role_id
from roles.learn import LearnRole

LOG = init_logger("learn")


def _parse_args():
    def add_role_id(parser):
        parser.add_argument(
            "--role-id",
            default=None,
            help="进程角色 ID，默认从 TOML 推导。",
        )

    return parse_config_args("完整分布式拓扑学习进程", extra_arguments=add_role_id)


if __name__ == "__main__":
    args = _parse_args()
    conf = load_training_config(args, PROJECT_ROOT)
    role_id = args.role_id or resolve_role_id(conf, "learn-0")
    artifacts_manager = initialize_role_runtime(
        conf, role="learn", is_master=False, role_id=role_id,
    )

    LearnRole(
        role_id=role_id,
        train_config=conf.train_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        step_agents_config=conf.step_agents_config,
        learners_config=conf.learners_config,
        task_manager_address=getattr(conf.rpc_config, "task_manager_address", ""),
        replay_buffer_address=getattr(conf.rpc_config, "replay_buffer_address", ""),
        model_manager_address=getattr(conf.rpc_config, "model_manager_address", ""),
        rpc_config=conf.rpc_config,
        artifacts_manager=artifacts_manager,
    ).run()
