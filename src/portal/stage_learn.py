# 分阶段分布式的学习和控制入口
from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args, resolve_role_id
from roles.stage_learn_control import StageLearnRole

LOG = init_logger("stage_learn")


def _parse_args():
    def add_role_id(parser):
        parser.add_argument(
            "--role-id",
            default=None,
            help="进程角色 ID，默认从 TOML [entry].role_id 或 design.task_name 推导。",
        )

    return parse_config_args("分布式分阶段中心学习", extra_arguments=add_role_id)


if __name__ == "__main__":
    args = _parse_args()
    conf = load_training_config(args, PROJECT_ROOT)
    role_id = args.role_id or resolve_role_id(conf, "learn-0")
    artifacts_manager = initialize_role_runtime(
        conf, role="stage_learn", is_master=True, role_id=role_id,
    )

    StageLearnRole(
        role_id=role_id,
        stage_task_config=conf.stage_task_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        replay_buffer_config=conf.replay_buffer_config,
        model_manager_config=conf.model_manager_config,
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
