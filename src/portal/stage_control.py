# 分阶段完整分布式拓扑的独立控制入口
from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args, resolve_role_id
from roles.stage_control import StageControlRole

LOG = init_logger("stage_control")


def _parse_args():
    def add_role_id(parser):
        parser.add_argument(
            "--role-id",
            default=None,
            help="进程角色 ID，默认从 TOML [entry].role_id 或 design.task_name 推导。",
        )

    return parse_config_args("分阶段分布式中心控制", extra_arguments=add_role_id)


if __name__ == "__main__":
    args = _parse_args()
    conf = load_training_config(args, PROJECT_ROOT)
    role_id = args.role_id or resolve_role_id(conf, "stage-control-0")
    artifacts_manager = initialize_role_runtime(
        conf, role="stage_control", is_master=True, role_id=role_id,
    )

    StageControlRole(
        role_id=role_id,
        stage_task_config=conf.stage_task_config,
        env_config=conf.env_config,
        replay_buffer_config=conf.replay_buffer_config,
        task_manager_address=getattr(conf.rpc_config, "task_manager_address", ""),
        replay_buffer_address=getattr(conf.rpc_config, "replay_buffer_address", ""),
        sim_manager_config=conf.sim_manager_config,
        sim_manager_address=getattr(conf.rpc_config, "sim_manager_address", ""),
        monte_carlo_method_config=conf.monte_carlo_method_config,
        monte_carlo_method_address=getattr(conf.rpc_config, "monte_carlo_method_address", ""),
        rpc_config=conf.rpc_config,
        artifacts_manager=artifacts_manager,
    ).run()
