from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from roles.sample import SampleRole

LOG = init_logger("sample")


def _parse_args():
    def add_sample_role(parser):
        parser.add_argument("sample_role_name", help="采样进程名称，例如 sample-0。")

    return parse_config_args("分布式采样进程", extra_arguments=add_sample_role)


if __name__ == "__main__":
    args = _parse_args()
    conf = load_training_config(args, PROJECT_ROOT)
    initialize_role_runtime(
        conf, role="sample", is_master=False, role_id=args.sample_role_name,
    )

    SampleRole(
        role_id=args.sample_role_name,
        train_config=conf.train_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        step_agents_config=conf.step_agents_config,
        learners_config=conf.learners_config,
        task_manager_address=getattr(conf.rpc_config, "task_manager_address", ""),
        replay_buffer_address=getattr(conf.rpc_config, "replay_buffer_address", ""),
        model_manager_address=getattr(conf.rpc_config, "model_manager_address", ""),
        sim_manager_address=getattr(conf.rpc_config, "sim_manager_address", ""),
        monte_carlo_method_address=getattr(conf.rpc_config, "monte_carlo_method_address", ""),
        rpc_config=conf.rpc_config,
    ).run()
