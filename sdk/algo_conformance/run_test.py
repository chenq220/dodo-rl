"""算法规范性发烟测试 CLI runner。

用法：
    python sdk/algo_conformance/run_test.py -f sdk/algo_conformance/qmix.toml
    python sdk/algo_conformance/run_test.py -f /path/to/my_algo.toml --only learner

选项：
    -f / --config   conformance TOML 配置文件路径（必填）。
    --only          只跑 learner 或 agent，默认两个都跑。

算法代码目录在 TOML 的 [conformance].algo_dir 中声明，无需命令行传入。

退出码：0=全部通过，1=有规范性检查失败。
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

# 确保项目根、src、sdk 在 sys.path（直接 python 运行时需要，且不经过 sdk/__init__.py）。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT / "src", _PROJECT_ROOT / "sdk", _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from algo_conformance.conformance import load_descriptor_from_toml
from algo_conformance.test_learner_conformance import run_learner_conformance
from algo_conformance.test_agent_conformance import run_agent_conformance

from common.exception import AlgoConformanceException
from common.plugins import activate_plugins
from portal.plugins.registry import project_plugin_registry
from utils.toml_config import TomlTrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="算法规范性发烟测试 runner",
    )
    parser.add_argument(
        "-f", "--config",
        required=True,
        help="conformance TOML 配置文件路径",
    )
    parser.add_argument(
        "--only",
        choices=["learner", "agent"],
        default=None,
        help="只跑 learner 或 agent；默认两个都跑",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    print(f"loading conformance config: {config_path}")

    # 先读 [conformance].algo_dir，加入 sys.path 后再加载描述符（加载时会 import 算法类）。
    with config_path.open("rb") as fp:
        raw = tomllib.load(fp)
    algo_dir = raw.get("conformance", {}).get("algo_dir", "")
    if algo_dir:
        algo_dir_resolved = Path(algo_dir).resolve()
        if str(algo_dir_resolved) not in sys.path:
            sys.path.insert(0, str(algo_dir_resolved))
        print(f"algo_dir added to sys.path: {algo_dir_resolved}")

    desc = load_descriptor_from_toml(config_path)
    print(f"algorithm: {desc.name}")
    print(f"  learner: {desc.learner_import_path}")
    print(f"  agent:   {desc.agent_import_path}")

    training_config = TomlTrainingConfig(config_path)
    activate_plugins(training_config.plugins_config, default_plugins=project_plugin_registry())

    targets = ["learner", "agent"] if args.only is None else [args.only]
    failures: list[str] = []

    for target in targets:
        print(f"\n{'=' * 60}")
        print(f"running {target} conformance smoke test: {desc.name}")
        print(f"{'=' * 60}")
        try:
            if target == "learner":
                run_learner_conformance(desc)
            else:
                run_agent_conformance(desc)
            print(f"PASS: {target} conformance smoke test passed: {desc.name}")
        except AlgoConformanceException as exc:
            failures.append(f"{target}: {exc.msg}")
            print(f"FAIL: {exc.msg}")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
