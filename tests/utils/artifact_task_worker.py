"""归档部署场景的子进程辅助工具；不需要 Docker 或外部仿真。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np

from envs.sense.paralle_examples import CounterParallelEnvironment


class ArtifactCounterEnvironment(CounterParallelEnvironment):
    """真实 PPO rollout 使用的可录制、可生成 GIF 的最小环境。"""

    def __init__(self, config, role_id=None, **kwargs):
        super().__init__(config.agent_names, max_steps=config.max_step_per_episode)
        self.role_id = role_id or "serial_train"

    def step(self, actions):
        # 给两个真实采样进程公平取得任务的机会，同时保持测试耗时很短。
        time.sleep(0.01)
        return super().step(actions)

    def render(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        frame[:, :, 0] = (self.step_num * 40) % 256
        frame[:, :, 1] = 80 if self.role_id == "sample-0" else 160
        return frame


def forbid_foreign_files(roots: list[Path]) -> None:
    """仅限制该子进程的文件访问；真实 RPC、算法和归档实现均照常运行。"""
    prefixes = tuple(os.path.abspath(root) + os.sep for root in roots)

    def audit(event, args):
        if event not in {"open", "os.listdir", "os.scandir"} or not args:
            return
        path = args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        absolute = os.path.abspath(os.fsdecode(path))
        if any(absolute == prefix[:-1] or absolute.startswith(prefix) for prefix in prefixes):
            raise PermissionError(f"simulated remote filesystem is not accessible: {absolute}")

    sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("serial_train", "learn_control", "sample"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--role-id", required=True)
    parser.add_argument("--deny-root", action="append", default=[], type=Path)
    args = parser.parse_args()
    forbid_foreign_files(args.deny_root)
    for root in args.deny_root:
        try:
            (root / "probe.txt").read_bytes()
        except PermissionError:
            continue
        raise AssertionError(f"foreign node file was visible: {root}")
    Path("node-proof.json").write_text(json.dumps({
        "pid": os.getpid(), "cwd": os.getcwd(), "role_id": args.role_id,
        "denied_roots": [str(root) for root in args.deny_root],
    }))
    sys.argv = [args.role, "-f", str(args.config)]
    if args.role == "learn_control":
        sys.argv += ["--role-id", args.role_id]
    elif args.role == "sample":
        sys.argv += [args.role_id]
    runpy.run_module(f"portal.{args.role}", run_name="__main__")


if __name__ == "__main__":
    main()
