"""Run the 2x2 partitioned cooperative-search game with random legal actions.

四分区科目可视化 demo：600×600 大地图按 2×2 划分为四个 300×300 责任区
（灰色虚线分界），每区 4 架无人机出生在本区左上角，可跨区自由移动并扫描
（传感器观测跟随当前位置所在区），每局各区独立随机生成 10 个信号源
（共 40 个）。运行时打开 pygame 窗口，每 10 步在终端打印各分区发现进度；
某区找齐本区 10 个信号源后该区固定分组的无人机会提前收工。

用法::

    python src/envs/games/cooperative_search/partition_demo.py            # 100 步
    python src/envs/games/cooperative_search/partition_demo.py --steps 200 --fps 5
    python src/envs/games/cooperative_search/partition_demo.py --threats  # 附加 3 个动态威胁
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from envs.games.cooperative_search.partition_game import (
    CooperativeSearchPartitionGame,
    PartitionedSearchConfig,
)


def _region_summary(game: CooperativeSearchPartitionGame) -> str:
    parts = []
    for region in range(game.num_regions):
        found = game.region_identified_target_count(region)
        done = "✓" if game.region_targets_fully_identified(region) else " "
        parts.append(f"R{region}:{done}{found}/{game.targets_per_region}")
    observed = (game.positive_counts + game.negative_counts) > 0
    coverage = np.count_nonzero(observed) / observed.size
    return f"regions_completed={game.completed_region_count()}/{game.num_regions} | " + \
        " ".join(parts) + f" | coverage={coverage:.1%}"


def play_random_demo(seed: int = 7, steps: int = 100, fps: int = 20,
                     threats: bool = False) -> None:
    config = PartitionedSearchConfig(
        map_width=600,
        map_height=600,
        belief_grid_width=200,
        belief_grid_height=200,
        num_drones=16,
        num_dynamic_threats=8 if threats else 0,
        num_targets=40,
        region_rows=2,
        region_cols=2,
        max_steps=steps,
        initial_threat_separation=60.0,
        threat_warning_radius=45.0,
        fps=fps,
    )
    game = CooperativeSearchPartitionGame(config=config, render_mode="human")
    rng = np.random.default_rng(seed)
    game.reset(seed=seed)
    print(
        f"四分区协同搜索 demo：{game.num_regions} 个责任区 × "
        f"{game.drones_per_region} 机（跨区自由移动），每区 "
        f"{game.targets_per_region} 个信号源（共 {game.config.num_targets}），"
        f"动态威胁={'开' if threats else '关'}，seed={seed}"
    )
    for region in range(game.num_regions):
        members = ", ".join(game.region_agents(region))
        bounds = game.region_bounds(region)
        print(
            f"  区 {region} 范围 x[{bounds[0]:.0f},{bounds[1]:.0f}] "
            f"y[{bounds[2]:.0f},{bounds[3]:.0f}]：{members}"
        )
    game.render()
    try:
        step = 0
        while game.running and game.agents and step < steps:
            game.handle_events()
            actions = {}
            for agent in game.agents:
                legal = game.available_commands(agent)
                actions[agent] = legal[int(rng.integers(0, len(legal)))]
            _, _, terminations, truncations, _ = game.step(actions)
            game.render()
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"[step {step:3d}] {_region_summary(game)}")
            if all(truncations.values()):
                break
        print(f"[final  ] {_region_summary(game)} after {game.step_count} steps")
        total_found = sum(
            game.region_identified_target_count(region)
            for region in range(game.num_regions)
        )
        print(
            f"全局完成：{game.completed_region_count()}/{game.num_regions} 区，"
            f"共确认 {total_found}/{game.config.num_targets} 个信号源"
        )
    finally:
        game.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20, help="渲染帧率，越小越慢")
    parser.add_argument("--threats", action="store_true",
                        help="启用 8 个动态威胁（默认关闭；正式科目 evade 阶段启用）")
    args = parser.parse_args()
    play_random_demo(args.seed, args.steps, args.fps, args.threats)


if __name__ == "__main__":
    main()
