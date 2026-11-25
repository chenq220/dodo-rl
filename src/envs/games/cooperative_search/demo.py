"""Run the continuous cooperative-search game with random legal actions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from envs.games.cooperative_search import CooperativeSearchConfig, CooperativeSearchGame


def play_random_demo(seed: int = 7, max_steps: int = 500) -> None:
    config = CooperativeSearchConfig(max_steps=max_steps)
    game = CooperativeSearchGame(config=config, render_mode="human")
    rng = np.random.default_rng(seed)
    game.reset(seed=seed)
    game.render()
    try:
        while game.running and game.agents:
            game.handle_events()
            actions = {}
            for agent in game.agents:
                legal = game.available_commands(agent)
                actions[agent] = legal[int(rng.integers(0, len(legal)))]
            _, _, terminations, truncations, _ = game.step(actions)
            game.render()
            if all(terminations.values()) or all(truncations.values()):
                break
        coverage = np.count_nonzero(game.visited_counts) / game.visited_counts.size
        print(f"Continuous cooperative-search finished: steps={game.step_count}, "
              f"coverage={coverage:.1%}, team_reward={game.total_reward:.2f}")
    finally:
        game.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    play_random_demo(args.seed, args.max_steps)


if __name__ == "__main__":
    main()
