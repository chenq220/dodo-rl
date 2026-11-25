"""离散目标选择塑形函数、十球小游戏和训练环境测试。"""

from __future__ import annotations

import pygame
import pytest

from config import EnvConfig
from envs.action.model import Action
from envs.games.shaping_test_games.target_selection_game import TargetSelectionGame
from envs.reward.shaping import target_selection_shaping
from envs.reward.utils import ShapingOutputRange
from envs.sense.target_selection_env import TargetSelectionEnv


ALL_TARGETS = list(range(10))
PRIMARY_TARGETS = [8]
SHAPING_TARGETS = [1, 2, 5, 7]


@pytest.mark.parametrize(
    ("output_range", "hit", "miss", "neutral"),
    [
        (ShapingOutputRange.SIGNED, 1.0, -1.0, 0.0),
        (ShapingOutputRange.UNIT, 1.0, -1.0, 0.0),
        (ShapingOutputRange.DELTA_UNIT, 1.0, 0.0, 0.5),
    ],
)
def test_output_range_contract(output_range, hit, miss, neutral):
    kwargs = dict(
        shaping_target_indices=SHAPING_TARGETS,
        all_target_indices=ALL_TARGETS,
        output_range=output_range,
    )
    assert target_selection_shaping(1, primary_target_indices=[], **kwargs) == hit
    assert target_selection_shaping(3, primary_target_indices=[], **kwargs) == miss
    assert target_selection_shaping(
        8,
        primary_target_indices=PRIMARY_TARGETS,
        **kwargs,
    ) == neutral
    assert target_selection_shaping(
        1,
        primary_target_indices=PRIMARY_TARGETS,
        **kwargs,
    ) == neutral


def test_game_never_mode_has_exactly_four_shaping_targets_and_no_primary():
    game = TargetSelectionGame(primary_mode="never", seed=9)
    for _ in range(30):
        assert len(game.shaping_target_indices) == 4
        assert len(set(game.shaping_target_indices)) == 4
        assert game.primary_target_indices == ()
        selected = game.shaping_target_indices[0]
        assert game.step_action(selected) == 1.0
        assert game.last_target_kind == "shaping"


def test_game_random_mode_spawns_zero_or_one_disjoint_primary_target():
    game = TargetSelectionGame(
        primary_mode="random",
        primary_probability=0.5,
        seed=17,
    )
    observed_primary_counts: set[int] = set()
    for _ in range(100):
        observed_primary_counts.add(len(game.primary_target_indices))
        assert len(game.shaping_target_indices) == 4
        assert set(game.primary_target_indices).isdisjoint(game.shaping_target_indices)
        game.step_action(0)
    assert observed_primary_counts == {0, 1}


def test_primary_target_disables_shaping_but_keeps_business_reward():
    game = TargetSelectionGame(
        primary_mode="random",
        primary_probability=1.0,
        seed=5,
    )
    primary_target = game.primary_target_indices[0]
    assert game.step_action(primary_target) == 1.0
    assert game.last_primary_reward == 1.0
    assert game.last_shaping_reward == 0.0


def test_game_renders_blue_controller_line_and_ten_targets():
    game = TargetSelectionGame(render_mode="rgb_array", primary_mode="never", seed=3)
    selected = game.shaping_target_indices[0]
    game.step_action(selected)
    surface = game.draw(show_last_decision=True)
    assert surface.get_size() == (game.WIDTH, game.HEIGHT)
    controller_pixel = surface.get_at(game.CONTROLLER_POS)[:3]
    assert controller_pixel == game.CONTROLLER_COLOR
    for position in game.target_positions():
        color_sample = (position[0] + 10, position[1] + 10)
        assert surface.get_at(color_sample)[:3] in {
            game.SHAPING_COLOR,
            game.OTHER_COLOR,
        }
    selected_position = game.target_positions()[selected]
    selected_color_sample = (selected_position[0] + 10, selected_position[1] + 10)
    assert surface.get_at(selected_color_sample)[:3] == game.SHAPING_COLOR
    pygame.quit()


def test_training_env_observation_and_reward_gate():
    config = EnvConfig(
        sense_name="target_selection",
        use_sim=False,
        max_step_per_episode=5,
        state_dim=TargetSelectionEnv.OBS_DIM,
        agent_names=["selector-0"],
        metadata={
            "primary_mode": "never",
            "output_range": "signed",
            "reward_weights": {"primary": 1.0, "selection": 0.15},
        },
    )
    env = TargetSelectionEnv(config, connect_sim=False)
    observations, _ = env.reset(seed=11)
    observation = observations["selector-0"]
    assert observation.shape == (20,)
    assert observation[0::2].sum() == 4.0
    assert observation[1::2].sum() == 0.0

    assert env.game is not None
    selected = env.game.shaping_target_indices[0]
    action = Action(
        action_key="target_select",
        parameters={"select": 1},
        target=selected,
        action_idx=selected,
    )
    _, rewards, _, _, infos = env.step({"selector-0": action})
    assert rewards["selector-0"] == pytest.approx(0.15)
    assert infos["selector-0"]["reward_breakdown"] == {
        "primary": 0.0,
        "selection": pytest.approx(0.15),
    }
    env.close()
