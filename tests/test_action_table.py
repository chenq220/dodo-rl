import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.action.base import ActionTable
from envs.action.model import (
    Action,
    ActionParameterDefined,
    CompositeAction,
    EpisodeActionDefined,
    StepActionDefined,
)

# fake episode动作：上层agent用于决策本局执行步数和后续agent执行顺序。
episode_action = EpisodeActionDefined(
    action_key="episode_plan",
    parameters=[
        ActionParameterDefined(key="plan", values=[0, 1]),
        ActionParameterDefined(key="priority", values=[10, 20]),
    ],
    step_num=3,
    agent_order_rules=["round_robin", "highest_priority"],
    role="commander",
)

# fake step动作：最终和环境交互的动作，支持目标选择。
step_action = StepActionDefined(
    action_key="step_attack",
    parameters=[
        ActionParameterDefined(key="power", values=[0, 1]),
        ActionParameterDefined(key="band", min=1, max=2, step=1),
    ],
    add_target=True,
    allow_null_targets=False,
    role="jammer",
)

action_table = ActionTable(
    action_config=[episode_action],
)

print('[episode_action]')
print(f"n_actions = {action_table.n_actions}")
for action_idx, action in action_table.action_map.items():
    print(action_idx, action.to_dict())


def test_allow_empty_action_prepends_a_single_empty_action():
    definitions = [
        StepActionDefined(
            action_key="move",
            parameters=[ActionParameterDefined(key="direction", values=["left", "right"])],
            add_target=False,
            allow_empty_action=True,
        ),
        StepActionDefined(
            action_key="fire",
            parameters=[],
            add_target=False,
            allow_empty_action=True,
        ),
    ]

    table = ActionTable(definitions)

    assert table.n_actions == 4
    assert table.action_map[0] == Action(
        action_key="",
        parameters={},
        target=-2,
        action_idx=0,
        is_empty_action=True,
    )
    assert [
        table.action_map[index].action_idx for index in range(table.n_actions)
    ] == [0, 1, 2, 3]
    assert all(
        not table.action_map[index].is_empty_action
        for index in range(1, table.n_actions)
    )


def test_empty_action_round_trips_through_dict():
    action = Action(
        action_key="",
        parameters={},
        target=-2,
        action_idx=0,
        is_empty_action=True,
    )

    assert Action.from_dict(action.to_dict()) == action
    assert Action.from_dict({
        "action_key": "move",
        "parameters": {},
        "target": -2,
        "action_idx": 1,
    }).is_empty_action is False


def test_allow_empty_action_precedes_combined_actions():
    definitions = [
        StepActionDefined(
            action_key="heading",
            parameters=[ActionParameterDefined(key="value", values=[0, 1])],
            add_target=False,
            allow_empty_action=True,
        ),
        StepActionDefined(
            action_key="speed",
            parameters=[ActionParameterDefined(key="value", values=[10, 20])],
            add_target=False,
        ),
    ]

    table = ActionTable(definitions, multi_action_mode="combination")

    assert table.n_actions == 5
    assert table.action_map[0].is_empty_action is True
    assert isinstance(table.action_map[1], CompositeAction)
    assert table.action_map[1].action_idx == 1
    assert all(action.action_idx == 1 for action in table.action_map[1].actions)


action_table = ActionTable(
    action_config=[step_action],
    targets=["radar_0", "radar_1"],
)
print('[step_action]')
print(f"n_actions = {action_table.n_actions}")
for action_idx, action in action_table.action_map.items():
    print(action_idx, action.to_dict())
