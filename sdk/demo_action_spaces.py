"""四类统一业务动作定义示例。业务代码只声明 ActionDefined。"""
import json
import numpy as np
from dodorl_sdk import (
    ActionDefined, ActionParameterDefined, ActionSpace,
    action_space_from_definition,
)


def examples():
    definitions = [
        ActionDefined(
            action_key=[{
                "key": "move",
                "parameters": [ActionParameterDefined("direction", values=["left", "right"])],
            }],
            head_type="single",
            action_type="discrete",
        ),
        ActionDefined(head_type="multi", action_type="discrete", action_key=[
            {"key": "move", "parameters": [
                ActionParameterDefined("direction", values=["left", "right"]),
            ]},
            {"key": "mode", "parameters": [
                ActionParameterDefined("value", values=[0, 1, 2]),
            ]},
        ]),
        ActionDefined(
            action_key=[{
                "key": "aim_at",
                "parameters": [
                    ActionParameterDefined("x", min=-1200, max=1200, unit="m"),
                    ActionParameterDefined("y", min=-1200, max=1200, unit="m"),
                    ActionParameterDefined("z", min=0, max=300, unit="m"),
                ],
            }],
            head_type="single",
            action_type="continuous",
        ),
        ActionDefined(head_type="multi", action_type="continuous", action_key=[
            {"key": "velocity", "parameters": [
                    ActionParameterDefined("vx", min=-10, max=10, unit="m/s"),
                    ActionParameterDefined("vy", min=-10, max=10, unit="m/s"),
            ]},
            {"key": "aim_at", "parameters": [
                    ActionParameterDefined("x", min=-1200, max=1200, unit="m"),
                    ActionParameterDefined("y", min=-1200, max=1200, unit="m"),
                    ActionParameterDefined("z", min=0, max=300, unit="m"),
            ]},
        ]),
    ]
    return [action_space_from_definition(definition) for definition in definitions]


def main():
    rng = np.random.default_rng(7)
    for space in examples():
        action = space.sample(rng)
        record = json.loads(json.dumps(space.record(action)))
        restored = ActionSpace.restore(record, space)
        assert restored.to_dict() == action.to_dict()
        print(space.kind, "layout=", space.layout, "action=", action.to_dict())


if __name__ == "__main__":
    main()
