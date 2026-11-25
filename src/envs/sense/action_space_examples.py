"""In-memory four-quadrant examples using the standard Parallel/AEC execution flow."""
from envs.action.spaces import ActionSpace, MultiHeadAction
from envs.sense.paralle_examples import CounterParallelEnvironment, _ParallelCounterSimulation
from envs.sense.aec_examples import CounterAECEnvironment, _CounterSimulation


def action_delta(action):
    if isinstance(action, MultiHeadAction):
        return sum(action_delta(value) for value in action.heads.values())
    if hasattr(action, "actions"):
        return sum(action_delta(value) for value in action.actions)
    return float(sum(action.parameters.values()))


class _ParallelSimulation(_ParallelCounterSimulation):
    _action_delta = staticmethod(action_delta)


class _AECSimulation(_CounterSimulation):
    def send_actions(self, actions):
        self.last_actor, action = next(iter(actions.items()))
        self.last_delta = action_delta(action)


class _ActionSpaceExample:
    def _bind_spaces(self, spaces):
        if set(spaces) != set(self.possible_agents):
            raise ValueError("one action space is required for every agent")
        if not all(isinstance(space, ActionSpace) for space in spaces.values()):
            raise ValueError("expected typed action spaces")
        self.action_space_models = dict(spaces)
        self.action_dim = {key: space.to_gym_space() for key, space in spaces.items()}

    def get_avail_agent_actions(self, agent_name, init=False):
        space = self.action_space_models[agent_name]
        if space.kind == "discrete":
            return [1] * space.layout["category_counts"][0]
        if space.kind == "multi_discrete":
            return {name: [1] * head.layout["category_counts"][0] for name, head in space.heads.items()}
        return None


class ActionSpaceCounterParallelEnvironment(_ActionSpaceExample, CounterParallelEnvironment):
    """Sum all physical parameters into a counter; receive full business objects."""
    def __init__(self, spaces, max_steps=3):
        super().__init__(list(spaces), max_steps=max_steps)
        self._bind_spaces(spaces)
        self.sim = _ParallelSimulation(list(spaces))


class ActionSpaceCounterAECEnvironment(_ActionSpaceExample, CounterAECEnvironment):
    def __init__(self, spaces, max_steps=3):
        super().__init__(list(spaces), max_steps=max_steps)
        self._bind_spaces(spaces)
        self.sim = _AECSimulation(list(spaces))
