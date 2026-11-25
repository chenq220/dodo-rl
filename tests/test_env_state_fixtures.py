from envs.state.tests.cooperative_search_state import CooperativeSearchState
from envs.state.tests.cooperative_search_threat_state import CooperativeSearchThreatState
from envs.state.tests.region_activity_state import RegionActivityState
from envs.state.tests.target_selection_state import TargetSelectionState
from envs.state.tests.threat_region_state import ThreatRegionState
from envs.state.tests.track_seek_state import TrackSeekState


def test_env_state_fixtures_construct():
    assert TrackSeekState(entities=[]).observer_x == 0.0
    assert RegionActivityState(entities=[]).observer_x == 0.0
    assert ThreatRegionState(entities=[]).observer_x == 0.0
    assert TargetSelectionState(entities=[]).selected_target == -1
    assert CooperativeSearchState(entities=[]).frame_index == 0
    assert CooperativeSearchThreatState(entities=[]).frame_index == 0
