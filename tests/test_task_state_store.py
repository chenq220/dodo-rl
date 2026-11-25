from scheduler.tasks.task_state_store import TaskStateStore


class MemoryTaskStateStore:
    def __init__(self):
        self.events = []

    def record_episode_claimed(self, **kwargs):
        self.events.append(("claimed", kwargs))

    def record_episode_completed(self, **kwargs):
        self.events.append(("completed", kwargs))

    def record_episode_failed(self, **kwargs):
        self.events.append(("failed", kwargs))

    def record_episode_reclaimed(self, **kwargs):
        self.events.append(("reclaimed", kwargs))

    def record_learn_token_issued(self, **kwargs):
        self.events.append(("issued", kwargs))

    def record_learn_token_dispatched(self, **kwargs):
        self.events.append(("dispatched", kwargs))

    def record_learn_token_completed(self, **kwargs):
        self.events.append(("learn_completed", kwargs))

    def record_metric_sample(self, **kwargs):
        self.events.append(("metric", kwargs))

    def record_progress(self, **kwargs):
        self.events.append(("progress", kwargs))


def test_memory_store_satisfies_protocol_and_records():
    store: TaskStateStore = MemoryTaskStateStore()
    store.record_episode_claimed(
        task_id="t", stage_name="s", generation=0, episode_num=1, lease_id="L",
    )
    store.record_progress(task_id="t", stage_name="s", generation=0, sample_completed=1)
    assert store.events[0][0] == "claimed"
    assert store.events[0][1]["lease_id"] == "L"
    assert store.events[1][0] == "progress"
