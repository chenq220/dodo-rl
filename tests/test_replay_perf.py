import importlib


def test_replay_perf_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("DODO_REPLAY_PERF", "")
    import sample.replay.replay_perf as replay_perf

    importlib.reload(replay_perf)
    assert replay_perf.now() == 0.0

    @replay_perf.timed("t")
    def add(value):
        return value + 1

    assert add(6) == 7
    assert replay_perf.snapshot() == {}


def test_replay_perf_enabled_records_and_snapshots(monkeypatch):
    monkeypatch.setenv("DODO_REPLAY_PERF", "1")
    import sample.replay.replay_perf as replay_perf

    importlib.reload(replay_perf)
    replay_perf.record("latency", 1.0)
    replay_perf.record("latency", 3.0)
    snap = replay_perf.snapshot()["latency"]
    assert snap["count"] == 2
    assert snap["total"] == 4.0
    importlib.reload(replay_perf)


def test_replay_perf_flush_ignores_closed_stream(monkeypatch):
    monkeypatch.setenv("DODO_REPLAY_PERF", "1")
    import sample.replay.replay_perf as replay_perf

    importlib.reload(replay_perf)

    def _raise_closed(*_args, **_kwargs):
        raise ValueError("I/O operation on closed file.")

    monkeypatch.setattr(replay_perf.LOG, "info", _raise_closed)
    replay_perf.record("latency", 1.0)
    replay_perf._flush()


def test_replay_perf_flush_ignores_closed_handler_stream(monkeypatch, capsys):
    monkeypatch.setenv("DODO_REPLAY_PERF", "1")
    import io
    import logging
    import sample.replay.replay_perf as replay_perf

    importlib.reload(replay_perf)

    closed = io.StringIO()
    closed.close()
    handler = logging.StreamHandler(closed)
    replay_perf.LOG.handlers = [handler]
    replay_perf.LOG.propagate = False
    replay_perf.record("latency", 1.0)
    replay_perf._flush()

    captured = capsys.readouterr()
    assert "Logging error" not in captured.err
    assert "Logging error" not in captured.out
