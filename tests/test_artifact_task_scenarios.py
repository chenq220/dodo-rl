"""真实任务归档：串行、同机共享目录、模拟跨机独立文件系统。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import msgpack
from PIL import Image
import pytest
import toml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).parent / "utils" / "artifact_task_worker.py"
EPISODES = 12
pytestmark = pytest.mark.slow


def _addresses():
    # 一次保留全部端口，防止本次配置自身分配到重复端口。
    sockets = [socket.socket() for _ in range(7)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [f"127.0.0.1:{sock.getsockname()[1]}" for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _config(runtime: Path, archive: Path, task: str, addresses: list[str]):
    return {
        "design": {"task_id": task, "task_name": "artifact_counter", "artifacts_dir": str(runtime)},
        "train_config": {
            "max_episode": EPISODES, "seed": 123, "record_sample": True,
            "model_save_dir": str(runtime / "models"), "model_load_dir": str(runtime / "models"),
            "result_save_dir": str(runtime / "results"), "render_video_dir": str(runtime / "recordings"),
            "artifacts_archive_save_dir": str(archive), "end_eval_episodes": 4,
        },
        "env_config": {
            "sense_name": "artifact_counter", "agent_names": ["agent_0"],
            "state_dim": 1, "max_step_per_episode": 4, "use_sim": False,
        },
        "agents": {"0": {
            "name": "counter", "agent_type": "sb3", "agent_names": ["agent_0"],
            "obs_dim": 1, "state_shape": 1, "hidden_dim": 16, "obs_last_action": False,
            "actions": {"0": {"action_key": "counter", "add_target": False,
                                "parameters": {"0": {"key": "delta", "values": [-1, 1]}}}},
        }},
        "learners": {"0": {
            "name": "counter", "device": "cpu", "algo_type": "ppo", "state_shape": 1,
            "algo_config": {"capacity": 64, "batch_size": 8, "n_steps": 8, "n_epochs": 1,
                            "minibatch_size": 4, "learn_times_per_episode": 1,
                            "sample_strategy": "on_policy_recent_steps"},
        }},
        "replay_buffer_config": {},
        "model_manager_config": {"tmp_local_dir": str(runtime / "model_pool")},
        "monte_carlo_method_config": {"tmp_local_dir": str(runtime / "samples")},
        "log_config": {"log_dir": str(runtime / "logs"), "enabled_handlers": ["file", "console"]},
        "torch_rpc_config": {"world_size": 3, "init_address": addresses[0],
                             "call_timeout_seconds": 10, "num_worker_threads": 8},
        "grpc_config": dict(zip(("task_manager_address", "replay_buffer_address", "model_manager_address",
                                 "sim_manager_address", "monte_carlo_method_address", "artifacts_manager_address"),
                                addresses[1:])) | {"call_timeout_seconds": 10},
        "plugins_config": {"enabled": ["environment.artifact_counter", "agent.sb3", "learner.ppo"],
                           "environment": {"artifact_counter": "artifact_task_worker:ArtifactCounterEnvironment"}},
    }


def _run_nodes(tmp_path: Path, deployment: str, rpc: str):
    task = f"archive-{deployment}-{rpc}"
    serial = deployment == "serial"
    isolated = deployment == "isolated"
    roles = ["serial_train"] if serial else ["center", "sample-0", "sample-1"]
    nodes = {role: tmp_path / role for role in roles}
    for role, node in nodes.items():
        node.mkdir()
        (node / "probe.txt").write_text(role)
    runtime_roots = {role: (nodes[role] if isolated else tmp_path / "shared") / task for role in roles}
    archive = tmp_path / "archives"
    addresses = _addresses()
    processes, streams = [], []
    try:
        for index, role in enumerate(roles):
            node = nodes[role]
            config = _config(runtime_roots[role], archive if role in {"center", "serial_train"}
                             else node / "unused-worker-archives", task, addresses)
            config_file = node / "task.toml"
            config_file.write_text(toml.dumps(config))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT_ROOT / "src"), str(WORKER.parent)]),
                       DODO_RPC=rpc, DODO_TASK_ID=task, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLCONFIGDIR=str(node / "mpl"), DODO_PROGRESS_FILE=str(node / "progress.env"))
            for key in ("DODORL_TEST_LOG_HANDLERS", "DODO_CENTER_FINISHED", "DODO_RPC_CONNECT_HOST", "DODO_RPC_LISTEN_HOST"):
                env.pop(key, None)
            env["RANK"] = str(index)
            interface = "lo0" if sys.platform == "darwin" else "lo"
            env.update(TP_SOCKET_IFNAME=interface, GLOO_SOCKET_IFNAME=interface)
            entry = "serial_train" if serial else "learn_control" if role == "center" else "sample"
            command = [sys.executable, str(WORKER), entry, str(config_file), "--role-id", role]
            if isolated:
                for other_role, other_node in nodes.items():
                    if other_role != role:
                        command += ["--deny-root", str(other_node)]
                if role != "center":
                    command += ["--deny-root", str(archive)]
            stream = (node / "console.log").open("w")
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=node, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 75
        while any(p.poll() is None for p in processes):
            failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
            logs = {role: (node / "console.log").read_text() for role, node in nodes.items()}
            foreign_access = any("simulated remote filesystem is not accessible" in log for log in logs.values())
            if failures or foreign_access or time.monotonic() > deadline:
                pytest.fail("task failed, accessed another node, or timed out\n" + "\n".join(
                    f"{role}:\n{log[-10000:]}" for role, log in logs.items()))
            time.sleep(0.1)
        assert [p.returncode for p in processes] == [0] * len(roles), "\n".join(
            f"{role}:\n{(node / 'console.log').read_text()[-10000:]}" for role, node in nodes.items())
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for stream in streams:
            stream.close()
    return task, nodes, runtime_roots, archive / task


@pytest.mark.parametrize(("deployment", "rpc"), [
    pytest.param("serial", "trpc", id="serial-task"),
    pytest.param("shared", "trpc", id="same-machine-trpc"),
    pytest.param("shared", "grpc", id="same-machine-grpc"),
    pytest.param("isolated", "trpc", id="different-machines-trpc"),
    pytest.param("isolated", "grpc", id="different-machines-grpc"),
])
def test_task_artifacts_archive_in_all_deployments(tmp_path, deployment, rpc):
    task, nodes, runtimes, archive = _run_nodes(tmp_path, deployment, rpc)
    manifest = json.loads((archive / "manifest.json").read_text())
    entries = manifest["artifacts"]
    assert manifest["task_id"] == task
    assert set(manifest["stages"]) == {"train"}
    assert {"model", "result", "sample", "render_gif", "log", "config"} <= {e["artifact_type"] for e in entries}
    assert len({e["relative_path"] for e in entries}) == len(entries)
    for entry in entries:
        data = (archive / entry["relative_path"]).read_bytes()
        assert len(data) == entry["size"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        if entry["artifact_type"] == "render_gif":
            with Image.open(archive / entry["relative_path"]) as image:
                assert image.format == "GIF" and image.n_frames > 1
    archived_files = {str(p.relative_to(archive)) for p in archive.rglob("*") if p.is_file()}
    assert archived_files == {e["relative_path"] for e in entries} | {"manifest.json"}
    configs = [e for e in entries if e["artifact_type"] == "config"]
    assert len(configs) == 1
    assert toml.loads((archive / configs[0]["relative_path"]).read_text())["train_config"]["max_episode"] == EPISODES
    assert any(Path(e["relative_path"]).name.startswith("final_") for e in entries if e["artifact_type"] == "model")
    samples = [e for e in entries if e["artifact_type"] == "sample"]
    for entry in samples:
        with (archive / entry["relative_path"]).open("rb") as stream:
            records = list(msgpack.Unpacker(stream, raw=False))
        assert len(records) == entry["metadata"]["record_count"] == (EPISODES + 4) * 5
        assert {record["episode"] for record in records} == set(range(1, EPISODES + 1))
    log_roles = {e["metadata"].get("role_id") for e in entries if e["artifact_type"] == "log"}
    expected_roles = {"serial_train"} if deployment == "serial" else {"center", "sample-0", "sample-1"}
    assert expected_roles <= log_roles
    assert len([e for e in entries if e["artifact_type"] == "render_gif"]) == 4
    proofs = [json.loads((node / "node-proof.json").read_text()) for node in nodes.values()]
    assert len({proof["pid"] for proof in proofs}) == len(nodes)
    if deployment == "isolated":
        assert all(proof["denied_roots"] for proof in proofs)
        original_gifs = []
        for role in ("sample-0", "sample-1"):
            assert not (nodes[role] / "unused-worker-archives").exists()
            originals = list((runtimes[role] / "logs").glob(f"{role}_*.log"))
            assert originals
            for entry in (e for e in entries if e["artifact_type"] == "log" and e["metadata"].get("role_id") == role):
                source = Path(entry["metadata"]["producer_path"])
                assert source.is_relative_to(runtimes[role])
                assert source.read_bytes().startswith((archive / entry["relative_path"]).read_bytes())
            for gif in (runtimes[role] / "recordings").glob("*.gif"):
                original_gifs.append(gif)
                match, = [e for e in entries if Path(e["relative_path"]).name == gif.name]
                assert gif.read_bytes() == (archive / match["relative_path"]).read_bytes()
        assert len(original_gifs) == 4
