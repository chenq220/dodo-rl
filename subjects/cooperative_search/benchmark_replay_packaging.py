"""样本封装与抽样模式微基准（v0.15.2 第 3.3 节配对正确性与微基准）。

固定同一份合成回放数据、同一有效组集合、成员顺序与初始权重，比较独立组
模式与同局多组封装模式的入池、抽样读取、张量组装与共享网络整次更新的耗时；
顺带验证相同有效组集合下两种封装铺平后的张量与更新后权重一致。结论须同时
报告容量三口径（物理局数/有效组轨迹数/有效步数）与字节数，不用相同的裸
capacity 数值暗示相同数据预算。

用法::

    python subjects/cooperative_search/benchmark_replay_packaging.py \
        [--episodes 16] [--steps 60] [--obs-dim 64] [--repeat 20] \
        [--tensor-pool memory|lazy_memmap] [--skip-update] [--out DIR]

在线端到端对照（多 seed、按有效组步/墙钟双预算）请使用配对科目
``cooperative_search_qmix_partitions.toml``（independent）与
``cooperative_search_qmix_partitions_bundled.toml``（episode_groups）在训练
集群执行；两者除抽样方式与容量/批单位换算外配置完全一致。
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch

from common.info_bridge import info_bridge
from config import EnvConfig, LearnerConfig, LocalStorageConfig, QmixConfig, ReplayBufferConfig
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import EpisodeTransition

AGENT_NAME = "uav_team"
GROUPS = tuple(
    tuple(f"drone_{group * 4 + slot}" for slot in range(4)) for group in range(4)
)


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _build_episodes(episode_num: int, steps: int, obs_dim: int, state_dim: int):
    """一个物理局 × 4 组 × K=4；组间数据可区分以便逐项比较。"""
    episodes = []
    for index, group in enumerate(GROUPS):
        transition = EpisodeTransition(agent_ids=group, group_index=index)
        transition.episode_num = episode_num
        for step in range(steps):
            obs = {
                agent_id: np.full(
                    obs_dim, float(step) + index * 0.25 + slot * 0.01, dtype=np.float32,
                )
                for slot, agent_id in enumerate(group)
            }
            transition.add_step(
                obs=obs,
                action={agent_id: _action((step + index + slot) % 12)
                        for slot, agent_id in enumerate(group)},
                last_action={agent_id: None for agent_id in group},
                reward={agent_id: 0.01 * (step + 1) for agent_id in group},
                next_obs={
                    agent_id: np.full(
                        obs_dim, float(step + 1) + index * 0.25 + slot * 0.01,
                        dtype=np.float32,
                    )
                    for slot, agent_id in enumerate(group)
                },
                termination={agent_id: False for agent_id in group},
                truncation={agent_id: step == steps - 1 for agent_id in group},
                state=np.full(state_dim, float(step) + index * 0.5, dtype=np.float32),
                next_state=np.full(
                    state_dim, float(step + 1) + index * 0.5, dtype=np.float32,
                ),
                avail_actions_next={agent_id: [1] * 12 for agent_id in group},
                avail_actions={agent_id: [1] * 12 for agent_id in group},
                metadata={"episode_num": episode_num, "step": step + 1,
                          "group_index": index},
                active_agent_ids=set(group),
                next_active_agent_ids=set(group) if step < steps - 1 else set(),
            )
        transition.set_hidden(
            {agent_id: torch.zeros(1, 4) for agent_id in group},
            {agent_id: torch.zeros(1, 4) for agent_id in group},
        )
        episodes.append(transition)
    return episodes


_ACTION_CACHE = {}


def _action(index: int):
    from envs.action.model import Action

    if index not in _ACTION_CACHE:
        _ACTION_CACHE[index] = Action(
            action_key="heading", parameters={"value": index},
            target=-2, action_idx=index,
        )
    return _ACTION_CACHE[index]


def _replay(path: Path, packaging: str, capacity: int, tensor_pool: str) -> ReplayBuffer:
    return ReplayBuffer(ReplayBufferConfig(
        agent_capacities={AGENT_NAME: capacity},
        tensor_pool=tensor_pool,
        storage=LocalStorageConfig(data_path=str(path)),
        batch_size=1,
        sample_packaging=packaging,
    ))


def _learner(obs_dim: int, state_dim: int, max_seq_len: int):
    from algos.modules.agents.rnn_agent import RNNAgent
    from algos.training.qmix_learner import QMixerLearner
    from test_agent_grouping import _build_rnn_agent

    config = copy.deepcopy(_build_rnn_agent(GROUPS, obs_dim=obs_dim).config)
    config.state_shape = state_dim
    agent = RNNAgent(
        agent_type="rnn", config=config, agent_net_source="blanco", device="cpu",
    )
    learner_config = LearnerConfig(
        name=AGENT_NAME, algo_type="qmix", state_shape=state_dim,
        algo_config=QmixConfig(),
        standardise_rewards=True, standardise_returns=False,
    )
    env = SimpleNamespace(
        config=EnvConfig(max_step_per_episode=max_seq_len),
        state_space=lambda: __import__("gymnasium").spaces.Box(
            -1.0, 1.0, shape=(state_dim,), dtype=np.float32,
        ),
    )
    return QMixerLearner(
        algo_type="qmix", config=learner_config, agent=agent, env=env, device="cpu",
    )


def _timed(repeat: int, fn):
    """先预热再计时；返回 (均值秒, 中位秒, 重复次数)。"""
    for _ in range(max(2, repeat // 4)):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.mean(samples)), float(np.median(samples)), len(samples)


def run_benchmark(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = 32
    bundles = [
        _build_episodes(episode_num, args.steps, args.obs_dim, state_dim)
        for episode_num in range(1, args.episodes + 1)
    ]

    workdir = Path(tempfile.mkdtemp(prefix="packaging_benchmark_"))
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "config": {
            "episodes": args.episodes,
            "groups_per_episode": len(GROUPS),
            "group_width": len(GROUPS[0]),
            "steps": args.steps,
            "obs_dim": args.obs_dim,
            "state_dim": state_dim,
            "tensor_pool": args.tensor_pool,
            "repeat": args.repeat,
            "seed": args.seed,
            "batch_group_trajectories": args.batch_groups,
        },
        "modes": {},
    }

    independent = _replay(
        workdir / "independent", "independent",
        capacity=args.episodes * len(GROUPS), tensor_pool=args.tensor_pool,
    )
    bundled = _replay(
        workdir / "bundled", "episode_groups",
        capacity=args.episodes, tensor_pool=args.tensor_pool,
    )
    try:
        # ---- 入池 ----
        push_timings = {}
        for name, replay in (("independent", independent), ("episode_groups", bundled)):
            mean, median, count = _timed(
                max(2, args.repeat // 4),
                lambda replay=replay: [
                    replay.push({AGENT_NAME: bundle}, notify_learning=False,
                                sample_role_name="benchmark")
                    for bundle in bundles
                ],
            )
            push_timings[name] = {
                "mean_secs": mean, "median_secs": median, "repeats": count,
                "episodes": args.episodes,
            }
        report["push"] = push_timings

        state_independent = independent.state()
        state_bundled = bundled.state()
        report["capacity_report"] = {
            "independent": {
                "sample_units": state_independent.agent_sizes[AGENT_NAME],
                "episodes": state_independent.agent_episode_counts[AGENT_NAME],
                "group_trajectories": state_independent.agent_trajectory_counts[AGENT_NAME],
                "steps": state_independent.agent_step_counts[AGENT_NAME],
                "tensor_bytes": independent.tensor_pool.memory_bytes,
            },
            "episode_groups": {
                "sample_units": state_bundled.agent_sizes[AGENT_NAME],
                "episodes": state_bundled.agent_episode_counts[AGENT_NAME],
                "group_trajectories": state_bundled.agent_trajectory_counts[AGENT_NAME],
                "steps": state_bundled.agent_step_counts[AGENT_NAME],
                "tensor_bytes": bundled.tensor_pool.memory_bytes,
            },
        }

        # ---- 抽样读取（随机选择 + 物化） ----
        independent_batch_groups = args.batch_groups
        bundled_batch_episodes = max(1, args.batch_groups // len(GROUPS))
        sample_timings = {}
        sampled_counts = {}
        for name, replay, batch_size in (
            ("independent", independent, independent_batch_groups),
            ("episode_groups", bundled, bundled_batch_episodes),
        ):
            def sample_once(replay=replay, batch_size=batch_size):
                return replay.sample(
                    AGENT_NAME, batch_size=batch_size,
                    strategy="recent_biased", seed=None,
                )

            mean, median, count = _timed(args.repeat, sample_once)
            sample_timings[name] = {
                "batch_units": batch_size,
                "mean_secs": mean, "median_secs": median, "repeats": count,
            }
            sampled = sample_once()
            sampled_counts[name] = len(sampled)
        report["sample"] = sample_timings
        report["sampled_group_trajectories"] = sampled_counts
        for name, count in sampled_counts.items():
            if count == 0:
                raise SystemExit(
                    f"sample returned no data for {name}: 需要至少 "
                    f"ceil(batch_groups / groups_per_episode)={int(np.ceil(
                        args.batch_groups / len(GROUPS)))} 个物理局，"
                    f"当前 --episodes={args.episodes}。"
                )

        # ---- 受控同集合物化 + 张量组装（分离选择差异） ----
        flat_entries = [
            meta
            for bundle in ReplayBuffer._split_bundles(bundled.buffer[AGENT_NAME])
            for meta in bundle
        ]
        assert [meta.write_seq for meta in flat_entries] == [
            meta.write_seq for meta in independent.buffer[AGENT_NAME]
        ]
        materialize_timings = {}
        for name, replay in (("independent", independent), ("episode_groups", bundled)):
            mean, median, count = _timed(
                args.repeat,
                lambda replay=replay: replay._materialize_episode_metas(flat_entries),
            )
            materialize_timings[name] = {
                "mean_secs": mean, "median_secs": median, "repeats": count,
                "group_trajectories": len(flat_entries),
            }
        report["materialize_same_set"] = materialize_timings

        to_batch_timings = {}
        batches = {}
        for name, replay, batch_size in (
            ("independent", independent, independent_batch_groups),
            ("episode_groups", bundled, bundled_batch_episodes),
        ):
            batch_data = replay._materialize_episode_metas(flat_entries)
            mean, median, count = _timed(
                args.repeat,
                lambda batch_data=batch_data: EpisodeTransition.to_batch(
                    list(GROUPS[0]), copy.deepcopy(batch_data),
                    expected_n_agents=len(GROUPS[0]), action_dim=12,
                ),
            )
            to_batch_timings[name] = {
                "mean_secs": mean, "median_secs": median, "repeats": count,
                "group_trajectories": len(batch_data),
            }
            batches[name] = batch_data
        report["to_batch"] = to_batch_timings

        # ---- 配对正确性：相同集合铺平张量一致 ----
        left_batch = EpisodeTransition.to_batch(
            list(GROUPS[0]), copy.deepcopy(batches["independent"]),
            expected_n_agents=4, action_dim=12,
        )
        right_batch = EpisodeTransition.to_batch(
            list(GROUPS[0]), copy.deepcopy(batches["episode_groups"]),
            expected_n_agents=4, action_dim=12,
        )
        for key in ("obs", "state", "actions", "actions_onehot", "avail_actions",
                    "reward", "terminated", "filled", "active_agents"):
            torch.testing.assert_close(
                left_batch[key], right_batch[key],
                msg=lambda m, key=key: f"paired tensor mismatch on {key}: {m}",
            )
        report["paired_tensor_equal"] = True

        # ---- 整次共享网络更新（QMIX 一次 optimizer.step） ----
        if not args.skip_update:
            left = _learner(args.obs_dim, state_dim, args.steps)
            right = _learner(args.obs_dim, state_dim, args.steps)
            right.restore_state(copy.deepcopy(left.snapshot_state()))
            update_timings = {}
            for name, learner, batch_data in (
                ("independent", left, batches["independent"]),
                ("episode_groups", right, batches["episode_groups"]),
            ):
                mean, median, count = _timed(
                    args.repeat,
                    lambda learner=learner, batch_data=batch_data: learner.update(
                        copy.deepcopy(batch_data)
                    ),
                )
                update_timings[name] = {
                    "mean_secs": mean, "median_secs": median, "repeats": count,
                }
            report["learner_update"] = update_timings
            for name, model in left.models.items():
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, right.models[name].state_dict()[key],
                        msg=lambda m, key=key, name=name: (
                            f"paired update mismatch on {name}.{key}: {m}"
                        ),
                    )
            torch.testing.assert_close(
                left.rew_ms.mean, right.rew_ms.mean,
            )
            report["paired_update_weights_equal"] = True

        valid_group_steps = args.batch_groups * args.steps
        report["throughput_group_steps_per_sec"] = {
            name: {
                "sample": valid_group_steps / timing["mean_secs"],
                **({
                    "learner_update": valid_group_steps / report["learner_update"][name]["mean_secs"],
                } if not args.skip_update else {}),
            }
            for name, timing in report["sample"].items()
        }
    finally:
        independent.close()
        bundled.close()

    out_dir = Path(args.out) if args.out else (
        ROOT / "artifacts/results/cooperative_search_partitions/packaging_benchmark"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{stamp}.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    markdown = _markdown(report)
    (out_dir / f"benchmark_{stamp}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nreport written to: {json_path}")
    return report


def _markdown(report: dict) -> str:
    lines = [
        "# 样本封装与抽样模式微基准报告",
        "",
        f"- 生成时间：{report['generated_at']}  代码版本：`{report['git_revision']}`",
        f"- 平台：{report['platform']}  torch：{report['torch_version']}",
        f"- 数据：{report['config']['episodes']} 物理局 × "
        f"{report['config']['groups_per_episode']} 组 × K="
        f"{report['config']['group_width']}，每局 {report['config']['steps']} 步，"
        f"obs_dim={report['config']['obs_dim']}，tensor_pool={report['config']['tensor_pool']}",
        f"- 批量：{report['config']['batch_group_trajectories']} 条组轨迹"
        f"（捆绑模式 = {report['config']['batch_group_trajectories'] // report['config']['groups_per_episode']}"
        " 个物理局），重复计时 "
        f"{report['config']['repeat']} 次取均值/中位",
        "",
        "## 容量口径（不用相同裸 capacity 暗示相同数据预算）",
        "",
        "| 模式 | 抽样单位 | 物理局数 | 组轨迹数 | 有效步数 | 张量字节 |",
        "|---|---|---|---|---|---|",
    ]
    for name, stats in report["capacity_report"].items():
        lines.append(
            f"| {name} | {stats['sample_units']} | {stats['episodes']} | "
            f"{stats['group_trajectories']} | {stats['steps']} | "
            f"{stats['tensor_bytes']:,} |"
        )
    lines += [
        "",
        "## 耗时（先预热；秒）",
        "",
        "| 阶段 | independent 均值 | independent 中位 | episode_groups 均值 | episode_groups 中位 |",
        "|---|---|---|---|---|",
    ]
    for stage, key in (
        ("入池（全部局）", "push"),
        ("抽样读取", "sample"),
        ("同集合物化", "materialize_same_set"),
        ("张量组装 to_batch", "to_batch"),
        ("整次更新 learner.update", "learner_update"),
    ):
        if key not in report:
            continue
        left = report[key].get("independent", {})
        right = report[key].get("episode_groups", {})
        if not left or not right:
            continue
        lines.append(
            f"| {stage} | {left['mean_secs']:.6f} | {left['median_secs']:.6f} | "
            f"{right['mean_secs']:.6f} | {right['median_secs']:.6f} |"
        )
    lines += [
        "",
        "## 配对正确性",
        "",
        f"- 相同有效组集合铺平后张量逐项一致：{report.get('paired_tensor_equal', False)}",
        f"- 相同初始权重下单次更新后权重一致："
        f"{report.get('paired_update_weights_equal', '(skipped)')}",
        "",
        "## 说明",
        "",
        "- 本基准回答“相同数据的封装开销”；不同抽样方式的在线效果差异须用配对",
        "  科目 TOML 做多 seed 在线训练对照，两者不能互相替代。",
        "- 在线对照科目：`cooperative_search_qmix_partitions.toml`（independent）与",
        "  `cooperative_search_qmix_partitions_bundled.toml`（episode_groups）。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=16, help="物理局数")
    parser.add_argument("--steps", type=int, default=60, help="每局环境步数")
    parser.add_argument("--obs-dim", type=int, default=64,
                        help="单机观测维度（真实科目为 615）")
    parser.add_argument("--batch-groups", type=int, default=64,
                        help="批量的组轨迹条数（捆绑模式按局换算）")
    parser.add_argument("--repeat", type=int, default=20, help="计时重复次数")
    parser.add_argument("--tensor-pool", choices=["memory", "lazy_memmap"],
                        default="memory")
    parser.add_argument("--skip-update", action="store_true",
                        help="跳过 QMIX 整次更新计时")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None, help="报告输出目录")
    args = parser.parse_args()

    with info_bridge.scope(toml_config={
        "env_config": SimpleNamespace(max_step_per_episode=args.steps),
        "stage_task_config": None,
    }):
        run_benchmark(args)


if __name__ == "__main__":
    main()
