"""Serial EvaluationRole and distributed EvalControlRole share the same report hooks."""
from __future__ import annotations

from common.logger import init_logger

LOG = init_logger("eval_report")


class EvalEpisodeReportMixin:
    """
    单局评估报告：入参对齐 ``report_sample_result`` 以及当时的令牌快照。

    串行 ``EvaluationRole`` 与分布式 ``EvalControlRole`` 都从本进程 TaskManager
    抽 ``get_eval_episode_records()``，再逐局调用 ``episode_report``。
    """

    episode_results: list[dict]

    def episode_report(
            self,
            episode_index: int,
            *,
            total_rewards: dict | None = None,
            episode_metrics: dict | None = None,
            episode_rewards_details: dict | None = None,
            artifacts: dict | None = None,
            episode_info: dict | None = None,
            **payload,
    ) -> dict:
        rewards = {
            str(name): float(value)
            for name, value in (total_rewards or {}).items()
        }
        metrics = dict(episode_metrics or {})
        steps = metrics.get("steps")
        record = {
            "episode": episode_index,
            "group_rewards": rewards,
            "total_rewards": rewards,
            "total_reward": float(sum(rewards.values())) if rewards else 0.0,
            "steps": int(steps) if steps is not None else 0,
            "episode_metrics": metrics,
            "episode_rewards_details": (
                dict(episode_rewards_details) if episode_rewards_details else None
            ),
            "artifacts": dict(artifacts) if artifacts else None,
            "episode_info": dict(episode_info or {}),
        }
        record.update(payload)
        LOG.info(
            f"Eval episode {episode_index} done: total_reward={record['total_reward']:.4f}, "
            f"steps={record['steps']}, "
            f"group_rewards={ {key: round(value, 4) for key, value in rewards.items()} }"
        )
        return record

    def finished_report(self):
        """全部评估局结束后的汇总钩子，项目定制化接口。"""
        pass

    def _collect_eval_episode_results(self) -> None:
        self.episode_results = [
            self.episode_report(
                int(record["episode"]),
                total_rewards=record.get("total_rewards") or {},
                episode_metrics=record.get("episode_metrics") or {},
                episode_rewards_details=record.get("episode_rewards_details"),
                artifacts=record.get("artifacts"),
                episode_info=record.get("episode_info") or {},
            )
            for record in self.task_manager.get_eval_episode_records()
        ]

    def _log_summary(self) -> None:
        if not self.episode_results:
            LOG.warning(f"{type(self).__name__} finished with no completed episodes.")
            return
        totals = [record["total_reward"] for record in self.episode_results]
        steps = [record["steps"] for record in self.episode_results]
        group_names = list(self.episode_results[0]["group_rewards"].keys())
        LOG.info("==== Evaluation summary ====")
        LOG.info(f"episodes={len(self.episode_results)}")
        LOG.info(
            f"total_reward: mean={_mean(totals):.4f}, min={min(totals):.4f}, max={max(totals):.4f}"
        )
        LOG.info(f"steps: mean={_mean(steps):.2f}, min={min(steps)}, max={max(steps)}")
        for name in group_names:
            group_values = [record["group_rewards"][name] for record in self.episode_results]
            LOG.info(
                f"  group {name}: mean={_mean(group_values):.4f}, "
                f"min={min(group_values):.4f}, max={max(group_values):.4f}"
            )


def _mean(values: list) -> float:
    return float(sum(values) / len(values)) if values else 0.0
