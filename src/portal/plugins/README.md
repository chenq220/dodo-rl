# Demo Plugins 二次开发说明

`src/portal/plugins` 用于项目定制化流程改造；运行时导入名仍为 `portal.plugins`。如果某项目要求执行完成某一个流程后调用外部接口上报数据，或保存数据文件等，优先在这里写 module 插件，并在 TOML 中显式启用。项目默认不内置 demo module。

## 一、模块定制改造二次开发手册

### 1. 选择可扩展的基线类

只有被 `@extensible` 装饰的基线类可以通过插件机制替换。示例：

```python
from scheduler.rollout_sample import RolloutSampleWorker


class ProjectRolloutSampleWorker(RolloutSampleWorker):
    def rollout_one_episode(self, evaluate=False, random_sample=False, step_limit=None, **kwargs):
        return super().rollout_one_episode(
            evaluate=evaluate,
            random_sample=random_sample,
            step_limit=step_limit,
        )
```

定制子类必须继承对应基线类，并保持构造参数兼容。优先只重写确实需要改造的方法，公共流程能复用时调用 `super()`，避免复制整段框架流程导致后续框架升级难合并。

### 2. 在插件入口注册替换关系

在插件模块中定义定制子类，并通过 `[plugins_config.module]` 指向该类。映射规则：

- 基线类必须被 `@extensible` 标记。
- 定制类必须是基线类的子类。
- 同一个基线类只能被一个已启用插件覆盖。
- 只在项目插件层做项目差异，通用能力应回收到框架模块。

### 3. 项目全局注册表与 TOML 选择策略

项目内置插件统一定义在 `src/portal/plugins/registry.py` 的
`PROJECT_PLUGIN_REGISTRY` 数据类中，包含 `simulation`、`environment`、
`agent`、`learner`、`module` 五类；已有 Agent 与 Learner 算法也在此清单中。
Portal 默认以它作为候选清单，不需要每份
训练 TOML 重复声明这些位置。未写 `enabled`/`disabled` 时，Portal 会按科目引用的
env/agent/learner 收窄激活集；`module` 仅在全局清单或 TOML 中声明且被启用时生效。

训练 TOML 可选择全局插件，或追加新的插件：

```toml
[plugins_config]
enabled = ["module.project_extensions"]

[plugins_config.module]
project_extensions = "my_project_plugins.worker:ProjectRolloutWorker"
```

`enabled` 是白名单、`disabled` 是黑名单，二者互斥。TOML 的分类表可以追加新的 `类型.名称`。
Portal 入口会在实例化框架对象前完成合并和注册。

### 4. 定制开发建议

- `Role` 层适合改造任务生命周期、组件组装、最终报告和退出策略。
- `Rollout` 层适合改造采样/学习执行细节，例如 episode 记录、动作后处理、学习结果上报。
- `Scheduler` 层适合改造任务计划，例如采样序列、学习序列、动态学习参数。
- `ReplayBuffer` 层适合改造存储、采样策略、持久化和监控。
- 采样/评估结束由 `TrainTaskManager.is_sample_completed()` 或 `rollout_one_episode()` 返回 `None` 表达；Role 主循环据此退出或等待下一阶段。内部唯一 tick 是 `poll_stage_lifecycle()`。
- 项目退出行为通过覆写 `RoleExitMethod` 定制，默认行为是记录日志并正常返回。
- 覆盖 `TrainTaskManager` 时优先用公共钩子，不要改 `_` 方法：
  - `build_learn_permit_policy()`：替换学习许可策略；`decide` 只判断当前 ReplayBuffer 可执行哪些 Learner
  - `check_sample_admission(completed_episodes, generation)`：采样准入闸门，返回字符串则阻塞
  - `enqueue_bypass_sample_task(episode_info, *, kind)`：插入旁路采样令牌（本次仅支持 `replay_write=False`）
  - `custom_monitor()` / `custom_monitor_interval_seconds()`：独立线程周期改状态；未覆盖时不启动线程
  - `should_trigger_mid_eval()` / `custom_episode()`：盘中评估门控与令牌元数据

## 二、可定制改造模块表

| 基线模块 | 基线类 | 主要职责 | 常见定制方向 |
| --- | --- | --- | --- |
| `roles` | `RoleExitMethod` | Role 捕获任务完成信号后的统一退出入口。 | 改造成进程退出、资源清理、状态上报、工作流回调。 |
| `roles.train` | `TrainRole` | 训练角色入口，组装 Agent、Env、Learner、ReplayBuffer、TaskManager，并驱动训练主循环。 | 定制训练生命周期、组件构建、日志/指标、模型保存、Agent/Learner 同步方式。 |
| `roles.evaluation` | `EvaluationRole` | 评估角色入口，加载模型并运行评估 episode。 | 定制模型加载、评估报告、轨迹录制、评估后处理。 |
| `roles.sample_learning_evaluation` | `SampleLearningEvaluationRole` | 先用历史样本学习，再切到评估流程验证模型。 | 定制样本加载、离线学习评估流程、结果保存和曲线输出。 |
| `scheduler.rollout_sample` | `RolloutSampleWorker` | 按 episode 驱动 Agent 与 Environment 交互，并把 transition 写入 ReplayBuffer。 | 定制 episode 记录、动作处理、异常重试、状态落盘、采样指标。 |
| `scheduler.rollout_learn` | `RolloutLearnWorker` | 按 `LearnInfo` 从 ReplayBuffer 抽样并调用 Learner 更新。 | 定制学习执行、`algo_metadata` 应用、学习结果汇总、分布式学习适配。 |
| `scheduler.train_task` | `TrainTaskManager` | 管理在线训练的采样计划、按样本更新发放的学习许可、奖励和学习结果。 | 覆盖 `build_learn_permit_policy` / `check_sample_admission` / `enqueue_bypass_sample_task` / `custom_monitor`；不要覆盖 `_` 方法。 |
| `scheduler.sample_learn_task` | `SampleLearnManager` | 管理历史样本学习任务、样本加载仲裁和学习结果回收。 | 定制样本缓存加载、离线学习任务分发、分布式锁/状态同步。 |
| `sample.replay_buffer` | `ReplayBuffer` | 以 episode 为单位保存、覆盖、抽样轨迹数据。 | 定制采样算法、持久化后端、容量策略、监控统计。 |
