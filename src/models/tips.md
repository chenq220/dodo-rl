# ModelManager 设计说明

## 目标

ModelManager 第一版只负责模型寄存和版本管理，不负责模型参数融合，也不负责模型评估。

学习进程并发提交的模型会被登记为不同版本。后续是否发布为主模型，由外部评估流程完成后显式调用 `promote_model` 决定。

## 生命周期

模型版本状态定义在 `models/model.py`：

- `candidate`：新提交版本，等待评估或人工处理。
- `champion`：当前发布版本，采样进程默认应拉取它。
- `specialist`：专项能力强但不适合作为全局主模型的版本。
- `archived`：历史归档版本，可回滚或继续作为训练分支 parent。
- `quarantined`：结构错误、加载失败或动作非法等硬错误版本。

## 核心接口

- `submit_model(...)`：提交一个候选模型版本，保存权重文件和元数据。
- `get_model(...)`：按模型名和版本读取元数据或 `state_dict`。
- `get_champion(...)`：读取当前发布版本。
- `promote_model(...)`：把某个版本晋升为 `champion`。
- `archive_model(...)`：归档版本。
- `quarantine_model(...)`：隔离硬错误版本。
- `mark_specialist(...)`：标记专项模型。
- `rollback_model(...)`：回滚到指定版本或当前 champion 的父版本。
- `list_versions(...)`：列出模型版本。

## 存储结构

本地运行期快照集中放在 `artifacts/models/.pool`；`artifacts/models` 根目录只保留
训练显式导出的 `final` / `optimo` 等可交付模型：

```text
artifacts/models/
  .pool/
    <task_name>/
      <model_name>__<task_id>__<version>__metadata.json
      <model_name>__<task_id>__<version>__agent_net.pth
      <model_name>__<task_id>__<version>__mixer_net.pth
      <model_name>__<task_id>__<version>__critic_net.pth
  final_<task>_<stage>_<generation>_<model_name>.pth
```

`<model_name>__<task_id>__<version>__metadata.json` 是单个版本的详细记录。运行期模型池默认每个
Agent 保留最近 8 个版本；champion、specialist、quarantined 及未完成自动晋升窗口的
candidate 总会保留。设置 `ModelManagerConfig.max_versions_per_model = -1` 可关闭该回收。

## 并发语义

ModelManager 内部使用本地锁保证单进程内提交、晋升和回滚的原子性。未来接 RPC 时，应把 ModelManager 放在独立控制进程中，由 RPC 服务端串行执行这些接口。

并发学习进程不共享写模型文件的职责，只通过 `submit_model` 提交到 ModelManager。

## 不做的事情

第一版不做以下事情：

- 不平均多个模型参数。
- 不自动判断模型好坏。
- 不自动跑仿真评估。
- 不清理新策略启用前遗留在旧目录中的历史模型；如需清理应单独执行审计后的迁移/删除任务。

这些能力后续应拆成 `ModelEvaluationWorker`、评估规则和模型池策略模块。
