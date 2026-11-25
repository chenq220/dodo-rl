# 产物归档部署场景测试

运行全部场景：

```bash
.venv/bin/python -m pytest -q tests/test_artifact_task_scenarios.py
```

测试文件由 pytest 默认收集，并标记为 `slow`。使用框架计数环境运行真实 PPO、
SerialTrainRole / LearnControlRole / SampleRole 和 ArtifactsManager，不需要 Docker、
第二台机器、仿真服务或已安装 SDK wheel。
每个任务训练 12 局，每局 4 步，并执行 4 局结束评估，保持回归成本可控。

## 三类部署场景

| 场景 | 进程与存储 | 协议 |
| --- | --- | --- |
| 串行任务 | 一个独立进程完成采样、学习、文件生成和归档 | 本机调用 |
| 同机分布式 | 一个控制进程、两个采样进程，共享运行目录 | Torch RPC、gRPC |
| 模拟不同机器 | 一个控制进程、两个采样进程，各自独立工作目录和运行目录 | Torch RPC、gRPC |

共 5 个参数化用例。辅助环境、子进程入口及隔离工具位于
`tests/utils/artifact_task_worker.py`；测试入口位于 `tests/test_artifact_task_scenarios.py`。

第三类使用本机真实 RPC 通信，但各节点使用不同的绝对产物路径。
子进程安装 Python 文件访问审计钩子，禁止打开、枚举其他节点的文件；采样进程
还被禁止访问控制端归档目录。每个进程先实际尝试读取被禁止节点的探针文件，
确认拒绝访问后才启动任务。控制端若尝试通过采样端路径读取日志或 GIF，测试立即失败。
这种方式验证归档不依赖跨节点共享文件，不用于验证物理多机网络部署。

## 每个场景的验收条件

- 所有任务进程正常退出，退出码为 0；超时或失败时回收子进程并输出各节点日志。
- 自动产生一个含 `train` 阶段的完整归档，包含配置、模型、结果图、录制样本、GIF 和角色日志。
- final 模型存在，4 个评估 GIF 均可解码且包含多帧。
- MsgPack 样本可解码，记录数与 manifest 一致；12 局训练与 4 局评估合计 80 条 reset/step 记录。
- 串行日志包含 serial_train；分布式日志包含 center、sample-0、sample-1。
- 每项产物的大小、SHA-256 都与 manifest 一致，没有重复条目、漏记文件或混入其他文件。
- 独立目录场景中，两端采样日志快照与原文件前缀一致，4 个 GIF 与采样端原文件逐字节一致；采样端不产生第二份归档。

任务配置和所有输出位于 pytest 的临时目录。正常采样阶段日志归档为快照，
提交快照后的 RPC 关停诊断保存在各子进程的 `console.log` 中。

只运行模拟不同机器的两个用例：

```bash
.venv/bin/python -m pytest -q tests/test_artifact_task_scenarios.py -k different-machines
```
