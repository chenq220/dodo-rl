# DodoRL

DodoRL 是一个面向单智能体与多智能体训练的开放式强化学习框架，覆盖环境接入、策略训练、评估和模型管理，支撑多智能体联合训练与中等规模的分布式仿真并发学习。支持在独立业务工程中使用 SDK，也支持基于源码直接二次开发。

## 项目特色

- **多智能体联合训练**：通过多 Agent、多 Learner 编排组合学习策略与固定策略，支持智能体分组、共享模型，以及各 Learner 独立的配置和回放分区。
- **模块化扩展**：环境、Agent、Learner、采样和调度模块可通过插件扩展；Learner 可用 Batch Schema 定义自己的训练数据契约。
- **统一业务建模**：支持 Parallel / AEC 环境，以及单头、多头的离散和连续动作空间；各算法的适用范围见[动作空间说明](docs/manuals/action-spaces.md)。
- **完整训练流程**：通过 TOML 配置多阶段课程、在线或离线样本学习、历史样本复习、自动调参、模型保存和结果归档。
- **中等规模分布式仿真并发学习**：支持多个采样进程并发驱动仿真实例，与学习进程协同运行，提供 gRPC、Torch RPC 和 Docker 部署，也可使用单进程串行训练。
- **面向持续训练的可靠性**：通过任务租约与回收、模型版本管理、仿真异常恢复、训练快照和产物归档，为联合训练和仿真并发学习提供健壮可靠的代码支撑。

内置原生多智能体算法 **QMIX、COMA、MAPPO、MADDPG**，并通过 Stable-Baselines3 适配 **PPO、A2C、DQN**。

## 内置科目

仓库中的科目位于 [`subjects/`](subjects/)，包含训练配置及配套的评估或调试工具。

| 科目 | 内容 | 入口 |
| --- | --- | --- |
| 协同目标搜索 | 多架无人机共享概率图搜索隐藏目标，包含搜索与威胁规避课程 | [QMIX 配置](subjects/cooperative_search/cooperative_search_qmix.toml) |
| 分区协同搜索 | 按职责区组织多组智能体，提供独立组轨迹和同局多组封装两套配置 | [独立组](subjects/cooperative_search/cooperative_search_qmix_partitions.toml) · [同局多组](subjects/cooperative_search/cooperative_search_qmix_partitions_bundled.toml) |
| 威胁策略专训 | 训练供搜索科目使用的背景威胁策略，包含普通和分区场景 | [普通场景](subjects/cooperative_search/cooperative_search_threat_qmix.toml) · [分区场景](subjects/cooperative_search/cooperative_search_partition_threat_qmix.toml) |
| 奖励塑形验证 | 目标追踪、区域活动、威胁规避和目标选择四类小游戏 | [科目说明](subjects/shaping_test/README.md) |

这些科目使用本地模拟环境。接入外部仿真时，可配合可选的 `dodo-sim` SDK 使用。

## 两种使用形态

| 形态 | 适合的工作 | 使用方式 |
| --- | --- | --- |
| **SDK 集成** | 在独立业务工程中开发自己的科目和应用 | 安装 `dodorl-sdk`，通过公开接口、TOML 和插件接入环境、算法及流程，使用 `dodorl_sdk.portal` 启动训练或评估。见 [SDK 使用说明](sdk/README.md)。 |
| **源码二次开发** | 直接运行内置科目，或扩展框架与训练流程 | 基于本仓库开发，在 `subjects/` 组织科目，通过 `src/` 中的接口和插件扩展实现，使用 [`startup.sh`](startup.sh) 或 `portal` 入口运行。见[算法二次开发手册](docs/manuals/算法二次开发手册.md)。 |

项目使用 **Python 3.12**。源码依赖见 [`requirements.txt`](requirements.txt)，SDK 打包入口见 [`sdk/build_sdk.sh`](sdk/build_sdk.sh)。

## 运行方式

建议使用 AOCI 或其他代码理解/Agent 工具，基于仓库源码与 `subjects/` 配置生成 `Agent.md`，由它说明框架能力、内置科目内容，以及单机、Docker Compose、Torch RPC / gRPC 等组网运行方式；生成后以该文档作为运行与二次开发入口。

## 文档导航

- [SDK 使用说明](sdk/README.md)：独立工程接入、角色入口与插件开发。
- [算法二次开发手册](docs/manuals/算法二次开发手册.md)：Agent、Learner、Batch Schema 和注册方式。
- [动作空间](docs/manuals/action-spaces.md)：四类动作空间、执行契约与动作建模。
- [TOML 配置手册](docs/manuals/toml配置手册.md)：训练、评估、分布式及自动调参配置。
- [Docker 镜像](docs/manuals/docker镜像.md)：基础镜像、带代码镜像与 compose 构建方式。
