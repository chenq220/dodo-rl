# Dodo SDK

0.17.0 内置原生 QMIX、COMA、MAPPO、MADDPG 和可定制的 Learner Batch schema。QMIX 支持单头离散动作；COMA 支持单头离散、单头连续和多头连续动作。MADDPG 支持 Parallel 环境下的单头和多头连续动作。原有 iql、vdn、ippo、ia2c、maa2c、pac、pac_dcg 不再作为内置算法提供，可继续使用 0.16.x 或注册自定义 Learner。旧 QMIX 同结构模型权重可加载，优化器和目标网络在新版本初始化；旧 COMA 需重新采样训练。

版本 0.16.0 的仿真、通信、资源池与仿真异常由独立的 **dodo-sim 0.3.0** 提供，SIM 是可选依赖，普通训练无需安装；只有场景工程使用仿真时才安装两个 SDK。
两边各自拥有当前代码一致的动作、态势基类，Simulation 支持注入任一 SDK 的 State 类。
环境无需转换 SIM 异常；RL 的 Rollout / TaskManager 自动完成失败识别、租约回收、探活和资源重申请。
ArkSim、SMAC 已废弃；7v3 科目已完整迁至 scenario 工程，环境、态势、训练配置和训练产物由 scenario 维护，RL SDK 不再内置该科目。


`dodorl_sdk` 是 DodoRL 面向业务工程发布的 Python SDK。业务工程可以通过它：

- 启动串行训练、分阶段训练、分布式学习控制、采样、串行评估和分布式评估角色；
- 实现并注册自己的 `Environment`，通过 dodo-sim 接入 `Simulation`；
- 使用 SDK 内置的 Agent/Learner，或开发强类型算法插件；
- 复用训练状态、动作、奖励、配置及训练数据契约；
- 通过训练环境驱动仿真。

## 1. 安装

安装本地 wheel：

```bash
python -m pip install --find-links=/path/to/wheels dodorl_sdk-*-py3-none-any.whl
```

检查版本和主要入口：

```bash
python -c "from dodorl_sdk.portal import SerialTrainSDK; from dodo_sim import ZMQConfig; import dodorl_sdk; print(dodorl_sdk.__version__)"
```

wheel 顶层只安装 `dodorl_sdk`。框架实现位于 `dodorl_sdk.dodorl` 内部，不会把 `sims`、`envs`、`common`、`config` 等名称安装成顶层包。

## 2. SDK 分层

### 2.1 运行入口

| 入口                    | 用途                  |
|-----------------------|---------------------|
| `SerialTrainSDK`      | 单进程、单阶段串行训练         |
| `SerialStageTrainSDK` | 单进程分阶段训练            |
| `ControlSDK`          | 完整分布式拓扑的独立控制角色      |
| `StageControlSDK`     | 完整分布式拓扑的独立分阶段任务控制角色 |
| `LearnSDK`            | 完整分布式拓扑的独立学习角色      |
| `ModelSDK`            | 完整分布式拓扑的独立模型服务      |
| `LearnControlSDK`     | 分布式训练中心学习控制角色       |
| `StageLearnSDK`       | 分布式分阶段学习控制角色        |
| `SampleSDK`           | 连接中心服务的采样角色；评估时传 `branch="eval"` |
| `EvaluationSDK`       | 单进程模型评估             |
| `EvalControlSDK`      | 分布式评估中心控制角色         |

### 2.2 业务建模接口

顶层可导入的主要接口包括：

```python
from dodorl_sdk import AECEnvironment, Action, ActionDefined, ActionParameterDefined, CGCS2000Frame, CoordinateSystem, Entity, EnvState, GeoPoint, LocalFrame, ParallelEnvironment, State, StateDescribe, EpisodeActionDefined, WGS84Frame
from dodo_sim import Simulation
```

仿真异常由独立的 dodo-sim 公开：

```python
from dodo_sim import SimulationException, SimulationCommandException, SimulationStateNotReceivedException, SimulationTimeoutException, ZMQMiddlewareException
```

### 2.3 高级扩展接口

- `dodorl_sdk.algorithms`：Agent/Learner 基类、算法配置基类和联合注册装饰器；
- `dodorl_sdk.modules`：可替换的训练框架模块基线，SimManager 由 dodo-sim 提供；
- `dodorl_sdk.contracts`：任务、样本、模型和 RPC Protocol；
- `dodorl_sdk.evaluation`：Evaluation 基类与评估扩展协议；
- `dodorl_sdk.config`：完整配置类型和枚举。

业务代码不要从 `dodorl_sdk.dodorl.*` 导入。该命名空间是 SDK 的内部实现，公共接口以 `dodorl_sdk` 及上述公共子模块为准。

## RL / SIM TOML 配置入口

训练 SDK 接受同时包含训练和仿真配置的一份 TOML。`TomlTrainingConfig` 统一处理 design、引用与表达式，将 `sim_config`、`sim_manager_config` 委托给 dodo-sim 装配；中间件、实例驱动和 metadata 的字段规则由 dodo-sim 维护。scenario 直接使用此入口，无须维护 TOML 分解器。

```python
from dodorl_sdk.portal import SerialTrainSDK

SerialTrainSDK(config_file="configs/scene.toml").run()
# 或 SerialTrainSDK(config_text=scene_toml_text).run()
```

dodo-sim 是可选依赖。自己实现 game / mock 环境时配置 `[env_config] use_sim = false`，安装 `dodorl-sdk` 即可，无须安装 dodo-sim。解析训练配置和注册插件时不会加载仿真 SDK；首次使用启用的仿真配置或创建仿真资源时才按需导入。客户端和管理器都关闭时，采样、学习、RPC 与全局关闭均不加载 dodo-sim。需要仿真时可安装 `dodorl-sdk[sim]`；缺少依赖会在首次使用处明确提示。

dodo-sim 继续提供独立入口，支持互斥的 TOML 文件路径和原始文本，不依赖训练 SDK：

```python
from dodo_sim import TomlSimulationConfig

simulation = TomlSimulationConfig(config_file="configs/simulation.toml")
# 或 simulation = TomlSimulationConfig(config_text=sim_toml_text)
simulation.activate_plugins()
```

需要分别维护训练与仿真配置时，可把已解析的 `simulation.sim_config` 和 `simulation.sim_manager_config` 通过角色 SDK 的同名参数显式注入。未传覆盖参数时保留统一 TOML 的解析结果。两个 SDK 的文本入口直接解析原文，使用显式工程根目录或工作目录解释运行路径。

分布式中心角色持有 dodo-sim 的 SimManager，采样进程通过现有 RPC 申请和归还仿真租约。采样任务失败先归还任务令牌，连续 reset 失败后再同步申请仿真重建；不需要额外的仿真服务。

## 3. 推荐的业务工程结构

```text
business_project/
├── plugins/
│   ├── simulation.py
│   ├── environment.py
│   ├── algorithms.py       # 可选
│   └── evaluation.py       # 可选
├── configs/
│   └── training.toml
├── artifacts/
├── sim_config.ini          # 使用 SimManager 时可选
└── main.py
```

入口运行前可以显式指定业务工程根目录：

```python
from pathlib import Path
from dodorl_sdk import configure_sdk
from dodorl_sdk.portal import SerialTrainSDK

project_root = Path(__file__).resolve().parent
configure_sdk(project_root=project_root)

SerialTrainSDK(
    plugin_dirs=[project_root / "plugins"],
    config_file=project_root / "configs" / "training.toml",
).run()
```

未调用 `configure_sdk()` 时，SDK 默认以进程当前工作目录作为业务工程根目录。也可以使用环境变量：

- `DODORL_SDK_PROJECT_ROOT`：业务工程根目录；
仿真资源路径由 dodo-sim 的 SimManagerConfig 管理，不通过 configure_sdk 设置。

归档目录按实际产物创建，仅含 manifest 和本次生成的模型、奖励图片、日志等类型目录。
`design.artifacts_dir` 若指向以当前 `task_id` 命名的任务独占目录，入口会登记该目录用于收尾清理。
全部已规划阶段归档成功后，进程关停时会核验归档文件的大小和 SHA-256，再删除整个任务运行目录，
包括空目录、日志锁和进程清单。归档失败、阶段未完成或归档位于运行目录内部时保留运行目录。
分阶段任务在初始化时登记从 `start_stage` 起本次要执行的全部阶段；单个阶段结束只复制归档，
保留源文件和目录供后续阶段使用。全部阶段完成并归档成功后才统一清理；中途退出保留原目录。
归档输出应放在运行目录之外，例如运行目录为 `artifacts/task-001`，归档根为 `artifacts/archives`。

## 4. Simulation 插件

外部项目继承公共 `Simulation`，直接实现公开的态势、推进和动作/通信接口。环境提交动作后调用
`sim.forward_step(step_interval)`，获得推进完成后的态势，再计算观测和奖励。

```python
from dodo_sim import Simulation

class ProjectSimulation(Simulation):
    def __init__(self, sim_type, config, sim_manager=None, role_id=None):
        super().__init__(sim_type, config, sim_manager, role_id)
        # 根据 self.config.middleware / self.config.metadata 建立连接

    def reset(self):
        # 完成初始化并返回可决策态势，按需调用 self._convert_state(raw_state)
        ...

    def forward_step(self, interval=1):
        # 真正推进 interval 个原生时间单位，完成后返回态势
        ...

    def get_state(self, time_point=None):
        # None 取最新，指定时间戳精确匹配；读取不推进
        ...

    def send_actions(self, actions, raw=False):
        # 转换、发送并确认动作已被原生服务接受
        ...

    def set_clock_rate(self, clock_rate):
        # 可选：支持原生时钟倍率时覆盖
        ...

    def close(self):
        # 必须实现：幂等释放连接、线程等持有资源
        ...
        super().close()
```

| 接口/属性 | 含义 |
|---|---|
| `reset()` | 子类直接实现，返回可决策初始态势 |
| `forward_step(interval=1)` | 子类直接实现，实际推进后返回态势 |
| `get_state(time_point=None)` | 子类直接实现；只读，指定时间戳精确匹配 |
| `state_convertor` | 默认 None；可注入具有 `from_json` 的态势类，是否使用由子类决定 |
| `_convert_state(raw_state)` | 可选辅助函数；无转换器原样返回，有转换器调用 `from_json`，失败直接抛出 |
| `send_actions(actions, raw=False)` | 动作转换、下发和协议完成确认由子类负责 |
| `set_clock_rate(clock_rate)` | 可选覆盖；未覆盖时调用抛 `NotImplementedError`，不影响子类实例化 |
| `config / sim_type / sim_manager / sim_id / has_sim_manager` | 当前配置、插件身份及资源占用；构造时基类初始化 |
| `close()` | 子类必须实现；幂等释放通信等资源并调用 `super().close()` 归还资源池占用 |
| `reacquire_simulation()` | manager 模式重新申请资源；子类随后重建连接，保留转换器 |

时间流子类内部恢复运行、等待首份时间 >= 当前位置 + interval 的态势、暂停后返回。
步进子类维护从 0 起的逻辑时间，`forward_step(2)` 必须实际完成两次原生 Step 并缓存各节点。
间隔单位由仿真定义。`get_state` 不改变推进基准；未匹配帧时有界等待并抛仿真异常。

基类不自动转换公开方法的返回值。子类可在取得原始态势后调用 `_convert_state`，也可直接返回原始数据或自行转换；仅赋值 `state_convertor` 不会自动触发转换。AFSIM、JSQL 与附带示例主动使用该辅助函数。

业务配置放 `[sim_config.metadata]`，通信配置放 `[sim_config.middleware]`。不再从环境编排
resume/pause；旧 `get_state(delay)` 的推进调用迁移为 `forward_step(interval)`。

详细 Agent 二次开发指南见 [Agent.md](Agent.md)，安装后也可读取：

```python
from importlib.resources import files
print(files("dodorl_sdk").joinpath("Agent.md").read_text(encoding="utf-8"))
```

AFSIM、JSQL 从 `dodo_sim` 导入；场景环境与态势在外部场景工程中实现和注册。新仿真插件继承 `dodo_sim.Simulation`。

## 5. Environment 插件

并行多智能体环境继承 `ParallelEnvironment`（同步联合转移），轮流行动环境继承 `AECEnvironment`。二者分别遵循 PettingZoo
Parallel 和 AEC 语义。二次开发不要直接继承内部的 `BaseEnvironment`。

`reset` / `step` 等运行时入口由基类提供默认实现；业务插件通常只需实现观测、奖励、结束判定等钩子。

### 5.1 Parallel（同步）环境

```python
from dodorl_sdk import ParallelEnvironment


class ProjectEnvironment(ParallelEnvironment):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        # 从 config.metadata 读取业务参数

    def done(self):
        # 返回 terminated, truncated, need_resume
        ...

    def state(self):
        # 返回全局状态向量
        ...

    def get_obs(self):
        # 返回 {agent_id: observation}
        ...

    def get_avail_agent_actions(self, agent_name, init=False):
        # 返回离散动作可用掩码
        ...

    def get_reward(self, **kwargs):
        # 返回 {agent_id: reward}
        ...
```

#### `ParallelEnvironment` 基类接口与属性

| 接口名称                                 | 接口功能                                          | 输入项目                                                                                | 返回项                                                | 是否有默认实现                    |
|--------------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------|----------------------------|
| `metadata`                           | PettingZoo 能力声明（名称、渲染模式等）                     | —                                                                                   | `dict`                                             | 是                          |
| `config`                             | 环境静态配置与业务 metadata                            | —                                                                                   | `EnvConfig`                                        | 是（`__init__` 赋值）           |
| `possible_agents`                    | 全量固定 AgentID 槽位                               | —                                                                                   | `list[AgentID]`                                    | 是（来自 `config.agent_names`） |
| `agents`                             | 当前需直接做动作的活动 AgentID                           | —                                                                                   | `list[AgentID]`                                    | 是（`__init__` / reset 维护）   |
| `stage_name`                         | 当前训练阶段名                                       | —                                                                                   | `str`                                              | 是                          |
| `episode_num` / `step_num`           | 当前局号与联合物理步号                                   | —                                                                                   | `int`                                              | 是                          |
| `max_step_per_episode`               | 当前 episode 实际步数上限                             | —                                                                                   | `int`                                              | 是                          |
| `action_dim` / `obs_dim`             | AgentID → 动作 / 观测空间                           | —                                                                                   | `dict[AgentID, Space]`                             | 是（空 dict，子类或组装阶段填充）        |
| `pre_state` / `post_state`           | 转移前 / 后业务态势                                   | —                                                                                   | `State \| None`                                    | 是                          |
| `rewards` / `current_actions`        | 最近一次转移的奖励与动作                                  | —                                                                                   | `dict`                                             | 是                          |
| `multi_agent_action_buffer`          | 本物理步已按序决策的动作上下文                               | —                                                                                   | `dict[AgentID, Action]`                            | 是                          |
| `sim` / `connect_sim`                | Simulation 实例及是否允许访问仿真                        | —                                                                                   | 实例 / `bool`                                        | 是                          |
| `__init__`                           | 初始化配置、Agent 列表，按需创建 Simulation                | `config`, `sim_config=None`, `connect_sim=True`, `sim_manager=None`, `role_id=None` | `None`                                             | 是                          |
| `reset`                              | 重置仿真与 Parallel episode，返回全量初始观测               | `seed=None`, `options=None`                                                         | `(observations, infos)`                            | 是                          |
| `step`                               | 一次下发全部活动 Agent 动作，产生一次联合转移                    | `actions: dict[AgentID, Action]`                                                    | `(obs, rewards, terminations, truncations, infos)` | 是                          |
| `done`                               | 判定环境终止、时限截断、是否需恢复仿真                           | 无                                                                                   | `(terminated, truncated, need_resume)`             | 否                          |
| `state`                              | 将 `post_state` 转为集中式训练全局状态向量                  | 无                                                                                   | `np.ndarray`                                       | 否                          |
| `get_obs`                            | 返回各 AgentID 观测                                | 无                                                                                   | `dict[AgentID, ObsType]`                           | 否                          |
| `get_avail_agent_actions`            | 返回指定 Agent 的离散动作可用掩码                          | `agent_name`, `init=False`                                                          | `list[int]`                                        | 否                          |
| `get_reward`                         | 根据 `pre_state` / `post_state` 计算即时奖励          | `**kwargs`                                                                          | `dict[AgentID, float]`                             | 否                          |
| `episode_agent_order_resolve`        | 解释 Episode Agent 动作，返回下一 Policy/Step Agent 组名 | `episode_agent_action`, `context`                                                   | `AgentName`                                        | 否（仅使用 Episode Agent 时需实现）  |
| `close`                              | 释放持有的 Simulation 连接                           | 无                                                                                   | `None`                                             | 是                          |
| `apply_env_metadata`                 | 在 episode reset 前合并任务级环境参数                    | `overrides: dict \| None`                                                           | `None`                                             | 是                          |
| `set_episode_step_limit`             | 在配置硬上限内设置本局步数上限                               | `step_limit: int \| None`                                                           | `int`                                              | 是                          |
| `action_space` / `observation_space` | 返回指定 Agent 的动作 / 观测空间                         | `agent`                                                                             | `gym.spaces.Space`                                 | 是                          |
| `state_space`                        | 返回全局状态空间                                      | 无                                                                                   | `gym.spaces.Box`                                   | 是                          |
| `render`                             | 渲染输出；默认无渲染                                    | 无                                                                                   | `None \| ndarray \| str \| list`                   | 是                          |
| `get_episode_rewards_details`        | 返回本局奖励构成；默认不支持                                | 无                                                                                   | `dict \| None`                                     | 是                          |
| `fake_reset` / `fake_step`           | 从录制数据恢复离线 reset / step                        | `payload: dict`                                                                     | reset/step 标准返回值                                   | 是                          |
| `send_actions` | 过滤活动实体后提交动作，并保存当前动作记录；推进由 step 调用 sim.forward_step 完成 | 动作映射 | 动作记录 | 是 |

### 5.2 AEC 环境

```python
from dodorl_sdk import AECEnvironment


class ProjectAECEnvironment(AECEnvironment):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        # 从 config.metadata 读取业务参数

    def done(self):
        ...

    def state(self):
        ...

    def get_obs(self):
        ...

    def get_avail_agent_actions(self, agent_name, init=False):
        ...

    def get_reward(self, **kwargs):
        ...
```

#### `AECEnvironment` 基类接口与属性

| 接口名称                                     | 接口功能                                         | 输入项目                                                                                | 返回项                                     | 是否有默认实现                    |
|------------------------------------------|----------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------|----------------------------|
| `metadata`                               | PettingZoo 能力声明；默认 `is_parallelizable=False` | —                                                                                   | `dict`                                  | 是                          |
| `config`                                 | 环境静态配置与业务 metadata                           | —                                                                                   | `EnvConfig`                             | 是（`__init__` 赋值）           |
| `possible_agents`                        | 全量固定 AgentID 槽位                              | —                                                                                   | `list[AgentID]`                         | 是（来自 `config.agent_names`） |
| `agents`                                 | 当前需直接做动作的活动 AgentID                          | —                                                                                   | `list[AgentID]`                         | 是                          |
| `agent_selection`                        | 当前原子步应执行动作的 AgentID                          | —                                                                                   | `AgentID`                               | 是                          |
| `terminations` / `truncations` / `infos` | 各 Agent 的终止、截断与附加信息                          | —                                                                                   | `dict`                                  | 是                          |
| `stage_name`                             | 当前训练阶段名                                      | —                                                                                   | `str`                                   | 是                          |
| `episode_num` / `step_num`               | 当前局号与 AEC 原子步号                               | —                                                                                   | `int`                                   | 是                          |
| `max_step_per_episode`                   | 当前 episode 允许的原子步上限                          | —                                                                                   | `int`                                   | 是                          |
| `action_dim` / `obs_dim`                 | AgentID → 动作 / 观测空间                          | —                                                                                   | `dict[AgentID, Space]`                  | 是                          |
| `pre_state` / `post_state`               | 原子转移前 / 后业务态势                                | —                                                                                   | `State \| None`                         | 是                          |
| `rewards` / `current_actions`            | 最近一次原子转移的奖励与动作                               | —                                                                                   | `dict`                                  | 是                          |
| `sim` / `connect_sim`                    | Simulation 实例及是否允许访问仿真                       | —                                                                                   | 实例 / `bool`                             | 是                          |
| `__init__`                               | 初始化配置、AEC 协议容器，按需创建 Simulation               | `config`, `sim_config=None`, `connect_sim=True`, `sim_manager=None`, `role_id=None` | `None`                                  | 是                          |
| `reset`                                  | 重置仿真与 AEC 协议状态；首个行动者为首个活动 Agent              | `seed=None`, `options=None`                                                         | `None`（经 `observe` / `last` 取观测）        | 是                          |
| `step`                                   | 对 `agent_selection` 执行一次原子动作并完成状态转移          | `action: Action \| None`                                                            | `None`                                  | 是                          |
| `observe`                                | 返回指定 Agent 的当前观测                             | `agent`                                                                             | `ObsType`                               | 是                          |
| `done`                                   | 判定环境终止、时限截断、是否需恢复仿真                          | 无                                                                                   | `(terminated, truncated, need_resume)`  | 否                          |
| `state`                                  | 将 `post_state` 转为集中式训练全局状态向量                 | 无                                                                                   | `np.ndarray`                            | 否                          |
| `get_obs`                                | 返回固定 AgentID 槽位对应的当前观测                       | 无                                                                                   | `dict[AgentID, ObsType]`                | 否                          |
| `get_avail_agent_actions`                | 返回指定 Agent 的离散动作可用掩码                         | `agent_name`, `init=False`                                                          | `list[int]`                             | 否                          |
| `get_reward`                             | 根据 `pre_state` / `post_state` 计算即时奖励         | `**kwargs`                                                                          | `dict[AgentID, float]`                  | 否                          |
| `episode_agent_order_resolve`            | 解释 Episode Agent 动作，返回下一环境实体 AgentID         | `episode_agent_action`, `context`                                                   | `AgentID`                               | 否（仅使用 Episode Agent 时需实现）  |
| `close`                                  | 释放持有的 Simulation 连接                          | 无                                                                                   | `None`                                  | 是                          |
| `apply_env_metadata`                     | 在 episode reset 前合并任务级环境参数                   | `overrides: dict \| None`                                                           | `None`                                  | 是                          |
| `set_episode_step_limit`                 | 在配置硬上限内设置本局步数上限                              | `step_limit: int \| None`                                                           | `int`                                   | 是                          |
| `action_space` / `observation_space`     | 返回指定 Agent 的动作 / 观测空间                        | `agent`                                                                             | `gym.spaces.Space`                      | 是                          |
| `state_space`                            | 返回全局状态空间                                     | 无                                                                                   | `gym.spaces.Box`                        | 是                          |
| `render`                                 | 渲染输出；默认无渲染                                   | 无                                                                                   | `None \| ndarray \| str \| list`        | 是                          |
| `get_episode_rewards_details`            | 返回本局奖励构成；默认不支持                               | 无                                                                                   | `dict \| None`                          | 是                          |
| `fake_reset` / `fake_step`               | 从录制数据恢复离线起点 / 转移                             | `payload: dict`                                                                     | reset 风格返回值 / Parallel step 五元组         | 是                          |
| `send_actions` | 过滤活动实体后提交动作，并保存当前动作记录；推进由 step 调用 sim.forward_step 完成 | 动作映射 | 动作记录 | 是 |

环境构造配置通过 `EnvConfig` 提供。业务自由配置放入 `[env_config.metadata]`。metadata 支持嵌套 TOML：

```toml
[env_config.metadata]
scenario = "demo"

[env_config.metadata.sensor]
channels = [1, 2, 3]
```

标准配置表中的未知字段会直接报错。SDK 不会自动将 metadata 转换成业务自定义的 `EnvConfig` 子类，插件应自行读取并转换。

## 6. 插件注册与默认算法

业务 TOML 只需要声明业务自己的 Simulation/Environment。SDK 默认自动启用以下算法插件。

内置 Agent：

- `rnn`
- `rnn_ns`
- `policy_rnn`
- `mappo_rnn`
- `q_rnn`
- `maddpg_mlp`
- `monte_carlo`
- `sb3`

内置 Learner：

- `qmix`
- `mappo`
- `coma`
- `maddpg`
- `ppo`
- `a2c`
- `dqn`

训练 TOML 中，同名 Learner 会为 Agent 选择默认类型。QMIX 默认使用 `rnn`，MAPPO 和 COMA 使用 `policy_rnn`，MADDPG 使用 `maddpg_mlp`，SB3 算法使用 `sb3`。显式填写 `agent_type` 可选择兼容的替代实现。独立推理时需填写 Agent 类型。

MADDPG 默认使用 `sample_strategy = "uniform_steps"`，从回放池的单步转移中均匀、有放回地抽样；`batch_size` 表示抽样次数。若业务 TOML 显式配置了其他策略，仍以该配置为准。

可在 `[learners.N.algo_config]` 中设置 `min_replay_size`，让 active 回放池达到指定数量后再发放学习令牌。它与 `capacity` 使用相同单位：独立模式按组轨迹计数，捆绑模式按物理局计数；默认 `None` 不增加等待条件。

各内置 Agent 共用 `get_actions(obs, agent_list, hidden_in)`，返回按成员索引的业务动作和下一步隐藏状态。首步可通过 `agent.get_init_hidden_state()` 初始化隐藏状态，后续步骤传入上一步返回的隐藏状态。

插件配置示例：

```toml
[plugins_config]
enabled = [
    "simulation.project_sim",
    "environment.project_env",
]

[plugins_config.simulation]
project_sim = "simulation:ProjectSimulation"

[plugins_config.environment]
project_env = "environment:ProjectEnvironment"

[env_config]
sense_name = "project_env"
use_sim = true
agent_names = ["agent_0"]
state_dim = 64

[sim_config]
sim_type = "project_sim"
config_source = "local"

[agents.0]
name = "policy"
agent_names = ["agent_0"]
obs_dim = 32

[learners.0]
name = "policy"
algo_type = "qmix"
```

SDK 自动注册 RNN/QMIX。训练 Agent 也可省略 `agent_type`，由同名 QMIX Learner 选用 `rnn`。

插件启停规则：

- SDK 默认 Agent/Learner 自动启用；
- `enabled` 只需要列出业务显式插件；
- `enabled` 与 `disabled` 不能同时使用；
- `disabled = ["agent.rnn"]` 可以禁用默认插件；
- 业务同名插件可以覆盖默认实现；
- 旧 TOML 中显式声明 SDK 内置算法路径仍然兼容。

## 7. 训练入口

角色 SDK 的规范入口是 `dodorl_sdk.portal`。SDK 顶层导入和
`dodorl_sdk.training` 在 0.14 仍保持兼容，但新项目应使用 `portal`。

源工程的 `portal.train` / `TrainRole` 仅用于工程内部调试，不包含在 SDK 安装包中。
业务单进程训练使用 `SerialTrainSDK`，分阶段训练使用 `SerialStageTrainSDK`。
公开角色入口以 `sdk/src/dodorl_sdk/portal.py` 的 `__all__` 为准，构建脚本通过
`SDK_ROLE_MODULES` 白名单复制对应角色实现，不打包源工程的 `src/portal` 命令行入口目录。
新增 SDK 角色时须显式补充公开接口、导出清单和打包白名单，并验证独立安装后的入口行为。

### 7.1 串行训练

```python
from dodorl_sdk.portal import SerialTrainSDK

SerialTrainSDK(
    plugin_dirs=["plugins"],
    config_file="configs/training.toml",
).run()
```

### 7.2 串行分阶段训练

```python
from dodorl_sdk.portal import SerialStageTrainSDK

SerialStageTrainSDK(
    plugin_dirs=["plugins"],
    config_file="configs/stage_training.toml",
).run()
```

### 7.3 分布式学习控制

```python
from dodorl_sdk.portal import LearnControlSDK, StageLearnSDK

LearnControlSDK(
    plugin_dirs=["plugins"],
    config_file="configs/training.toml",
    role_id="learn-control-0",
).run()

# 分阶段中心角色使用 StageLearnSDK，参数形式相同。
```

完整分布式拓扑可以把控制、学习和模型服务拆成三个独立进程：

```python
from dodorl_sdk.portal import ControlSDK, LearnSDK, ModelSDK

# 进程 1：任务、回放、仿真资源与中心样本录制服务。
ControlSDK(
    plugin_dirs=["plugins"],
    config_file="configs/training.toml",
).run()

# 进程 2：Learner；可通过 role_id 区分多个学习进程。
LearnSDK(
    plugin_dirs=["plugins"],
    config_file="configs/training.toml",
    role_id="learn-0",
).run()

# 进程 3：模型版本与权重发布服务。
ModelSDK(
    plugin_dirs=["plugins"],
    config_file="configs/training.toml",
).run()
```

分阶段完整拓扑使用独立的 `StageControlSDK`，不能用单阶段的 `ControlSDK` 代替：

```python
from dodorl_sdk.portal import LearnSDK, ModelSDK, StageControlSDK

# 进程 1：按 [stage_task_config] / [stages.*] 依次调度阶段。
StageControlSDK(
    plugin_dirs=["plugins"],
    config_file="configs/stage_training.toml",
    role_id="stage-control-0",
).run()

# 进程 2、3 分别运行 LearnSDK 与 ModelSDK；采样进程仍运行 SampleSDK。
```

`StageControlSDK` 在阶段切换时关闭任务令牌入口、按配置清理回放池，并等待当前阶段
采样与学习任务都排空后再替换 `TrainTaskManager`，防止迟到的学习结果写入下一阶段。

这些独立入口支持 `[grpc_config]` 和 `[torch_rpc_config]`。Torch RPC 的 `RANK`
由对应启动脚本分配并在协议 runtime 中统一读取，Role 不直接读取环境变量。

### 7.4 分布式采样

```python
from dodorl_sdk.portal import SampleSDK

SampleSDK(
    plugin_dirs=["plugins"],
    config_file="configs/training.toml",
    sample_role_name="sample-0",
).run()
```

分布式角色从 `[grpc_config]` 读取服务地址：

```toml
[grpc_config]
task_manager_address = "127.0.0.1:53151"
replay_buffer_address = "127.0.0.1:53152"
model_manager_address = "127.0.0.1:53153"
sim_manager_address = "127.0.0.1:53154"
monte_carlo_method_address = "127.0.0.1:53155"
```

## 8. ZMQ 中间件

SDK 公开通用 `ZMQClient`、`ZMQConfig` 和 `ZMQSocketType`：

```python
from dodo_sim import ZMQClient, ZMQConfig, ZMQSocketType

config = ZMQConfig(
    remote_ip="127.0.0.1",
    remote_port=9094,
    remote_mode="connect",
    send_socket_type=ZMQSocketType.REQ,
    local_ip="127.0.0.1",
    local_port=8001,
    local_mode="bind",
    receive_socket_type=ZMQSocketType.PULL,
)

client = ZMQClient(config)
try:
    reply = client.send({"command": "reset"})
finally:
    client.close()
```

TOML 配置：

```toml
[sim_config.middleware]
remote_ip = "127.0.0.1"
remote_port = 9094
remote_mode = "connect"
send_socket_type = "REQ"

local_ip = "127.0.0.1"
local_port = 8001
local_mode = "bind"
receive_socket_type = "PULL"
```

支持的类型包括 `REQ`、`REP`、`PUB`、`SUB`、`PAIR`、`DEALER`、`ROUTER`、`PULL`、`PUSH`、`XPUB`、`XSUB`、`STREAM` 等 PyZMQ
当前运行时提供的类型。

默认行为：

- `send_socket_type=REQ`：`send()` 默认等待一条回复；
- 其他发送类型：`send()` 默认只发送，可通过 `wait_reply=True` 修改；
- `receive_socket_type=PULL`：后台接收并写入一级缓存；
- `SUB` 使用前通过 `subscribe()` 设置主题；
- `recv_remote()` 用于主动读取远端 socket。

## 9. 自定义 Agent/Learner

自定义算法必须把实现类与强类型配置类联合声明：

```python
from dataclasses import dataclass
from typing import ClassVar
from dodorl_sdk.algorithms import (
    Agent,
    AgentAlgoConfig,
    Learner,
    LearnerAlgoConfig,
    agent_plugin,
    learner_plugin,
)


@dataclass
class ProjectAgentConfig(AgentAlgoConfig):
    feature_width: int = 128


@agent_plugin(ProjectAgentConfig, default_obs_agent_id=True)
class ProjectAgent(Agent):
    ...


@dataclass
class ProjectLearnerConfig(LearnerAlgoConfig):
    default_agent_type: ClassVar[str] = "project"
    business_loss_scale: float = 1.0


@learner_plugin(ProjectLearnerConfig)
class ProjectLearner(Learner):
    ...
```

Learner 配置类可用 `default_agent_type` 声明配对 Agent 的插件标识。下例根据同名 Learner 选择 `project`，并按各插件声明构造具体的 `algo_config`。

```toml
[plugins_config]
enabled = ["agent.project", "learner.project"]

[plugins_config.agent]
project = "algorithms:ProjectAgent"

[plugins_config.learner]
project = "algorithms:ProjectLearner"

[agents.0]
name = "policy"

[agents.0.algo_config]
feature_width = 256

[learners.0]
name = "policy"
algo_type = "project"

[learners.0.algo_config]
business_loss_scale = 0.5
```

缺少联合配置声明、配置类继承错误或出现未知算法字段时，SDK 会在配置加载阶段报错。

## 10. Evaluation 扩展

### 10.1 串行评估

默认评估入口：

```python
from dodorl_sdk.portal import EvaluationSDK

EvaluationSDK(
    plugin_dirs=["plugins"],
    config_file="configs/evaluation.toml",
).run()
```

单进程评估支持 `sim_config.config_source="local"` 直连，也支持 `"manager"`：评估角色使用本进程 SimManager，按 `sim_manager_config` 读取资源文件并向环境传入资源池，结束时先关闭环境、归还仿真，再停止资源池。manager 模式需要启用 `SimManagerConfig.use_sim`。自定义评估构造器应保留基类的 `sim_manager_config` 参数。

自定义评估应继承公共 `Evaluation`，通常只覆盖单局统计和结束报告：

```python
from dodorl_sdk import EvalConfig, Evaluation


class ProjectEvaluation(Evaluation):
    def episode_report(self, episode_index: int, *, total_rewards=None, **payload) -> dict:
        report = super().episode_report(
            episode_index,
            total_rewards=total_rewards,
            **payload,
        )
        report["project_metric"] = report["total_reward"]
        return report

    def finished_report(self):
        return {"episode_count": len(self.episode_results)}


def build_eval_config(training_config) -> EvalConfig:
    config = training_config.eval_config
    config.deterministic = True
    return config
```

在 TOML 中注册：

```toml
[entry]
evaluation_class = "evaluation:ProjectEvaluation"
eval_config_builder = "evaluation:build_eval_config"
```

Evaluation 类必须继承 `dodorl_sdk.Evaluation`；builder 必须返回 `EvalConfig`。

### 10.2 分布式评估中心

1+N 组网与训练侧 `LearnControlSDK` + `SampleSDK` 相同：一个中心进程托管任务总线和 RPC，N 个采样进程连中心。评估中心不创建
Learner、不消费学习令牌；采样复用 `SampleRole`，由任务令牌决定 `epsilon=0` 和 `deterministic`。

中心入口是 `EvalControlSDK`。它按 `branch="eval"` 加载 TOML，解析 `[entry].distributed_evaluation_class`（缺省
`EvalControlRole`），再启动中心角色：

```python
from dodorl_sdk.portal import EvalControlSDK, SampleSDK

# 进程 1：评估中心。role_id 默认 eval-control-0。
EvalControlSDK(
    plugin_dirs=["plugins"],
    config_file="configs/evaluation.toml",
    role_id="eval-control-0",
).run()

# 进程 2..N：仍是 SampleRole。branch="eval" 只做评估装配（EvalConfig、模型文件、learners_config=[]）。
SampleSDK(
    plugin_dirs=["plugins"],
    config_file="configs/evaluation.toml",
    sample_role_name="sample-0",
    branch="eval",
).run()
```

自定义中心报告应继承 `EvalControlRole`，覆盖同一套 `episode_report` / `finished_report`。入参对齐采样上报的
`total_rewards`、`episode_metrics`、`episode_rewards_details`、`artifacts`，不要从 `self.env` 抠字段：

```python
from dodorl_sdk import EvalControlRole


class ProjectEvalControl(EvalControlRole):
    def episode_report(self, episode_index: int, *, episode_metrics=None, **payload) -> dict:
        report = super().episode_report(
            episode_index,
            episode_metrics=episode_metrics,
            **payload,
        )
        report["project_metric"] = (episode_metrics or {}).get("success")
        return report

    def finished_report(self):
        return {"episode_count": len(self.episode_results)}
```

```toml
[entry]
evaluation_class = "evaluation:ProjectEvaluation"
distributed_evaluation_class = "evaluation:ProjectEvalControl"
eval_config_builder = "evaluation:build_eval_config"
```

`distributed_evaluation_class` 必须是 `EvalControlRole` 的子类。Torch RPC 用 `[torch_rpc_config].world_size` 的 `branch:`
区分训练/评估采样数；评估 SDK 固定 `branch="eval"`：

```toml
[design]
train_sample_count = 3
eval_sample_count = 6

[torch_rpc_config]
world_size = "branch: train=1 + train_sample_count, eval=1 + eval_sample_count"
init_address = "127.0.0.1:53151"
```

这些入口同样支持 `[grpc_config]`。Torch RPC 的 `RANK` 仍由启动脚本注入。

## 11. 配置接口

全部配置类型集中在 `dodorl_sdk.config`：

```python
from dodorl_sdk.config import AgentConfig, EntryConfig, EnvConfig, EvalConfig, LearnerConfig, GrpcConfig, StageConfig, StageTaskConfig, TrainConfig
from dodo_sim import SimConfig, ZMQConfig, ZMQSocketType
```

常用配置同时从 `dodorl_sdk` 顶层公开。更完整的算法、存储、AutoML、模型管理和采样配置清单，以 `dodorl_sdk.config.__all__` 为准。

### 11.1 业务配置扩展

`EnvConfig` 和 `SimConfig` 的标准字段由 SDK 统一管理。业务插件的自由配置统一放在
`metadata` 中，不需要继承或替换 SDK 的 `EnvConfig`、`SimConfig`。

环境插件配置示例：

```toml
[env_config]
sense_name = "jammer"
max_step_per_episode = 50
state_dim = 22
agent_names = ["jammer-1", "jammer-2"]

[env_config.metadata]
target_names = ["radar-1", "radar-2", "radar-3"]
fighter_name = "fighter-1"

[env_config.metadata.goal]
longitude = 120.8847
latitude = 24.1653
altitude = 3000.0
```

环境从 `config.metadata` 读取业务字段：

```python
from dodorl_sdk import EnvConfig, ParallelEnvironment


class JammerEnvironment(ParallelEnvironment):
    def __init__(self, config: EnvConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        metadata = config.metadata or {}
        self.target_names = list(metadata["target_names"])
        self.fighter_name = str(metadata.get("fighter_name", "fighter-1"))
        self.goal = dict(metadata.get("goal", {}))
```

仿真插件使用 `[sim_config.metadata]`：

```toml
[sim_config]
sim_type = "custom_sim"
clock_rate = 5.0
config_source = "local"

[sim_config.metadata]
command_timeout = 10.0
state_buffer_size = 2000
fragment_expire_seconds = 20.0
```

```python
from dodo_sim import SimConfig, Simulation


class CustomSimulation(Simulation):
    def __init__(self, sim_type: str, config: SimConfig, **kwargs):
        super().__init__(sim_type, config, **kwargs)
        metadata = config.metadata or {}
        self.command_timeout = float(metadata.get("command_timeout", 10.0))
        self.state_buffer_size = int(metadata.get("state_buffer_size", 2000))
```

`[sim_config.middleware]` 是 SDK 标准中间件配置，不属于业务 `metadata`：

```toml
[sim_config.middleware]
remote_ip = "127.0.0.1"
remote_port = 12000
remote_mode = "bind"
send_socket_type = "PUSH"
local_ip = "127.0.0.1"
local_port = 11000
local_mode = "connect"
receive_socket_type = "PULL"
```

`metadata` 支持 TOML 嵌套表、数组和普通标量。配置较复杂时，建议在业务插件内转换为
强类型 dataclass：

```python
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class JammerOptions:
    target_names: tuple[str, ...]
    fighter_name: str = "fighter-1"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "JammerOptions":
        return cls(
            target_names=tuple(str(name) for name in metadata["target_names"]),
            fighter_name=str(metadata.get("fighter_name", "fighter-1")),
        )
```

标准配置表中的未知字段仍会报错，避免拼写错误被静默忽略。例如自定义
`target_names` 必须写在 `[env_config.metadata]` 下，不能直接写在 `[env_config]` 下。

## 12. 模块扩展与公共契约

`dodorl_sdk.modules` 提供以下可替换基线：

- `AutoML`
- `ModelManager`
- `MonteCarloMethod`
- `ReplayBuffer`
- `RolloutLearnWorker`
- `RolloutSampleWorker`
- `SampleLearnManager`
- `SimManager`
- `StageTask`
- `TrainTaskManager`
- `LearnPermitRequest`、`LearnPermitGrant`、`LearnPermitPolicy`
- `ReplayReadyLearnPermitPolicy`

`dodorl_sdk.contracts` 提供稳定的数据模型和 Protocol，包括：

- `EpisodeTransition`、`SAEpisodeTransition`；
- `EpisodeInfo`、`LearnInfo`、`LearnTaskResult`、`StageLifecycleState`；
- `ModelInfo`、`ModelArtifactInfo`、`ModelStatus`、`ModelUsage`；
- `TrainTaskManagerPort`、`ReplayBufferPort`、`ModelManagerPort`、`SimManagerPort`、`MonteCarloMethodPort`。

业务工程只应依赖这些公共模块，不应依赖内部实现路径。

覆盖 `TrainTaskManager` 时使用以下公共钩子，不要改 `_` 方法：

- `build_learn_permit_policy()`：注入 `LearnPermitPolicy`；`decide` 只根据当前 ReplayBuffer 状态
  返回可执行的 Learner，不建立按 episode 累积的学习债务。类型与默认策略
  `ReplayReadyLearnPermitPolicy` 从 `dodorl_sdk.modules` 导出。
- `check_sample_admission(completed_episodes, generation)`：采样准入闸门；返回 `None` 放行，返回字符串则阻塞并进入
  `sample_backpressure_state.reason`。
- `enqueue_bypass_sample_task(episode_info, *, kind)`：旁路采样插队。令牌只带执行参数（`replay_write` / `record_render` /
  `record_name` 等）；本次仅支持 `replay_write=False`。
- `custom_monitor()` / `custom_monitor_interval_seconds()`：独立线程周期改状态。未覆盖 `custom_monitor` 时不启动线程。
- `poll_stage_lifecycle()`：阶段结束的唯一外部驱动 tick。`is_task_finished()` / `is_sample_completed()` 是它的薄封装，RPC
  白名单不新增此方法。

`RolloutSampleWorker` 只执行令牌参数（`replay_write` / `record_render` / `record_name` 等），不读取来源字段。旁路种类由 TM 按
`lease_id` 内部索引。

## 13. 构建 SDK

在 DodoRL 仓库根目录执行：

```bash
./sdk/build_sdk.sh
```

默认使用项目的 Python 3.12 解释器；特殊环境通过 `DODORL_BUILD_PYTHON` 覆盖。构建产物：

```text
sdk/dist/dodorl_sdk-*-py3-none-any.whl
```

构建脚本会把运行时实现收纳到 wheel 的 `dodorl_sdk/dodorl/`，安装后顶层包仍然只有 `dodorl_sdk`。

## 14. 最小检查清单

业务工程接入时建议至少检查：

```bash
python -c "from dodorl_sdk.portal import SerialTrainSDK; from dodorl_sdk import ParallelEnvironment; from dodo_sim import Simulation"
python -c "from dodo_sim.middleware import ZMQClient, ZMQSocketType"
python -c "from dodorl_sdk.config import EnvConfig, SimConfig, EntryConfig, GrpcConfig"
```

如果训练使用 SDK 默认 RNN/QMIX，业务 TOML 中不需要再次注册 `agent.rnn` 和 `learner.qmix`。

## 15. 离线样本微调（v0.13.0）

训练阶段通过 `TrainConfig.sample_type = "offline"` 驱动真机轨迹；轨迹入口仍使用顶层
`[monte_carlo_method_config]`，现有 TOML 无需迁移。有效轨迹数由 MC 索引决定，忽略
`TrainConfig.max_episode`。业务环境需要实现 `apply_recorded_state(payload)`，并确保
`fake_reset/fake_step` 水合后可以正常计算 observation、state、reward、done 和 action mask。
真机字段映射应在 `MonteCarloMethod.adapt_record` 子类中完成，标准 payload 必须包含完整状态和动作信息。

### 可选仿真依赖与开发解释器

纯算法训练无需安装 dodo-sim，独立仿真无需安装 dodorl-sdk。使用仿真的训练进程安装两个 SDK，也可通过 `pip install "dodorl-sdk[sim]"` 声明仿真依赖。训练模块、RPC 模块和类型注解不会在导入时加载 dodo_sim；实际启用仿真配置或资源池时才加载，缺包错误会提示在场景工程安装。关闭仿真时不创建资源池和对应 RPC 服务。Rollout 对已加载 SIM 异常的识别保持自动处理。

## 四类动作空间

见 [四类动作空间](../docs/manuals/action-spaces.md)；四象限定义与编解码示例见 `sdk/demo_action_spaces.py`。


## TOML 进程环境变量

统一训练配置支持顶层表：

```toml
[globe_env_vars]
OMP_NUM_THREADS = "4"
MKL_NUM_THREADS = "4"
```

值必须为字符串，显式覆盖当前进程的同名环境变量，子进程默认继承；不会修改系统永久设置、父shell或已有容器。
角色启动时应用，纯配置读取、插件预读和 `--check` 只校验。完整配置对象通过 `globe_env_vars` 属性暴露校验后的字面值，不对这些值执行 design/eval 或 shell 展开。

`from dodorl_sdk.portal import SerialTrainSDK` 本身不加载计算运行时，角色构造时会先应用配置再导入训练组件。
如果使用者会先导入 PyTorch、环境类或其他重依赖，可在这些导入前显式预加载：

```python
from dodorl_sdk import bootstrap_environment
bootstrap_environment("training.toml")

from dodorl_sdk.portal import SerialTrainSDK
SerialTrainSDK(config_file="training.toml").run()
```

该预加载入口也支持 `config_text=...`；`apply=False` 只读取校验。
已经初始化的 OpenMP/MKL/CUDA 不保证响应事后修改，应在相关库初始化前设置。
分布式角色各自在启动进程中应用自己的配置；不会自动向远端主机或 Docker 容器注入变量。
