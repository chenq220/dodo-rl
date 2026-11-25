# DodoRL SDK — 二次开发指南

DodoRL 0.17.0 提供训练调度、环境、算法、样本及模型管理。仿真、通信和仿真资源池由独立的 dodo-sim 提供。使用仿真时，`sim` 可选依赖的版本约束为 `>=0.4.0,<0.5`。
业务工程通过 SDK 的公共接口开发。
`portal.train` 与 `TrainRole` 仅用于源工程内部调试，不随 SDK 分发；业务训练使用公开的角色 SDK。
SDK 公开角色入口由 `dodorl_sdk.portal.__all__` 显式声明，内部角色按构建脚本
`SDK_ROLE_MODULES` 白名单分发；源码 `src/portal` 命令行入口目录不随 SDK 分发。

## 公共入口

```python
from dodorl_sdk import Action, State, EnvConfig, ParallelEnvironment, AECEnvironment
from dodorl_sdk.config import PluginsConfig, TomlTrainingConfig
from dodorl_sdk.portal import SerialTrainSDK, SerialStageTrainSDK, EvaluationSDK, SDKPluginRegistry
from dodo_sim import Simulation, SimConfig, SimManager, SimManagerConfig
from dodo_sim import AFSim, AFSimConfig, JsqlSim, JsqlSimConfig
```

Simulation、中间件、仿真配置、仿真异常和 SimManager 不再从 RL SDK 导出。
State 与 Action 在两个工程中各自实现，本版本保持基类代码、字段、默认值和序列化逻辑一致；两份类身份不同。
RL 提供 ActionTable 和训练动作编排；SIM 按动作字段契约使用对象，环境开发者无需手工在两份同结构动作间转换。

## 环境与态势注入

环境继承 ParallelEnvironment 或 AECEnvironment，实现观测、奖励、结束条件、状态向量与可用动作等业务钩子。
环境构造参数保留 config、sim_config、connect_sim、sim_manager 和 role_id，并向父类转发。
仿真连接由父类按插件配置创建，无仿真环境设置 EnvConfig.use_sim=False。
离线回放时，业务环境应实现 `_state_from_record(raw_state)`，恢复自己的态势类型；框架不再根据字典字段猜测科目态势。

```python
class MyEnvironment(ParallelEnvironment):
    def __init__(self, config, sim_config=None, connect_sim=True,
                 sim_manager=None, role_id=None):
        super().__init__(config=config, sim_config=sim_config, connect_sim=connect_sim,
                         sim_manager=sim_manager, role_id=role_id)
        if self.sim is not None:
            self.sim.state_convertor = MyState
    # 实现环境要求的 state/get_obs/get_reward/done/get_avail_agent_actions 钩子。
```

MyState 可继承 RL 或 SIM 的 State，也可提供独立的 from_json(payload) 解析接口。
Simulation 不检查它是否属于某个 SDK 的继承体系。需要原生仿真解析能力时可注入 SIM 的 SimState 类，
或 SimState.converter(sim_type="afsim", parser_options={})。类工厂按当前仿真实例绑定配置，不修改全局属性。
reset、forward_step、get_state 返回注入类的态势实例；资源重申请后保留转换器配置。
转换错误正常向上传播，不回退为旧帧或成功状态。

## 仿真与中间件开发

从 dodo_sim 导入 Simulation 和仿真异常，从 dodo_sim.middleware 导入 MessageMiddleWare、HTTPClient、ZMQClient 或 TcpUdpSocket。
Simulation 实现 reset、send_actions、forward_step、get_state 和 close，使用 _convert_state 自动转换 JSON。
get_state 不推进；forward_step 确认推进完成后返回终点态势；动作回复或推进结果不确定时不能自动重发。
支持渲染的适配器覆盖 render，返回当前已确认节点的图像；render 不推进仿真。

中间件只处理传输、原始消息缓存和幂等关闭，不承担暂停、恢复、推进、业务过滤或态势解析。
所有原始消息进入有界缓存，快照与接收使用相同的并发保护；异常和资源关闭要求见 dodo-sim 的 README。

## 框架自动识别仿真失败

**环境开发者正常调用 SIM SDK，让其异常向上传播，无需捕获后改抛 EpisodeInterruptedException，
也无需手工归还采样令牌、探活或重新申请仿真。**

在线采样的 Rollout 直接识别 dodo-sim 的 SimulationException 异常族，沿用 RL 的回合中断处理：

1. 使用当前 lease_id 上报本局失败，TaskManager 结算租约并允许重新领取任务，不增加成功局数。
2. 通过 env.reset 探活，期间临时关闭录制回调，防止写入健康检查样本。
3. 连续探活失败达到阈值（默认 10 次）后，调用 sim.reacquire_simulation 释放旧资源并申请资源、重建连接。
4. reset 验证成功后恢复调度，从新回合采样。

采样任务令牌与仿真资源独立管理。资源重申请需要 SimManager 管理模式；固定连接模式保留原连接探活。
异常恢复依赖 RL 在线采样流程驱动；只使用 SIM 编写自定义循环时，由调用方驱动恢复。
业务参数、态势模型、奖励和算法程序错误不能全部包装成可恢复仿真故障。

## 插件与训练

```toml
[plugins_config.simulation]
afsim = "dodo_sim.simulations.afsim:AFSim"
[plugins_config.environment]
my_env = "plugins.environment:MyEnvironment"
[sim_config]
sim_type = "afsim"
config_source = "manager"
[sim_manager_config]
use_sim = true
config_file = "./sim_config.ini"
```

SimManager 的资源池实现在 SIM，分布式训练的 RPC 适配由 RL 维护。
纯 game / mock 环境配置 env_config.use_sim=false，不要求安装 dodo-sim。仿真配置在首次读取启用的属性时按需装配；插件只注册路径，RPC、异常识别和关闭流程不得主动加载未使用的可选仿真 SDK。
角色 SDK 接受完整场景 TOML，统一配置入口将仿真块委托给 dodo-sim 的 TomlSimulationConfig 装配，scenario 无须拆分配置。dodo-sim 也可独立解析 TOML 文件或字符串；已装配的 sim_config / sim_manager_config 仍可显式注入角色 SDK。configure_sdk 仅设置算法 SDK 的工程根目录；INI 路径由 dodo-sim 管理。
仿真配置类从 dodo_sim 导入，业务自由参数放入 metadata。训练角色从 dodorl_sdk.portal 导入。

## 验证与迁移边界

SDK 的框架测试使用最小状态、动作、环境和协议模拟服务；7v3 科目已迁至 scenario，科目源码、配置和验收由该工程维护。
ArkSim 与 SMAC 已废弃，不再提供实现、选择器或旧别名。AFSIM 与 JSQL 继续支持。
测试应验证两侧态势类注入、独立安装、真实租约失败回收、探活及重申请；不依赖场景逻辑。


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

## 原生MAPPO按步游标训练

`[learners.<id>.algo_config]` 可设置 `sample_strategy="on_policy_cursor_steps"`、`batch_size=1000`。
每批消费样本池中1000个新环境步，允许跨回合。循环策略对中段片段使用池中附带的局首 `burn_in`，
每轮PPO用当前参数无梯度重建hidden；前缀不参与loss或batch步数。段尾保留价值自举，不能把普通
窗口边界标为任务终止。现有采样端仍按整局提交，按步抽样不等于全进程同步的在线步屏障。
配置中不要启用会阻塞攒批的按局背压；batch_size按数据量配置，并非固定为1000。
