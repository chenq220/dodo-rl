# 训练 TOML 配置手册

科目训练、评估和分布式角色共用一份 TOML。标准表里的未知键会报错；业务自由字段只能写在 `metadata` 或 `[game_config]`。

配套文档：

- AutoML 搜索空间、Optuna、归档：见本页 [§18 `[auto_ml_config]`](#18-auto_ml_config)
- 算法插件与 `algo_config` 二次开发：[算法二次开发手册.md](./算法二次开发手册.md)
- SDK 侧插件与 metadata 约定：`sdk/README.md` 第 9、11 节

完整科目示例：`subjects/cooperative_search/cooperative_search_qmix.toml`、`subjects/shaping_test/region_activity.toml`。

---

## 1. 顶层段一览

| TOML 段 | 必填 | 作用 |
| --- | --- | --- |
| `[design]` | 否 | 任务设计变量及框架保留的 `task_id`；供 `${name}` / `eval:` / `branch:` 引用 |
| `[plugins_config]` | 内置科目可省略；SDK 工程须声明 | 插件白名单、黑名单和类路径 |
| `[entry]` | 否 | 角色 id、评估类、评估配置构建器 |
| `[train_config]` | 建议有 | 训练局数、epsilon、模型路径、盘中评估 |
| `[env_config]` | 建议有 | 场景名、步数、全局状态维度、智能体清单 |
| `[env_config.metadata]` | 否 | 环境插件自由配置 |
| `[game_config]` | 否 | 复制进 `env_config.metadata["game_config"]` |
| `[replay_buffer_config]` | 否 | 回放池默认容量、抽样策略与张量池后端；训练分区容量由各 Learner 配置 |
| `[model_manager_config]` | 可选段 | 模型池晋升与回收 |
| `[monte_carlo_method_config]` | 可选段 | 样本读写 / 随机策略采样 |
| `[grpc_config]` | 与 Torch RPC 可并存；`DODO_RPC` 选择 | 分布式 gRPC 地址 |
| `[torch_rpc_config]` | 与 gRPC 可并存；`DODO_RPC` 选择 | 静态 TensorPipe world |
| `[agents.N]` | 建议有 | 编号表：策略 Agent |
| `[learners.N]` | 建议有 | 编号表：学习器，`name` 与 Agent 对齐 |
| `[stages.N]` | 分阶段必填 | 编号表：课程阶段 |
| `[stage_task_config]` | 否 | `start_stage`、跳过阶段加载 |
| `[eval_config]` | 否 | 独立评估 / 盘中评估共用 |
| `[auto_ml_config]` | 可选段 | AutoML 全局默认 |
| `[log_config]` | 否 | 日志目录、级别、Handler |

可选段统一支持 `enabled`（默认 `true`）。`enabled=false` 时不生成对应配置对象。当前走这套语义的是：`auto_ml_config`、`model_manager_config`、`monte_carlo_method_config`，以及嵌套的 `seed_search` / `business_metric` / `auto_ml_overrides`。

`[rpc_config]` 已删除，见 §23。`[grpc_config]` 与 `[torch_rpc_config]` 可以同时写在一份 TOML 里；环境变量 `DODO_RPC=grpc|trpc` 选择启用哪一段。未设置且两段都在时用 `[torch_rpc_config]`。都不写时分布式地址可忽略（串行任务）。

---

## 2. 解析约定

### 2.1 `[design]` 变量引用

`[design]` 不映射到某个固定配置类。除框架保留的 `task_id` 外，其余字段作为任务设计变量。`root_dir = "{{project_root}}"`（或 `"{{root_dir}}"`）会替换成加载时的工程根目录。

其它段里的字符串支持：

| 写法 | 含义 |
| --- | --- |
| `"${name}"` | 整值替换为 design 中的 `name`（保留原类型） |
| `"prefix-${name}"` | 内嵌替换，结果是字符串 |
| `"eval: width * count"` | 受限表达式：`+ - * / // %`、一元正负、`len(...)` |
| `"branch: train=..., eval=..."` | 按加载分支选一支；默认 `train` |

`eval:` 只能引用已经解析完的 design 变量。循环引用无法解析。

**配置示例：**

```toml
[design]
width = 8
count = 2
obs_dim = "eval: width * count"
path_prefix = "prefix-${width}"
```

### 2.2 编号表

`agents`、`learners`、`stages`、动作参数、AutoML 搜索空间都用数字键：

```toml
[agents.0]
[agents.1]
[agents.0.actions.0]
[agents.0.actions.0.parameters.0]
[stages.0.auto_ml_overrides.learner_search_space.0]
```

只收集纯数字键，按整数排序。不要写成 `[agents.uav_team]`。

### 2.3 特殊字段归一

| 键 | 行为 |
| --- | --- |
| `device` | `"auto"` 按机器选 CUDA/CPU；也可直接写 `"cpu"` / `"cuda"` |
| `seed` | `"none"` / `"null"` / 空串 → 不固定；否则取模到 32 位正整数 |
| `model_version` | `"none"` / `"null"` / 空串 → 不指定版本 |

### 2.4 标准字段 vs metadata

`EnvConfig` 的标准字段是封闭的。场景专有项必须写在：

- `[env_config.metadata]`
- `[train_config.env_metadata]` / `[stages.N.train_overrides.env_metadata]`（任务级增量覆盖）
- `[game_config]`（整表并入 `env_config.metadata["game_config"]`）

不要把场景专有键（如 `target_names`）直接写在 `[env_config]` 顶层。

---

## 3. `[design]`

配置任务级设计变量：路径根、维度、名单，以及可选的稳定任务身份。供其它段用 `${}` / `eval:` / `branch:` 引用。

| 键 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `task_name` | `str` | - | 无固定默认 | 任务逻辑名；常作为 AutoML `subject_name` |
| `task_id` | `str` | - | `""` | 框架保留的进程任务身份；非空值覆盖环境变量 `DODO_TASK_ID` |
| `root_dir` | `str` | - | 无固定默认 | 工程根；写 `"{{project_root}}"` 时加载阶段替换成实际路径 |
| `resource_dir` | `str` | - | 无固定默认 | 产物根，科目里常写 `"${root_dir}/artifacts"` |
| 维度/名单等 | 任意 | - | 无 | 如 `obs_dim`、`state_dim`、`agent_names`，可用 `eval:` / `len(...)` 计算 |

`task_id` 优先级：非空 `design.task_id` > 非空环境变量 `DODO_TASK_ID` > 自动生成的 8 位十六进制 ID。省略 `task_id`，或配置为空字符串、`"none"`、`"null"`（大小写不敏感），表示不覆盖环境变量或自动生成值。TOML 没有原生 `null`，不要写成未加引号的 `null`。同一分布式任务的各角色进程应加载相同的非空 `design.task_id`。显式 `DODO_LOG_DIR` 仍优先于按任务 ID 推导的默认日志目录。

**配置示例：**

```toml
[design]
task_name = "demo"
task_id = ""
root_dir = "{{project_root}}"
resource_dir = "${root_dir}/artifacts"
obs_dim = 8
state_dim = 16
agent_names = ["agent_0"]
```

---

## 4. `[plugins_config]`

配置本次运行启用哪些插件及其实现类路径。内置科目通常可省略本段，使用工程默认插件清单；独立 SDK 工程须显式声明本次用到的类路径。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `enabled` | `list[str]` | - | `[]` | 白名单，形如 `"agent.rnn"`；与 `disabled` 互斥 |
| `disabled` | `list[str]` | - | `[]` | 黑名单 |

子表（值必须是 `module:ClassName`）：

- `[plugins_config.simulation]`
- `[plugins_config.environment]`
- `[plugins_config.agent]`
- `[plugins_config.learner]`
- `[plugins_config.module]`

`enabled` 与 `disabled` 不能同时非空；两者皆空时启用当前清单内全部插件。同一子表内插件名不可重复。

**配置示例：**

```toml
[plugins_config]
# 内置科目可整段省略；SDK 示例：
# enabled = ["environment.demo", "agent.rnn", "learner.qmix"]

[plugins_config.environment]
demo = "my_project.env:DemoEnv"

[plugins_config.agent]
rnn = "dodorl.agents:RnnAgent"

[plugins_config.learner]
qmix = "dodorl.learners:QmixLearner"
```

---

## 5. `[entry]`

配置入口侧元数据：本进程角色名，以及评估实现与评估配置的构建入口。串行训练常可省略。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `role_id` | `str` | - | `""` | 角色进程名；也可由 CLI / 环境变量覆盖 |
| `evaluation_class` | `str` | - | `""` | 单机评估类，`module:Class` |
| `distributed_evaluation_class` | `str` | - | `""` | 分布式评估控制类，`module:Class` |
| `eval_config_builder` | `str` | - | `""` | 从完整配置构造评估配置的函数路径 |

**配置示例：**

```toml
[entry]
role_id = "serial_train"
evaluation_class = "my_project.eval:DemoEvaluator"
```

---

## 6. `[train_config]`

配置训练任务：总局数、探索、模型读写路径、盘中/结束评估、产物归档与采样相关字段。有训练主路时建议配置。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `max_episode` | `int` | - | `2000` | 训练总局数 |
| `sample_type` | `str` | `online` / `offline` | `"online"` | 采样方式；训练可用在线或离线 |
| `record_sample` | `bool` | - | `false` | 是否录制每局态势与动作的 JSON 结构样本；不控制 GIF，也不保存回放池 |
| `artifacts_archive_enabled` | `bool` | - | `true` | 任务正常结束后是否自动归档本任务已生成的模型、结果、GIF、样本、回放池和报告；关闭后文件保留在原运行目录 |
| `artifacts_archive_save_dir` | `str` | - | `artifacts/archives` | 任务归档根目录；最终归档写入其下的 `<TASK_ID>/` |
| `result_save_dir` | `str` | - | `artifacts/results` | 结果目录 |
| `render_video_dir` | `str` | - | `artifacts/recordings` | GIF 目录 |
| `render_gif_fps` | `int` | - | `15` | GIF 帧率 |
| `render_gif_frame_stride` | `int` | - | `1` | 隔多少环境步采一帧 |
| `render_gif_max_frames` | `int?` | - | `None` | GIF 最大帧数；`None` 表示不截断 |
| `agent_compose_type` | `str` | `order` / `decision` | `"order"` | `order`=按环境默认顺序步进；`decision`=由 Episode Agent 动作交给环境解释顺序 |
| `env_metadata` | `table` | - | `{}` | 对本局环境 `metadata` 的递归覆盖 |
| `sample_task_timeout` | `float` | - | `300` | 采样令牌无结果占用超时（秒） |
| `monte_carlo_method_config` | 子表/可选 | - | `None` | 样本读写配置；也可由顶层 `[monte_carlo_method_config]` 注入 |
| `seed` | `int?` | - | `None` | 训练种子；分布式按 `role_id` 派生 |
| `initialization_seed` | `int?` | - | `None` | 网络初始化种子；空则沿用 `seed` |
| `save_optimo_models` | `bool` | - | `true` | 刷新本代历史最大回报时覆盖保存 optimo 模型 |
| `agent_reload_model_cycle` | `int` | - | `1` | 每隔多少轮采样刷新 Agent 网络 |
| `center_model_cycle` | `int` | - | `10` | 分布式学习每隔多少轮更新总线网络 |
| `model_save_dir` | `str` | - | `artifacts/models` | 可交付模型目录 |
| `model_load_dir` | `str?` | - | `artifacts/models` | 加载目录 |
| `model_version` | `str?` | - | `None` | 从哪个版本前缀开始训 |
| `start_epsilon` | `float` | - | `1` | ε 起点 |
| `end_epsilon` | `float` | - | `0.05` | ε 终点 |
| `epsilon_decay_episodes` | `int` | - | `200` | 在多少局内衰减 |
| `temperature` | `float` | - | `1.0` | Categorical 采样温度 |
| `temperature_episodes` | `int` | - | `0` | 使用温度的局数 |
| `mid_eval_enabled` | `bool` | - | `false` | 是否开启盘中评估 |
| `mid_eval_episode_interval` | `int` | - | `100` | 每完成多少局触发一次盘中评估 |
| `end_eval_episodes` | `int` | - | `0` | 主路结束后自动评估局数；`0` 表示不做 |
| `sample_learn_max_lag` | `int?` | - | `None` | 采样领先学习的最大局数；`None` 关闭背压 |
| `auto_ml_overrides` | 子表/可选 | - | `None` | 单阶段训练上覆盖全局 `[auto_ml_config]`；字段同 §18；编号表出现则整表替换。兼容旧键名 `auto_ml_config` |

自动归档只收集当前任务已登记的输出，不扫描整个 `artifacts`，也不会移动 `model_load_dir`、`agent_model_path` 等输入文件。分阶段任务每阶段结束增量归档一次，所有阶段须使用相同的 `artifacts_archive_save_dir`。单阶段用 `[train_config.auto_ml_overrides]`；多阶段用 `[stages.N.auto_ml_overrides]`。全局默认仍写顶层 `[auto_ml_config]`。

**配置示例：**

```toml
[train_config]
max_episode = 2000
sample_type = "online"
record_sample = false
artifacts_archive_enabled = true
artifacts_archive_save_dir = "${resource_dir}/archives"
result_save_dir = "${resource_dir}/results/demo"
render_video_dir = "${resource_dir}/recordings"
model_save_dir = "${resource_dir}/models/demo"
model_load_dir = "${resource_dir}/models/demo"
start_epsilon = 1.0
end_epsilon = 0.05
epsilon_decay_episodes = 200
mid_eval_enabled = false
end_eval_episodes = 0
```

---

## 7. `[env_config]`

配置环境侧：场景名、单局步数、实体清单、是否依赖仿真，以及环境自由 metadata。几乎所有科目都需要。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `sense_name` | `str` | - | `""` | 场景 / 环境插件名；科目须显式配置 |
| `max_step_per_episode` | `int` | - | `50` | 单局最大步数 |
| `agent_names` | `list[str]` | - | `[]` | 环境实体智能体清单 |
| `step_interval` | `float` | - | `1` | 每次动作后推进的原生时间跨度；时间流单位由仿真定义，步进仿真表示实际推进步数 |
| `step_interval_secs` | `float` | - | `1` | `step_interval` 的兼容读写别名；两种名称共用同一值，新配置建议使用正式名称 |
| `state_dim` | `int` | - | `0` | 全局状态维度；会同步到 Agent/Learner 的运行期状态维 |
| `use_sim` | `bool` | - | `true` | 是否依赖 Simulation；本地游戏可 `false` |
| `metadata` | `table` | - | `{}` | 环境自由配置 |

`[game_config]` 无固定 schema，整表深拷贝到 `metadata["game_config"]`，由环境自己解读。

**配置示例：**

```toml
[env_config]
sense_name = "demo_env"
use_sim = false
max_step_per_episode = 50
state_dim = "${state_dim}"
agent_names = ["agent_0"]

[env_config.metadata]
difficulty = 1

[game_config]
map_name = "default"
```

---

[game_config] 的 `show_threat_roaming` 控制活动圈、航点和引导线的渲染，默认 `false`；威胁专训科目显式设为 `true`，正式搜索科目设为 `false`。

### 7.1 `[game_config.threat_roaming]`

普通协同搜索与四分区科目用此表配置每个威胁的独立临时游走圈、随机航点和同伴避让。环境在全图按责任区和等面积子格生成分散圈心，默认每区2个圈、每子格1个。圈完整位于子格内，圈心间距至少为直径；圈心不跟随威胁位置；每40～80步联合刷新布局并跨子格分配，单次圈心移动100～200公里。实体到达时仅在原圈内换航点。运动仍允许跨区。训练与正式搜索应使用同一观察参数，并使用新训练的单威胁策略模型。省略此表时，正式搜索仍使用旧威胁观察；四分区威胁专训默认使用本表的默认值。

| 字段 | 类型 | 可选值 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `circle_radius_km` | `float` | 正数 | `50.0` | 分层活动圈半径，各子格宽高须至少容纳一个直径；圈完整位于场内，不作为运动硬边界 |
| `circle_center_min_shift_km` | `float` | 非负 | `100.0` | 联合刷新时每个圈心的最小位移，须小于最大位移 |
| `circle_center_max_shift_km` | `float` | 正数 | `200.0` | 联合刷新时每个圈心的最大位移；每实体分配到不同子格，布局不可行时明确报错 |
| `waypoint_reached_km` | `float` | 正数 | `5.0` | 距航点达到此距离即完成一次并换点 |
| `waypoint_min_distance_km` | `float` | 正数 | `10.0` | 新随机航点距所属圈心的最小距离；大于到达距离、小于最大选点距离 |
| `waypoint_max_distance_km` | `float` | 正数 | `25.0` | 新航点距所属圈心的最大距离，不能超过活动圈半径 |
| `waypoint_refresh_min_steps` | `int` | 正整数 | `40` | 全局布局联合刷新间隔的随机下限，不大于上限 |
| `waypoint_refresh_max_steps` | `int` | 正整数 | `80` | 全局布局联合刷新间隔的随机上限，刷新后所有实体换航点；受timeout硬上限限制 |
| `waypoint_timeout_steps` | `int` | 正整数 | `200` | 同一航点最多追踪的物理步数，超时换点不发到达奖励 |
| `perception_radius_km` | `float` | 正数 | `100.0` | 以自身为中心的圆形同伴感知范围；边界及范围外不进入网格 |
| `radial_bins` | `int` | 正整数 | `10` | 极坐标径向环数，等距离分环 |
| `angular_bins` | `int` | 正整数 | `16` | 极坐标角向扇区数，以自身正前方为第一个中心并循环排列 |
| `representation_output_dim` | `int` | 环数×扇区数 | `160` | 逐格保留的同伴表征长度；完整观察长度为本值加13 |
| `separation_distance_km` | `float` | 正数 | `30.0` | 低于此距离时扣分，不能超过同伴感知半径 |
| `danger_distance_km` | `float` | 正数 | `15.0` | 低于此距离时追加危险接近惩罚，须小于期望间距 |
| `progress_reward_weight` | `float` | 非负 | `0.2` | 每步接近当前航点的奖励权重，远离则扣分 |
| `arrival_reward` | `float` | 非负 | `0.5` | 每完成一个航点的一次性奖励 |
| `circle_penalty` | `float` | 非负 | `1.0` | 离开自身当期临时圈的平方惩罚上限，距自身圈心达到两倍半径时达到上限；换点前按旧圈结算 |
| `separation_penalty` | `float` | 非负 | `0.5` | 对最近同伴过近的平方惩罚上限 |
| `danger_penalty` | `float` | 非负 | `2.0` | 对最近同伴危险接近的追加平方惩罚上限 |

四分区科目启用后威胁数量必须按区均分、每区至少2个；普通科目按全图分层，不限制每区数量。配置威胁 Agent 时使用单成员 `agent_groups`、`obs_agent_id=false`，默认 `obs_dim=173`、`state_shape=173`。正式科目的威胁另设 `fixed_mode=true` 和新模型路径。该观察不含目标或无人机坐标，各威胁使用自己的奖励。

```toml
[game_config.threat_roaming]
circle_radius_km = 50.0
circle_center_min_shift_km = 100.0
circle_center_max_shift_km = 200.0
waypoint_reached_km = 5.0
waypoint_min_distance_km = 10.0
waypoint_max_distance_km = 25.0
waypoint_refresh_min_steps = 40
waypoint_refresh_max_steps = 80
waypoint_timeout_steps = 200
perception_radius_km = 100.0
radial_bins = 10
angular_bins = 16
representation_output_dim = 160
separation_distance_km = 30.0
danger_distance_km = 15.0
progress_reward_weight = 0.2
arrival_reward = 0.5
circle_penalty = 1.0
separation_penalty = 0.5
danger_penalty = 2.0
```

---

## 8–9. 仿真配置与统一入口

同一份 TOML 可以同时包含训练配置与 `sim_config`、`sim_manager_config`（包括 `instances`）、`plugins_config.simulation`。`TomlTrainingConfig` 统一解析 design 引用与表达式，将仿真块委托给 dodo-sim 的 `TomlSimulationConfig.from_dict()` 装配；中间件和实例驱动字段的规则由 dodo-sim 维护。联合运行时插件仍通过现有统一注册器激活。

客户端与管理器配置分别按需装配并缓存；读取缺省管理器只返回 None，不连带装配客户端。读取训练和环境配置本身不加载 dodo-sim；首次读取启用的仿真配置时才委托 SIM。纯本地 game 配置 `env_config.use_sim = false`，且不启用 sim_manager_config，此时两个仿真配置均为 None，无须安装 dodo-sim。若显式启用管理器，则仍需要仿真依赖。

scenario 直接使用这一入口，不拆分 TOML。分布式中心角色创建 dodo-sim 的 SimManager，通过现有 RPC 向采样进程分发连接；实例重建与租约恢复沿用中心管理流程。

RL 配置入口及各角色 SDK 接受二选一的 `config_file` 和 `config_text`，例如 `TomlTrainingConfig(config_text=scene_text, project_root=project_root)`。dodo-sim 保留独立的仿真 TOML 文件/字符串解析能力；也可分别解析两份 TOML，再通过角色 SDK 的 `sim_config` / `sim_manager_config` 显式覆盖参数注入。

---

## 10. `[replay_buffer_config]`

配置经验回放池的默认容量、抽样策略、落盘后端与张量池类型。`capacity=1000` 是未单独指定 Agent 分区时的初始默认值；有学习器的训练任务由 Role 从各 Learner 的 `algo_config.capacity` 设置对应分区。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `capacity` | `int` | - | `1000` | 未指定 Agent 分区时的默认容量；Role 可按 Learner 容量覆盖对应分区。容量单位随 `sample_packaging` 变化：独立组模式为组轨迹数，捆绑模式为物理局数 |
| `agent_capacities` | `dict[str,int]` | - | `{}` | 运行期按 Agent 分区装配的容量；科目 TOML 通常不写，训练容量请写在各 Learner 的 `algo_config.capacity` |
| `batch_size` | `int` | - | `64` | 默认抽样批大小；学习时通常被 Learner `algo_config.batch_size` 覆盖。批量单位由抽样策略决定：按局策略为组轨迹数或物理局数，按步策略为步数 |
| `strategy` | `str` | `recent_biased` / `time_slice` / `on_policy` / `on_policy_recent_steps` / `on_policy_cursor` / `on_policy_cursor_steps` / `sliding_window` / `uniform` / `uniform_steps` | `"recent_biased"` | 默认抽样策略，各取值含义见下表；捆绑封装模式支持 `uniform` / `uniform_steps` / `recent_biased` / `time_slice` |
| `sample_packaging` | `str` | `independent` / `episode_groups` | `"independent"` | 样本封装/抽样单位。`independent`：保存与抽取独立组轨迹，允许不同局、不同组混采。`episode_groups`：同一物理 episode、同一 Agent 的多组轨迹封成一个张量样本并按该样本抽取，网络消费时把 `[B,G,T+1,K,D]` 的有效组铺平为 `[N,T+1,K,D]`；缺组槽位不作为训练样本。捆绑模式下 `capacity` 的单位是物理局，按局抽样的 `batch_size` 也是物理局数，按步抽样则是物理决策步数——相同数据预算需要按每局组数换算（例如 500 局×4 组 ≈ 2000 条组轨迹），不要用相同的裸数值配置两种模式；on-policy 与 sliding_window 策略在捆绑模式下会明确报错 |
| `recent_ratio` | `float` | - | `0.3` | `recent_biased` 中最新数据比例 |
| `seed` | `int?` | - | `None` | 归档/抽样种子；空则沿用 `train_config.seed` |
| `storage_type` | `str` | `local` / `s3` | `"local"` | 阶段快照落盘/加载后端 |
| `storage` | 子表 | - | 按类型填充 | 嵌套 `[replay_buffer_config.storage]`；见 §21 |
| `tensor_pool` | `str` | `memory` / `lazy_memmap` | `"memory"` | `memory`=进程内 CPU 张量；`lazy_memmap`=磁盘 mmap，省内存、占磁盘 |
| `keep_scratch` | `bool` | - | `false` | `true` 时任务结束后保留本地 mmap scratch；业务落盘用阶段 `save_replay_buffer` |

`tensor_pool = "lazy_memmap"` 时，scratch 落在 `storage.data_path/scratch_subdir/<task_id>/...`（S3 回退到 `artifacts/samples/replay_scratch`）。行预算由程序按容量、各阶段 rehearsal 比例与 `env.max_step_per_episode` 推算，不是 TOML 字段；同局多组捆绑模式下每局占多倍行，入池时按真实步数动态扩容。

`sample_packaging` 的补充规则：

- 两种模式共用同一 AgentNet、Learner 与组宽固定的 QMIX；loss 归约规则相同，每批仍只执行一次 optimizer 更新，不改变推理协同机制。
- 捆绑模式下 `uniform_steps` 按物理局及时间步选择决策，各组使用同一时间索引；提前结束的组不产生该步样本。
- 捆绑模式（`episode_groups`）的 `time_slice` 按共同物理时间选窗，各组只保留窗口内自己的实际决策帧，中途开始的组沿用同组同局 burn-in；窗口内没有实际帧的组不产生样本。
- 阶段快照（DRBF）的存储布局与封装模式无关，跨模式恢复时捆绑样本由"连续同局轨迹"重新派生：独立快照在捆绑模式下得到同局捆绑，捆绑快照在独立模式下退化为逐条组轨迹。
- 回放池状态同时汇报物理局数、有效组轨迹数与有效步数三口径，监控与实验报告不要只用单一 `size` 数值比较两种模式的数据预算。

抽样策略取值：

| 值 | 含义 |
| --- | --- |
| `recent_biased` | 近期偏好：一部分取最新，其余从历史均匀抽。适合 QMIX / VDN / DQN 等 off-policy |
| `time_slice` | 同一局内连续切片 `[t:t+L]`；`batch_size` 是连续步数 L。适合需要时序片段的算法 |
| `on_policy` | 取池尾最近 `batch_size` 个 episode。适合 PPO / MAPPO / A2C |
| `on_policy_recent_steps` | 从池尾向前累计 `batch_size` 个 transition；`batch_size` 按步数不是局数 |
| `on_policy_cursor` | 游标取最近 `batch_size` 个 episode 后前进，每条样本只抽一次 |
| `on_policy_cursor_steps` | 按步游标切 `batch_size` 步，已返回步不会重复抽 |
| `sliding_window` | 从队头设游标，每次后移一格；到末尾再从头。用于离线学习 |
| `uniform` | 从该 Agent 全部缓存 episode 中均匀随机抽 `batch_size` 个 |
| `uniform_steps` | 从有效单步转移中均匀、有放回地抽取 `batch_size` 次；可重复抽到同一步，轨迹越长被抽中的概率越高。适合 MADDPG 等离策略算法 |

**配置示例：**

```toml
[replay_buffer_config]
batch_size = 64
strategy = "recent_biased"
recent_ratio = 0.3
sample_packaging = "independent"
storage_type = "local"
tensor_pool = "memory"
keep_scratch = false
```

---

## 11. `[model_manager_config]`

可选段。配置模型池：候选快照目录、自动晋升与版本回收。分布式学习或多版本模型池场景使用；空表表示启用默认值。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `tmp_local_dir` | `str` | - | `artifacts/models/_snapshot` | 候选快照工作目录 |
| `auto_promote_first_model` | `bool` | - | `true` | 首个提交是否自动晋升 champion |
| `auto_promote_interval` | `int` | - | `1` | 每完成多少个学习批次晋升；`-1` 关闭 |
| `auto_promote_optimo_reward` | `bool` | - | `false` | 晋升窗口内选 reward 最高候选，否则选最后一次 |
| `max_versions_per_model` | `int` | - | `8` | 每 Agent 最多保留版本；`-1` 不回收 |
| `storage_type` | `str` | `local` / `s3` | `"local"` | 后端存储类型 |

**配置示例：**

```toml
[model_manager_config]
enabled = true
tmp_local_dir = "artifacts/models/_snapshot"
auto_promote_first_model = true
auto_promote_interval = 1
auto_promote_optimo_reward = false
max_versions_per_model = 8
storage_type = "local"
```

---

## 12. `[monte_carlo_method_config]`

可选段。配置结构化样本的读写、上传与轨迹校验。随机采样科目或离线样本驱动时常配这一段。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `sample_size` | `int` | - | `50000` | 最大样本容量 |
| `upload_interval` | `int` | - | `100` | 每采样多少次上传一次 |
| `tmp_local_dir` | `str` | - | `artifacts/samples` | 临时文件目录 |
| `storage_type` | `str` | `local` / `s3` | `"local"` | 后端存储类型 |
| `write_enabled` | `bool` | - | `true` | `false` 时只读历史样本 |
| `load_sample_name` | `str?` | - | `None` | 读取已有样本文件名或 S3 key |
| `load_file_path` | `str?` | - | `None` | 本地完整路径，优先于 `tmp_local_dir` + 文件名 |
| `accept_truncated_episodes` | `bool` | - | `false` | 是否接受最后一步未达到局步数上限的连续片段 |
| `split_on_step_gap` | `bool` | - | `false` | 发现步号缺口时是否拆成多个连续候选轨迹 |

`load_file_path` 优先于 `tmp_local_dir` + `load_sample_name`。只读历史样本时设 `write_enabled = false` 并指定加载路径或文件名。

**配置示例：**

```toml
[monte_carlo_method_config]
enabled = true
sample_size = 50000
upload_interval = 100
write_enabled = true
accept_truncated_episodes = false
split_on_step_gap = false
storage_type = "local"
```

---

## 13. RPC

分布式角色之间的进程通信地址。`[grpc_config]` 与 `[torch_rpc_config]` 可同时写在 TOML 中，由环境变量 `DODO_RPC` 选择启用哪一段（见 §1）。串行任务可省略。

### 13.1 `[grpc_config]`

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `task_manager_address` | `str` | - | `""` | 任务管理地址 `host:port` |
| `replay_buffer_address` | `str` | - | `""` | 回放池角色地址 |
| `model_manager_address` | `str` | - | `""` | 模型池角色地址 |
| `sim_manager_address` | `str` | - | `""` | 仿真资源池角色地址 |
| `monte_carlo_method_address` | `str` | - | `""` | 蒙特卡洛/样本角色地址 |
| `artifacts_manager_address` | `str` | - | `"127.0.0.1:53156"` | 产物台账角色地址；worker 转发 `put`/`put_log`/`complete` |
| `call_timeout_seconds` | `float` | - | `60` | 单次 RPC deadline；`<=0` 不设 |

**配置示例：**

```toml
[grpc_config]
task_manager_address = "127.0.0.1:50051"
replay_buffer_address = "127.0.0.1:50052"
model_manager_address = "127.0.0.1:50053"
call_timeout_seconds = 60
```

### 13.2 `[torch_rpc_config]`

静态 TensorPipe world。`RANK` 由启动脚本注入，不要写在 TOML。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `world_size` | `int` | - | `0` | RPC world 进程数，须 `>= 2` |
| `init_address` | `str` | - | `""` | Rank 0 监听地址 `host:port`，可带或不带 `tcp://` |
| `call_timeout_seconds` | `float` | - | `60` | 单次 RPC deadline（秒）；`<=0` 不设 |
| `num_worker_threads` | `int` | - | `16` | TensorPipe worker 线程数，须 `> 0` |
| `max_concurrent_replay_pushes` | `int` | - | `1` | 整局轨迹进入 TensorPipe 前的中心准入数 |
| `replay_push_permit_poll_interval_seconds` | `float` | - | `0.1` | 未拿到准入时的重试间隔，范围 `[0.01, 60]` |
| `replay_push_permit_lease_timeout_seconds` | `float` | - | `120` | 准入租约超时（秒），范围 `[1, 86400]` |
| `replay_push_chunk_bytes` | `int` | - | `31457280` | 单次 push 切分包上限；`0` 关闭切分，否则约 `[1MiB, 512MiB]` |

`world_size` 常用 `branch:` 区分训练与评估采样进程数。

**配置示例：**

```toml
[torch_rpc_config]
world_size = "branch: train=4, eval=2"
init_address = "127.0.0.1:29500"
call_timeout_seconds = 60
num_worker_threads = 16
max_concurrent_replay_pushes = 1
```

---

## 14. `[agents.N]`

所有算法使用同一套 Agent 配置，填写成员、观测维度和业务动作定义。训练时，`[learners.N]` 通过 `name` 配对，框架按同名 Learner 选择默认 Agent。

| Learner `algo_type` | 默认 Agent |
| --- | --- |
| `qmix` | `rnn` |
| `mappo`、`coma` | `policy_rnn` |
| `maddpg` | `maddpg_mlp` |
| `ppo`、`a2c`、`dqn` | `sb3` |

需要选用兼容的替代网络时，可显式填写 `agent_type`，例如让 QMIX 使用 `q_rnn`。独立推理、固定策略或其他没有同名 Learner 的 Agent 须显式填写类型，网络型 Agent 还须提供 `model_file_path`。分阶段任务使用顶层同名 Learner 的默认类型，各阶段的 `learners` 列表决定谁参与本阶段更新。

各内置 Agent 共用 `get_actions(obs, agent_list, hidden_in)`，返回按成员索引的业务动作和下一步隐藏状态。初始隐藏状态由 `get_init_hidden_state()` 提供。调用方可通过命名参数传入动作掩码、探索系数、上一动作及评估或采样选项。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | - | 必填 | 与配对 Learner 的 `name` 一致 |
| `device` | `str` | - | `"cpu"` | `"cpu"` / `"cuda"` / `"auto"` 等 |
| `agent_type` | `str` | `rnn` / `rnn_ns` / `policy_rnn` / `mappo_rnn` / `q_rnn` / `maddpg_mlp` / `monte_carlo` / `sb3` | 随同名 Learner | 可选的 Agent 插件标识。无同名 Learner 时必填 |
| `action_defined` | `table?` | - | `None` | 一个智能体唯一的业务动作定义；由 `head_type × action_type` 表达四类空间，见 §14.2 |
| `multi_action_mode` | `str` | `single` / `combination` / `free` | `"single"` | 旧单头离散入口字段；使用 `action_defined` 时在定义内部配置 |
| `agent_names` | `list[str]` | - | `[]` | 该策略控制的环境实体（旧版单组入口） |
| `agent_groups` | `list[list[str]]` | - | `[]` | 无名二维分组入口：外层为组声明顺序，内层为组内固定网络槽位顺序。与 `agent_names` 互斥且必须恰好一个非空、所有组等宽、成员不得跨组重复；显式分组第一版仅支持 `rnn` 类型 Agent（可训练须配 `qmix` Learner），组宽即网络 `n_agents`，所有组共享同一套网络与 Learner。使用分组的 QMIX 科目还须让同名 Agent/Learner 显式声明一致的组状态维度 `state_shape`（不再要求等于 `env_config.state_dim`） |
| `agent_target_names` | `list[str]` | - | `[]` | 旧单头离散入口字段；使用 `action_defined` 时由定义的 `targets` 配置 |
| `obs_dim` | `int` | - | `0` | 单实体观测维度 |
| `hidden_dim` | `int` | - | `64` | 网络隐藏层 |
| `num_layers` | `int` | - | `1` | 循环层数 |
| `obs_last_action` | `bool` | - | `true` | 观测是否拼上一动作 |
| `obs_agent_id` | `bool` | - | 按类型 | 是否拼 agent id；`rnn` 默认 true，其余 false |
| `agent_output_type` | `str` | `q` / `pi_logits` / `continuous_action` | `"q"` | 分别表示动作价值、策略 logits 和连续 Actor 输出。配对 Learner 会同步输出语义 |
| `model_file_path` | `str?` | - | `None` | 固定策略权重；无 Learner 的网络型 Agent **必须**提供 |
| `fixed_mode` | `bool` | - | `false` | 对该 Agent 强制 `epsilon=0`、确定性动作，且局末不推入回放池 |

不要在 TOML 写运行期字段：`n_agents`、`n_actions`、`define_action`、`action_space_spec`。业务动作只写一个 `action_defined`；框架在 Agent 装配时自动生成动作表、连续边界、Gymnasium 空间和环境编解码器。`state_shape` 仅在使用 `agent_groups` 显式分组的科目需要显式声明（组状态维度配置真值），单组科目由运行期按环境回填。

### 14.1 `[agents.N.algo_config]`

网络/控制器结构。基类字段：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `use_net` | `bool` | - | `true`（蒙特卡洛为 `false`） | 是否神经网络 Agent |
| `agent` | `str` | - | 随 `agent_type` | 网络注册名，如 `rnn` / `rnn_ns` |
| `mac` | `str` | - | 随类型 | controller 注册名，如 `basic_mac` / `non_shared_mac` |
| `use_rnn` | `bool` | - | 随类型 | 网络是否使用 RNN；`rnn`/`rnn_ns` 为 true，蒙特卡洛和 SB3 为 false |
| `sharing_scope` | `str` | `all` / `none` | 随类型 | `all`=全部实体共享一套参数；`none`=每槽位独立 |
| `mask_before_softmax` | `bool` | - | `true` | softmax 前是否屏蔽不可用动作 |
| `action_selector` | `str?` | - | `None` | 项目侧 Agent 通常不设 |

自定义 Agent 的额外字段也写在这一子表。

### 14.2 `[agents.N.action_defined]`

每个 `[agents.N]` 只配置一个 `action_defined`。两个正交字段决定具体实现：

| `head_type` | `action_type` | 自动装配的实现 | 策略输出 |
| --- | --- | --- | --- |
| `single` | `discrete` | `DiscreteActionSpace` + `ActionTable` | 一个整数索引 |
| `multi` | `discrete` | `MultiDiscreteActionSpace` + 每头独立 `ActionTable` | 按头排列的整数向量 |
| `single` | `continuous` | `ContinuousActionSpace` | 一个归一化浮点向量 |
| `multi` | `continuous` | `MultiContinuousActionSpace` | 按头展平的浮点向量 |

`action_key` 是编号列表。每项的 `key` 同时作为业务动作名和动作头名，该项其余字段按 `action_type` 使用下面的离散或连续定义。父表只负责选择头数和数值类型：

| 字段 | 类型 | 枚举 | 默认 | 约束 |
| --- | --- | --- | --- | --- |
| `head_type` | `str` | `single` / `multi` | `"single"` | `single` 断言 `action_key` 恰好一项；`multi` 断言至少两项 |
| `action_type` | `str` | `discrete` / `continuous` | `"discrete"` | 决定所有 `action_key` 项使用离散表还是连续表；多头第一版不能混合 |
| `action_key` | `list[table]` | - | `[]` | 使用 `action_key.0`、`action_key.1` 编号声明；列表顺序决定多头策略向量布局 |

### 14.3 离散动作定义

当 `action_type="discrete"` 时，每个 `[agents.N.action_defined.action_key.K]` 使用下表。参数可以给出显式枚举，也可以给出带步长的范围；所有参数组合会自动展开成该头的 `ActionTable`。

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `key` | `str` | 必填 | 业务动作名，同时作为多头动作的头名；在整个列表中必须唯一 |
| `parameters` | `table` | `{}` | 编号参数表，路径为 `parameters.K` |
| `add_target` | `bool` | `false` | 是否为该动作增加目标选择 |
| `targets` | `list[str]` | `[]` | 可选目标名称；`add_target=true` 时使用 |
| `allow_null_targets` | `bool` | `false` | 是否允许选中空目标，仅在 `add_target=true` 时有效 |
| `allow_empty_action` | `bool` | `false` | 是否在该离散动作表中增加空动作 |
| `role` | `str?` | `None` | 业务角色标签 |
| `multi_action_mode` | `str` | `"single"` | `single` / `combination` / `free`，沿用旧动作表展开语义 |
| `strict_masks` | `bool` | `true` | 动作掩码长度是否必须和该头动作表完全一致 |

离散参数表：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `key` | `str` | 必填 | 参数名 |
| `values` | `list?` | `None` | 显式枚举值 |
| `min` | `float?` | `None` | 按范围枚举时的起点 |
| `max` | `float?` | `None` | 按范围枚举时的终点 |
| `step` | `float?` | `None` | 按范围枚举时的步长；使用范围时必须提供 |
| `unit` | `str?` | `None` | 描述单位 |

### 14.4 连续动作定义

当 `action_type="continuous"` 时，每个 `[agents.N.action_defined.action_key.K]` 使用下表。一个三维坐标是一个动作项下的三个连续参数，仍然只算一个头。

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `key` | `str` | 必填 | 业务动作名，同时作为多头动作的头名；在整个列表中必须唯一 |
| `parameters` | `table` | 必填 | 编号参数表，路径为 `parameters.K`，至少包含一个参数 |
| `role` | `str?` | `None` | 业务角色标签 |

连续参数表：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `key` | `str` | 必填 | 参数名 |
| `min` | `float` | 必填 | 有限物理下界 |
| `max` | `float` | 必填 | 有限物理上界，必须严格大于 `min` |
| `unit` | `str?` | `None` | 描述单位 |

连续参数禁止填写 `values` 和 `step`，也禁止 `add_target`、`targets`、`allow_null_targets`、`allow_empty_action` 及非默认 `multi_action_mode`。框架把物理值线性映射到策略侧 `[-1, 1]`，环境收到解码后的物理值。

### 14.5 四类配置示例

单头离散：

```toml
[agents.0]
name = "policy"
agent_type = "rnn"
agent_names = ["agent_0"]
obs_dim = "${obs_dim}"

[agents.0.action_defined]
head_type = "single"
action_type = "discrete"

[agents.0.action_defined.action_key.0]
key = "move"

[agents.0.action_defined.action_key.0.parameters.0]
key = "dir"
values = [0, 1, 2, 3]
```

多头离散：

```toml
[agents.0.action_defined]
head_type = "multi"
action_type = "discrete"

[agents.0.action_defined.action_key.0]
key = "move"

[agents.0.action_defined.action_key.0.parameters.0]
key = "dir"
values = [0, 1, 2, 3]

[agents.0.action_defined.action_key.1]
key = "jammer_mode"

[agents.0.action_defined.action_key.1.parameters.0]
key = "mode"
values = ["off", "search", "track"]
```

单头连续三维坐标：

```toml
[agents.0.action_defined]
head_type = "single"
action_type = "continuous"

[agents.0.action_defined.action_key.0]
key = "aim_at"

[agents.0.action_defined.action_key.0.parameters.0]
key = "x"
min = -1200.0
max = 1200.0
unit = "m"

[agents.0.action_defined.action_key.0.parameters.1]
key = "y"
min = -1200.0
max = 1200.0
unit = "m"

[agents.0.action_defined.action_key.0.parameters.2]
key = "z"
min = 0.0
max = 300.0
unit = "m"
```

多头连续：

```toml
[agents.0.action_defined]
head_type = "multi"
action_type = "continuous"

[agents.0.action_defined.action_key.0]
key = "velocity"

[agents.0.action_defined.action_key.0.parameters.0]
key = "vx"
min = -10.0
max = 10.0
unit = "m/s"

[agents.0.action_defined.action_key.0.parameters.1]
key = "vy"
min = -10.0
max = 10.0
unit = "m/s"

[agents.0.action_defined.action_key.1]
key = "aim_at"

[agents.0.action_defined.action_key.1.parameters.0]
key = "x"
min = -1200.0
max = 1200.0
unit = "m"

[agents.0.action_defined.action_key.1.parameters.1]
key = "y"
min = -1200.0
max = 1200.0
unit = "m"

[agents.0.action_defined.action_key.1.parameters.2]
key = "z"
min = 0.0
max = 300.0
unit = "m"
```

旧 `[agents.N.actions.M]` 仍会解析为单头离散动作，用于已有科目兼容；新科目统一使用 `action_defined`。动作定义只负责业务动作结构和编解码，所选 Agent/Learner 仍必须声明支持对应的动作空间类型。

---

## 15. `[learners.N]`

配置学习器：算法类型、通用训练超参，以及 `[learners.N.algo_config]` 中的算法专用项。`name` 必须与要训练的 Agent 相同。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | - | 必填 | 绑定 Agent |
| `algo_type` | `str` | `qmix` / `mappo` / `coma` / `maddpg` / `ppo` / `a2c` / `dqn` | `"qmix"` | Learner 插件标识 |
| `device` | `str` | - | `"cpu"` | 学习计算设备 |
| `common_reward` | `bool` | - | `true` | 公共团队奖励 |
| `standardise_returns` | `bool` | - | `false` | 标准化回报 |
| `standardise_rewards` | `bool` | - | `true` | 标准化奖励 |
| `grad_norm_clip` | `float` | - | `5.0` | 骨架梯度裁剪 |
| `learner_log_interval` | `int` | - | `1` | 日志间隔 |
| `gamma` | `float` | - | `0.99` | 折扣因子 |

不要写：`n_agents`、`n_actions`、`scheme`、`use_cuda`（运行期填充）。`state_shape` 仅在配对 Agent 使用 `agent_groups` 显式分组的科目需要显式声明，且必须与同名 Agent 的 `state_shape` 一致（组状态维度配置真值）。

所有 Learner 的 `algo_config` 都包含：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `lr` | `float` | - | 随算法 | 学习率 |
| `batch_size` | `int` | - | 随算法 | 一次学习的抽样量，单位由抽样策略决定，见 §10 |
| `learn_times_per_episode` | `int` | - | 随算法 | 每个学习令牌调用更新的次数 |
| `capacity` | `int` | 正整数 | `1000` | 该 Learner 同名 Agent 分区最多保留的 active episode 数；不同 Learner 可独立设置 |
| `min_replay_size` | `int?` | `0` 至 `capacity` | `None` | 学习前 active 回放池须达到的数量；单位与 `capacity` 相同，不计 rehearsal 存档。`None` 不增加等待条件；低于门槛时暂停学习令牌 |
| `agent_output_type` | `str` | `q` / `pi_logits` / `continuous_action` | 随算法 | 会同步到配对 Agent |
| `sample_strategy` | `str` | 同 `replay_buffer_config.strategy` | 随算法 | 覆盖回放池默认策略；含义见 §10 |

### 15.1 QMIX / IQL / VDN

| 字段 | 枚举 | QMIX 默认 | IQL 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `mixer` | `qmix` / `vdn` / 空 | `"qmix"` | `None`（VDN 为 `"vdn"`） | `qmix`=超网络混合；`vdn`=各 agent Q 求和；空=无 mixer（IQL） |
| `lr` | - | `1e-3` | `3e-4` | 学习率 |
| `target_update_interval_or_tau` | - | `200` | `200` | `>1` 硬更新间隔（局），`<=1` 软更新 tau |
| `hidden_dim` | - | `64` | `128` | 网络隐藏层；运行期会与配对 Agent 对齐 |
| `double_q` | - | `true` | `true` | 是否用 Double Q |
| `td_lambda` | - | `0.6` | `0.6` | TD(λ) |
| `mixing_embed_dim` | - | `32` | - | 仅 QMIX：mixer 内部嵌入维度 |
| `hypernet_layers` | - | `2` | - | 仅 QMIX：超网络层数 |
| `hypernet_embed` | - | `64` | - | 仅 QMIX：超网络隐藏层维度 |
| `sample_strategy` | 同 §10 | `recent_biased` | `recent_biased` | 覆盖回放池默认策略 |
| `batch_size` | - | `16` | `16` | 一次更新抽取的 episode 数 |
| `learn_times_per_episode` | - | `4` | `4` | 每个学习令牌更新次数 |

**配置示例：**

```toml
[learners.0]
name = "policy"
algo_type = "qmix"
gamma = 0.99

[learners.0.algo_config]
lr = 0.0005
batch_size = 16
learn_times_per_episode = 4
mixer = "qmix"
double_q = true
sample_strategy = "recent_biased"
```

### 15.2 MAPPO

| 字段 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- |
| `lr` | - | `1e-4` | 学习率 |
| `actor_lr` / `critic_lr` | 正数或省略 | 省略 | 原生 MAPPO 的 Actor / Critic 独立学习率；各自省略时回退到 `lr` |
| `target_update_interval_or_tau` | - | `0.01` | 软更新 tau（`<=1`） |
| `hidden_dim` | - | `128` | critic MLP 隐藏层 |
| `critic_type` | - | `"cv_critic"` | critic 注册名 |
| `eps_clip` | - | `0.1` | PPO 策略比率裁剪范围 |
| `epochs` | - | `4` | 同一批 on-policy 样本重复优化轮数 |
| `actor_epochs` / `critic_epochs` | 正整数或省略 | 省略 | 原生 MAPPO 每批 Actor / Critic 独立优化轮数；各自省略时回退到 `epochs` |
| `critic_warmup_updates` | 非负整数 | `0` | 前 N 次成功的 Learner update 只训练 Critic；Actor 权重和优化器状态保持不变。计数随学习快照保存和恢复 |
| `warmup_use_mc_returns` | `bool` | `false` | 仅在 Critic 预热期间用 `lambda=1` 的回报目标；预热结束后恢复 `gae_lambda`。真终止不 bootstrap，时间截断或按步窗口边界仍用末帧 V bootstrap |
| `entropy_coef` | - | `0.005` | 策略熵奖励系数 |
| `q_nstep` | - | `10` | n-step return 的步数 |
| `obs_individual_obs` | - | `true` | critic 是否拼接各 agent 局部观测 |
| `obs_last_action` | - | `false` | critic 是否拼接上一时刻联合动作 |
| `add_value_last_step` | - | `true` | 轨迹最后一步是否用 value 做 bootstrap |
| `obs_agent_id` | - | `true` | critic 是否拼接 agent id |
| `mask_before_softmax` | - | `true` | softmax 前是否屏蔽不可用动作 |
| `sample_strategy` | 同 §10 | `on_policy` | 覆盖回放池默认策略 |
| `batch_size` | - | `64` | 一次更新抽取的 episode 数 |
| `learn_times_per_episode` | - | `1` | 每个学习令牌更新次数 |

IPPO / IA2C / MAA2C / COMA 共用 actor-critic 公共项；相对基类默认差异如下（未列出的项与 `EpymarlActorCriticConfig` 相同：`lr=5e-4`、`target_update_interval_or_tau=0.01`、`hidden_dim=128`、`entropy_coef=0.001`、`q_nstep=5`、`obs_individual_obs=false`、`obs_last_action=false`、`obs_agent_id=true`、`add_value_last_step=true`、`mask_before_softmax=true`、`sample_strategy=on_policy`、`batch_size=10`、`learn_times_per_episode=1`）。

| 算法 | 覆盖字段与默认 | 说明 |
| --- | --- | --- |
| IA2C | `critic_type="ac_critic"` | 独立局部 V critic |
| MAA2C | `critic_type="cv_critic"`，`entropy_coef=0.01` | 集中式 V critic |
| IPPO | `learner=ppo_learner`，`lr=3e-4`，`critic_type="ac_critic"`，`eps_clip=0.2`，`epochs=4` | 独立局部 V + PPO clip |
| COMA | `learner=coma_learner`，`lr=3e-4`，`target_update_interval_or_tau=200`，`hidden_dim=64`，`critic_type="coma_critic"`，`entropy_coef=0.001`，`q_nstep=10`；另见下表 | 反事实优势 + 集中式动作价值 critic |

COMA 另有：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `critic_q_fn` | `"coma"` | critic Q 函数实现名 |
| `critic_baseline_fn` | `"coma"` | 反事实 baseline 实现名 |
| `critic_train_mode` | `"seq"` | critic 训练模式 |
| `critic_train_reps` | `1` | 每个 update 内 critic 重复训练次数 |

**配置示例：**

```toml
[learners.0]
name = "policy"
algo_type = "mappo"

[learners.0.algo_config]
lr = 1.0e-4
eps_clip = 0.1
epochs = 4
entropy_coef = 0.005
sample_strategy = "on_policy"
batch_size = 64
```

原生 MAPPO 支持分别配置 Actor 和 Critic，适用于已有策略配合新初始化的价值网络，
或两者需要不同优化速度的情况。例如，保留 `lr`、`epochs` 作为回退值，只覆盖
`critic_lr` 和 `critic_epochs`，即可让价值网络获得更多拟合量。省略新增字段时，
两者仍使用原来的 `lr` 和 `epochs`，并且不进行 Critic 预热。运行期动态修改 `lr`
只影响未设置独立学习率的优化器；将独立学习率重新设为 `None` 后恢复回退。

`critic_warmup_updates` 以成功执行的 `update()` 为单位，不按采样回合、优化器步数
或模型发布次数计数。预热会消费学习样本并更新 Critic，但不调用 Actor 优化器。
学习状态快照中的 `update_cnt` 决定剩余预热次数；只加载 Actor 权重不包含这项
进度。`warmup_use_mc_returns=true` 对完整终止轨迹给出 Monte Carlo 回报；对于
时间截断或轨迹片段，目标仍含 Critic 的末帧估值，不能视为不依赖价值网络的完整回报。

更新诊断增加以下数值项：

- `pre_approx_kl` / `post_approx_kl`、`pre_clip_fraction` / `post_clip_fraction`：
  整批有效决策在优化前、最后一次 Actor 优化之后，相对采样行为策略的近似 KL
  和裁剪比例。原有 `approx_kl`、`clip_fraction` 保留优化过程的统计口径，不能替代
  `post_*` 检查最终策略变化。
- `pre_log_ratio_mean/std/abs_max`、`post_log_ratio_mean/std/abs_max`：行为策略
  对数概率之差的均值、标准差和最大绝对值，可辅助发现少量极端比率。
- `raw_advantage_mean/std`、`return_mean/std`、`value_mean/std`：优势标准化前的
  尺度和回报、价值估计的尺度，不能只看标准化后的优势或回合奖励净和。
- `batch_value_explained_variance_pre/post`、`batch_value_mse_pre/post`：Critic 在
  本批目标上的优化前后拟合情况；这是训练批指标，不代表独立评估集上的泛化。
  `batch_value_explained_variance_valid` 表明回报方差是否足以计算有效解释方差，
  回报近似常量时不应把其数值当作拟合优劣证据。
- `mc_value_explained_variance_pre`、`mc_value_mse_pre`、`mc_return_mean/std`：
  对本批从 reset 开始、无 burn-in 且真正终止的完整轨迹，用原始奖励的折扣
  Monte Carlo 回报检查更新前的 V。这个诊断不依赖本次 GAE 目标，不改变学习
  目标；`mc_complete_samples` 为参与比较的有效决策数，`mc_value_explained_variance_valid`
  表明解释方差是否有效。截断和部分轨迹排除；`standardise_rewards=true` 时
  原始奖励与 V 单位不一致，暂不计算，计数和 valid 均为 `0`。
- `actor_updates` / `critic_updates`：本次 update 实际调用各优化器 `step()` 的
  次数，受独立 epochs 和实际 minibatch 数影响；`critic_warmup_active`、
  `mc_returns_used` 用数值 `0/1` 标记本次模式。

这些控制不自动改变采样节拍或奖励函数，也不保证固定 KL 上限。应结合采样版本
滞后、最终 `post_*` 和独立评估判断是否增加优化强度。

### 15.3 MADDPG

配置 `algo_type = "maddpg"` 后，同名 Agent 默认使用 `maddpg_mlp`。同一个 Learner 的
`agent_names` 定义联合观测和动作的固定顺序，每个槽位有独立的高斯 Actor 和集中式 Q Critic。
训练支持 Parallel 环境，以及同组同构的单头或多头连续动作。Actor 推理只读取本槽位的输入。

以下片段为两个智能体配置航向与速度两个连续动作头。根据实际环境设置 `obs_dim` 和 `state_shape`。

```toml
[agents.0]
name = "team"
agent_names = ["a", "b"]
obs_dim = 2
state_shape = 2
obs_last_action = false

[agents.0.algo_config]
exploration_std = 0.2

[agents.0.action_defined]
head_type = "multi"
action_type = "continuous"

[[agents.0.action_defined.action_key]]
key = "turn"
parameters = [{ key = "angle", min = -30, max = 30, unit = "deg" }]

[[agents.0.action_defined.action_key]]
key = "speed"
parameters = [{ key = "value", min = 0, max = 12, unit = "m/s" }]

[learners.0]
name = "team"
algo_type = "maddpg"
state_shape = 2
common_reward = false
standardise_rewards = false

[learners.0.algo_config]
lr = 0.0005
target_update_interval_or_tau = 0.01
sample_strategy = "uniform_steps"
batch_size = 32
```

`common_reward = false` 使用各智能体自身的奖励，设为 `true` 使用回放记录的团队奖励。
Actor 根据观测预测每个动作维度的均值和 `log_std`。Rollout、Actor 更新和目标动作使用高斯采样，
再经 `tanh` 约束到归一化范围，由动作空间换算为各头的业务单位。固定策略、评估和确定性采样使用 `tanh(mean)`。
`[agents.N.algo_config].exploration_std` 是 `tanh` 前高斯分布的初始标准差，默认 `0.2`。
采样时将 `log_std` 约束在 `[-5, 2]`，配置为 `0` 时从最小允许标准差 `exp(-5)` 起步。
DQN、QMIX 等 Q 策略继续使用 `epsilon` 探索日程。
[OpenAI 参考实现](https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py#L35-L64)

此前保存的连续 MADDPG 权重可加载，框架会保留均值网络，并按配置初始化新增的标准差输出层。旧版 EPyMARL 离散 MADDPG 模型需要重新训练。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `lr` | `0.0005` | Actor 和 Critic 的 Adam 学习率 |
| `hidden_dim` | `128` | Actor 和 Critic 的隐藏层宽度 |
| `target_update_interval_or_tau` | `0.01` | `(0, 1]` 内为目标网络软更新系数，大于 1 的整数为硬更新间隔 |
| `reg` | `0.001` | Actor 输出的均值与 `log_std` 的 L2 正则系数 |
| `sample_strategy` | `uniform_steps` | 从有效单步转移中均匀、有放回抽样，含义见 §10 |
| `batch_size` | `32` | 一次更新抽取的单步转移数 |

### 旧版 PAC / PAC-DCG 配置

PAC 特有项（仅非共享版本）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `critic_type` | `"pac_critic_ns"` | PAC critic |
| `state_value_type` | `"cv_critic_ns"` | 状态价值网络类型 |
| `initial_entropy_coef` | `20.0` | 熵系数起点 |
| `final_entropy_coef` | `0.01` | 熵系数终点 |
| `entropy_end_ratio` | `0.8` | 熵退火结束位置，相对 `t_max` 的比例 |
| `t_max` | `500500` | 熵退火时间尺度（环境步） |

PAC-DCG 另有：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `critic_type` | `"pac_dcg_critic_ns"` | 深度协调图 critic |
| `cg_edges` | `"full"` | 协调图边：`full`=全连接，也可给整数或边列表 |
| `cg_utilities_hidden_dim` | `None` | utility 网络隐藏层；空则沿用 `hidden_dim` |
| `cg_payoffs_hidden_dim` | `None` | payoff 网络隐藏层 |
| `cg_payoff_rank` | `None` | payoff 低秩分解秩；空表示不分解 |
| `duelling` | `false` | 是否用 dueling 结构 |
| `msg_anytime` | `true` | 消息传递是否 anytime |
| `msg_iterations` | `8` | 消息传递迭代次数 |
| `msg_normalized` | `true` | 是否归一化消息 |

### 15.4 单智能体 PPO（SB3）

| 字段 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- |
| `policy` | - | `"MlpPolicy"` | SB3 策略类名；向量观测用 `MlpPolicy` |
| `n_steps` | - | `2048` | SB3 占位 rollout 长度；离线训练按实际样本量重建 buffer |
| `n_epochs` | - | `10` | 同一批样本重复优化轮数 |
| `gae_lambda` | - | `0.95` | GAE 的 λ |
| `clip_range` | - | `0.2` | PPO 策略比率裁剪范围 |
| `clip_range_vf` | - | `None` | 价值函数裁剪范围；`None` 表示不裁 |
| `ent_coef` | - | `0.0` | 策略熵奖励系数 |
| `vf_coef` | - | `0.5` | 价值损失系数 |
| `max_grad_norm` | - | `0.5` | SB3 内部梯度裁剪，与骨架 `grad_norm_clip` 分开 |
| `normalize_advantage` | - | `true` | 是否对优势做标准化 |
| `target_kl` | - | `None` | KL 早停阈值；`None` 表示不早停 |
| `sample_strategy` | 同 §10 | `on_policy` | 默认 on-policy；科目示例也可能用 `on_policy_recent_steps` |
| `batch_size` | - | `10` | 框架从回放池抽的 episode 数 |
| `minibatch_size` | - | `256` | SB3 每次梯度更新使用的 transition 数 |
| `learn_times_per_episode` | - | `1` | 每个学习令牌更新次数 |

**配置示例：**

```toml
[learners.0]
name = "policy"
algo_type = "ppo"

[learners.0.algo_config]
lr = 3.0e-4
n_epochs = 10
clip_range = 0.2
sample_strategy = "on_policy"
batch_size = 10
minibatch_size = 256
```

### 15.5 A2C / DQN（SB3）

A2C 相对 PPO 少 clip，多 RMSprop 相关项：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `n_steps` | `5` | SB3 占位 rollout 长度 |
| `gae_lambda` | `1.0` | GAE 的 λ |
| `ent_coef` | `0.0` | 策略熵奖励系数 |
| `vf_coef` | `0.5` | 价值损失系数 |
| `max_grad_norm` | `0.5` | SB3 内部梯度裁剪 |
| `rms_prop_eps` | `1e-5` | RMSprop epsilon |
| `use_rms_prop` | `true` | `true`=原论文 RMSprop；`false`=Adam |
| `normalize_advantage` | `false` | 是否对优势做标准化 |
| `sample_strategy` | `on_policy` | 含义见 §10 |
| `batch_size` | `10` | 每次学习令牌抽的 episode 数 |

DQN 用框架回放池存 episode，再同步到 SB3 内部 transition buffer：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `buffer_size` | `100000` | SB3 内部 replay 的 transition 容量 |
| `learning_starts` | `1000` | 内部 buffer 至少积累这么多 transition 后才训练 |
| `tau` | `1.0` | 目标网络软更新系数；`1.0` 表示硬同步 |
| `train_freq` | `1` | SB3 训练频率占位；本框架手动训练，一般不用改 |
| `gradient_steps` | `1` | 每次更新调用 `DQN.train` 的梯度步数 |
| `train_batch_size` | `64` | 从 SB3 内部 buffer 抽的 transition 数 |
| `target_update_interval` | `1000` | 目标网络同步间隔 |
| `exploration_fraction` | `0.1` | SB3 内部 ε 衰减占位；执行侧 ε 仍由训练令牌控制 |
| `exploration_initial_eps` | `1.0` | SB3 内部 ε 起点占位 |
| `exploration_final_eps` | `0.05` | SB3 内部 ε 终点占位 |
| `max_grad_norm` | `10.0` | SB3 内部梯度裁剪 |
| `sample_strategy` | `recent_biased` | off-policy；含义见 §10 |
| `batch_size` | `16` | 从框架回放池抽的 episode 数 |

**配置示例：**

```toml
[learners.0]
name = "policy"
algo_type = "dqn"

[learners.0.algo_config]
lr = 1.0e-4
buffer_size = 100000
learning_starts = 1000
train_batch_size = 64
sample_strategy = "recent_biased"
batch_size = 16
```

---

## 16. 分阶段：`[stages.N]` 与 `[stage_task_config]`

配置课程阶段：谁参与更新、回放池清空/rehearsal、阶段结束是否落盘，以及对本阶段训练/Agent/Learner/AutoML 的覆盖。分阶段训练必填 `[stages.N]`。

阶段表**只允许**下列键，其它训练字段必须放进 `train_overrides` / `agent_overrides` / `learner_overrides` / `auto_ml_overrides`。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | - | 必填 | 阶段名，写入局/学习信息 |
| `learners` | `list[str]` | - | 全部 Learner | 本阶段参与更新的 Learner 名 |
| `clear_replay_buffer` | `bool` | - | `true` | 进入本阶段时是否清空 **active** 环；**不会**删除已有 rehearsal 存档 |
| `rehearsal_archive_ratio` | `float` | - | `0` | **离开本阶段时**从 active 冻结进 `rehearsal[<本阶段名>]` 的条数比例，取值 `[0, 1]`。基准是回放池 `capacity`：每个 Agent 目标归档条数 = `round(capacity × ratio)`，再与当时该 Agent active 实际条数取较小值。`0`=不归档。被选中的局从 active **移出**到该阶段存档，不是复制；后续阶段仍可读这些存档 |
| `rehearsal_archive_select` | `str` | `even_span` / `newest` / `uniform` | `"even_span"` | 在 active 时序队列上**怎么挑**要归档的那批局（挑满 `rehearsal_archive_ratio` 算出的条数）。`even_span`=从最早到最晚等间距铺开，覆盖整段历史；`newest`=只留队列最末（最新）那一段；`uniform`=在全部 active 局中均匀随机抽。仅在 `rehearsal_archive_ratio>0` 时有意义 |
| `rehearsal_sample_ratio` | `float` | - | `0` | **进入并训练本阶段时**，每个 off-policy 学习 batch 里来自 rehearsal 存档的比例，取值 `[0, 1]`。基准是本次学习的 `batch_size`：约 `round(batch_size × ratio)` 条来自各阶段 rehearsal 合计，其余来自当前 active。`0`=只用 active。仅对抽样策略 `recent_biased` / `uniform` / `uniform_steps` 生效；PPO/MAPPO 等 on-policy 策略忽略本项，始终只采 active |
| `save_replay_buffer` | `bool` | - | `false` | 阶段结束是否落盘回放池快照 |
| `train_overrides` | 子表 | - | `{}` | 覆盖本阶段 `train_config`（字段同 §6） |
| `agent_overrides.<name>` | 子表 | - | `{}` | 按 Agent `name` 覆盖骨架/`algo_config`（字段同 §14） |
| `learner_overrides.<name>` | 子表 | - | `{}` | 按 Learner `name` 覆盖骨架/`algo_config`（字段同 §15） |
| `auto_ml_overrides` | 子表/可选 | - | `None` | 本阶段 AutoML；未写或 `enabled=false` 表示本阶段关闭。字段同 §18 |

rehearsal 系列配置用于阶段切换时保留旧经验。归档比例按各 Learner 的 `algo_config.capacity` 分区计算，混采比例按学习 `batch_size` 计算。阶段 A 结束时 `rehearsal_archive_ratio=0.25` 冻结旧经验，阶段 B 的 `rehearsal_sample_ratio=0.25` 让 off-policy batch 约 25% 来自旧经验。

覆盖子表示例：

```toml
[stages.0.train_overrides]
max_episode = 500

[stages.0.train_overrides.env_metadata]
difficulty = 2

[stages.0.agent_overrides.policy]
hidden_dim = 128

[stages.0.learner_overrides.policy.algo_config]
lr = 0.0003

[stages.0.auto_ml_overrides]
enabled = true
```

`[stage_task_config]`：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `start_stage` | `int` | - | `0` | 从第几个阶段开始（0 起始） |
| `skip_stage_model_load_dir` | `str?` | - | `None` | `start_stage>0` 时的模型目录 |
| `skip_stage_model_version` | `str?` | - | `None` | 跳过入口的模型版本前缀 |
| `skip_stage_replay_path` | `str?` | - | `None` | 回放池快照；相对路径相对 `artifacts/samples` |
| `stage_model_versions` | `list[str]` | - | `[]` | **兼容旧配置**；仅有此项且 `start_stage>0` 时取 `[0]` 映射为 `skip_stage_model_version` |

Agent 列表各阶段必须相同；阶段只决定谁更新、超参和课程 metadata。无 Learner 的固定策略 Agent 靠 `model_file_path` + `fixed_mode`。出现 `[stages.N.auto_ml_overrides]` 且未 `enabled=false` 表示该阶段启用 AutoML。

**配置示例：**

```toml
[stages.0]
name = "main"
learners = ["policy"]
clear_replay_buffer = true
rehearsal_archive_ratio = 0.0
save_replay_buffer = false

[stage_task_config]
start_stage = 0
```

---

## 17. `[eval_config]`

配置独立评估或盘中评估：评估局数、模型路径、样本/GIF 输出、产物归档与采样相关字段。未写 `agent_model_path` 时通常沿用训练模型目录。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `max_episode` | `int` | - | `50` | 评估局数 |
| `sample_type` | `str` | `online` | `"online"` | 评估仅支持 `online` |
| `record_sample` | `bool` | - | `false` | 是否录制结构化样本 |
| `artifacts_archive_enabled` | `bool` | - | `true` | 评估任务正常结束后是否自动归档已生成产物 |
| `artifacts_archive_save_dir` | `str` | - | `artifacts/archives` | 任务归档根目录 |
| `result_save_dir` | `str` | - | `artifacts/results` | 评估奖励曲线等结果目录 |
| `render_video_dir` | `str` | - | `artifacts/recordings` | GIF 目录 |
| `render_gif_fps` | `int` | - | `15` | GIF 帧率 |
| `render_gif_frame_stride` | `int` | - | `1` | 隔多少环境步采一帧 |
| `render_gif_max_frames` | `int?` | - | `None` | GIF 最大帧数 |
| `agent_compose_type` | `str` | `order` / `decision` | `"order"` | `order`=按环境默认顺序步进；`decision`=由 Episode Agent 动作交给环境解释顺序 |
| `env_metadata` | `table` | - | `{}` | 对本局环境 `metadata` 的递归覆盖 |
| `sample_task_timeout` | `float` | - | `300` | 采样令牌无结果占用超时（秒） |
| `monte_carlo_method_config` | 子表/可选 | - | `None` | 样本读写配置；也可由顶层 `[monte_carlo_method_config]` 注入 |
| `deterministic` | `bool` | - | `true` | 是否使用确定性动作（argmax） |
| `agent_model_path` | `str` | - | `artifacts/models` | 模型根路径 |
| `agent_model_file` | `str?` | - | `None` | 单组 Agent 文件名 |
| `agent_model_files` | `table` | - | `{}` | `AgentConfig.name` → 文件名（可多组） |
| `output_dir` | `str` | - | `artifacts/samples` | 样本目录 |
| `output_prefix` | `str` | - | `"eval"` | 样本文件前缀 |
| `render_gif_enabled` | `bool` | - | `false` | 须显式开启才会录 GIF |
| `render_gif_episode_interval` | `int` | - | `1` | 局号整除该间隔才录 |
| `metric_outputs` | 子表 | - | `{}` | 逐局业务指标；见下 |

`[eval_config.metric_outputs.<name>]`（`name` 须为合法标识符）：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `info_path` | `str?` | - | `None` | 从 env `infos` 取值的路径 |
| `metadata_path` | `str?` | - | `None` | 从本局生效 metadata 取值 |
| `agent_reduce` | `str` | `sum` / `mean` / `min` / `max` / `latest` | `"mean"` | 多 agent 归约 |
| `episode_reduce` | `str` | `sum` / `mean` / `min` / `max` / `latest` | `"mean"` | 局内逐步归约；评估常用 `latest` |

`render_gif_enabled` 未开启时，仅改 `render_gif_episode_interval` 不会录制。

**配置示例：**

```toml
[eval_config]
max_episode = 50
sample_type = "online"
deterministic = true
agent_model_path = "${resource_dir}/models/demo"
output_dir = "${resource_dir}/samples"
output_prefix = "eval"
render_gif_enabled = false
artifacts_archive_enabled = true
result_save_dir = "${resource_dir}/results/demo"

[eval_config.metric_outputs.coverage]
info_path = "coverage"
episode_reduce = "latest"
```

---

## 18. `[auto_ml_config]`

可选段。配置 AutoML 全局默认：主指标、坍塌/毕业线、归档与 Optuna、搜索空间。科目里常写 `enabled = false` 表示整份配置关闭 AutoML。

TOML 段上的 `enabled` **只决定是否构造本对象**，不会写入配置类。对象内部运行开关是 `monitor_enabled`（默认 `true`）。

阶段出现 `[stages.N.auto_ml_overrides]` 且未 `enabled=false`，表示该阶段启用，并在全局默认上合并。编号表 `learner_search_space` / `env_metadata_search_space` / `metric_outputs` 若在 overrides 里出现，则**整表替换**，不与全局编号拼接。

各算法允许的参数 key、类型与默认范围由 [`ALGO_TUNABLE_PARAM`](../../src/scheduler/model.py) 注册，Learner 搜索空间可按下表收窄范围。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `subject_name` | `str` | - | 必填 | 科目名，参与 Optuna study 命名 |
| `monitor_enabled` | `bool` | - | `true` | 运行期监控开关；`false` 时不启动监控 |
| `poll_interval_episodes` | `int` | - | `10` | 每隔多少局检查一次 |
| `min_episodes_before_check` | `int` | - | `30` | 首次坍塌/毕业检查前至少完成的局数 |
| `min_episodes_between_restart` | `int` | - | `20` | 两次 restart 之间的最小间隔局数 |
| `metric_primary` | `str` | `reward_mean` / `reward_min_agent` / `reward_sum` / `stage_success_rate` | `"reward_mean"` | 主指标聚合方式 |
| `reward_window` | `int` | - | `50` | 计算主指标时回看的最近局数 |
| `target_metric` | `float?` | - | `None` | 毕业线；到 deadline 仍未达到则 trial 记为 PRUNED |
| `target_metric_deadline_episodes` | `int` | - | `0` | 毕业线截止局数；`<=0` 关闭 |
| `collapse_restart_ratio` | `float` | - | `0.35` | 主指标跌到本 trial 峰值的该比例以下视为坍塌；`<=0` 关闭 |
| `collapse_min_episodes` | `int` | - | `80` | 坍塌检测生效前至少完成的局数 |
| `archive_enabled` | `bool` | - | `true` | restart 时是否落盘归档 |
| `archive_root_dir` | `str` | - | `artifacts/automl/archives` | 本地归档根目录 |
| `storage_type` | `str` | `local` / `s3` | `"local"` | 归档后端 |
| `extra_complete_rounds` | `int` | - | `0` | 主路跑满后额外完整探索轮数 |
| `metric_outputs` | 编号/命名子表 | - | `{}` | 逐局业务指标提取；见下 |
| `business_metric` | 子表/可选 | - | `None` | 将多个逐局指标合成 Optuna 主指标；见下 |
| `optuna` | 子表 | - | 见下 | Optuna 采样与剪枝 |
| `learner_search_space` | 编号表 | - | `[]` | Learner 超参搜索项 |
| `env_metadata_search_space` | 编号表 | - | `[]` | 环境 metadata 搜索项 |
| `seed_search` | 子表/可选 | - | `None` | 全局运行种子搜索；段级 `enabled=false` 可不构造 |

`[auto_ml_config.optuna]`：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `direction` | `str` | `maximize` / `minimize` | `"maximize"` | 优化方向 |
| `load_if_exists` | `bool` | - | `true` | 已有 sqlite study 时是否续跑 |
| `sampler_name` | `str` | `tpe` / `random` / `cmaes` | `"tpe"` | 采样器 |
| `sampler_n_startup_trials` | `int` | - | `10` | 冷启动随机 trial 数 |
| `pruner_name` | `str` | `median` / `hyperband` / `none` | `"median"` | 剪枝器 |
| `pruner_n_startup_trials` | `int` | - | `5` | MedianPruner：前 N 个 trial 不参与中位数比较 |
| `pruner_n_warmup_steps` | `int` | - | `30` | MedianPruner：单 trial 局数低于此值不剪枝 |
| `pruner_min_resource` | `int` | - | `1` | Hyperband：最低 rung 资源 |
| `pruner_max_resource` | `int` | - | `200` | Hyperband：最高 rung 资源 |
| `pruner_reduction_factor` | `int` | - | `3` | Hyperband：相邻 rung 资源递增因子 |

`[auto_ml_config.learner_search_space.N]`：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `agent_name` | `str` | - | `""` | 目标 Learner / Agent 名 |
| `path` | `str` | - | `""` | 相对 `LearnerConfig` 的字段路径，如 `algo_config.lr` |
| `low` / `high` | `number?` | - | `None` | 连续/整数范围；空则沿用框架注册默认 |
| `log` | `bool?` | - | `None` | 是否对数采样 |
| `step` | `number?` | - | `None` | 离散步长 |
| `choices` | `list?` | - | `None` | `categorical` 参数的离散候选值 |

`[auto_ml_config.env_metadata_search_space.N]`：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `path` | `str` | - | `""` | metadata 内路径 |
| `type` | `str` | `float` / `int` / `categorical` | 必填 | `float`/`int` 需 `low`/`high`；`categorical` 需 `choices` |
| `low` / `high` / `log` / `step` / `choices` | 同左 | - | `None` | 分布范围 |

`[auto_ml_config.seed_search]`：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `low` | `int` | - | `0` | 种子下界 |
| `high` | `int` | - | `2147483647` | 种子上界 |

`[auto_ml_config.metric_outputs.<name>]`（`name` 须为合法标识符）：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `info_path` | `str?` | - | `None` | 从 env `infos` 取值 |
| `metadata_path` | `str?` | - | `None` | 从本局生效 metadata 取值 |
| `agent_reduce` | `str` | `sum` / `mean` / `min` / `max` / `latest` | `"mean"` | 多 agent 归约 |
| `episode_reduce` | `str` | 同上 | `"mean"` | 局内逐步归约 |

`[auto_ml_config.business_metric]`：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `enabled` | `bool` | - | `false` | 是否启用表达式业务主指标 |
| `name` | `str` | - | `""` | 指标名 |
| `expression` | `str` | - | `""` | 组合 `metric_outputs` 名的安全表达式 |
| `aggregation` | `str` | `mean` / `min` / `max` / `latest` | `"mean"` | 窗口聚合 |
| `window` | `int` | - | `50` | 聚合窗口局数 |
| `target` | `float?` | - | `None` | 可选业务目标值 |
| `deadline_episodes` | `int` | - | `0` | 业务目标截止局数；`<=0` 关闭 |

顶层 `[auto_ml_search_space]` 已删除。Optuna SQLite 路径不由 TOML 配置。

**配置示例：**

```toml
[auto_ml_config]
enabled = false
subject_name = "demo"
monitor_enabled = true
poll_interval_episodes = 10
metric_primary = "reward_mean"
reward_window = 50
archive_enabled = true
storage_type = "local"

[auto_ml_config.optuna]
direction = "maximize"
sampler_name = "tpe"
pruner_name = "median"

[auto_ml_config.learner_search_space.0]
agent_name = "policy"
path = "algo_config.lr"
low = 1.0e-4
high = 5.0e-4
log = true

[auto_ml_config.env_metadata_search_space.0]
path = "reward_weights.navigation"
type = "float"
low = 0.05
high = 0.50

[auto_ml_config.seed_search]
enabled = true
low = 0
high = 2147483647
```

---

## 19. `[log_config]`

配置训练、评估期间日志的保存位置、输出方式和详细程度。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `log_dir` | `str` | - | `{cwd}/logs` | 运行日志保存目录 |
| `log_file_name` | `str` | - | `"rl.log"` | 日志基础文件名，实际追加任务 ID；同任务多进程共写该文件，约每 10 MiB 切分 |
| `console_log_format` | `str` | - | `%(asctime)s \| %(filename)s:%(lineno)d \| %(levelname)s \| %(message)s` | 控制台 Formatter 格式串 |
| `file_log_format` | `str` | - | `%(asctime)s \| %(task_id)s \| %(name)s \| %(filename)s:%(lineno)d \| %(levelname)s \| %(message)s` | 文件 Formatter 格式串 |
| `log_level` | `str` | - | `"INFO"` | 日志级别 |
| `enabled_handlers` | `list[str]` | `console` / `file` | `["console", "file"]` | 可只开其中一个 |
| `log_debug_enable` | `list[str]` | - | `[]`（加载后自动补 `"show"`） | debug 白名单；未写 `"show"` 时会自动加入 |

多个进程使用相同的任务 ID、日志目录和基础文件名，且目录位于同一共享文件系统时，日志会合写到同一文件。启用任务产物归档后，任务／阶段结束时将共享日志和已生成分片复制到 `<artifacts_archive_save_dir>/<task_id>/logs/<阶段名>/`；源文件继续保留。仅启用控制台输出时不会生成文件日志；关闭产物归档时日志只保存在 `log_dir`。

**配置示例：**

```toml
[log_config]
log_dir = "logs"
log_file_name = "demo.log"
log_level = "INFO"
enabled_handlers = ["console", "file"]
```

---

## 20. `[artifacts_manager_config]`

配置产物台账（ArtifactsManager）模块自身的保存位置。产物归档的开关和归档根目录仍由 `[train_config]` / `[eval_config]` 的 `artifacts_archive_enabled` / `artifacts_archive_save_dir` 控制（见 §6、§17）；本表只在不满足于归档目录默认值、想让台账与其余运行产物分开存放时才需要配置。

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `save_dir` | `str` | - | `artifacts/archives` | 产物台账根目录；归档与远端输送缓存（`_remote_cache/`）都写在它下面 |

整张表完全可选：不写 `[artifacts_manager_config]` 时台账直接沿用 `artifacts_archive_save_dir`，两者都写时以本表 `save_dir` 为准。现有科目 TOML 无需为引入本表做任何改动。

**配置示例：**

```toml
[artifacts_manager_config]
save_dir = "${resource_dir}/archives"
```

---

## 21. 存储子表

`ReplayBuffer` / `ModelManager` / `MonteCarlo` / `AutoML` 等在 `storage_type = "local"` 时，`[*.storage]` 使用：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `data_path` | `str` | - | 随模块：samples / models / automl archives | 本地数据根目录 |
| `scratch_subdir` | `str` | - | `"replay_scratch"` | mmap 等工作目录相对 `data_path` 的子目录；目前主要 ReplayBuffer 使用 |

`storage_type = "s3"` 时：

| 字段 | 类型 | 枚举 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `endpoint_url` | `str` | - | `http://localhost:9000` | S3/MinIO 服务地址 |
| `access_key` | `str` | - | `minioadmin` | Access Key |
| `secret_key` | `str` | - | `minioadmin` | Secret Key |
| `bucket_name` | `str` | - | 按模块：`models-pool` / `replay-buffer` / `automl-archives` / `monte-carlo-samples` | 对象存储桶名；不填时按模块给默认桶 |

AutoML 的 `[auto_ml_config.storage]`：local 模式会忽略 S3 专用字段，便于预置 endpoint 后切换。

**配置示例：**

```toml
[replay_buffer_config.storage]
data_path = "artifacts/samples"
scratch_subdir = "replay_scratch"
```

---

## 22. 不要写进 TOML 的内容

这些由运行期自动填写，不要写进科目 TOML：

- Agent/Learner：`n_agents`、`n_actions`、`scheme`、`use_cuda`、`define_action`（`state_shape` 仅显式分组科目需要声明，见第 14 节）
- 训练/阶段上的完整 `auto_ml_config` 对象本身（用顶层 `[auto_ml_config]` + overrides）
- Torch RPC 的 `RANK`
- `SampleLearningTasksConfig` / `SampleLearningEvalConfig`（样本学习入口单独注入，不是训练 TOML 段）
- mmap 行预算 / `max_steps_per_agent`（按各 Learner 的 `algo_config.capacity`、阶段 `rehearsal_archive_ratio` 和 `env.max_step_per_episode` 推算）

已禁止：

- `[rpc_config]` → 改用 `[grpc_config]` 或 `[torch_rpc_config]`
- 顶层 `[auto_ml_search_space]` → 改到 `auto_ml_config` / `auto_ml_overrides` 的编号表
- 阶段根上直接写 `max_episode` 等训练字段 → 必须 `[stages.N.train_overrides]`

---

## 23. 最小骨架

串行训练至少需要 design 变量、环境、一个 Agent、一个 Learner、一份 `train_config`。分阶段再加 `[stages.N]`。

```toml
[design]
task_name = "demo"
task_id = ""
root_dir = "{{project_root}}"
resource_dir = "${root_dir}/artifacts"
obs_dim = 8
state_dim = 16

[train_config]
max_episode = 100
model_save_dir = "${resource_dir}/models/demo"
model_load_dir = "${resource_dir}/models/demo"
result_save_dir = "${resource_dir}/results/demo"

[env_config]
sense_name = "demo_env"
use_sim = false
max_step_per_episode = 50
state_dim = "${state_dim}"
agent_names = ["agent_0"]

[replay_buffer_config]

[agents.0]
name = "policy"
agent_names = ["agent_0"]
obs_dim = "${obs_dim}"

[agents.0.action_defined]
head_type = "single"
action_type = "discrete"

[agents.0.action_defined.action_key.0]
key = "move"

[agents.0.action_defined.action_key.0.parameters.0]
key = "dir"
values = [0, 1, 2, 3]

[learners.0]
name = "policy"
algo_type = "qmix"

[learners.0.algo_config]
capacity = 256
lr = 0.0005
batch_size = 16

[stages.0]
name = "main"

[eval_config]
max_episode = 5
deterministic = true
```

字段以 `src/config.py`（及对应 `*AlgoConfig`）为准；本说明与代码不一致时以代码为准。


### 按模型版本同步采样与学习

`train_config.versioned` 默认 `true`。设为 `false` 时，分布式采样进程领到令牌后加载当前
`champion`，Learner 按原回放策略取样。旧的 `sample_dispatch_*` 配置已被移除。

`versioned = true` 适用于在线 `learn_control` 与 `stage_learn`。
TaskManager 给主路采样令牌绑定模型版本。上一批收齐并交给 Learner 后，
下一批可先领票，等 ModelManager 发布指定版本再采样。每个参与的
Learner 需设置 `learn_times_per_episode = 1`，模型发布后推进各自的版本。
共享 rollout 的令牌按 Agent 携带版本，同一局可使用不同轮次的模型。
若某个 Learner 在 rollout 结束前已换版，回放池保留仍匹配当前版本的分区。

批次以各 Learner 的 `algo_config.sample_strategy` 和 `batch_size` 为准。

每次成功发布学习后的模型时，发布日志同时显示 `learner`、`model_version`、
`sample_episode_num` 和 `learn_id`；同样的信息写入模型元数据（另含阶段和 generation）。
其中 `sample_episode_num` 是学习令牌领取时的采样完成水位，不是保证该批样本均来自该局。
模型版本仍由各 Learner 独立生成，同一个采样水位可以对应多个不同版本。

- 滚动（`on_policy`、`on_policy_recent_steps`、`recent_biased`、`sliding_window`、
  `uniform`）先收集到回放池能组满一批，之后每个版本补一张有有效数据的采样票。
  学习批量保持不变，并包含本版的新数据。历史窗口与滑动游标可跨版本；
  近期偏好和均匀抽样的 rehearsal 配额最多占用批量减一，且受实际存档数量限制。
- `uniform_steps` 有放回抽样，只需一条有效步即可组成批次；每版仍等待当前版本的新轨迹，但不强制批次包含新轨迹，以保持均匀分布。
- 游动（`on_policy_cursor`、`on_policy_cursor_steps`）每版分别凑满一批，
  按当前版本未消费的轨迹数或有效 transition 数发票。超额数据可留作历史回放。
- `time_slice` 每版等待一条可切片轨迹，`batch_size` 表示局内窗口长度。

`train_config.allow_sample_overlap = true` 在版本模式下允许滚动 Learner 的
采样并发。空闲 worker 使用当时已发布的版本，回传时按实际版本入库，
同阶段同代次的旧版轨迹也可进入滚动回放。窗口凑满后，每条仍在池中的
新入库轨迹推进一次学习窗口。同一批中可以同时出现多个采样版本。
游动和局内切片的学习批次仍按当前版本取样。

学习日志同时输出 `Learner metrics: { ... }` JSON 行，包含完整精度的全部学习指标、
Learner、批量、局数和耗时。控制台表格最多显示前 12 项指标，作为阅读摘要；
需要分析小 KL、价值解释方差或额外诊断时，以 JSON 行中的 `metrics` 为准，
避免四位小数格式把小数值显示为零。

### 阶段仿真实例配置

`TaskConfig.sim_metadata` 默认 `{}`，表示本任务不修改实例配置。原生分布式阶段可使用：

```toml
[stages.1.train_overrides.sim_metadata]
environment = ["PROCESS_PARAMETER=12"]
```

该字典覆盖 `SimInstancesConfig.metadata`，不覆盖 `SimConfig.metadata`。
中心阶段角色在准备门闩内以初始实例 metadata 为基线递归覆盖字典，列表整体替换，
再调用 `SimManager.instances.configure({"metadata": target})`。需要保留的其他环境变量
应完整写入列表。非空覆盖要求在线托管实例和 dodo-sim>=0.4.1；外部 INI 或离线任务报错。
Compose driver 配置相同不重建，配置变化后等待容器重建就绪再开放新阶段。
采样进程在新一局前校验租约，重新连接已换代的实例；阶段切换不重建 SimManager。
其他 driver 保留各自 configure 语义，不自动继承 Compose 的重建行为。
