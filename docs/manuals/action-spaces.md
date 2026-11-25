# 四类动作空间与执行契约

业务侧为每个智能体声明一个 `ActionDefined`。装配过程根据 `head_type`（`single`/`multi`）和 `action_type`（`discrete`/`continuous`）自动选择 `DiscreteActionSpace`、`MultiDiscreteActionSpace`、`ContinuousActionSpace` 或 `MultiContinuousActionSpace`。业务代码不创建 `ActionTable` 或 `ActionSpace`。一个头是同一智能体一步内同时执行的一个业务分量，一个头可包含多个连续参数；一个三维瞄准点是一个三维连续头，两台独立智能体不因此成为两个头。

| 空间 | 业务值 | 算法数值 / Gymnasium | 上一动作编码 |
| --- | --- | --- | --- |
| 单头离散 | 既有 `Action` 或 `CompositeAction` | 整数 / `Discrete(n)` | n 维 one-hot |
| 多头离散 | `MultiHeadAction` | int64 头索引向量 / `MultiDiscrete(nvec)` | 各头 one-hot 拼接 |
| 单头连续 | `ContinuousAction` | float32 向量 / `Box([-1,1], (d,))` | d 维归一化值 |
| 多头连续 | `MultiHeadAction` | float32 展平向量 / `Box([-1,1], (sum(d),))` | 各头归一化值拼接 |

`CompositeAction` 仍由一个旧离散索引选择一组指令。`MultiHeadAction` 没有全局索引，各头同时执行；多头离散只维护每个头的动作表，不构造头之间的笛卡尔积。v1 要求多头至少两个头，所有头同为离散或同为连续，不支持混合、条件头与跨头联合约束。

## 定义与配置

```python
from dodorl_sdk import (
    ActionDefined, ActionParameterDefined, AgentConfig,
)

definition = ActionDefined(
    head_type="multi",
    action_type="continuous",
    action_key=[
        {"key": "velocity", "parameters": [
            ActionParameterDefined("vx", min=-10, max=10, unit="m/s"),
            ActionParameterDefined("vy", min=-10, max=10, unit="m/s"),
        ]},
        {"key": "aim_at", "parameters": [
            ActionParameterDefined("x", min=-1200, max=1200, unit="m"),
            ActionParameterDefined("y", min=-1200, max=1200, unit="m"),
            ActionParameterDefined("z", min=0, max=300, unit="m"),
        ]},
    ],
)
config = AgentConfig(
    name="device", agent_type="monte_carlo", agent_names=["device_0"],
    obs_dim=512, action_defined=definition,
)
```

`ActionDefined.action_key` 是列表，每项的 `key` 同时作为业务指令名和动作头名，列表顺序决定策略向量布局。`head_type="single"` 要求恰好一项，`multi` 要求至少两项，所有 `key` 必须唯一。连续参数的 `min/max` 是物理边界，不填写 `values/step`；连续动作不能声明目标。旧 `StepActionDefined` 仅保留为兼容别名，新业务代码不再使用。完整四象限示例见 `sdk/demo_action_spaces.py`。

`Agent` 初始化时自动把 `action_defined` 装配为对应动作空间，生成动作表或连续边界、Gymnasium 空间、上一动作编码和环境使用的业务编解码器。`inject_env_dims` 会把同一个空间对象按 AgentID 注入环境，因此科目环境直接接收 `Action`、`ContinuousAction` 或 `MultiHeadAction`。底层 `action_space_spec` 和旧 `define_action` 仅保留历史配置兼容，不能和 `action_defined` 同时使用。

没有新配置时继续使用旧定义；保留 `Action`、动作表索引、`combination`/`free` 模式和旧 JSON 格式。旧适配器接受环境提供的超集掩码并截取当前表的范围；显式创建的新离散空间要求掩码长度精确匹配。所有路径遇到没有可用动作均报错，不随机执行被禁用的动作。

## 空间接口与数值语义

- `sample(rng=None, mask=None)` 返回业务对象；可传 NumPy `Generator` 或 `np.random`，同一个多头采样共享 RNG。
- `validate(action)` 校验动作类型、名称、参数键、维度、取值与边界，不修改动作。
- `encode(action)` 把物理业务对象转成算法数值；`decode(value)` 执行反向转换，不进行随机采样。
- `encode_previous_action(action)` 返回 float32 编码，`None` 返回全零向量。
- `layout` 包含 `kind`、`head_count`、`category_counts`、`continuous_dim`、`value_dim`、`previous_action_dim`；多头额外含各头名称和切片。
- `to_gym_space()` 返回算法坐标的 Gymnasium 空间；`to_dict()` / `ActionSpace.from_dict()` 保存和恢复空间定义。

连续动作对象始终保存物理值。编码使用 `u = 2 * (x-low)/(high-low) - 1`，解码使用相应的线性映射。空间定义保留参数名、物理上下界、可选单位；所有上下界有限且严格递增，包含端点。NaN/Inf、布尔值、字符串、维度错误、缺失/多余参数或越界都会报错；不裁剪、不取整、不吸附。

连续空间不接收离散掩码。单头离散掩码是二进制列表；多头离散掩码是覆盖所有命名头的字典，每个头至少有一个合法选项。头声明顺序决定编码顺序，不依赖动作字典或掩码字典的插入顺序；头名不得重复。

## Agent、环境与记录

Agent 暴露 `action_space`（业务编解码器）、`action_layout`、`action_dim`（Gym 空间）。`action_table` 仅在单头离散时存在有效对象，其他空间为 `None`。`AgentInputShape.action_dim` 保留旧离散类别数含义，其他空间为 0；新代码读取 `action_layout`，不能把连续数值维度误当作类别数。

`RandomAgent`、调度器的两条随机采样路径均使用统一空间接口。自定义规则 Agent 通过类属性声明能力：

```python
supported_action_spaces = ("discrete", "multi_discrete", "continuous", "multi_continuous")
```

规则策略通过 `self.action_space.decode(...)` 构造业务对象；其 `get_actions` 仍返回 `(agent_id -> business_action, hidden_out)`。公共 `record_action_warp` 装饰器会校验动作。环境通过 `inject_env_dims` 获取 Gym 空间和业务编解码器；环境接收物理业务值，坐标到角度等业务转换由具体科目实现。

Parallel/AEC 的内存执行示例分别是 `ActionSpaceCounterParallelEnvironment` 和 `ActionSpaceCounterAECEnvironment`（`envs.sense.action_space_examples`），两者都接收 `agent_id -> ActionSpace` 映射，用所有物理参数之和更新计数器。它们用于演示完整业务对象传递，不赋予参数实际控制含义。

`space.record(action)` 产生可 JSON 往返的记录，包含版本、空间定义、空间签名、算法数值和原始物理业务对象；`ActionSpace.restore(record, expected_space=None)` 同时校验签名、空间顺序和物理值与编码的一致性。保留原始物理对象可避免 float32 编码往返损失精度，也保留业务 metadata。当前采用自包含记录，后续可以在不改变动作值语义的前提下将重复的空间定义提取为文件头。

环境录制与 Transition 均保持旧单头离散表示；新空间保存上述版本化记录。`action_from_dict()`、环境 `_actions_from_record()` 可以恢复新记录、动作值及历史离散/组合指令。新 Transition 第一帧的上一动作记录为 `None`，旧离散仍为 0。`EpisodeTransition(action_spaces={agent_id: space})` 和 `SAEpisodeTransition(action_space=space)` 支持直接使用；调度器自动绑定。

## 算法边界与验收

Agent/Learner 用 `supported_action_spaces` 声明能力，默认仅 `("discrete",)`，不匹配时在初始化报出算法名、请求空间和支持列表。内置 `policy_rnn` + `mappo` 支持全部四类动作空间，使用独立的循环 MAPPO batch 和分布实现。`rnn`/`rnn_ns` + QMIX、其他 EPyMARL 算法与 SB3 保持原有单头离散能力；EPyMARL/SB3 张量适配器仍拒绝其他空间。

MAPPO 使用各头 Categorical 或展平的 tanh Gaussian；多头联合 log probability 对头/动作维度求和。采样时保存行为策略概率、当前 mask 和温度，经通用 ReplayBuffer 传输后训练。连续输出位于归一化 [-1,1] 内，并在 log probability 中计算 tanh Jacobian。单头/多头连续共享展平分布，各头按动作空间编解码还原。详见 [MAPPO 配置说明](toml配置手册.md#152-mappo)。

```bash
.venv/bin/python -m pytest tests/test_action_spaces.py -q
PYTHONPATH=src:sdk/src .venv/bin/python sdk/demo_action_spaces.py
```

验收覆盖四类空间、连续端点与非法值、多头顺序和掩码、无联合枚举、上一动作、JSON/签名、配置兼容、Parallel/AEC 随机和规则执行、环境与轨迹记录恢复、算法能力拒绝，以及现有离散动作和训练回归。
