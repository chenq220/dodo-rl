# 四区域训练过程基准资产

测试入口：`tests/test_baseline_training_partitions.py`。原四机
`tests/test_baseline_training.py` 保留不变。

- `threat_agent_6f548e77.pth`：威胁专训任务 `6f548e77` 的最终固定 Agent，
  从 `final_6f548e77_roam_0_dynamic_threats_agent_net.pth` 冻结复制。
  SHA256：`c88b839cd24ea6913e4040b20673bc9b528126696cdb6218937e4a68646cfc5a`。
  它是测试所需资产，不依赖可清理的 `artifacts/`，不得自动替换成最新模型。
- `expected_trace.json` 和 `expected_trace_linux.json`：seed=42 的 CPU 单线程真实串行训练过程，
  2 个 search 物理局加 2 个 evade 物理局。每局最多600步、每物理局四条组轨迹。
  含奖励、组长度、观察/状态/动作与掩码轨迹摘要、环境指标及实际QMIX更新统计。
  运行过程必须完整匹配其中一个平台基准；离散轨迹摘要仍严格一致。

所有实验参数在测试中用配置类直接声明，不读取科目TOML。为保证短课程执行真实更新，
回放容量固定8条组轨迹、batch固定4条，每阶段从第一局就更新网络；epsilon衰减保持2250/750物理局，
不会因只跑4局而变成快速衰减课程。回放张量使用临时目录中的lazy_memmap。

运行比对（会执行完整4局）：

本机定向验证约4分钟，两个阶段每局均更新网络，共6次QMIX更新。

```bash
.venv/bin/python -m pytest tests/test_baseline_training_partitions.py -q
```

显式录制（仅输出标记包围的JSON，**不会覆盖**现有基准）：

```bash
DODO_RECORD_BASELINE=1 .venv/bin/python -m pytest \
  tests/test_baseline_training_partitions.py::test_baseline_training_partitions -s -q
```

只有有意修改训练语义并审阅差异后才更新平台基准；Linux 数值记录到 `expected_trace_linux.json`，原基准保留。
正常测试比较数值统计（允许很小的浮点误差），离散轨迹摘要要求一致；
所有连续数值录制时四舍五入到小数点后10位。观察和状态记录逐步均值及整体统计，
不使用浮点字节哈希；数值比较使用相对容差1e-7、绝对容差1e-10，以兼容CPU计算舍入差异。
动作、动作掩码和终止标志的离散轨迹哈希仍严格比较。
本基准固定CPU单线程、Python哈希种子与模型初始化；更换底层数值实现或硬件时仍需核对差异，
不能用自动重录掩盖变化。
