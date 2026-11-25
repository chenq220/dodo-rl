# 塑形函数专项测试

用 pygame 模拟单一场景测试 `envs/reward/shaping.py` 里的塑形函数。

## 命名约定

**一游戏一环境一配置**：每个小游戏独立 `{name}_game.py` + `{name}_env.py` + `{name}.toml`，不在同一 Env 里切换多种塑形场景（例如 d=0 与 d>0 应拆成两个 game/env）。

| 层 | 路径 / 命名 |
|---|---|
| Game | `envs/games/shaping_test_games/{name}_game.py` → `{Name}Game` |
| Env | `envs/sense/{sense_name}_env.py` → `{SenseName}Env`，`use_sim=false`，`self.game` |
| State | `envs/state/{sense_name}_state.py` → `{SenseName}State` |
| 注册 | `envs/__init__.py` 与 `envs/sense/__init__.py` 的 `ENV_MAPPING` |
| 配置 | `subjects/shaping_test/{sense_name}.toml` |

`sense_name`、`task_name`、TOML 文件名、checkpoint 子目录保持一致（如 `track_seek`）。

## 游戏说明

| 名称 | 游戏位置 | 环境位置 | 塑形函数 | 说明 |
|---|---|---|---|---|
| track_seek | `envs/games/shaping_test_games/target_seek_game.py`（`TargetSeekGame`） | `envs/sense/track_seek_env.py`（`TrackSeekEnv`） | `region_visible_target_seek_shaping`（d=0） | 640×480 平面，蓝点四向移动追踪一个或多个红点，目标可静态或随机移动。 |
| region_activity | `envs/games/shaping_test_games/region_activity_game.py`（`RegionActivityGame`） | `envs/sense/region_activity_env.py`（`RegionActivityEnv`） | `region_activity_shaping` | 640×480 平面，每局随机位置与大小的六边形目标框（面积≤屏幕 1/4），蓝点四向移动且不可停止，需在框内活动；目标框可静态或平移。 |
| threat_region | `envs/games/shaping_test_games/threat_region_game.py`（`ThreatRegionGame`） | `envs/sense/threat_region_env.py`（`ThreatRegionEnv`） | `region_threat_shaping` + `region_visible_target_seek_shaping`（d=0） | 640×480 平面，蓝点生于北侧、南侧红点作 seek 引导（权重为 threat 的 1/3），场中宽 1/3 屏、高 24px 的横向威胁区域需绕开；`threat_speed` 控制横向速度（0 静止）；观测 14 维含威胁矩形四顶点相对位移。 |
| target_selection | `envs/games/shaping_test_games/target_selection_game.py`（`TargetSelectionGame`） | `envs/sense/target_selection_env.py`（`TargetSelectionEnv`） | `target_selection_shaping` | 上方蓝球每步连接下方 10 个目标球之一；每步随机 4 个黄球作为塑形目标。`primary_mode=random` 时按概率出现 0/1 个红色主要目标，出现时塑形保持中性、由主业务奖励选红球；`never` 时红球永不出现。 |

## 新增小游戏 checklist

1. 在 `shaping_test_games/` 添加 `{name}_game.py`（逻辑 + `python -m` 演示入口）。
2. 添加 `{name}_state.py`、`{name}_env.py`，注册到 `ENV_MAPPING`。
3. 添加 `{name}.toml`：`sense_name = task_name`，`use_sim = false`，`[auto_ml_config]` 中设 `enabled = false`（TOML 解析时不创建 AutoML 配置）；在 `obs_dim` 上方注释各维含义。
4. 在「游戏说明」表格追加一行。
5. 塑形超参：activity 用 `leave_decay_coef`；threat 用 `approach_decay_coef`（KAZ defense 用 5/H；`threat_region` 小游戏为「踏入/穿越才扣分」，toml 里用 `1.0` 使逼近带≈1px）；`shaping_gamma = 1.0` 避免 PBRS 负漂移。

## Target selection 运行

交互小游戏（按 `M` 切换主目标永不出现 / 随机出现）：

```bash
python -m envs.games.shaping_test_games.target_selection_game
```

正式训练：

```bash
python -m portal.train -f subjects/shaping_test/target_selection.toml
```
