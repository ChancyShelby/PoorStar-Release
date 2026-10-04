# PoorStar

面向《星际争霸 II》的持久、可干预 Intent 分层智能体研究。

核心问题：在控制循环记忆、任务分层和训练资源的影响后，显式 Intent 是否能改善长期决策，并提供可解释、可干预的强化学习接口？当前实验域为 ZvZ。

```text
玩家可见观测 → 共享编码与循环记忆 → Intent → Task → 结构化动作
                                                     ↓
可见效果反馈 ← 游戏执行 ← 当前帧合法性检查 ←────────────┘
```

Preference 表达战略优先级；IntentState 保存承诺期、年龄、置信度与生命周期；Task 将战略状态转成具体目标和单位成员。主要比较为 flat recurrent、direct Task、persistent Intent，共用底层观测、动作和执行接口。

当前提供数据处理、模型、BC/PPO、采集、联盟和评估代码。正式训练与游戏能力验证尚未完成，发布包不提供训练完成的可玩模型。

## 代码导航

| 模块 | 职责 |
| --- | --- |
| `poorstar/engine` | SC2 协议、玩家视角观测、replay 读取与查询 |
| `poorstar/geometry` | 实体排序、稳定槽位与坐标语义 |
| `poorstar/actions` | 结构化动作、时序、当前帧合法性与目标解析 |
| `poorstar/data` | 轨迹、BC 分片、索引、语料划分与数据读取 |
| `poorstar/models` | 共享观测编码、循环记忆与自回归动作策略 |
| `poorstar/hierarchy` | Preference、Intent、Task、干预与生命周期 |
| `poorstar/training` | BC/PPO、采集、检查点、恢复与阶段转换 |
| `poorstar/league` | 对手池、任务调度、独立评估与晋级 |
| `poorstar/evaluation` | 行为指标、因果干预、置信度与实验比较 |
| `poorstar/cli` | 按用途组织的命令行入口 |
| `poorstar/common` | 文件身份、参数身份与源码定位 |

## 安装与检查

使用 Python 3.10：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,sc2,storage]'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python examples/inspect_hierarchy.py
python -m pytest --collect-only -q
python -m pytest -q
python scripts/verify_release.py
```

配置示例与测试使用有界 fixture；不会启动 SC2 或正式训练。SC2 游戏、地图、replay、数据分片和模型权重需单独准备。

训练、数据处理与评估入口均提供 `--help`，例如：

```bash
poorstar-train --help
poorstar-build-data --help
poorstar-collect --help
poorstar-league --help
poorstar-evaluate --help
```

进一步阅读：[架构](docs/ARCHITECTURE.md)、[运行与数据接口](docs/USAGE.md)、[研究设计](docs/RESEARCH.md)。

本项目作者保留原创代码及文档的著作权，所有权利保留。仓库公开用于展示、学术交流与代码审阅，目前不提供开源许可或一般复用授权。完整说明见 [LICENSE_STATUS.md](LICENSE_STATUS.md)，依赖与参考来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
