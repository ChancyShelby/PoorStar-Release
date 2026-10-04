# 运行与数据接口

## 环境

Python 3.10；基础模型依赖 NumPy 1.x 与 PyTorch 2.2。`sc2` extra 提供游戏协议和 replay 工具，`storage` extra 提供 Zstandard，`test` extra 提供测试工具。SC2 游戏和地图不随 Python 包提供。

```bash
python -m pip install -e '.[test,sc2,storage]'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SC2PATH=/absolute/path/to/StarCraftII
```

源码安装用于数据与正式实验身份检查。若从其他位置执行安装后的命令，可设置 `POORSTAR_SOURCE_ROOT` 指向包含 `poorstar/` 的源码目录。

## 命令

| 工作 | 入口 |
| --- | --- |
| 单 replay 提取 | `poorstar-extract` |
| 并行数据构建与恢复 | `poorstar-build-data` |
| BC 语料聚合与划分 | `poorstar-prepare-data`、`poorstar-partition-data` |
| 高层 sidecar 标注 | `poorstar-annotate` |
| 共享底层 BC | `poorstar-train-foundation` |
| 分层 BC、VALUE_ONLY、MANAGER_RL、JOINT_RL | `poorstar-train` |
| 固定阶段 actor 采集 | `poorstar-collect` |
| 联盟与 worker | `poorstar-league`、`poorstar-worker` |
| BC 开发集评估与研究指标 | `poorstar-evaluate-bc`、`poorstar-evaluate` |

其他工具按用途位于 `poorstar/cli/{data,engine,training,league,evaluation}`，可用 `python -m poorstar.cli.<用途>.<模块> --help` 查看参数。

## 产物顺序

1. 准备原始 replay、匹配 SC2 build 和地图；构建并核验提取计划及源码 manifest。
2. 提取轨迹和传感器；关闭完整 writer 后聚合 BC 数据、监督 census 与压缩索引。
3. 按 replay 分组冻结划分与 action profile；定义模型配置及数字训练计划。
4. 为分层 BC 准备与数据窗口绑定的 Task/Intent sidecar。
5. 创建包含源码、数据、划分、模型、动作、训练计划身份的 launch plan；运行相应阶段。
6. RL 阶段从已验证 actor 采集 rollout，核验完整回执与时间窗后更新策略；独立评估再进入后续阶段。

各入口的 `--help` 列出实际所需 JSON 和产物路径。配置示例仅展示三种架构参数，不代替数据计划或训练资格。输出目录应位于源码树之外。

## 数据约束

- policy 输入仅包含该玩家在当前时刻可见的信息；未来动作和终局结果作为明确时间的目标使用。
- 同一 replay 的双方视角和相邻窗口不能跨越数据划分。
- raw 单位身份与当前帧槽位显式绑定；缺失目标不伪造为 WAIT。
- padding、burn-in、无监督字段与观察不足分别处理。
- 源码 manifest 的路径相对于当前源码根目录，内容包括文件大小和 SHA-256；`RELEASE_MANIFEST.json` 可用于核验此发布包的源码身份。
- 数据、模型、optimizer、采集回执和实验输出保存在外部目录；本发布包不提供这些实验产物。
