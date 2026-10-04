# 架构

## 观测、数据与动作

`engine/observation.py` 将 SC2 raw protocol 转为当前玩家可见观测。`geometry/entities.py` 按确定顺序分配实体槽位；raw tag 仅用于当前帧目标绑定，不作为模型数值特征。单位位置采用 protocol world coordinates，图像保留原生尺寸。

`actions/frame.py` 绑定观测、实体、引擎动作目录与查询结果；`actions/contract.py` 和 `actions/coarse.py` 定义动作顺序、参数与条件候选集。模型提出动作，执行依据当前帧检查可用性与目标。派发、引擎接受、可见效果分别记录。

`data/trajectory.py` 保存 raw trajectory；`data/bc` 负责因果决策样本、压缩分片和时间窗。`data/corpus` 管理 replay 身份、语料范围和划分。同一 replay 的双方视角与全部窗口归于同一划分。padding 和 burn-in 不参与训练损失。

## 共享模型与分层控制

`models/policy.py` 提供实体/空间/标量编码、循环记忆、自回归结构化动作，以及动作时序预测。`models/encoder.py` 提供批量编码。

`hierarchy/state.py` 定义 Preference、IntentState 与控制状态；`hierarchy/goals.py` 定义 typed Task。`hierarchy/model.py` 将高层状态接入共享策略；`hierarchy/session.py` 管理更新、续期、终止和循环状态。`hierarchy/observer.py` 只根据玩家可见反馈确认任务效果，观察不足保持 censored。

flat recurrent、direct Task 与 persistent Intent 共用观测、动作和执行接口。Intent 调节 Task，不代替具体动作，也不绕过合法性检查。

## 学习与采集

`training/foundation.py` 与 `training/bc.py` 分别实现共享底层和分层 BC。`training/learner.py` 实现 PPO；`training/returns.py` 计算时间相关与 SMDP targets；`training/scorer.py` 对齐采样和重评分语义。

阶段顺序为 BC → VALUE_ONLY → MANAGER_RL → JOINT_RL。`training/stages.py` 管理阶段参数与转换；`training/checkpoint.py` 保存优化器和随机状态；`training/export.py` 导出 actor。训练身份绑定源码、数据、划分、模型与数字计划，阶段验证由 `training/gates.py` 执行。

`training/actor.py` 调度双玩家引擎，双方共享游戏时钟。`training/deadline.py` 管理采集时间界限；`training/admission.py` 验证完整、成功、身份匹配的采集产物后才允许学习。自然终局与基础设施截断分别处理。

## 联盟与评估

`league/registry.py` 保存不可变模型快照；`league/scheduler.py` 调度对手；`league/ownership.py`、`dispatch.py` 与 `coordinator.py` 管理任务归属、执行和累计资源；`promotion.py` 使用独立评估结果决定晋级。

`evaluation` 提供高层标注、干预、可见行为测量、置信度校准、检查点选择与多种子比较。评估结果不回流为同一测试集上的训练选择。协议 fixture 验证接口与数值行为；游戏能力需用真实引擎实验验证。
