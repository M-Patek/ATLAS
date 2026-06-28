# SSFR Kitchen Integration

> SSFR（结构自发现与复用）与物理厨房的集成应用

## 概述

将SSFR的认知架构能力应用于物理厨房环境，实现：

1. **物理感知** → 离散认知空间
2. **结构发现** → 环境结构理解
3. **空间竞争** → 最优空间选择
4. **动作生成** → 物理动作执行

## 架构

```
Physical Kitchen → KitchenSpaceAdapter → SSFR → ActionDecoder → Physical Kitchen
       ↑____________________________________________________________|
```

## 核心组件

### 1. KitchenSpaceAdapter（空间适配器）

将物理厨房的连续状态映射到SSFR可用的离散网格表示。

```python
from experiments.demo_ssfr_kitchen import KitchenSpaceAdapter

adapter = KitchenSpaceAdapter(kitchen, grid_resolution=0.5)

# 坐标转换
grid_pos = adapter.world_to_grid((5.0, 3.0))
world_pos = adapter.grid_to_world(grid_pos)

# 编码观测
observation = adapter.encode_observation(robot_id, task)
# {
#     'position': (10, 6),           # 网格位置
#     'goal_position': (3, 13),       # 目标位置
#     'obstacles': [(5, 5), ...],     # 障碍物
#     'uncertainty': 0.033,           # 不确定性
#     'velocity': (0.1, 0.0),         # 速度
#     'speed': 0.1,                   # 速度大小
#     'nearby_objects_count': 3,      # 附近物体数
#     'robot_angle': 0.0,            # 机器人角度
#     'task_step': 'move_to',         # 当前步骤
#     'task_target': 'coffee_machine', # 当前目标
#     'physical_position': (5.0, 3.0), # 物理位置
#     'physical_goal': (1.5, 6.5),    # 物理目标
#     'grid_resolution': 0.5,         # 网格分辨率
# }
```

### 2. PhysicalSSFR（物理SSFR）

封装SSFR用于物理厨房环境。

```python
from experiments.demo_ssfr_kitchen import PhysicalSSFR

physical_ssfr = PhysicalSSFR(
    kitchen,
    grid_resolution=0.5,
    space_names=['ricci', 'fisher', 'wasserstein']
)

# 感知：生成结构假设
hypotheses = physical_ssfr.perceive(robot_id, task, active_space='ricci')

# 竞争：选择最优结构
winner = physical_ssfr.compete(robot_id, actual_position)

# 获取空间有效性
validity = physical_ssfr.get_space_validity(robot_id)
# {'ricci': 0.975, 'fisher': 0.371, 'wasserstein': 0.654}

# 获取统计
stats = physical_ssfr.get_statistics()
```

### 3. SSFRTaskPlanner（任务规划器）

使用SSFR进行任务级规划。

```python
from experiments.demo_ssfr_kitchen import SSFRTaskPlanner

planner = SSFRTaskPlanner(physical_ssfr)

# 分配任务
planner.assign_task(robot_id, 'make_coffee')

# 执行一步规划
result = planner.step(robot_id)
# {
#     'status': 'running',
#     'task': 'Make Coffee',
#     'step': 'move_to',
#     'target': 'coffee_machine',
#     'progress': 0.0,
#     'best_space': 'ricci',
#     'space_validity': {...},
#     'num_hypotheses': 3,
#     'winner_id': 'abc123',
#     'step_result': 'moving',
# }
```

## 空间导航策略

不同认知空间提供不同的导航策略：

| 空间 | 特性 | 导航策略 |
|------|------|----------|
| **Ricci** | 曲率感知 | 避开高曲率区域，沿测地线移动 |
| **Fisher** | 信息几何 | 选择信息增益最大的路径 |
| **Wasserstein** | 最优传输 | 选择能量消耗最小的路径 |

## 测试

```bash
# 运行集成测试
python experiments/test_ssfr_kitchen.py
```

**测试结果：** 7/7 通过

| 测试 | 描述 | 状态 |
|------|------|------|
| 空间适配器 | 坐标转换、障碍物、观测编码 | ✅ PASS |
| 物理SSFR | 感知、竞争、空间有效性 | ✅ PASS |
| 任务规划器 | 任务分配、步骤执行 | ✅ PASS |
| SSFR导航 | 物理导航、移动控制 | ✅ PASS |
| 结构演化 | 结构池增长、演化 | ✅ PASS |
| 空间比较 | 不同空间配置比较 | ✅ PASS |
| 端到端集成 | 完整任务执行 | ✅ PASS |

## 演示

```bash
# 运行演示
python experiments/demo_ssfr_kitchen.py
```

演示内容：
1. **SSFR感知** - 在不同位置生成结构假设
2. **任务规划** - 使用SSFR指导任务执行
3. **结构演化** - 观察结构池随时间增长
4. **空间比较** - 比较不同空间配置的效果

## 性能

| 指标 | 数值 |
|------|------|
| 感知时间 | ~1.3ms |
| 结构池大小 | 24 (300步后) |
| 坐标转换误差 | <0.5m |
| 空间有效性计算 | <1ms |

## 下一步

1. **结构复用** - 实现跨任务的结构复用
2. **在线学习** - 从物理交互中学习最优空间选择
3. **多机器人** - 多机器人协作的SSFR集成
4. **可视化** - 结构发现和空间有效性的可视化
