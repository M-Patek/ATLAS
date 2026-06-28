# ATLAS Physical Kitchen - 物理厨房沙盒

## 概述

基于 pymunk 的2D物理模拟厨房环境，用于研究机器人在物理世界中的感知、决策和交互。

## 系统架构

```
┌─────────────────────────────────────────┐
│         ATLAS Physical Kitchen          │
│           物理厨房沙盒                  │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  Robot  │  │ Kitchen│  │  Task   │ │
│  │ 机器人   │  │  厨房   │  │  任务   │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │            │            │      │
│  ┌────┴────────────┴────────────┴────┐ │
│  │         Physics Engine              │ │
│  │    pymunk (2D物理)                  │ │
│  │    重力 | 碰撞 | 摩擦 | 质量         │ │
│  └────────────────────────────────────┘ │
│       │                               │
│  ┌────┴────────────────────────────┐   │
│  │         SSFR Integration        │   │
│  │   感知编码 → 结构发现 → 动作解码  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## 核心组件

### 1. 物理环境 (Kitchen)

```python
from atlas.kitchen import Kitchen

kitchen = Kitchen(width=10.0, height=8.0)
kitchen.setup_default_kitchen()
```

**特性：**
- 重力: -9.8 m/s²
- 碰撞检测: 圆形/矩形刚体
- 摩擦: 地板 0.8, 机器人 0.5, 物体 0.3
- 弹性: 0.2 (低弹性)

### 2. 机器人 (Robot)

```python
robot = kitchen.add_robot('Chef-1', (2.0, 2.0))
robot.move_forward()
robot.turn_left()
robot.stop()
```

**能力：**
- 移动: 前进/后退/转向
- 机械臂: 角度控制 + 伸展控制
- 传感器: 射线投射视觉
- 最大速度: 2.0 m/s

### 3. 物体系统 (Object Library)

| 物体 | 类型 | 质量 | 可抓取 | 可堆叠 |
|------|------|------|--------|--------|
| Coffee Cup | CUP | 0.2kg | ✅ | ✅ |
| Plate | PLATE | 0.3kg | ✅ | ✅ |
| Apple | INGREDIENT | 0.15kg | ✅ | ✅ |
| Coffee Machine | APPLIANCE | 2.0kg | ❌ | ❌ |
| Water Kettle | CONTAINER | 0.5kg | ✅ | ✅ |

### 4. 任务系统 (Task System)

```python
from atlas.kitchen import TASK_LIBRARY

task = TASK_LIBRARY['make_coffee']
# Steps: move_to → grab → place → press_button → wait
```

**预定义任务：**
- `make_coffee`: 做咖啡
- `prepare_breakfast`: 准备早餐
- `clean_table`: 清理桌子

### 5. SSFR 集成

```python
from atlas.kitchen.controller import SSFRKitchenController

controller = SSFRKitchenController(kitchen)
controller.register_robot(robot_id)
controller.assign_task(robot_id, 'make_coffee')

# 运行
while running:
    kitchen.step()
    controller.step()
```

**接口：**
- `KitchenSSFRInterface`: 观察编码
- `StateEncoder`: 状态向量编码
- `ActionDecoder`: 动作解码
- `NavigationController`: 导航控制

## 使用示例

### 基础使用

```python
from atlas.kitchen import Kitchen, create_demo_kitchen

# 创建厨房
kitchen = create_demo_kitchen()

# 添加机器人
robot = kitchen.add_robot('Chef', (5.0, 2.0))

# 运行模拟
for _ in range(60):
    robot.move_forward()
    kitchen.step()
```

### 可视化

```python
from atlas.kitchen import KitchenRenderer

renderer = KitchenRenderer(kitchen)
while running:
    kitchen.step()
    running = renderer.render()
```

### 任务执行

```python
from atlas.kitchen.controller import SSFRKitchenController

controller = SSFRKitchenController(kitchen)
controller.register_robot(robot_id)
controller.assign_task(robot_id, 'make_coffee')

for _ in range(300):
    kitchen.step()
    controller.step()
```

## 测试

```bash
# 运行基础设施测试
python experiments/test_physical_kitchen.py

# 运行演示
python experiments/demo_physical_kitchen.py --demo all
```

**测试结果：** 10/10 通过

| 测试 | 描述 | 状态 |
|------|------|------|
| 物理环境 | 重力、碰撞 | ✅ PASS |
| 机器人移动 | 前进、转向 | ✅ PASS |
| 碰撞检测 | 物体碰撞 | ✅ PASS |
| 摩擦 | 减速效果 | ✅ PASS |
| 传感器 | 射线投射 | ✅ PASS |
| 任务系统 | 动作序列 | ✅ PASS |
| SSFR接口 | 观察编码 | ✅ PASS |
| 厨房布局 | 默认布局 | ✅ PASS |
| 质量惯性 | 加速度差异 | ✅ PASS |
| 机械臂 | 角度/伸展 | ✅ PASS |

## 性能

| 指标 | 数值 |
|------|------|
| 物理步进 | 60 FPS |
| 物体数量 | 200+ (无压力) |
| 内存占用 | < 100MB |
| CPU占用 | < 10% (单核) |

## 文件结构

```
src/atlas/kitchen/
├── __init__.py          # 核心: Kitchen, Robot, Object, Sensor
└── controller.py        # SSFR集成: TaskExecutor, NavigationController

experiments/
├── test_physical_kitchen.py   # 测试
└── demo_physical_kitchen.py   # 演示
```

## 下一步

1. **抓取交互**: 实现机械臂抓取物体的完整物理约束
2. **SSFR策略**: 训练小Transformer做决策
3. **多机器人**: 多机器人协作任务
4. **复杂任务**: 更复杂的厨房任务（如烹饪）
5. **3D升级**: 可选的3D渲染
