---
id: schema-design
title: "Schema Design Principles and Best Practices"
status: complete
complexity: high
related:
  - "../04-data-ecosystem/06-data-formats.md"
  - "../05-integration/01-pipeline-patterns.md"
prerequisites:
  - "Robot kinematics"
  - "Data serialization (JSON, HDF5, TFRecord)"
  - "Temporal data structures"
last_validated: 2026-06-27
---

# Schema设计原则与最佳实践

## §0 — One-liner

机器人数据标注Schema是连接原始感知数据与模型训练输入的契约层，其设计质量直接决定下游模型的泛化能力与多数据集兼容性。

## §1 — 核心概念

### 1.1 什么是机器人数据Schema

机器人数据Schema是对训练数据结构的正式定义，包含三个维度：

| 维度 | 说明 | 示例 |
|------|------|------|
| **语义维度** | 每个字段的业务含义 | `joint_positions`表示关节角度 |
| **类型维度** | 数据类型与形状约束 | `float32[7]`表示7维浮点数组 |
| **时序维度** | 时间戳与采样率定义 | `timestamp_ns`纳秒级时间戳 |

与通用数据Schema（如数据库Schema）不同，机器人数据Schema必须同时满足：
- **实时性约束**：传感器数据流的低延迟处理
- **异构性兼容**：视觉、力觉、本体感受等多模态统一表示
- **可扩展性**：支持新传感器、新任务类型的增量添加

### 1.2 Schema设计的核心挑战

```
Challenge 1: 动作空间异构
  ├─ 机械臂: 6-7 DOF joint angles
  ├─ 移动底盘: SE(2) pose (x, y, theta)
  ├─ 灵巧手: 16+ DOF tendon positions
  └─ 全身机器人: 30+ DOF 混合表示

Challenge 2: 时序对齐
  ├─ Camera: 30-60 Hz
  ├─ Force sensor: 1 kHz
  ├─ Joint encoder: 100-500 Hz
  └─ 需要统一时间轴与插值策略

Challenge 3: 层级语义
  ├─ Frame-level: 单帧状态 (100Hz)
  ├─ Clip-level: 动作片段 (5-30s)
  └─ Session-level: 完整任务 (1-10min)
```

## §2 — 动作空间Schema

### 2.1 关节角度空间 (Joint Space)

最常用的动作表示，直接对应机器人关节配置。

**Schema定义示例：**
```yaml
joint_positions:
  type: float32[n_joints]
  description: "关节角度，单位弧度"
  range: [-pi, pi]
  semantics:
    - "joint_0: base rotation"
    - "joint_1: shoulder pitch"
    # ...

joint_velocities:
  type: float32[n_joints]
  description: "关节角速度，单位rad/s"
  derived_from: "joint_positions"  # 可通过差分计算

joint_torques:
  type: float32[n_joints]
  description: "关节力矩，单位Nm"
  source: "motor_current_feedback"
```

**关键设计决策：**

| 决策点 | 选项A | 选项B | 推荐 |
|--------|-------|-------|------|
| 角度单位 | 弧度 (rad) | 度 (deg) | **弧度** — 与数学库一致 |
| 连续角处理 | 原始值 (累积) | 模2π (wrap) | **模2π** — 避免歧义 |
| 缺失关节 | NaN填充 | 省略字段 | **NaN填充** — 保持张量形状 |
| 速度表示 | 显式存储 | 运行时差分 | **显式存储** — 减少计算误差 |

### 2.2 末端位姿空间 (Task Space / Cartesian Space)

用于表示末端执行器在笛卡尔空间中的目标位置。

**Schema定义：**
```yaml
cartesian_position:
  type: float32[3]
  description: "末端执行器位置，单位米"
  coordinate_frame: "base_link"

cartesian_orientation:
  type: float32[4]
  description: "四元数表示 (x, y, z, w)"
  normalization: "unit_quaternion"
  # 备选: rotation_matrix (float32[3,3]) 或 euler_angles (float32[3])

cartesian_pose:
  type: struct
  fields:
    position: float32[3]
    orientation: float32[4]  # quaternion
  transform_from: "joint_positions"  # 通过FK计算
```

**旋转表示对比：**

| 表示法 | 存储大小 | 奇异性 | 插值友好 | 约束 | 适用场景 |
|--------|----------|--------|----------|------|----------|
| 欧拉角 (ZYX) | 3 | 有 (万向锁) | 困难 | 无 | 调试显示 |
| 旋转矩阵 | 9 | 无 | 困难 | 正交约束 | 坐标变换 |
| **四元数** | 4 | 无 | 球面线性插值 | 单位约束 | **推荐默认** |
| 轴角 | 4 | 无 | 中等 | 无 | 小角度旋转 |
| 李代数 so(3) | 3 | 无 | 线性 | 无 | 优化问题 |

### 2.3 图像空间动作 (Image-space Actions)

用于视觉伺服或基于图像的模仿学习。

**Schema定义：**
```yaml
image_space_action:
  type: struct
  fields:
    pixel_coordinates:
      type: int32[2]  # 或 float32[2] 支持亚像素
      description: "目标像素位置 (u, v)"
      reference: "camera_0_rgb"
    depth_offset:
      type: float32
      description: "相对于当前深度的偏移量，单位米"
      optional: true
    gripper_state:
      type: int8
      description: "0=open, 1=closed"
```

**图像空间 vs 笛卡尔空间对比：**

| 特性 | 图像空间 | 笛卡尔空间 |
|------|----------|------------|
| 相机标定依赖 | 不需要 | **必须** |
| 视角泛化 | 同视角内好 | 视角变化鲁棒 |
| 遮挡处理 | 天然处理 | 需要显式处理 |
| 深度信息 | 可选 | 必须 |
| 典型应用 | UMI, RT-1 | ALOHA, Diffusion Policy |

## §3 — 时序Schema设计

### 3.1 时间戳对齐策略

机器人数据的核心挑战是多传感器异步采样。推荐采用**统一时间轴 + 最近邻/线性插值**策略。

**时间戳Schema：**
```yaml
timestamp:
  type: int64
  description: "Unix时间戳，单位纳秒"
  # 避免使用float秒，防止累积精度损失

timestamp_source:
  type: string
  description: "时间戳来源: 'ros_time', 'ntp_synced', 'hardware_clock'"
  # 关键: 记录同步精度

sync_accuracy_ns:
  type: int64
  description: "多传感器同步精度，单位纳秒"
  example: 1000000  # 1ms
```

**对齐策略对比：**

| 策略 | 实现复杂度 | 精度 | 计算开销 | 适用场景 |
|------|------------|------|----------|----------|
| 最近邻 (Nearest) | 低 | 中 | 低 | 高帧率、低精度要求 |
| 线性插值 (Linear) | 中 | 高 | 中 | **通用推荐** |
| 三次样条 (Spline) | 高 | 很高 | 高 | 轨迹优化后处理 |
| 零阶保持 (ZOH) | 低 | 低 | 极低 | 离散事件 (夹爪开关) |

### 3.2 采样率设计

**推荐采样率矩阵：**

| 数据类型 | 推荐频率 | 说明 |
|----------|----------|------|
| RGB图像 | 30 Hz | 标准视频帧率，平衡带宽与信息量 |
| 深度图像 | 30 Hz | 与RGB同步，便于配准 |
| 关节状态 | 50-100 Hz | 捕捉快速运动，需≥2×动作带宽 |
| 力/力矩 | 500 Hz - 1 kHz | 接触检测需要高频 |
| 夹爪状态 | 事件驱动 | 状态变化时记录，节省存储 |
| 末端位姿 | 50-100 Hz | 通过FK从关节状态计算 |

**降采样策略：**
- 视觉数据：随机间隔降采样，保持时序覆盖
- 动作数据：低通滤波后降采样，避免混叠
- 事件数据：保留所有事件，不降采样

### 3.3 时序数据存储模式

**模式A：列式存储 (推荐)**
```python
# HDF5/TFRecord 结构
episode_001/
  ├── observations/
  │   ├── images/          # [T, H, W, C]  uint8
  │   ├── joint_pos/       # [T, N]        float32
  │   └── timestamps/      # [T]           int64
  └── actions/
      ├── joint_pos_target/ # [T, N]       float32
      └── gripper_cmd/      # [T]          int8
```

**模式B：行式存储 (JSONL)**
```json
{"timestamp": 1234567890000000000, "obs": {...}, "action": {...}}
{"timestamp": 1234567890033333333, "obs": {...}, "action": {...}}
```

| 特性 | 列式 (HDF5/Zarr) | 行式 (JSONL) |
|------|------------------|--------------|
| 读取效率 | 高 (可按列加载) | 低 (需解析整行) |
| 压缩率 | 高 (同类数据连续) | 中 |
| 随机访问 | 支持 | 需索引 |
| 流式写入 | 需预分配 | 天然支持 |
| **推荐用于** | **大规模训练数据** | **调试/小规模数据** |

## §4 — 层级Schema

### 4.1 三层架构

```
Session (会话级)
  └── 描述: 完整任务执行，包含多个片段
  └── 示例: "整理桌面" = 抓取杯子 + 放置到托盘 + 重复
  └── Metadata: 机器人配置、环境描述、操作员ID

  └── Clip (片段级)
    └── 描述: 原子动作单元，有明确起止
    └── 示例: "抓取杯子" = 接近 + 夹取 + 提升
    └── 标注: 动作类别、成功/失败、语言指令

    └── Frame (帧级)
      └── 描述: 单个时间步的完整状态
      └── 示例: t=1.23s 时刻的图像+关节状态+动作目标
      └── 存储: 原始传感器数据
```

### 4.2 各层级Schema示例

**Frame级Schema：**
```yaml
frame:
  timestamp_ns: int64
  observation:
    camera_rgb: uint8[H, W, 3]
    camera_depth: uint16[H, W]  # 或 float32 米单位
    joint_positions: float32[N]
    joint_velocities: float32[N]
    end_effector_pose: float32[7]  # [x,y,z,qx,qy,qz,qw]
    gripper_width: float32
  action:
    target_joint_positions: float32[N]
    target_gripper_width: float32
  metadata:
    frame_id: int64
    is_keyframe: bool  # 用于关键帧标注
```

**Clip级Schema：**
```yaml
clip:
  clip_id: string
  start_frame: int64
  end_frame: int64
  start_timestamp_ns: int64
  end_timestamp_ns: int64
  language_instruction: string
  action_category: string  # 如 "pick", "place", "pour"
  success: bool
  objects_involved: list[string]  # 参与物体ID
  contact_events: list[ContactEvent]  # 接触事件列表
```

**Session级Schema：**
```yaml
session:
  session_id: string
  robot_config:
    robot_type: string  # "franka", "ur5", "bimanual_aloha"
    n_joints: int
    gripper_type: string
    camera_configs: list[CameraConfig]
  environment:
    scene_description: string
    object_inventory: list[ObjectDef]
  clips: list[Clip]
  quality_metrics:
    tracking_accuracy: float
    sync_quality_ns: int64
```

### 4.3 层级间引用策略

**扁平引用 (Flat References) — 推荐：**
```python
# 每个frame包含session_id和clip_id
frame = {
    "session_id": "sess_001",
    "clip_id": "clip_003",
    "frame_id": 145,
    # ... 实际数据
}
```

优势：便于分布式处理，无需遍历层级结构即可获取上下文。

## §5 — Schema版本管理与兼容性

### 5.1 版本化策略

**语义化版本 (SemVer) 应用于Schema：**
```
Schema版本: major.minor.patch
  ├─ major: 不兼容的结构变更 (字段删除/类型改变)
  ├─ minor: 向后兼容的扩展 (新增可选字段)
  └─ patch: 文档/注释更新，无结构变化
```

**Schema版本声明：**
```yaml
schema_version: "2.1.0"
schema_name: "atlas_robot_episode"
compatible_with: ["2.0.x", "2.1.x"]  # 运行时兼容性检查
```

### 5.2 兼容性处理模式

| 场景 | 处理策略 | 实现方式 |
|------|----------|----------|
| 新增可选字段 | 向后兼容 | 旧数据读取时填充默认值 |
| 字段重命名 | 版本适配层 | 维护字段映射表 |
| 类型变更 | 不兼容 | 数据迁移脚本 (major版本升级) |
| 单位变更 | 显式标注 | 字段名包含单位 (如 `position_m`) |
| 坐标系变更 | 显式标注 | `coordinate_frame` 元数据字段 |

### 5.3 多数据集兼容性

**Open X-Embodiment 兼容性层：**

Open X-Embodiment (OXE) 定义了跨机器人数据集的统一Schema。与其兼容的关键映射：

```python
# OXE Schema 映射示例
OXE_FIELD_MAP = {
    "observation/image": "observations/camera_rgb",
    "observation/state": "observations/joint_positions",
    "action": "actions/target_joint_positions",
    "language_instruction": "clips/*/language_instruction",
    "dataset_name": "session/robot_config/robot_type",
}
```

**关键兼容性原则：**
1. **最小公共子集**：定义所有数据集共有的核心字段
2. **扩展字段命名空间**：机器人特有字段使用 `robot_specific/` 前缀
3. **单位标准化**：统一为 SI 单位 (米、弧度、秒、牛顿)
4. **坐标系约定**：默认使用 ROS 标准 (x前, y左, z上)

## §6 — 业界标准与工具

### 6.1 现有Schema标准对比

| 标准/数据集 | 来源 | 动作空间 | 时序设计 | 层级结构 | 语言标注 |
|-------------|------|----------|----------|----------|----------|
| **RLDS** | Google | 通用张量 | 事件流 | Episode→Step | 支持 |
| **Open X-Embodiment** | Google/DeepMind | 多形态统一 | 固定频率 | Episode级 | 自然语言指令 |
| **ALOHA** | Stanford | 关节位置 | 50Hz | Episode级 | 无 (纯模仿) |
| **UMI** | Stanford | SE(3)位姿 | 事件驱动 | 轨迹段 | 无 |
| **RoboTurk** | USC | 关节+夹爪 | 30Hz | Episode级 | 有限 |
| **ARMBench** | Amazon | 关节位置 | 10Hz | Episode级 | 无 |
| **LIBERO** | 多机构 | 关节位置 | 20Hz | Episode→Subtask | 自然语言 |

### 6.2 推荐工具链

| 工具 | 用途 | 优点 | 局限 |
|------|------|------|------|
| **RLDS (TensorFlow)** | 数据存储与加载 | 与TF生态集成好 | 仅TensorFlow |
| **Zarr** | 大规模数组存储 | 并行读写、云存储 | 需额外元数据层 |
| **HDF5** | 通用科学数据 | 成熟、多语言绑定 | 并发写入受限 |
| **Apache Arrow** | 内存列式格式 | 零拷贝、跨语言 | 持久化需额外层 |
| **DuckDB** | 结构化查询 | SQL接口、分析友好 | 非时序原生 |

## §7 — 最佳实践与常见陷阱

### 7.1 最佳实践清单

1. **显式优于隐式**：每个字段都应有明确的语义定义，不依赖上下文推断
2. **自包含原则**：单个Frame应包含理解其状态所需的所有信息
3. **不变量约束**：在Schema层定义约束（如四元数归一化），并在加载时验证
4. **向后兼容测试**：每次Schema变更都需验证旧数据的可读性
5. **文档即代码**：Schema定义与验证代码同处一个文件，避免文档漂移

### 7.2 常见陷阱

| 陷阱 | 描述 | 后果 | 规避方法 |
|------|------|------|----------|
| **浮点时间戳** | 使用float64秒存储时间 | 微秒级精度损失 | 使用int64纳秒 |
| **隐式坐标系** | 未标注参考坐标系 | 不同模块理解不一致 | 所有位姿必须含`frame_id` |
| **角度歧义** | 欧拉角顺序未说明 | 旋转解释错误 | 使用四元数或显式标注顺序 |
| **混合单位** | 同一字段不同数据集单位不同 | 训练数值不稳定 | 标准化为SI单位，字段名含单位 |
| **缺失值处理不一致** | 有的用NaN，有的用0，有的省略 | 模型学习到错误模式 | 统一使用NaN，定义`valid_mask` |
| **硬编码维度** | 假设7关节机械臂 | 无法支持其他机器人 | 使用`n_joints`元数据，动态形状 |

### 7.3 Schema验证代码模板

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class FrameSchema:
    """帧级数据Schema，带运行时验证"""
    timestamp_ns: int
    joint_positions: np.ndarray  # shape: (n_joints,)
    joint_velocities: Optional[np.ndarray] = None
    camera_rgb: Optional[np.ndarray] = None  # shape: (H, W, 3)
    end_effector_pose: Optional[np.ndarray] = None  # shape: (7,) [x,y,z,qx,qy,qz,qw]

    def validate(self, n_joints: int, image_shape: tuple):
        assert self.joint_positions.shape == (n_joints,), \
            f"joint_positions shape mismatch: {self.joint_positions.shape} vs ({n_joints},)"
        assert np.all(np.abs(self.joint_positions) <= np.pi), \
            "joint_positions out of [-pi, pi] range"

        if self.end_effector_pose is not None:
            assert self.end_effector_pose.shape == (7,), \
                "end_effector_pose must be (7,) [x,y,z,qx,qy,qz,qw]"
            q = self.end_effector_pose[3:]
            assert np.abs(np.linalg.norm(q) - 1.0) < 1e-3, \
                "quaternion not normalized"

        if self.camera_rgb is not None:
            assert self.camera_rgb.shape[:2] == image_shape[:2], \
                f"image shape mismatch"
            assert self.camera_rgb.dtype == np.uint8, \
                "camera_rgb must be uint8"
```

## §8 — 与DVAS项目的关联

DVAS (Data-Centric Visual Action System) 是本项目的核心数据基础设施，Schema层为其提供：

1. **统一数据契约**：DVAS各模块（采集、清洗、训练、评估）通过Schema定义接口
2. **多数据源聚合**：Schema兼容性层使DVAS能够消费Open X-Embodiment、ALOHA、UMI等多源数据
3. **质量门控基础**：Schema验证是DVAS数据质量检查的第一道防线
4. **版本演进支持**：Schema版本管理使DVAS能够平滑升级数据格式而不中断训练流程

**DVAS Schema继承关系：**
```
BaseSchema (通用机器人数据)
  ├── DVASObservation (观测定义)
  ├── DVASAction (动作定义)
  └── DVASEpisode (会话定义)
      ├── DVASALOHAEpisode (ALOHA特化)
      ├── DVASUMIEpisode (UMI特化)
      └── DVASOXEEpisode (OXE兼容层)
```

---

*References:*
- OXE: Padalkar et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", 2023
- RLDS: Google, "Reinforcement Learning Datasets (RLDS)"
- Zarr: https://zarr.dev/
- ROS Coordinate Conventions: http://www.ros.org/reps/rep-0103.html

*Related: [02-action-annotation](12-action-annotation.md) | [03-scene-annotation](13-scene-annotation.md) | [04-physics-annotation](14-physics-annotation.md)*
