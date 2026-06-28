---
id: action-annotation
title: "Action Annotation Standards"
status: complete
complexity: high
related:
  - "./01-schema-design.md"
  - "../04-data-ecosystem/02-umi-systems.md"
  - "../04-data-ecosystem/06-data-formats.md"
prerequisites:
  - "Robot kinematics"
  - "Imitation learning basics"
  - "Open X-Embodiment dataset"
last_validated: 2026-06-27
---

# 动作标注标准

## §0 — One-liner

动作标注将连续机器人运动转化为结构化训练信号，其表示方法的选择直接影响模仿学习算法的收敛速度与泛化性能。

## §1 — 核心概念

### 1.1 动作标注的定义与范围

动作标注是机器人学习数据流水线中的关键环节，涵盖：

| 层次 | 内容 | 输出形式 |
|------|------|----------|
| **低级动作** | 关节目标位置、末端位姿、速度指令 | 连续向量 |
| **中级动作** | 技能/子任务标识 (如 "grasp", "place") | 离散标签 + 时间区间 |
| **高级动作** | 自然语言指令 (如 "把杯子放到托盘上") | 文本字符串 |

本文档聚焦于**低级动作标注**——即模型直接学习的控制信号表示。

### 1.2 动作表示的关键维度

```
动作表示设计空间:
  ├─ 空间维度
  │   ├─ Joint Space (关节空间)
  │   ├─ Task Space / Cartesian Space (任务空间)
  │   └─ Image Space (图像空间)
  ├─ 时间维度
  │   ├─ Absolute (绝对位置/姿态)
│   ├─ Relative (相对偏移)
│   ├─ Velocity (速度指令)
│   └─ Force/Torque (力/力矩)
  └─ 参考系维度
      ├─ World Frame (世界坐标系)
      ├─ Base Frame (机器人基座)
      ├─ End-effector Frame (末端执行器)
      └─ Object Frame (物体坐标系)
```

## §2 — Open X-Embodiment数据集标注规范

### 2.1 OXE统一Schema

Open X-Embodiment (OXE) 是目前最大的跨机器人模仿学习数据集，包含22个数据集、超过100万条轨迹。其核心贡献是定义了跨形态机器人的统一动作表示。

**OXE标准字段：**
```python
{
  "observation": {
    "image": np.ndarray,      # uint8[H, W, 3] — 主相机RGB
    "wrist_image": np.ndarray, # uint8[H, W, 3] — 腕部相机 (可选)
    "state": np.ndarray,       # float32[N] — 本体感受状态
    "language_instruction": str  # 自然语言任务描述
  },
  "action": np.ndarray,         # float32[M] — 统一动作向量
  "reward": float,              # 稀疏奖励 (通常为0/1)
  "is_terminal": bool           # 片段终止标志
}
```

### 2.2 OXE动作标准化

OXE将不同机器人的动作统一映射到固定维度的向量。以RT-X模型为例：

```python
# RT-X 统一动作空间 (维度通常为 7-8)
action = [
    delta_x,      # 末端/基座 x方向位移 (米)
    delta_y,      # 末端/基座 y方向位移 (米)
    delta_z,      # 末端/基座 z方向位移 (米)
    delta_roll,   # 绕x轴旋转 (弧度)
    delta_pitch,  # 绕y轴旋转 (弧度)
    delta_yaw,    # 绕z轴旋转 (弧度)
    gripper_cmd,  # 夹爪指令: -1(开) ~ 1(闭)
    # 可选: terminate Episode (终止信号)
]
```

**关键设计决策：**
- **相对动作 (delta)**：而非绝对位置，增强泛化性
- **统一维度**：通过填充或投影将不同DOF映射到相同空间
- **夹爪归一化**：将不同夹爪的开度映射到 [-1, 1]

### 2.3 OXE数据集构成

| 数据集 | 机器人 | 动作空间 | 规模 | 特点 |
|--------|--------|----------|------|------|
| RT-1 RobotLang | SayCan | 相对SE(3)+夹爪 | 130k | 厨房任务，语言条件 |
| Bridge Data | WidowX | 相对SE(3)+夹爪 | 50k | 桌面操作，低成本 |
| ALOHA | Bimanual Franka | 关节位置 | 500+ | 双臂精细操作 |
| UMI | Single arm | SE(3)位姿 | 6k+ | 手持夹具，轨迹跟踪 |
| RoboTurk | Sawyer/Franka | 关节+夹爪 | 100k+ | 众包遥操作 |
| Language Table | xArm | 相对2D+夹爪 | 200k+ | 语言条件推物 |

## §3 — ALOHA/UMI数据格式详解

### 3.1 ALOHA数据格式

ALOHA (A Low-cost Hardware for Dual-arm teleOperation) 是斯坦福大学提出的双臂遥操作系统，其数据格式被广泛用于双臂模仿学习研究。

**ALOHA HDF5 Schema：**
```
episode_xxx.hdf5
├── /observations/
│   ├── /images/
│   │   ├── /cam_high        # [T, 480, 640, 3] uint8 — 顶部相机
│   │   ├── /cam_left_wrist  # [T, 480, 640, 3] uint8 — 左腕相机
│   │   ├── /cam_right_wrist # [T, 480, 640, 3] uint8 — 右腕相机
│   │   └── /cam_front       # [T, 480, 640, 3] uint8 — 正面相机
│   ├── /qpos/               # [T, 14] float64 — 14关节位置 (双臂)
│   └── /qvel/               # [T, 14] float64 — 14关节速度
├── /action/                 # [T, 14] float64 — 目标关节位置 (来自主臂)
└── /base_pose/              # [T, 2] float64 — 移动底盘 (可选)
```

**ALOHA动作表示特点：**

| 特性 | 说明 | 影响 |
|------|------|------|
| 绝对关节位置 | 直接记录主臂关节角度作为动作目标 | 简单直接，但跨机器人迁移困难 |
| 双臂同步 | 左右臂各7关节，共14维 | 需考虑双臂协调约束 |
| 50Hz固定频率 | 每个时间步都有完整观测和动作 | 数据量大，但时序完整 |
| 主从映射 | 动作 = 主臂当前位置 (遥操作模式下) | 存在延迟和噪声 |

**ALOHA数据加载示例：**
```python
import h5py
import numpy as np

def load_aloha_episode(filepath):
    with h5py.File(filepath, 'r') as f:
        episode = {
            'images': {
                'top': f['observations/images/cam_high'][:],
                'left_wrist': f['observations/images/cam_left_wrist'][:],
                'right_wrist': f['observations/images/cam_right_wrist'][:],
            },
            'joint_positions': f['observations/qpos'][:],  # [T, 14]
            'actions': f['action'][:],  # [T, 14] — 目标关节位置
        }
    return episode
```

### 3.2 UMI数据格式

UMI (Universal Manipulation Interface) 是斯坦福大学提出的手持式数据采集系统，使用GoPro相机记录操作者手持夹具的轨迹。

**UMI数据特点：**

| 特性 | UMI设计 | 与传统遥操作对比 |
|------|---------|------------------|
| 动作空间 | SE(3)位姿 (手持夹具) | 传统: 关节空间 |
| 视角 | 手持ego-centric | 传统: 固定外部相机 |
| 标定 | 基于ArUco码的在线标定 | 传统: 预标定 |
| 同步 | 相机内建IMU + 视觉SLAM | 传统: ROS时间同步 |

**UMI数据文件结构：**
```
episode_xxx/
├── videos/
│   ├── 0.mp4          # GoPro主视频 (ego-centric)
│   └── 1.mp4          # 可选第二视角
├── metadata.json      # 相机内参、标定信息
├── traj.yaml          # 关键帧轨迹 (SE(3)位姿序列)
└── annotations.json   # 语言指令、成功/失败标注
```

**UMI轨迹表示 (traj.yaml)：**
```yaml
# 关键帧轨迹，非均匀采样
trajectory:
  - timestamp: 0.0
    pose:
      position: [x, y, z]           # 米
      orientation: [qx, qy, qz, qw]  # 四元数
    gripper: 0.0  # 0=开, 1=闭
  - timestamp: 0.5
    pose: ...
    gripper: 0.0
  # ... 关键帧之间通过插值得到密集轨迹
```

**UMI与ALOHA核心差异：**

| 维度 | ALOHA | UMI |
|------|-------|-----|
| **动作表示** | 绝对关节位置 (14维) | SE(3)位姿 + 夹爪 (7维) |
| **相机配置** | 4个固定相机 | 1-2个手持ego相机 |
| **频率** | 固定50Hz | 事件驱动，关键帧插值 |
| **双臂支持** | 原生支持 | 需扩展 |
| **跨机器人迁移** | 困难 (关节空间绑定) | 较容易 (任务空间) |
| **典型模型** | ACT, Diffusion Policy | Diffusion Policy + 投影 |

## §4 — 动作表示方法详解

### 4.1 绝对位置 (Absolute Position)

**定义：** 直接指定目标关节角度或末端位姿。

```python
# 关节空间绝对位置
action = [q1, q2, q3, q4, q5, q6, q7]  # 目标关节角度 (rad)

# 任务空间绝对位置
action = [x, y, z, qx, qy, qz, qw, gripper]  # 目标位姿 + 夹爪
```

**优点：**
- 直观易懂，与机器人控制器接口直接对应
- 单步决策，无需累积

**缺点：**
- 跨机器人迁移困难（不同机器人关节配置不同）
- 对初始状态敏感，小误差可能导致大偏差
- 难以表示速度/力约束

**适用场景：** 同类型机器人的直接模仿学习 (如ALOHA)。

### 4.2 相对位置 (Relative Position / Delta Action)

**定义：** 指定相对于当前状态的位移。

```python
# 关节空间相对动作
delta_q = [dq1, dq2, dq3, dq4, dq5, dq6, dq7]  # 关节角度增量
next_q = current_q + delta_q

# 任务空间相对动作 (OXE/RT-X标准)
delta_pose = [dx, dy, dz, droll, dpitch, dyaw, dgripper]
next_pose = current_pose * delta_pose  # SE(3)合成
```

**优点：**
- 跨机器人迁移更友好（相对运动语义一致）
- 对初始状态不敏感
- 天然支持闭环控制

**缺点：**
- 需要当前状态作为参考
- 累积误差问题（多步预测时）
- 旋转增量表示需注意奇异性

**旋转增量表示对比：**

| 表示法 | 公式 | 优点 | 缺点 |
|--------|------|------|------|
| 欧拉角增量 | [dα, dβ, dγ] | 直观、3维 | 万向锁、非线性 |
| 轴角增量 | axis * angle | 4维、无奇异 | 大角度时不直观 |
| **四元数增量** | **q_delta = q_target * q_current⁻¹** | **插值友好** | **4维、需归一化** |
| 李代数增量 | exp(ω) | 线性化好 | 需指数映射 |

**适用场景：** 跨机器人迁移学习 (OXE, RT-X)、闭环视觉伺服。

### 4.3 速度控制 (Velocity Control)

**定义：** 指定关节速度或末端速度作为控制指令。

```python
# 关节速度
joint_vel_cmd = [v1, v2, v3, v4, v5, v6, v7]  # rad/s

# 笛卡尔速度 (空间速度)
cartesian_vel = [vx, vy, vz, wx, wy, wz]  # m/s, rad/s
# 通过雅可比矩阵映射: dq = J⁺ * cartesian_vel
```

**优点：**
- 与机器人底层控制器接口一致 (大多数机器人都支持速度控制)
- 自然表达运动平滑性
- 易于施加速度/加速度约束

**缺点：**
- 需要积分得到位置，累积误差
- 低频控制时稳定性差
- 难以直接对应视觉目标

**适用场景：** 实时遥操作、模型预测控制 (MPC)。

### 4.4 力/力矩控制 (Force/Torque Control)

**定义：** 指定末端力/力矩或关节力矩。

```python
# 末端力/力矩 (Wrench)
wrench = [fx, fy, fz, tx, ty, tz]  # N, Nm

# 关节力矩
tau = [t1, t2, t3, t4, t5, t6, t7]  # Nm
```

**优点：**
- 适用于接触丰富的任务 (装配、打磨)
- 对位置精度要求低，对力精度要求高
- 安全性好（可设置力上限）

**缺点：**
- 需要力传感器（增加成本）
- 力信号噪声大，需滤波
- 与视觉目标难以直接关联

**适用场景：** 接触任务、柔顺控制、人机协作。

### 4.5 动作表示方法综合对比

| 表示法 | 维度 | 跨机器人迁移 | 视觉关联 | 接触任务 | 累积误差 | 推荐场景 |
|--------|------|--------------|----------|----------|----------|----------|
| 绝对关节位置 | N_joints | 差 | 间接 | 差 | 无 | 同构机器人模仿 |
| 绝对SE(3) | 7 | 中 | 直接 | 差 | 无 | 单臂任务空间控制 |
| **相对关节位置** | **N_joints** | **中** | **间接** | **中** | **有** | **双臂协调** |
| **相对SE(3)** | **7** | **好** | **直接** | **中** | **有** | **跨机器人迁移 (推荐)** |
| 关节速度 | N_joints | 中 | 间接 | 中 | 严重 | 遥操作、MPC |
| 笛卡尔速度 | 6 | 好 | 直接 | 中 | 严重 | 视觉伺服 |
| 关节力矩 | N_joints | 差 | 间接 | 好 | 无 | 柔顺控制 |
| 末端力/力矩 | 6 | 好 | 间接 | 好 | 无 | 装配任务 |

## §5 — 动作质量评估指标

### 5.1 轨迹级指标

| 指标 | 定义 | 计算方式 | 用途 |
|------|------|----------|------|
| **MSE (Mean Squared Error)** | 预测与真值动作差的平方均值 | `mean((pred - gt)²)` | 基本拟合度 |
| **MAE (Mean Absolute Error)** | 平均绝对误差 | `mean(\|pred - gt\|)` | 对异常值鲁棒 |
| **末端误差 (End-effector Error)** | 预测与真值末端位姿差 | `\|FK(pred) - FK(gt)\|` | 物理意义明确 |
| **平滑度 (Smoothness)** | 轨迹二阶导数范数 | `mean(\|a_t\|²)` | 运动质量 |
| **成功率 (Success Rate)** | 任务完成比例 | `n_success / n_total` | 最终目标 |

### 5.2 分布级指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **FID (Fréchet Action Distance)** | 生成与真实动作分布的距离 | 生成模型评估 |
| **Coverage** | 生成分布对真实分布的覆盖度 | 多样性评估 |
| **Precision/Recall** | 生成样本的真实性与多样性权衡 | 生成质量 |

### 5.3 动作标注质量检查清单

```python
# 自动化质量检查函数
def check_action_quality(actions, observations):
    issues = []

    # 1. 检查关节限位
    if np.any(np.abs(actions) > JOINT_LIMITS):
        issues.append("Joint limit violation")

    # 2. 检查速度突变 (平滑性)
    velocities = np.diff(actions, axis=0)
    if np.any(np.abs(velocities) > MAX_VELOCITY):
        issues.append("Excessive velocity")

    # 3. 检查自碰撞 (需机器人模型)
    # if check_self_collision(actions):
    #     issues.append("Self-collision detected")

    # 4. 检查时间一致性
    if not monotonically_increasing(timestamps):
        issues.append("Non-monotonic timestamps")

    # 5. 检查动作-观测对齐
    if len(actions) != len(observations):
        issues.append("Action-observation mismatch")

    return issues
```

## §6 — 自动化动作标注技术

### 6.1 基于遥操作的自动标注

**原理：** 操作者通过遥操作设备控制机器人，系统自动记录操作者输入作为动作标签。

| 遥操作方式 | 动作来源 | 优点 | 缺点 |
|------------|----------|------|------|
| 同构主从 (ALOHA) | 主臂关节位置 | 直观、精确 | 需专用设备 |
| 手持夹具 (UMI) | 视觉跟踪位姿 | 低成本、灵活 | 精度受限 |
| VR手柄 | 手柄位姿映射 | 沉浸式 | 映射延迟 |
| 键盘/游戏手柄 | 离散指令 | 最简单 | 不自然 |

### 6.2 基于视觉的自动标注

**原理：** 从视频中提取人手/工具运动，映射到机器人动作空间。

**技术路线：**
1. **人手姿态估计**：MediaPipe, HaMeR 等模型提取人手3D位姿
2. **工具跟踪**：基于ArUco/SIFT的6DoF物体跟踪
3. **动作映射**：人手/工具位姿 → 机器人末端目标位姿

**局限：**
- 人手与机器人末端运动学差异大
- 遮挡严重时跟踪失败
- 需要精确的相机-机器人标定

### 6.3 基于仿真的自动标注

**原理：** 在仿真环境中执行策略，自动记录完美标注数据。

| 仿真平台 | 物理引擎 | 机器人支持 | 标注能力 |
|----------|----------|------------|----------|
| MuJoCo | 自身 | 广泛 | 完整状态、接触力 |
| Isaac Sim | PhysX | NVIDIA生态 | GPU并行、照片级渲染 |
| PyBullet | Bullet | 广泛 | 完整状态 |
| Habitat | 可选 | 移动为主 | 语义标注 |

**Sim2Real Gap 补偿：**
- 域随机化 (Domain Randomization)
- 系统识别 (System Identification)
- 适配层训练 (Adaptation Layer)

### 6.4 半自动化标注工具

| 工具/方法 | 功能 | 自动化程度 |
|------------|------|------------|
| **关键帧插值** | 标注关键帧，自动插值中间动作 | 中 |
| **轨迹分割** | 自动检测动作边界，人工确认 | 中高 |
| **语言-动作对齐** | 自动对齐语言指令与动作片段 | 中 |
| **异常检测过滤** | 自动标记可疑轨迹，人工审核 | 高 |

## §7 — 与DVAS项目的关联

DVAS项目的动作标注模块直接采用以下设计决策：

1. **默认动作表示**：采用 **相对SE(3) + 夹爪** (与OXE/RT-X兼容)
2. **多形态支持**：通过配置化的动作投影层支持不同机器人
3. **质量门控**：集成上述自动化质量检查，拦截低质量数据
4. **标注来源**：支持遥操作 (ALOHA风格)、手持 (UMI风格)、仿真三种模式

**DVAS动作标注流水线：**
```
原始遥操作数据
  ├── 时间戳对齐与插值
  ├── 坐标系转换 (设备坐标系 → 机器人基座)
  ├── 动作表示转换 (绝对 → 相对)
  ├── 质量检查 (限位、平滑性、碰撞)
  ├── 语言指令对齐 (如适用)
  └── 输出: 标准化训练样本 (RLDS格式)
```

---

*References:*
- OXE: Padalkar et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", 2023
- ALOHA: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", 2024
- UMI: Chi et al., "Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots", 2024
- Diffusion Policy: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", 2023
- ACT: Zhao et al., "Learning Visuomotor Policies via Action Chunking with Transformers", 2023

*Related: [01-schema-design](11-schema-design.md) | [03-scene-annotation](13-scene-annotation.md) | [04-physics-annotation](14-physics-annotation.md)*
