---
id: physics-annotation
title: "Physical Interaction Annotation"
status: complete
complexity: high
related:
  - "./02-action-annotation.md"
  - "./03-scene-annotation.md"
  - "../03-perception/06-sensor-fusion.md"
prerequisites:
  - "Contact mechanics"
  - "Rigid body dynamics"
  - "Causal inference basics"
last_validated: 2026-06-27
---

# 物理交互标注

## §0 — One-liner

物理交互标注捕获机器人与环境接触过程中的力学与因果关系信息，是构建具有物理常识的操作模型的核心数据层。

## §1 — 核心概念

### 1.1 为什么需要物理标注

传统机器人学习数据主要包含运动学信息（在哪里、去哪里），但物理交互任务需要理解：

```
物理交互理解的三个层次:
  ├─ 接触层 (Contact)
  │   ├─ 哪里接触？ (接触点/区域)
  │   ├─ 接触几何？ (点/线/面接触)
  │   └─ 接触状态？ (滑动/滚动/固定)
  ├─ 力学层 (Mechanics)
  │   ├─ 力多大？ (接触力/力矩)
  │   ├─ 力怎么变？ (力曲线/冲量)
  │   └─ 力从哪里来？ (重力/惯性/接触)
  └─ 因果层 (Causality)
      ├─ 什么导致接触？ (动作→接触因果)
      ├─ 接触导致什么？ (接触→状态变化因果)
      └─ 如果...会怎样？ (反事实推理)
```

**物理标注对机器人学习的价值：**

| 学习任务 | 物理信息作用 | 典型应用 |
|----------|------------|----------|
| 抓取稳定性预测 | 接触力分布 | 判断抓取是否牢固 |
| 装配策略学习 | 接触状态 + 力反馈 | 轴孔装配、插拔 |
| 工具使用 | 因果关系 + 物理属性 | 用锤子敲击、用刀切割 |
| 异常检测 | 力信号模式 | 碰撞检测、滑移检测 |
| Sim2Real迁移 | 物理参数标定 | 仿真参数辨识 |

### 1.2 物理标注的特殊挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| **传感器噪声** | 力传感器信号含高频噪声 | 接触检测困难 |
| **间接测量** | 关节力矩 ≠ 末端接触力 (需雅可比映射) | 末端力估计误差 |
| **多接触点** | 同时多点接触，力信号耦合 | 难以分解单点力 |
| **时序精细** | 接触事件发生在毫秒级 | 需要高频同步 |
| **标注主观性** | "接触"vs"接近"的边界模糊 | 标注一致性差 |
| **物理属性未知** | 质量、摩擦系数等难以直接观测 | 需要估计或假设 |

## §2 — 接触点/接触区域标注

### 2.1 接触几何表示

接触的几何表示取决于接触类型和精度需求：

```yaml
contact_annotation:
  type: struct
  fields:
    contact_id: string
    timestamp_ns: int64
    bodies_involved: [string, string]  # ["gripper_finger_left", "mug_handle"]
    contact_geometry:
      type: enum  # "point", "line", "patch", "surface"
      # 点接触
      point_contact:
        position: float32[3]       # 接触点3D坐标
        normal: float32[3]          # 接触法线方向
      # 面接触 (更一般)
      patch_contact:
        contact_polygon: list[float32[3]]  # 接触区域多边形顶点
        centroid: float32[3]
        area: float32               # 接触面积 (m²)
    contact_state: enum  # "approaching", "sliding", "rolling", "fixed"
```

**接触类型对比：**

| 接触类型 | 几何特征 | 力学特征 | 典型场景 | 标注难度 |
|----------|----------|----------|----------|----------|
| **点接触** | 单点 | 3维力 | 指尖触碰 | 中 |
| **线接触** | 一维 | 分布力 | 边缘接触 | 高 |
| **面接触** | 二维区域 | 分布力+力矩 | 平面贴合 | 高 |
| **软接触** | 变形区域 | 非线性力-变形 | 海绵抓取 | 很高 |
| **多点接触** | 多个分离区域 | 力耦合 | 多指抓取 | 很高 |

### 2.2 接触检测方法

**基于力信号的接触检测：**

```python
def detect_contact_force_based(
    force_history,
    threshold=2.0,      # 力阈值 (N)
    window_size=5,      # 滑动窗口
    noise_band=0.5      # 噪声带 (N)
):
    """基于力突变检测接触事件"""
    contacts = []
    force_norm = np.linalg.norm(force_history, axis=1)

    # 1. 基线估计 (无接触时的力)
    baseline = np.median(force_norm[:window_size])

    # 2. 接触 onset 检测
    for t in range(window_size, len(force_norm)):
        window_mean = np.mean(force_norm[t-window_size:t])
        if window_mean > baseline + threshold and \
           force_norm[t-1] <= baseline + noise_band:
            contacts.append({
                "type": "contact_onset",
                "timestamp": t,
                "force": force_norm[t]
            })
            baseline = window_mean  # 更新基线

        # 3. 接触 release 检测
        elif window_mean < baseline - threshold and \
             force_norm[t-1] >= baseline - noise_band:
            contacts.append({
                "type": "contact_release",
                "timestamp": t,
                "force": force_norm[t]
            })

    return contacts
```

**基于视觉的接触检测：**

| 方法 | 原理 | 优点 | 局限 |
|------|------|------|------|
| **变形检测** | 物体表面变形指示接触 | 直观、无需力传感器 | 仅适用于可变形物体 |
| **阴影分析** | 接触区域产生阴影 | 简单 | 光照敏感 |
| **触觉传感器** | GelSight等光学触觉 | 高分辨率接触几何 | 需专用硬件 |
| **运动学一致性** | 预期vs实际位置偏差 | 无需额外传感器 | 精度受限 |

### 2.3 接触区域标注实践

**GelSight触觉传感器标注：**

GelSight提供接触区域的高分辨率几何与力学信息：

```yaml
gelsight_contact:
  sensor_type: "GelSight Mini"
  raw_image: uint8[240, 320, 3]  # 弹性体变形图像
  contact_mask: bool[240, 320]    # 接触区域分割
  contact_depth: float32[240, 320] # 压痕深度 (mm)
  # 从标定得到的力学信息
  contact_force_distribution: float32[240, 320]  # 分布力 (N)
  total_force: float32[3]         # 合力 (N)
  contact_center: float32[2]      # 接触中心 (像素)
```

**从力信号反推接触区域 (无触觉传感器时)：**

```python
def estimate_contact_region(
    end_effector_force,
    end_effector_torque,
    gripper_geometry,
    object_model
):
    """
    基于合力/力矩和几何模型估计接触区域。
    这是一个欠定问题，需要正则化假设。
    """
    # 假设1: 接触力垂直于表面
    # 假设2: 接触区域在已知几何约束内
    # 假设3: 力分布均匀或高斯

    # 建立优化问题
    # min ||F_measured - sum(f_i)||² + lambda * R(f_distribution)
    # s.t. f_i perpendicular to surface
    #      contact_points on object surface

    # 使用高斯过程或神经网络学习从力到接触区域的映射
    pass
```

## §3 — 力/力矩标注方法

### 3.1 力传感器类型与数据特性

| 传感器类型 | 测量量 | 量程 | 分辨率 | 频率 | 典型应用 |
|------------|--------|------|--------|------|----------|
| **六维力/力矩传感器** | Fx,Fy,Fz,Mx,My,Mz | 50-500N | 0.1N | 1kHz | 腕部力控 |
| **关节力矩传感器** | 单轴力矩 | 10-100Nm | 0.01Nm | 500Hz | 碰撞检测 |
| **触觉阵列** | 分布压力 | 0-1MPa | 1kPa | 100Hz | 抓取稳定性 |
| **电流环估计** | 关节力矩 | 额定值 | 较低 | 控制频率 | 低成本方案 |
| **应变片** | 局部应变 | 自定义 | 高 | 10kHz+ | 结构监测 |

### 3.2 力/力矩数据标注Schema

```yaml
wrench_annotation:
  type: struct
  fields:
    timestamp_ns: int64
    reference_frame: string    # 测量坐标系
    force: float32[3]          # [Fx, Fy, Fz] N
    torque: float32[3]         # [Mx, My, Mz] Nm
    sensor_info:
      sensor_id: string
      sensor_type: string      # "ati_ft", "robotiq_ft", "estimated"
      calibration_date: string
      noise_characteristics:
        force_std: float32[3]   # 各轴力噪声标准差
        torque_std: float32[3]  # 各轴力矩噪声标准差
    quality_flags:
      saturated: bool          # 是否饱和
      drift_compensated: bool  # 是否漂移补偿
      gravity_compensated: bool # 是否重力补偿
```

### 3.3 重力补偿与工具动力学

**关键预处理步骤：**

```python
def compensate_wrench(raw_wrench, robot_pose, tool_mass, tool_com):
    """
    从原始力/力矩测量中补偿重力和工具惯性。
    """
    # 1. 重力在传感器坐标系中的表示
    gravity_world = np.array([0, 0, -9.81 * tool_mass])
    R_sensor_world = robot_pose.rotation  # 传感器到世界的旋转
    gravity_sensor = R_sensor_world.T @ gravity_world

    # 2. 重力产生的力矩
    com_offset = tool_com  # 质心相对于传感器的位置
    gravity_torque = np.cross(com_offset, gravity_sensor)

    # 3. 惯性力 (加速度已知时)
    # inertial_force = -tool_mass * end_effector_acceleration
    # inertial_torque = -inertia_tensor @ angular_acceleration

    compensated_force = raw_wrench.force - gravity_sensor
    compensated_torque = raw_wrench.torque - gravity_torque

    return Wrench(compensated_force, compensated_torque)
```

**常见错误：**

| 错误 | 描述 | 后果 |
|------|------|------|
| 未重力补偿 | 静态时力读数不为零 | 接触阈值误判 |
| 忽略工具质量 | 更换夹爪后未更新参数 | 系统性偏差 |
| 坐标系混淆 | 力在传感器系，位姿在基座系 | 重力方向错误 |
| 未考虑惯性 | 快速运动时惯性力显著 | 虚假接触检测 |

### 3.4 力信号特征标注

从原始力信号中提取高层语义特征：

```yaml
force_features:
  contact_events:
    - type: "contact_onset"
      timestamp: float
      force_magnitude: float      # 接触瞬间力大小
      approach_velocity: float    # 接触前接近速度
    - type: "slip_detected"
      timestamp: float
      slip_direction: float32[3]
      friction_estimate: float     # 实时摩擦估计

  force_profile:
    phase: enum  # "approach", "contact", "manipulation", "release"
    mean_force: float32[3]
    force_variance: float32[3]
    spectral_features:  # 力信号频域特征
      dominant_frequency: float
      high_freq_ratio: float      # 高频成分比例 (振动指示)

  stability_metrics:
    grasp_stability_index: float   # 基于力闭合的稳定性
    force_closure: bool            # 是否力闭合
    min_wrench_quality: float     # 最小 wrench 质量
```

## §4 — 因果关系标注

### 4.1 因果链标注

机器人操作中的因果关系形成链条：动作 → 接触 → 力变化 → 物体状态变化 → 任务结果。

```yaml
causal_chain:
  episode_id: string
  events:
    - event_id: "e1"
      type: "action"
      description: "gripper_close"
      timestamp: 1.23
      agent: "robot"

    - event_id: "e2"
      type: "contact"
      description: "finger_contact_with_mug"
      timestamp: 1.25
      cause: ["e1"]  # 由e1导致
      objects: ["gripper_finger", "mug"]

    - event_id: "e3"
      type: "force_change"
      description: "normal_force_increase"
      timestamp: 1.26
      cause: ["e2"]
      magnitude: 5.0  # N

    - event_id: "e4"
      type: "state_change"
      description: "mug_lifted"
      timestamp: 2.10
      cause: ["e1", "e3"]
      pre_state: {position: [0.5, 0.2, 0.03], supported_by: "table"}
      post_state: {position: [0.5, 0.2, 0.20], supported_by: null}

    - event_id: "e5"
      type: "outcome"
      description: "grasp_success"
      timestamp: 2.10
      cause: ["e4"]
      success: true
```

### 4.2 时序因果标注

因果关系的时序维度——理解"A在B之前"与"A导致B"的区别。

**因果 vs 时序关联：**

| 关系类型 | 定义 | 判断标准 | 机器人示例 |
|----------|------|----------|------------|
| **时序先后** | A发生在B之前 | 时间戳比较 | 夹爪闭合在提升之前 |
| **因果导致** | A是B的充分/必要原因 | 干预实验/反事实 | 夹爪闭合导致提升成功 |
| **使能条件** | A是B的前提 | 无A则无B | 接触是抓取的前提 |
| **并发伴随** | A与B同时发生 | 时间重叠 | 力增加与位置变化同时 |

**因果发现方法：**

```python
def discover_causality(event_sequence, intervention_data=None):
    """
    从事件序列中发现因果关系。
    结合观察数据与干预实验。
    """
    causal_graph = {}

    # 方法1: 时序Granger因果 (统计关联)
    for event_a in event_sequence:
        for event_b in event_sequence:
            if event_a.timestamp < event_b.timestamp:
                if granger_causality_test(event_a, event_b):
                    causal_graph.add_edge(event_a, event_b, "temporal")

    # 方法2: 干预验证 (真正因果)
    if intervention_data:
        for edge in causal_graph.edges:
            # 反事实: 如果A不发生，B是否还会发生？
            if counterfactual_test(edge.source, edge.target, intervention_data):
                edge.confidence += 1
            else:
                edge.confidence -= 1

    return causal_graph
```

### 4.3 反事实标注

反事实推理是物理理解的核心——"如果我没用力推，物体会动吗？"

```yaml
counterfactual_annotation:
  actual_event:
    action: "push_box"
    force: 10.0  # N
    result: "box_moved_0.5m"

  counterfactuals:
    - hypothetical: "no_force_applied"
      expected_result: "box_stays_stationary"
      causal_strength: "necessary"  # 力是移动的必要条件

    - hypothetical: "half_force_applied (5N)"
      expected_result: "box_moved_0.2m"
      causal_strength: "proportional"  # 因果关系是连续的

    - hypothetical: "push_from_different_angle"
      expected_result: "box_rotates"
      causal_strength: "direction_dependent"
```

**反事实数据在仿真中的生成：**

```python
# 在物理仿真中生成反事实数据
def generate_counterfactuals(sim_env, actual_trajectory):
    counterfactuals = []

    # 反事实1: 不执行动作
    cf1 = sim_env.simulate(
        initial_state=actual_trajectory.initial_state,
        action=None
    )
    counterfactuals.append({"type": "no_action", "trajectory": cf1})

    # 反事实2: 改变力的大小
    for force_scale in [0.5, 1.5, 2.0]:
        cf = sim_env.simulate(
            initial_state=actual_trajectory.initial_state,
            action=scale_action(actual_trajectory.action, force_scale)
        )
        counterfactuals.append({"type": f"scaled_force_{force_scale}", "trajectory": cf})

    # 反事实3: 改变接触位置
    for offset in [(0.01, 0, 0), (0, 0.01, 0)]:
        cf = sim_env.simulate(
            initial_state=actual_trajectory.initial_state,
            action=perturb_contact_point(actual_trajectory.action, offset)
        )
        counterfactuals.append({"type": f"perturbed_contact_{offset}", "trajectory": cf})

    return counterfactuals
```

## §5 — RoboTurk、ARMBench物理标注实践

### 5.1 RoboTurk

RoboTurk是USC提出的众包遥操作数据采集框架，其物理标注特点：

| 特性 | 描述 |
|------|------|
| 数据来源 | 人类通过网页界面遥操作Sawyer/Franka |
| 物理信息 | 关节力矩 (来自机器人电流估计) |
| 标注内容 | 成功/失败标签、语言描述 |
| 局限性 | 无精确末端力测量、无接触标注 |

**RoboTurk数据格式扩展 (加入物理标注)：**
```json
{
  "trajectory_id": "rt_001",
  "robot_type": "sawyer",
  "joint_positions": [[...], [...]],
  "joint_velocities": [[...], [...]],
  "joint_torques": [[...], [...]],  // 电流估计力矩
  "estimated_end_force": [[...], [...]],  // 通过雅可比映射
  "contact_annotations": [
    {"timestamp": 2.3, "type": "contact_onset", "confidence": 0.8}
  ],
  "success": true,
  "language_annotation": "pick up the red block"
}
```

### 5.2 ARMBench

ARMBench是Amazon发布的机器人操作基准，专注于仓储拣选任务。

| 特性 | 描述 |
|------|------|
| 任务 | 从货架抓取商品放入 tote |
| 传感器 | RGB-D + 腕部力传感器 |
| 物理标注 | 接触检测、滑移检测、抓取失败分析 |
| 规模 | 超过200k次抓取尝试 |

**ARMBench物理标注Schema：**
```yaml
armbench_trial:
  trial_id: string
  item_sku: string           # 商品SKU
  grasp_attempt:
    pre_grasp_image: Image
    post_grasp_image: Image
    gripper_force: float     # 夹爪夹持力 (N)
    wrist_force: float32[3]   # 腕部力传感器读数

  physical_outcomes:
    contact_detected: bool
    slip_detected: bool      # 基于力信号变化率
    drop_detected: bool       # 基于视觉
    collision_detected: bool  # 与货架碰撞

  success_criteria:
    item_in_tote: bool
    item_damaged: bool
    execution_time: float
```

**ARMBench中的失败模式分析 (物理视角)：**

| 失败类型 | 物理原因 | 力信号特征 |
|----------|----------|------------|
| 未接触目标 | 定位误差/遮挡 | 力无变化 |
| 滑移 | 摩擦力不足/夹持力不够 | 力下降+位置偏差 |
| 碰撞 | 路径规划不当 | 力突变 (高频) |
| 挤压变形 | 夹持力过大 | 力持续增加+物体变形 |
| 多抓 | 相邻物体粘连 | 力异常高 |

## §6 — 物理属性标注

### 6.1 物理属性类型

```yaml
physical_properties:
  object_id: string
  properties:
    # 质量与惯性
    mass:
      value: float32           # kg
      uncertainty: float32      # 测量不确定度
      method: enum             # "scale", "dynamics_estimation", "cad_model"

    center_of_mass:
      position: float32[3]      # 相对于物体坐标系
      uncertainty: float32[3]

    inertia_tensor:
      matrix: float32[3, 3]
      frame: string             # 计算坐标系

    # 表面属性
    friction:
      static_coefficient: float    # 静摩擦系数
      kinetic_coefficient: float   # 动摩擦系数
      measurement_method: string    # "inclined_plane", "pull_test", "estimation"

    restitution:
      value: float             # 弹性恢复系数 (0-1)

    # 材料属性
    material:
      category: string         # "metal", "plastic", "wood", "ceramic", "fabric"
      hardness: float          # 莫氏硬度或邵氏硬度
      deformability: enum      # "rigid", "elastic", "plastic", "fluid"

    # 几何属性
    bounding_box:
      dimensions: float32[3]   # [length, width, height]
      volume: float            # m³ (可计算密度)
```

### 6.2 物理属性估计方法

| 属性 | 直接测量 | 基于视觉估计 | 基于交互估计 |
|------|----------|--------------|--------------|
| **质量** | 电子秤 (最准确) | 不可能 | 通过加速度和力: F=ma |
| **摩擦系数** | 斜面实验 | 表面纹理分析 (粗糙度) | 推拉实验测最大静摩擦 |
| **重心** | 悬挂法 | 形状假设 (均匀密度) | 多姿态平衡实验 |
| **刚度** | 万能试验机 | 材料类别推断 | 按压实验测力-位移曲线 |
| **惯性张量** | 三线摆 | 不可能 | 旋转实验测角加速度 |

**基于交互的物理参数估计 (系统辨识)：**

```python
def estimate_physical_parameters(trajectory, forces, known_model):
    """
    通过观测轨迹和力，估计物体物理参数。
    使用非线性最小二乘或贝叶斯推断。
    """
    # 动力学模型: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
    # 未知参数: 质量、摩擦系数、质心偏移等

    def residual(params):
        predicted_trajectory = simulate_with_params(trajectory[0], forces, params)
        return trajectory - predicted_trajectory

    # 优化求解
    from scipy.optimize import least_squares
    result = least_squares(residual, x0=initial_guess,
                          bounds=(lower_bounds, upper_bounds))
    return result.x
```

### 6.3 物理属性标注的不确定性

物理属性标注必须包含不确定性估计：

```yaml
physical_property_with_uncertainty:
  property_name: "mass"
  value: 0.5        # kg
  uncertainty:
    type: "gaussian"
    std: 0.02       # kg
    confidence_interval_95: [0.46, 0.54]
  source:
    method: "scale_measurement"
    sensor_model: "Ohaus_PX2202"
    calibration_date: "2024-01-15"
    n_samples: 5
  # 或基于估计的不确定性
  estimation_method:
    type: "bayesian"
    posterior_samples: [...]  # 后验分布采样
    credible_interval_95: [0.45, 0.56]
```

## §7 — 最佳实践与常见陷阱

### 7.1 物理标注最佳实践

| 实践 | 说明 | 理由 |
|------|------|------|
| ** always重力补偿** | 所有力数据必须标注是否已重力补偿 | 未补偿数据无法比较 |
| **标注坐标系** | 每个物理量必须明确坐标系 | 避免方向理解错误 |
| **记录传感器信息** | 传感器型号、校准日期、噪声特性 | 数据溯源与质量评估 |
| **多模态交叉验证** | 力信号+视觉+运动学一致性检查 | 提高标注可靠性 |
| **不确定性量化** | 所有估计参数附带置信区间 | 下游决策考虑不确定性 |
| **反事实数据** | 在仿真中生成对比数据 | 增强因果推理能力 |

### 7.2 常见陷阱

| 陷阱 | 描述 | 后果 | 规避方法 |
|------|------|------|----------|
| **力=接触误解** | 有力信号就认为有接触 | 惯性力误判为接触力 | 结合位置一致性验证 |
| **忽略动态效应** | 快速运动时忽略惯性力 | 接触力估计偏差大 | 加速度补偿 |
| **单点接触假设** | 假设只有一个接触点 | 力矩解释错误 | 多点接触检测 |
| **过度平滑** | 力信号过度滤波 | 丢失接触事件 | 保留原始数据，滤波后数据作为衍生 |
| **因果混淆** | 将时序先后当作因果 | 学习到错误因果 | 干预实验验证 |
| **属性静态假设** | 假设物理属性不变 | 磨损/形变导致失效 | 定期重新标定 |

### 7.3 与DVAS项目的关联

DVAS项目的物理标注模块设计：

1. **分层物理理解：**
   - 底层：原始力/力矩信号 + 接触检测
   - 中层：力特征提取 (相位、频域、稳定性)
   - 高层：因果链构建 + 物理属性估计

2. **多模态融合策略：**
   ```
   视觉 (接触区域估计)
        ↘
         融合 → 完整物理标注
        ↗
   力觉 (接触力估计)
        ↘
         融合 → 物理一致性校验
        ↗
   运动学 (运动状态)
   ```

3. **Sim2Real物理对齐：**
   - 从真实交互数据估计物理参数
   - 将估计参数注入仿真环境
   - 在仿真中生成反事实数据
   - 反哺真实策略训练

**DVAS物理标注流水线：**
```
输入: 机器人轨迹 + 力信号 + 视觉数据
  ├── 预处理
  │   ├── 重力补偿
  │   ├── 工具动力学补偿
  │   └── 坐标系统一
  ├── 接触检测
  │   ├── 基于力的突变检测
  │   ├── 基于视觉的变形检测
  │   └── 多模态融合确认
  ├── 物理事件标注
  │   ├── 接触 onset/release
  │   ├── 滑移检测
  │   ├── 碰撞检测
  │   └── 稳定/不稳定状态
  ├── 因果链构建
  │   ├── 动作→接触→力→状态变化
  │   ├── 反事实仿真生成
  │   └── 因果强度评分
  ├── 物理属性估计
  │   ├── 质量/重心 (动力学)
  │   ├── 摩擦系数 (推拉实验)
  │   └── 刚度 (按压实验)
  └── 输出: 物理标注数据
      ├── 接触事件序列
      ├── 力特征时间线
      ├── 因果图
      └── 物体物理属性表
```

---

*References:*
- RoboTurk: Mandlekar et al., "Roboturk: A crowdsourcing platform for robotic skill learning through imitation", 2018
- ARMBench: Suresh et al., "ARMbench: A benchmark for robotic manipulation in ambiguous and novel real-world environments", 2023
- GelSight: Yuan et al., "GelSight: High-Resolution Robot Tactile Sensors for Estimating Geometry and Force", 2017
- Contact Mechanics: Johnson, "Contact Mechanics", Cambridge University Press, 1985
- Causal Inference: Pearl, "Causality: Models, Reasoning, and Inference", 2009
- Force Control: Siciliano & Khatib, "Springer Handbook of Robotics", Chapter on Force Control

*Related: [01-schema-design](11-schema-design.md) | [02-action-annotation](12-action-annotation.md) | [03-scene-annotation](13-scene-annotation.md)*
