---
id: slam-systems
title: "SLAM系统"
status: complete
complexity: high
related:
  - "./01-stereo-imu.md"
  - "./02-depth-estimation.md"
  - "./06-sensor-fusion.md"
prerequisites:
  - "视觉几何 (PnP, epipolar geometry)"
  - "状态估计 (EKF, BA)"
  - "李群与李代数 (SO(3), SE(3))"
last_validated: 2026-06-27
---

# SLAM 系统

## §0 — One-liner

SLAM（Simultaneous Localization and Mapping）通过融合视觉与惯性传感器，在未知环境中实时估计相机位姿并构建环境地图，是机器人自主导航与 ego-centric 场景理解的核心技术。

## §1 — SLAM 问题数学定义

### 1.1 状态变量

SLAM 的状态向量通常包含：

$$\mathbf{X} = \{ \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N, \mathbf{l}_1, \mathbf{l}_2, ..., \mathbf{l}_M \}$$

其中：
- $\mathbf{x}_i \in SE(3)$：第 $i$ 时刻的相机位姿
- $\mathbf{l}_j \in \mathbb{R}^3$：第 $j$ 个路标点的 3D 坐标

### 1.2 最大后验估计（MAP）

SLAM 的求解目标：

$$\mathbf{X}^* = \arg\max_{\mathbf{X}} P(\mathbf{X} | \mathbf{Z}, \mathbf{U})$$

通过贝叶斯分解：

$$P(\mathbf{X} | \mathbf{Z}, \mathbf{U}) \propto P(\mathbf{X}_0) \prod_{t=1}^{T} P(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{u}_t) \prod_{k=1}^{K} P(\mathbf{z}_k | \mathbf{x}_{i_k}, \mathbf{l}_{j_k})$$

其中：
- $P(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{u}_t)$：运动模型（IMU 预积分）
- $P(\mathbf{z}_k | \mathbf{x}_{i_k}, \mathbf{l}_{j_k})$：观测模型（特征重投影）

---

## §2 — ORB-SLAM3：特征点法 SLAM 的集大成者

### 2.1 系统架构

```
ORB-SLAM3 系统架构:

[输入: 双目/单目/RGBD + IMU]
         ↓
    [Tracking]
    ├── 特征提取 (ORB)
    ├── 初始位姿估计
    ├── 局部地图跟踪
    └── 关键帧决策
         ↓
    [Local Mapping]
    ├── 新关键帧插入
    ├── 地图点剔除/创建
    ├── 局部 BA
    └── 关键帧剔除
         ↓
    [Loop & Map Merging]
    ├── 回环检测 (DBoW2)
    ├── 回环校正
    ├── 地图合并
    └── 全局 BA
         ↓
    [Multi-Map]
    └── 地图集管理
```

### 2.2 特征提取与匹配

#### 2.2.1 ORB 特征

ORB (Oriented FAST and Rotated BRIEF) 特征：

- **FAST 角点检测**：
  $$S = \{ p | \exists \text{ 连续 } n \text{ 个像素满足 } |I(p) - I(p_i)| > \epsilon \}$$

- **方向计算**：通过图像矩计算特征方向
  $$\theta = \arctan\left(\frac{m_{01}}{m_{10}}\right)$$

- **BRIEF 描述子**：256 位二进制描述子
  $$d_i = \begin{cases} 1 & \text{if } I(p_a) < I(p_b) \\ 0 & \text{otherwise} \end{cases}$$

#### 2.2.2 特征匹配策略

| 匹配层级 | 方法 | 描述 |
|---------|------|------|
| 粗匹配 | 词袋模型 (BoW) | 通过视觉词汇加速匹配 |
| 精匹配 | 汉明距离 | 二进制描述子的距离度量 |
| 几何验证 | RANSAC + 本质矩阵 | 剔除误匹配 |

### 2.3 视觉-惯性融合

#### 2.3.1 IMU 预积分

Forster et al. 提出的 IMU 预积分：

$$\Delta \mathbf{R}_{ij} = \prod_{k=i}^{j-1} \text{Exp}\left( (\tilde{\omega}_k - b_g^k) \Delta t \right)$$

$$\Delta \mathbf{v}_{ij} = \sum_{k=i}^{j-1} \Delta \mathbf{R}_{ik} (\tilde{a}_k - b_a^k) \Delta t$$

$$\Delta \mathbf{p}_{ij} = \sum_{k=i}^{j-1} \left[ \Delta \mathbf{v}_{ik} \Delta t + \frac{1}{2} \Delta \mathbf{R}_{ik} (\tilde{a}_k - b_a^k) \Delta t^2 \right]$$

**预积分的优势**：
- 与初始位姿解耦，可重复计算
- 避免积分漂移累积
- 支持因子图优化

#### 2.3.2 视觉-惯性联合优化

ORB-SLAM3 的优化目标函数：

$$\min_{\mathbf{X}} \sum_{i,j} \rho_h(e_{v}^{ij^T} \mathbf{\Sigma}_{v}^{-1} e_{v}^{ij}) + \sum_{k} \rho_h(e_{imu}^{k^T} \mathbf{\Sigma}_{imu}^{-1} e_{imu}^{k})$$

其中：
- $e_v$：视觉重投影误差
- $e_{imu}$：IMU 预积分残差
- $\rho_h$：Huber 鲁棒核函数

### 2.4 多地图系统

#### 2.4.1 地图集 (Atlas) 架构

```
Atlas
├── Active Map (当前活跃地图)
│   ├── KeyFrame 1
│   ├── KeyFrame 2
│   ├── MapPoint 1
│   └── ...
├── Map 1 (历史地图)
├── Map 2 (历史地图)
└── ...
```

**关键创新**：
- 跟踪丢失时自动创建新地图
- 检测到回环时合并地图
- 支持长期运行而不丢失信息

#### 2.4.2 回环检测与地图合并

**回环检测流程**：
1. **候选检索**：DBoW2 词袋模型快速检索
2. **几何验证**：RANSAC + Sim(3) 变换估计
3. **融合优化**：位姿图优化 + 全局 BA

### 2.5 ORB-SLAM3 性能指标

| 场景 | 精度 (RMSE) | 实时性 | 鲁棒性 |
|------|------------|--------|--------|
| EuRoC MAV (室内) | 0.03-0.05m | 30-50 FPS | 高 |
| TUM-VI (室内) | 0.04-0.08m | 20-30 FPS | 高 |
| KITTI (室外) | 0.5-1.0% | 15-20 FPS | 中 |
| 快速旋转 | 0.1-0.2m | 20-30 FPS | 中 |
| 低纹理 | 0.2-0.5m | 15-25 FPS | 低 |

---

## §3 — Kimera：语义 SLAM 与 Mesh 重建

### 3.1 系统概述

Kimera 是由 MIT 开发的视觉-惯性 SLAM 系统，核心特点：
- **Kimera-VIO**：基于 GTSAM 的 VIO 模块
- **Kimera-Semantics**：语义分割 + 3D 重建
- **Kimera-Mesher**：实时 mesh 生成

### 3.2 Kimera-VIO：基于 GTSAM 的优化

#### 3.2.1 因子图表示

```
Factor Graph:

  X1 --- X2 --- X3 --- X4  ...  (位姿节点)
  |      |      |      |
  L1     L2     L3     L4       (路标点)
  |      |      |      |
  IMU1   IMU2   IMU3   IMU4    (IMU 因子)
```

**因子类型**：
- **视觉因子**：重投影误差
- **IMU 因子**：预积分约束
- **先验因子**：初始位姿约束

#### 3.2.2 增量式求解

Kimera 使用 GTSAM 的 **iSAM2** 算法：

1. **变量排序**：COLAMD 或 METIS
2. **增量式 QR 分解**：仅更新受影响的因子
3. **回退策略**：定期执行完整 BA

### 3.3 Kimera-Semantics：语义 3D 重建

#### 3.3.1 语义分割集成

```
输入图像 → [CNN 分割] → 语义掩码
    ↓
[3D 投影] → 体素语义投票
    ↓
[语义 Mesh] → 带标签的 3D 模型
```

**支持的语义分割网络**：
- Mask R-CNN
- DeepLabV3+
- SegNet

#### 3.3.2 语义 Mesh 生成

Kimera 使用 **Marching Cubes** 算法从 TSDF (Truncated Signed Distance Function) 提取 mesh：

$$M = \{ x \in \mathbb{R}^3 | TSDF(x) = 0 \}$$

**语义信息融合**：
- 每个体素存储语义概率分布
- 使用贝叶斯更新融合多帧观测

### 3.4 Kimera vs ORB-SLAM3

| 特性 | Kimera | ORB-SLAM3 |
|------|--------|-----------|
| **后端优化** | GTSAM (因子图) | g2o/Ceres (BA) |
| **语义支持** | 原生支持 | 需外部集成 |
| **Mesh 重建** | 实时 | 离线/后处理 |
| **多地图** | 不支持 | 支持 (Atlas) |
| **回环检测** | 支持 | 支持 (DBoW2) |
| **代码复杂度** | 较高 | 中等 |
| **依赖** | GTSAM, OpenCV | OpenCV, DBoW2 |

---

## §4 — OpenVINS：滤波法 VIO

### 4.1 系统概述

OpenVINS 是由 University of Delaware 开发的开源视觉-惯性里程计，基于 **MSCKF (Multi-State Constraint Kalman Filter)** 框架。

### 4.2 MSCKF 核心原理

#### 4.2.1 状态向量

MSCKF 的状态向量：

$$\mathbf{x} = [\mathbf{x}_{IMU}^T, \mathbf{x}_{C_1}^T, \mathbf{x}_{C_2}^T, ..., \mathbf{x}_{C_n}^T]^T$$

其中：
- $\mathbf{x}_{IMU}$：IMU 状态（位姿、速度、bias）
- $\mathbf{x}_{C_i}$：第 $i$ 个相机位姿（滑动窗口）

#### 4.2.2 滤波更新流程

```
预测 (Prediction):
  IMU 测量 → 状态传播 → 协方差传播

更新 (Update):
  特征跟踪 → 多帧约束 → EKF 更新
```

**MSCKF 的关键创新**：
- 不估计 3D 路标点（与 EKF-SLAM 不同）
- 利用多帧观测的几何约束
- 避免路标点维度爆炸

### 4.3 OpenVINS 特性

| 特性 | 描述 |
|------|------|
| **模块化设计** | 易于扩展和定制 |
| **多相机支持** | 支持任意数量相机 |
| **在线标定** | 支持相机-IMU 外参在线估计 |
| **仿真支持** | 内置仿真环境 |
| **ROS 集成** | 完整的 ROS/ROS2 接口 |

### 4.4 滤波法 vs 优化法

| 维度 | 滤波法 (OpenVINS) | 优化法 (ORB-SLAM3) |
|------|------------------|-------------------|
| **计算复杂度** | O(N) | O(N^3) 或 O(N)（稀疏优化） |
| **精度** | 中等 | 高 |
| **回环检测** | 不支持 | 支持 |
| **全局一致性** | 差 | 好 |
| **实时性** | 优秀 | 良好 |
| **内存占用** | 低 | 高 |
| **适用场景** | 资源受限、实时性要求高 | 精度要求高、长期运行 |

---

## §5 — 各 SLAM 系统在 Ego-centric 场景的表现对比

### 5.1 Ego-centric 场景特点

| 特点 | 影响 | 对 SLAM 的要求 |
|------|------|--------------|
| **快速运动** | 运动模糊、特征跟踪困难 | 高帧率、鲁棒特征 |
| **剧烈旋转** | 纯旋转退化、尺度丢失 | IMU 辅助、多目 |
| **动态物体** | 特征污染、地图污染 | 动态物体检测与剔除 |
| **遮挡频繁** | 跟踪丢失 | 多地图、快速重定位 |
| **光照变化** | 特征不稳定 | 光照不变特征、自适应 |

### 5.2 性能对比

#### 在 Ego-centric 数据集上的表现

| 系统 | EuRoC MH | TUM-VI | ADVIO | 实时性 |
|------|----------|--------|-------|--------|
| **ORB-SLAM3** | 0.04m | 0.06m | 0.8m | 30 FPS |
| **Kimera** | 0.05m | 0.07m | 1.2m | 25 FPS |
| **OpenVINS** | 0.06m | 0.08m | 1.5m | 45 FPS |
| **VINS-Fusion** | 0.08m | 0.10m | 2.0m | 30 FPS |
| **OKVIS** | 0.15m | 0.20m | 3.0m | 20 FPS |

*注：数值为 RMSE (m)，越低越好*

### 5.3 选型建议

| 场景 | 推荐系统 | 理由 |
|------|---------|------|
| **通用机器人** | ORB-SLAM3 | 功能全面、多地图、回环 |
| **语义导航** | Kimera | 语义 mesh、场景理解 |
| **资源受限** | OpenVINS | 轻量、高效、模块化 |
| **长期运行** | ORB-SLAM3 | 多地图、地图合并 |
| **快速原型** | OpenVINS | 易于定制、ROS 集成 |

---

## §6 — SLAM 在动态场景的挑战

### 6.1 动态物体检测方法

#### 6.1.1 几何方法

| 方法 | 原理 | 局限 |
|------|------|------|
| **RANSAC 外点剔除** | 假设静态场景占多数 | 动态物体占比较大时失效 |
| **多视图几何** | 三角化一致性检验 | 计算量大 |
| **光流约束** | 光流与相机运动不一致 | 需要稠密光流 |

#### 6.1.2 深度学习方法

| 方法 | 原理 | 代表工作 |
|------|------|---------|
| **语义分割** | 先验知识标记动态物体 | Mask R-CNN + SLAM |
| **实例分割** | 实例级动态检测 | DynaSLAM |
| **光流学习** | 学习光流与静态场景的关系 | DROID-SLAM |
| **端到端** | 直接学习动态/静态分类 | EMVS |

### 6.2 动态 SLAM 系统

#### DynaSLAM

```
输入图像 → [Mask R-CNN] → 动态物体掩码
    ↓
[静态特征提取] → 仅使用静态区域特征
    ↓
[标准 SLAM Pipeline]
```

**性能提升**：
- 动态场景 RMSE 降低 50%+
- 支持动态物体的跟踪与重建

#### DROID-SLAM

基于学习的 SLAM 系统：
- 使用 **RAFT** 光流网络
- 端到端训练
- 对动态场景更鲁棒

### 6.3 动态场景挑战总结

| 挑战 | 影响 | 解决方案 |
|------|------|---------|
| **特征污染** | 位姿估计偏差 | 动态物体检测 + 剔除 |
| **地图污染** | 地图包含动态物体 | 语义过滤、时序一致性 |
| **跟踪丢失** | 动态物体遮挡 | 多地图、预测跟踪 |
| **尺度漂移** | 动态物体导致尺度错误 | IMU 辅助、深度约束 |
| **实时性** | 动态检测增加计算量 | 轻量网络、GPU 加速 |

---

## §7 — 与 DVAS 项目的关联

### 7.1 在 DVAS 中的定位

```
DVAS 系统架构:

[传感器输入]
    ├── 双目图像 → [深度估计] → [SLAM]
    ├── IMU 数据 ──────────────→ [SLAM]
    └── RGBD 数据 ─────────────→ [SLAM]
                                      ↓
                              [相机轨迹]
                              [稀疏/稠密地图]
                                      ↓
                              [手势估计]
                              [场景重建]
                              [导航规划]
```

### 7.2 DVAS 特定需求

| 需求 | SLAM 要求 | 推荐方案 |
|------|----------|---------|
| **Ego-centric 采集** | 鲁棒快速运动 | ORB-SLAM3 (VI mode) |
| **手-物交互** | 高精度局部地图 | Kimera (语义 mesh) |
| **长时间采集** | 多地图、不丢失 | ORB-SLAM3 (Atlas) |
| **实时反馈** | 低延迟 | OpenVINS |
| **离线处理** | 高精度、全局一致 | ORB-SLAM3 + 全局 BA |

### 7.3 与下游模块的接口

```python
class SLAMInterface:
    """
    SLAM 系统统一接口
    """
    def __init__(self, system_type: str):
        if system_type == "orb_slam3":
            self.slam = ORBSLAM3Wrapper()
        elif system_type == "kimera":
            self.slam = KimeraWrapper()
        elif system_type == "openvins":
            self.slam = OpenVINSWrapper()
    
    def process_frame(self, img_left, img_right, imu_data):
        """
        处理一帧数据
        返回: 当前位姿 (SE3), 跟踪状态
        """
        return self.slam.track(img_left, img_right, imu_data)
    
    def get_trajectory(self):
        """获取完整相机轨迹"""
        return self.slam.get_keyframe_trajectory()
    
    def get_map(self):
        """获取地图点云"""
        return self.slam.get_map_points()
```

---

## §8 — 参考与资源

### 8.1 关键论文

1. **ORB-SLAM3** - Campos et al. (2021) - "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM" (IEEE T-RO)
2. **Kimera** - Rosinol et al. (2020) - "Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping" (ICRA)
3. **OpenVINS** - Geneva et al. (2020) - "OpenVINS: A Research Platform for Visual-Inertial Estimation" (IROS Workshop)
4. **MSCKF** - Mourikis & Roumeliotis (2007) - "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation" (ICRA)
5. **DynaSLAM** - Bescos et al. (2018) - "DynaSLAM: Tracking, Mapping, and Inpainting in Dynamic Scenes" (IEEE T-RO)
6. **DROID-SLAM** - Teed & Deng (2021) - "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras" (NeurIPS)

### 8.2 开源实现

| 系统 | 链接 | 许可证 |
|------|------|--------|
| ORB-SLAM3 | https://github.com/UZ-SLAMLab/ORB_SLAM3 | GPL-3.0 |
| Kimera | https://github.com/MIT-SPARK/Kimera | BSD-3 |
| OpenVINS | https://github.com/rpng/open_vins | MIT |
| VINS-Fusion | https://github.com/HKUST-Aerial-Robotics/VINS-Fusion | GPL-3.0 |
| DROID-SLAM | https://github.com/princeton-vl/DROID-SLAM | BSD-3 |

### 8.3 相关文档

- [双目+IMU 协同标定](21-stereo-imu.md) — SLAM 的输入依赖
- [深度估计](22-depth-estimation.md) — 深度图可用于稠密 SLAM
- [多传感器融合](26-sensor-fusion.md) — SLAM 是多传感器融合的核心应用

---

*Layer: 03-perception | Prev: [深度估计](22-depth-estimation.md) | Next: [手势估计](24-hand-pose.md)*
