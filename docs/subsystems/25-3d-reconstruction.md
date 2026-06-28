---
id: 3d-reconstruction
title: "3D重建"
status: complete
complexity: high
related:
  - "./02-depth-estimation.md"
  - "./03-slam.md"
  - "./04-hand-pose.md"
prerequisites:
  - "三维几何与变换"
  - "计算机图形学基础"
  - "深度学习优化"
last_validated: 2026-06-27
---

# 3D 重建

## §0 — One-liner

3D 重建从多视角图像或视频中恢复场景的三维几何与外观，Gaussian Splatting 和 NeRF 代表了当前神经渲染与实时重建的前沿范式。

## §1 — 3D 重建基础

### 1.1 问题分类

| 分类维度 | 类型 | 描述 |
|---------|------|------|
| **输入** | 多视角图像、视频、RGBD、LiDAR | 不同输入对应不同方法 |
| **输出** | 点云、Mesh、体素、NeRF、Gaussian | 不同表示各有优劣 |
| **动态性** | 静态、动态 | 动态重建更复杂 |
| **规模** | 物体级、场景级、大规模 | 方法复杂度递增 |

### 1.2 经典方法回顾

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| **SfM (Structure from Motion)** | 特征匹配 + 三角化 | 稀疏、慢但鲁棒 |
| **MVS (Multi-View Stereo)** | 稠密匹配 + 深度图融合 | 稠密但计算量大 |
| **TSDF Fusion** | 体素网格 + SDF 更新 | 实时但内存大 |
| **Poisson Reconstruction** | 隐式表面重建 | 高质量但需要法向 |

---

## §2 — Gaussian Splatting：原理、优化与实时渲染

### 2.1 论文信息

| 属性 | 内容 |
|------|------|
| **标题** | 3D Gaussian Splatting for Real-Time Radiance Field Rendering |
| **作者** | Kerbl et al. |
| **机构** | INRIA / Universite Cote d'Azur / Max Planck Institute |
| **发表** | SIGGRAPH 2023 (Best Paper) |
| **代码** | https://github.com/graphdeco-inria/gaussian-splatting |

### 2.2 核心原理

#### 2.2.1 3D Gaussian 表示

场景由大量 **3D 高斯椭球** 表示：

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

其中每个高斯包含以下参数：

| 参数 | 符号 | 维度 | 含义 |
|------|------|------|------|
| 中心位置 | $\boldsymbol{\mu}$ | $\mathbb{R}^3$ | 高斯中心 |
| 协方差矩阵 | $\boldsymbol{\Sigma}$ | $\mathbb{R}^{3 \times 3}$ | 形状与方向 |
| 颜色 | $\mathbf{c}$ | $\mathbb{R}^3$ | RGB 颜色 |
| 不透明度 | $\alpha$ | $\mathbb{R}$ | [0, 1] |

#### 2.2.2 协方差矩阵的参数化

为避免优化过程中的非正定性问题，协方差矩阵参数化为：

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

其中：
- $\mathbf{R} \in SO(3)$：旋转矩阵（四元数表示）
- $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$：尺度矩阵

**参数化优势**：
- 自动保证正定性
- 物理意义明确（旋转 + 各向异性缩放）
- 优化稳定

#### 2.2.3 Splatting 渲染

将 3D 高斯投影到 2D 图像平面：

$$\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^T\mathbf{J}^T$$

其中：
- $\mathbf{W}$：视图变换矩阵
- $\mathbf{J}$：投影变换的雅可比矩阵

**渲染方程**（按深度排序的 $\alpha$-混合）：

$$C(\mathbf{p}) = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

### 2.3 优化策略

#### 2.3.1 自适应密度控制

Gaussian Splatting 的核心创新之一：**自适应增删高斯点**

```
初始化: SfM 点云
    ↓
训练迭代:
    ├── 渲染图像
    ├── 计算损失
    ├── 梯度下降更新参数
    └── 自适应密度控制:
        ├── 梯度大的区域 → 分裂 (Split)
        ├── 不透明度低的点 → 克隆 (Clone)
        └── 不透明度极低的点 → 删除 (Prune)
```

**分裂策略**：
- 条件：梯度 > 阈值 且 尺度较大
- 操作：将一个大高斯分裂为两个较小的高斯
- 新尺度：$s' = s / \sqrt{2}$

**克隆策略**：
- 条件：梯度 > 阈值 且 尺度较小
- 操作：复制当前高斯并微小扰动位置

#### 2.3.2 训练流程

```python
# 伪代码
def train_gaussian_splatting(images, cameras):
    # 初始化
    gaussians = initialize_from_sfm(images, cameras)
    
    for iteration in range(max_iterations):
        # 随机采样视角
        camera = sample_camera(cameras)
        
        # 渲染
        rendered = render(gaussians, camera)
        
        # 计算损失
        loss = l1_loss(rendered, ground_truth) + \
               lambda_ssim * (1 - ssim(rendered, ground_truth))
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 自适应密度控制
        if iteration % densify_interval == 0:
            gaussians.adaptive_density_control()
    
    return gaussians
```

### 2.4 性能特点

| 指标 | 数值 | 说明 |
|------|------|------|
| **渲染速度** | 100-300 FPS @ 1080p | RTX 4090 |
| **训练时间** | 5-15 分钟 | 典型场景 |
| **内存占用** | 1-5 GB | 取决于高斯数量 |
| **重建质量** | PSNR > 30 dB | 高质量 |
| **高斯数量** | 100K - 5M | 自适应 |

### 2.5 优缺点分析

| 优点 | 缺点 |
|------|------|
| 实时渲染 | 内存占用大 |
| 训练速度快 | 编辑困难 |
| 高质量重建 | 动态场景支持有限 |
| 无需神经网络推理 | 存储空间大 |
| 可解释性强 | 大规模场景扩展性待提升 |

---

## §3 — NeRF：神经辐射场基础

### 3.1 论文信息

| 属性 | 内容 |
|------|------|
| **标题** | NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis |
| **作者** | Mildenhall et al. |
| **机构** | UC Berkeley / Google Research |
| **发表** | ECCV 2020 (Best Paper) |
| **代码** | https://github.com/bmild/nerf |

### 3.2 核心原理

#### 3.2.1 场景表示

NeRF 将场景表示为 **5D 神经辐射场**：

$$F_{\Theta}: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

其中：
- $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$：空间位置
- $\mathbf{d} = (\theta, \phi) \in S^2$：视角方向
- $\mathbf{c} = (r, g, b)$：RGB 颜色
- $\sigma \in \mathbb{R}^+$：体密度

#### 3.2.2 位置编码

为捕捉高频细节，NeRF 使用 **位置编码**：

$$\gamma(p) = [\sin(2^0 \pi p), \cos(2^0 \pi p), ..., \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)]$$

**作用**：
- 将低频坐标映射到高频空间
- 使 MLP 能学习高频细节
- $L = 10$ 用于位置，$L = 4$ 用于方向

#### 3.2.3 体渲染

从相机光线采样点，通过体渲染方程合成像素颜色：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i$$

其中：
- $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$：累积透射率
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$：不透明度
- $\delta_i$：采样点间距

### 3.3 训练策略

#### 3.3.1 损失函数

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{C}(\mathbf{r}) - C(\mathbf{r}) \right\|^2$$

#### 3.3.2 分层采样

NeRF 采用 **粗-细** 两级采样策略：

```
第一阶段 (Coarse):
  均匀采样 N_c = 64 个点
  → 估计粗略的体密度分布

第二阶段 (Fine):
  根据 coarse 网络的概率密度采样 N_f = 128 个点
  → 在重要区域集中采样
```

### 3.4 NeRF 变体

| 变体 | 创新点 | 性能 |
|------|--------|------|
| **Instant-NGP** | 哈希编码 + 小 MLP | 训练 5 分钟 |
| **Mip-NeRF** | 锥形光线 + 积分位置编码 | 抗锯齿 |
| **NeRF-W** | 外观嵌入 + 瞬态处理 | 无界场景 |
| **NeRF++** | 内外空间分解 | 360° 场景 |
| **D-NeRF** | 时间编码 | 动态场景 |
| **NeRFstudio** | 统一框架 | 易用性 |

### 3.5 NeRF vs Gaussian Splatting

| 维度 | NeRF | Gaussian Splatting |
|------|------|------------------|
| **表示** | 隐式 MLP | 显式高斯点 |
| **渲染** | 光线行进 (慢) | Splatting (快) |
| **训练** | 数小时 | 数分钟 |
| **内存** | 小 (仅 MLP 参数) | 大 (高斯点云) |
| **质量** | 高 | 高 |
| **实时性** | 否 | 是 |
| **编辑性** | 差 | 中等 |
| **动态** | 需特殊处理 | 需特殊处理 |

---

## §4 — 动态场景重建

### 4.1 挑战

| 挑战 | 描述 |
|------|------|
| **时间一致性** | 相邻帧的重建应连续 |
| **变形建模** | 非刚性物体的变形 |
| **遮挡处理** | 动态物体的遮挡与显露 |
| **计算效率** | 动态场景计算量更大 |

### 4.2 方法分类

#### 4.2.1 基于变形场的方法

$$\mathbf{x}(t) = \mathbf{x}_0 + \mathbf{D}(\mathbf{x}_0, t)$$

其中 $\mathbf{D}$ 为变形场。

**代表工作**：
- **D-NeRF**：时间作为额外输入
- **Neural 3D Video**：动态高斯点云

#### 4.2.2 基于骨骼的方法

$$\mathbf{x}(t) = \sum_{k} w_k(\mathbf{x}) \cdot \mathbf{T}_k(t) \cdot \mathbf{x}_0$$

其中：
- $w_k$：骨骼权重
- $\mathbf{T}_k$：骨骼变换

**代表工作**：
- **HumanNeRF**：人体动态重建
- **NeuMan**：人体 + 场景联合重建

#### 4.2.3 基于高斯变形的方法

Dynamic Gaussian Splatting：

$$\mu(t) = \mu_0 + \Delta\mu(t)$$
$$\Sigma(t) = R(t) S(t) S(t)^T R(t)^T$$

**代表工作**：
- **4D Gaussian Splatting**：时间维度扩展
- **Deformable 3D Gaussians**：变形场 + 高斯

### 4.3 动态重建性能对比

| 方法 | 表示 | 实时性 | 质量 | 适用场景 |
|------|------|--------|------|---------|
| **D-NeRF** | MLP + 时间 | 否 | 高 | 小规模动态 |
| **4D GS** | 时序高斯 | 部分 | 高 | 动态场景 |
| **Neural 3D Video** | 动态高斯 | 是 | 中 | 视频渲染 |
| **Luma AI** | 高斯 + 优化 | 是 | 高 | 商业应用 |

---

## §5 — 在机器人场景中的应用

### 5.1 场景理解

#### 5.1.1 语义 3D 重建

结合语义分割与 3D 重建：

```
输入: 多视角 RGB 图像
    ↓
[2D 语义分割] → 每像素语义标签
    ↓
[3D 重建] → 点云 / 高斯点云
    ↓
[语义融合] → 语义 3D 地图
```

**应用**：
- 物体识别与定位
- 场景分类
- 导航路径规划

#### 5.1.2 可变形物体建模

对于机器人操作中的可变形物体：

| 物体类型 | 重建方法 | 应用 |
|---------|---------|------|
| **刚性物体** | 标准 NeRF / GS | 抓取规划 |
| **可变形物体** | D-NeRF / 动态 GS | 操作仿真 |
| **流体** | 粒子法 + 神经渲染 | 倾倒任务 |
| **衣物** | 基于物理的模拟 | 折叠任务 |

### 5.2 导航应用

#### 5.2.1 占据地图构建

从重建结果提取占据信息：

$$O(v) = \begin{cases} \text{occupied} & \text{if } SDF(v) < 0 \\ \text{free} & \text{if } SDF(v) > \epsilon \\ \text{unknown} & \text{otherwise} \end{cases}$$

#### 5.2.2 路径规划

```python
def plan_path(start, goal, occupancy_map):
    """
    基于占据地图的路径规划
    """
    # A* 或 RRT* 算法
    path = a_star(start, goal, occupancy_map)
    return path
```

### 5.3 机器人场景重建案例

| 场景 | 方法 | 效果 |
|------|------|------|
| **室内导航** | Gaussian Splatting + 语义 | 实时语义地图 |
| **物体抓取** | NeRF + 6D 姿态 | 精确物体模型 |
| **场景编辑** | GS + 交互式工具 | 实时编辑 |
| **遥操作** | 动态 GS | 远程场景重建 |

---

## §6 — 与 DVAS 项目的关联

### 6.1 在 DVAS 中的定位

```
DVAS 3D 重建 Pipeline:

[ ego-centric 视频采集]
         ↓
[相机位姿估计] (SLAM / SfM)
         ↓
[3D 重建]
    ├── [Gaussian Splatting] → 实时渲染
    ├── [NeRF] → 高质量离线重建
    └── [TSDF Fusion] → 稠密地图
         ↓
[应用]
    ├── 场景理解
    ├── 导航规划
    ├── 物体操作
    └── 数据可视化
```

### 6.2 DVAS 特定需求

| 需求 | 技术方案 | 预期效果 |
|------|---------|---------|
| **实时场景重建** | Gaussian Splatting | 30 FPS 渲染 |
| **高质量离线重建** | NeRF / Instant-NGP | PSNR > 35 dB |
| **动态场景** | 4D Gaussian Splatting | 支持人体/物体运动 |
| **语义地图** | GS + 语义分割 | 带标签的 3D 地图 |
| **大规模场景** | 分块重建 + 融合 | 房间/楼层级别 |

### 6.3 与下游模块的接口

```python
class ReconstructionPipeline:
    """
    3D 重建 Pipeline 接口
    """
    def __init__(self, method="gaussian_splatting"):
        if method == "gaussian_splatting":
            self.reconstructor = GaussianSplatting()
        elif method == "nerf":
            self.reconstructor = NeRF()
    
    def process_sequence(self, images, cameras):
        """
        处理图像序列进行重建
        
        输入:
            - images: 图像列表
            - cameras: 相机参数列表
        
        输出:
            - scene: 重建的场景表示
        """
        return self.reconstructor.train(images, cameras)
    
    def render_view(self, camera):
        """
        从新视角渲染图像
        """
        return self.reconstructor.render(camera)
    
    def export_mesh(self, output_path):
        """
        导出 mesh 模型
        """
        return self.reconstructor.extract_mesh(output_path)
```

---

## §7 — 参考与资源

### 7.1 关键论文

1. **Gaussian Splatting** - Kerbl et al. (2023) - "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH)
2. **NeRF** - Mildenhall et al. (2020) - "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (ECCV)
3. **Instant-NGP** - Muller et al. (2022) - "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" (SIGGRAPH)
4. **Mip-NeRF** - Barron et al. (2021) - "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields" (ICCV)
5. **D-NeRF** - Pumarola et al. (2021) - "D-NeRF: Neural Radiance Fields for Dynamic Scenes" (CVPR)
6. **4D Gaussian Splatting** - Wu et al. (2024) - "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering" (CVPR)

### 7.2 开源实现

| 项目 | 链接 | 说明 |
|------|------|------|
| Gaussian Splatting | https://github.com/graphdeco-inria/gaussian-splatting | 官方实现 |
| NeRF | https://github.com/bmild/nerf | 官方实现 |
| Instant-NGP | https://github.com/NVlabs/instant-ngp | NVIDIA 官方 |
| nerfstudio | https://github.com/nerfstudio-project/nerfstudio | 统一框架 |
| gsplat | https://github.com/nerfstudio-project/gsplat | 高效 GS 库 |

### 7.3 相关文档

- [深度估计](22-depth-estimation.md) — 深度图可用于初始化重建
- [SLAM 系统](23-slam.md) — 提供相机位姿
- [手势估计](24-hand-pose.md) — 动态重建的应用场景

---

*Layer: 03-perception | Prev: [手势估计](24-hand-pose.md) | Next: [多传感器融合](26-sensor-fusion.md)*
