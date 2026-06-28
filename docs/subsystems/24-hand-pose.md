---
id: hand-pose-estimation
title: "手势估计"
status: complete
complexity: high
related:
  - "./02-depth-estimation.md"
  - "./03-slam.md"
  - "./05-3d-reconstruction.md"
prerequisites:
  - "人体姿态估计基础"
  - "3D 几何与变换"
  - "卷积神经网络与 Transformer"
last_validated: 2026-06-27
---

# 手势估计

## §0 — One-liner

手势估计从单目或双目图像中重建手部 3D 骨骼结构，是 ego-centric 场景下理解人类操作意图、实现自然交互的核心感知技术。

## §1 — 手势估计问题定义

### 1.1 手部骨骼模型

标准手部骨骼模型包含 21 个关键点（MANO 模型扩展至 16 个关节、45 个自由度）：

```
手部骨骼层级结构:

Wrist (腕关节)
├── Thumb (拇指)
│   ├── MCP (掌指关节)
│   ├── IP (近端指间关节)
│   └── Tip (指尖)
├── Index (食指)
│   ├── MCP
│   ├── PIP (近端指间关节)
│   ├── DIP (远端指间关节)
│   └── Tip
├── Middle (中指)
│   ├── MCP
│   ├── PIP
│   ├── DIP
│   └── Tip
├── Ring (无名指)
│   ├── MCP
│   ├── PIP
│   ├── DIP
│   └── Tip
└── Pinky (小指)
    ├── MCP
    ├── PIP
    ├── DIP
    └── Tip
```

### 1.2 参数化表示

#### 1.2.1 3D 关节位置

$$J = \{ \mathbf{j}_i \in \mathbb{R}^3 \}_{i=1}^{21}$$

#### 1.2.2 关节角度 (MANO 模型)

$$\theta = \{ \theta_{global}, \theta_{pose}, \theta_{shape} \}$$

其中：
- $\theta_{global} \in \mathbb{R}^3$：全局旋转
- $\theta_{pose} \in \mathbb{R}^{45}$：关节角度（15 个关节 × 3 DoF）
- $\theta_{shape} \in \mathbb{R}^{10}$：手型参数（PCA 系数）

#### 1.2.3 2D 热图表示

$$H_i(u, v) = \exp\left(-\frac{(u - u_i)^2 + (v - v_i)^2}{2\sigma^2}\right)$$

---

## §2 — HaMeR：从单图重建 3D 手姿

### 2.1 论文信息

| 属性 | 内容 |
|------|------|
| **标题** | Reconstructing Hands in 3D with Transformers |
| **作者** | Bardia et al. |
| **机构** | EPFL / Meta |
| **发表** | ICCV 2023 |
| **代码** | https://github.com/geopavlakos/HaMeR |

### 2.2 核心架构

HaMeR 采用 **Transformer-based** 架构，核心创新在于：

```
输入单张 RGB 图像 (H×W×3)
    ↓
[图像编码器] (ViT / ResNet)
    ↓
[Transformer 编码器] ──→ 全局特征聚合
    ↓
[3D 手姿解码器]
    ├── 2D 热图预测
    ├── 3D 关节位置预测
    └── MANO 参数预测
    ↓
[后处理] (可选)
    ├── 时序平滑
    └── 多视角融合
```

### 2.3 关键技术细节

#### 2.3.1 图像编码器

HaMeR 使用 **ViT (Vision Transformer)** 作为图像编码器：

- **输入**：手部区域裁剪图像（224×224）
- **分块**：将图像分割为 16×16 的 patch
- **编码**：通过 Transformer encoder 提取特征
- **输出**：全局特征向量 + 空间特征图

#### 2.3.2 3D 手姿解码

HaMeR 同时预测三种表示：

| 表示 | 维度 | 作用 |
|------|------|------|
| **2D 热图** | 21 × H' × W' | 提供 2D 位置约束 |
| **3D 关节位置** | 21 × 3 | 直接 3D 坐标 |
| **MANO 参数** | 61 (16+45) | 参数化手型 |

**多任务损失**：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{2D} + \lambda_2 \mathcal{L}_{3D} + \lambda_3 \mathcal{L}_{MANO}$$

其中：
- $\mathcal{L}_{2D}$：2D 热图交叉熵损失
- $\mathcal{L}_{3D}$：3D 关节位置 L2 损失
- $\mathcal{L}_{MANO}$：MANO 参数 L2 损失

#### 2.3.3 训练策略

**大规模预训练**：
- 数据集：100DOH + DexYCB + FreiHAND + HO-3D
- 总样本数：> 2M 张手部图像
- 数据增强：颜色抖动、随机裁剪、遮挡模拟

### 2.4 性能指标

| 数据集 | MPJPE (mm) | PA-MPJPE (mm) | 2D PCK@0.05 |
|--------|-----------|--------------|------------|
| **FreiHAND** | 6.2 | 4.8 | 0.98 |
| **HO-3D** | 8.5 | 6.2 | 0.95 |
| **DexYCB** | 7.8 | 5.5 | 0.96 |
| **100DOH** | 9.1 | 7.2 | 0.93 |

*注：MPJPE = Mean Per Joint Position Error，PA-MPJPE = Procrustes Aligned MPJPE*

---

## §3 — 100DOH：100 Days of Hands 数据集

### 3.1 数据集概述

| 属性 | 内容 |
|------|------|
| **全称** | 100 Days of Hands: A Dataset for Egocentric Hand Detection and Pose Estimation |
| **作者** | Shan et al. |
| **机构** | UNC Chapel Hill / Google |
| **规模** | 131,060 张图像，28,624 个视频片段 |
| **来源** | YouTube  ego-centric 视频 |

### 3.2 数据特点

#### 3.2.1 多样性

| 维度 | 分布 |
|------|------|
| **场景** | 室内 (65%)、室外 (35%) |
| **活动** | 烹饪、运动、手工、日常活动 |
| **手部数量** | 单手 (60%)、双手 (40%) |
| **遮挡** | 无遮挡 (45%)、部分遮挡 (40%)、严重遮挡 (15%) |
| **光照** | 明亮 (50%)、正常 (35%)、暗光 (15%) |

#### 3.2.2 标注内容

- **边界框 (BBox)**：手部区域检测
- **2D 关键点**：21 个关节点
- **3D 关键点**：通过多视角重建获得
- **可见性标签**：每个关键点的可见性

### 3.3 在手势估计中的作用

100DOH 作为 **预训练数据源** 的价值：

1. **大规模**：131K 张图像，覆盖广泛场景
2. **ego-centric**：与机器人/AR 场景一致
3. **多样性**：各种光照、遮挡、姿态
4. **弱监督**：可用于半监督学习

---

## §4 — DexYCB：基于 YCB 的手部数据集

### 4.1 数据集概述

| 属性 | 内容 |
|------|------|
| **全称** | DexYCB: A Benchmark for Capturing Hand Grasping of Objects |
| **作者** | Chao et al. |
| **机构** | NVIDIA |
| **规模** | 582K 帧，10 名参与者，20 个 YCB 物体 |
| **采集设备** | 8 个 RGBD 相机 (RealSense D415) |

### 4.2 数据特点

#### 4.2.1 多视角采集设置

```
        [Camera 1]
           │
    [Camera 8]    [Camera 2]
         │          │
    [Camera 7] ── [Hand] ── [Camera 3]
         │          │
    [Camera 6]    [Camera 4]
           │
        [Camera 5]
```

- 8 个同步 RGBD 相机环绕手部
- 提供多视角深度图
- 支持 3D 重建验证

#### 4.2.2 标注精度

| 标注类型 | 方法 | 精度 |
|---------|------|------|
| **2D 关键点** | 人工标注 + 自动验证 | ±3 px |
| **3D 关键点** | 多视角三角化 | ±5 mm |
| **物体 6D 姿态** | ArUco marker + ICP | ±2 mm, ±1° |
| **手-物接触** | 人工标注 | 像素级 |

### 4.3 在手-物交互研究中的价值

DexYCB 是研究 **手-物交互 (Hand-Object Interaction)** 的核心数据集：

| 研究方向 | 应用 |
|---------|------|
| **抓取姿态生成** | 给定物体生成自然抓取姿态 |
| **手-物接触检测** | 检测手指与物体的接触区域 |
| **力估计** | 从视觉估计接触力 |
| **抓取稳定性** | 评估抓取是否稳定 |

---

## §5 — Ego-centric 视角下的手势估计挑战

### 5.1 核心挑战分析

| 挑战 | 描述 | 影响 |
|------|------|------|
| **严重自遮挡** | 手掌遮挡手指、手指间相互遮挡 | 关键点检测失败 |
| **快速运动** | 手部快速移动导致运动模糊 | 特征跟踪困难 |
| **光照变化** | 室内/室外、阴影变化 | 特征不稳定 |
| **尺度变化** | 手在图像中的大小变化大 | 检测困难 |
| **视角受限** | 只能看到手背或手心 | 3D 重建歧义 |
| **背景复杂** | 与物体交互时背景多变 | 分割困难 |

### 5.2 解决方案

#### 5.2.1 遮挡处理

| 方法 | 原理 | 代表工作 |
|------|------|---------|
| **多视角融合** | 利用多相机减少遮挡 | DexYCB 采集方案 |
| **先验知识** | MANO 模型约束 | HaMeR, FrankMocap |
| **注意力机制** | 关注可见区域 | TransHand |
| **时序信息** | 利用前后帧补全 | VideoPose3D |

#### 5.2.2 快速运动处理

```python
# 时序平滑策略
def temporal_smooth(current_pose, history_poses, alpha=0.7):
    """
    指数移动平均平滑
    """
    if len(history_poses) == 0:
        return current_pose
    
    smoothed = alpha * current_pose + (1 - alpha) * history_poses[-1]
    return smoothed

# 卡尔曼滤波
def kalman_filter(pose_measurement, state, covariance):
    """
    利用运动模型预测和测量更新
    """
    # 预测
    state_pred = A @ state  # A: 状态转移矩阵
    cov_pred = A @ covariance @ A.T + Q  # Q: 过程噪声
    
    # 更新
    K = cov_pred @ H.T @ np.linalg.inv(H @ cov_pred @ H.T + R)
    state_updated = state_pred + K @ (pose_measurement - H @ state_pred)
    
    return state_updated
```

### 5.3 Ego-centric 手势估计性能

| 方法 | 数据集 | MPJPE (mm) | 实时性 | 备注 |
|------|--------|-----------|--------|------|
| **HaMeR** | 100DOH | 9.1 | 15 FPS | 单图 |
| **FrankMocap** | FreiHAND | 7.5 | 10 FPS | 需要全身 |
| **HandMesh** | FreiHAND | 8.2 | 25 FPS | 轻量网络 |
| **METRO** | FreiHAND | 6.5 | 5 FPS | Transformer |
| **MeshGraphormer** | FreiHAND | 6.3 | 8 FPS | 图 Transformer |

---

## §6 — 手-物交互检测

### 6.1 问题定义

手-物交互检测需要同时估计：
- **手部 3D 姿态**：21 个关节点
- **物体 6D 姿态**：旋转 + 平移
- **接触区域**：手指与物体的接触点
- **交互类型**：抓取、触摸、推动等

### 6.2 方法分类

| 方法类型 | 原理 | 代表工作 | 优缺点 |
|---------|------|---------|--------|
| **两阶段** | 先检测手/物，再估计交互 | Hand-Object CNN | 简单但误差累积 |
| **联合估计** | 同时估计手和物 | HO-3D, DexYCB baseline | 更准确但复杂 |
| **基于接触** | 显式建模接触 | ContactPose | 物理一致性好 |
| **基于生成** | 生成式模型 | GraspTTA | 可生成多样姿态 |

### 6.3 接触检测

#### 6.3.1 基于距离的方法

$$C_{ij} = \mathbb{I}(\|\mathbf{j}_i - \mathbf{o}_j\| < \epsilon)$$

其中：
- $C_{ij}$：第 $i$ 个关节与第 $j$ 个物体点的接触指示
- $\epsilon$：接触阈值

#### 6.3.2 基于学习的方法

```python
class ContactDetector(nn.Module):
    """
    手-物接触检测网络
    """
    def __init__(self):
        self.hand_encoder = HandEncoder()
        self.object_encoder = ObjectEncoder()
        self.contact_head = ContactHead()
    
    def forward(self, hand_img, object_img):
        hand_feat = self.hand_encoder(hand_img)
        object_feat = self.object_encoder(object_img)
        
        # 特征融合
        fused = torch.cat([hand_feat, object_feat], dim=-1)
        
        # 接触预测
        contact_map = self.contact_head(fused)
        return contact_map
```

### 6.4 手-物交互数据集对比

| 数据集 | 场景 | 手-物对数 | 3D 标注 | 接触标注 |
|--------|------|----------|--------|---------|
| **HO-3D** | 室内 | 77,558 | 是 | 否 |
| **DexYCB** | 室内 | 582K | 是 | 是 |
| **ContactPose** | 室内 | 2.3M | 是 | 是 |
| **100DOH** | 室内外 | 131K | 部分 | 否 |
| **H2O** | 室内 | 571K | 是 | 是 |

---

## §7 — 与 DVAS 项目的关联

### 7.1 在 DVAS 中的定位

```
DVAS 手势估计 Pipeline:

[ ego-centric 视频采集]
         ↓
[手部检测] (YOLO / MediaPipe)
         ↓
[手势估计] (HaMeR / 其他方法)
         ↓
[3D 手姿输出]
         ├──> [动作识别]
         ├──> [手-物交互分析]
         ├──> [遥操作控制]
         └──> [数据采集标注]
```

### 7.2 DVAS 特定需求

| 需求 | 技术方案 | 预期效果 |
|------|---------|---------|
| **实时手势交互** | HaMeR + TensorRT 优化 | 30 FPS |
| **高精度操作** | DexYCB 预训练 + 领域微调 | < 5mm 关节误差 |
| **手-物交互** | 联合手-物姿态估计 | 接触检测准确率 > 90% |
| **长时间采集** | 时序平滑 + 重检测 | 稳定跟踪 |
| **多设备同步** | 统一时间戳 + 空间标定 | 跨设备一致性 |

### 7.3 与下游模块的接口

```python
class HandPoseEstimator:
    """
    手势估计器接口
    """
    def __init__(self, model_type="hamer"):
        if model_type == "hamer":
            self.model = HaMeR()
        elif model_type == "frankmocap":
            self.model = FrankMocap()
    
    def estimate(self, image, bbox=None):
        """
        输入: RGB 图像 (H, W, 3)
        可选: 手部边界框 (x, y, w, h)
        
        输出: 
            - joints_3d: (21, 3) 3D 关节位置
            - joints_2d: (21, 2) 2D 关节位置
            - mano_params: MANO 参数
            - confidence: 置信度
        """
        if bbox is None:
            bbox = self.detect_hand(image)
        
        hand_img = self.crop_hand(image, bbox)
        result = self.model.infer(hand_img)
        
        return {
            'joints_3d': result.joints_3d,
            'joints_2d': result.joints_2d,
            'mano_params': result.mano_params,
            'confidence': result.confidence
        }
```

---

## §8 — 参考与资源

### 8.1 关键论文

1. **HaMeR** - Bardia et al. (2023) - "Reconstructing Hands in 3D with Transformers" (ICCV)
2. **100DOH** - Shan et al. (2020) - "Understanding Human Hands in Contact at Internet Scale" (CVPR)
3. **DexYCB** - Chao et al. (2021) - "DexYCB: A Benchmark for Capturing Hand Grasping of Objects" (CVPR)
4. **MANO** - Romero et al. (2017) - "Embodied Hands: Modeling and Capturing Hands and Bodies Together" (SIGGRAPH Asia)
5. **FrankMocap** - Rong et al. (2020) - "FrankMocap: Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration" (arXiv)
6. **ContactPose** - Brahmbhatt et al. (2020) - "ContactPose: A Dataset of Grasps with Object Contact and Hand Pose" (ECCV)

### 8.2 开源实现

| 项目 | 链接 | 说明 |
|------|------|------|
| HaMeR | https://github.com/geopavlakos/HaMeR | 官方实现 |
| FrankMocap | https://github.com/facebookresearch/frankmocap | Meta 官方 |
| MediaPipe Hands | https://mediapipe.dev | Google 轻量方案 |
| HandMesh | https://github.com/... | 轻量 3D 手势 |
| DexYCB Toolkit | https://github.com/NVlabs/DexYCB | 数据集工具 |

### 8.3 相关文档

- [深度估计](22-depth-estimation.md) — 深度图可辅助手势 3D 重建
- [3D 重建](25-3d-reconstruction.md) — 手势估计可与场景重建结合
- [多传感器融合](26-sensor-fusion.md) — 多视角融合提高手势估计精度

---

*Layer: 03-perception | Prev: [SLAM 系统](23-slam.md) | Next: [3D 重建](25-3d-reconstruction.md)*
