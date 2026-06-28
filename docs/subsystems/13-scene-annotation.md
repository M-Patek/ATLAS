---
id: scene-annotation
title: "Scene and Object Annotation"
status: complete
complexity: high
related:
  - "./01-schema-design.md"
  - "../03-perception/05-3d-reconstruction.md"
  - "../04-data-ecosystem/01-ego-collection.md"
prerequisites:
  - "Computer vision basics"
  - "Object detection and segmentation"
  - "3D geometry"
last_validated: 2026-06-27
---

# 场景与物体标注

## §0 — One-liner

场景与物体标注将原始视觉数据转化为结构化的环境理解表示，是机器人感知-决策闭环中连接像素级输入与语义级推理的关键桥梁。

## §1 — 核心概念

### 1.1 场景标注的层次结构

机器人场景标注不同于通用计算机视觉标注，需要在多个抽象层次上建立表示：

```
场景理解层次:
  ├─ 像素级 (Pixel-level)
  │   ├─ 语义分割: 每个像素的类别标签
  │   ├─ 实例分割: 每个像素的实例ID
│   └─ 深度估计: 每个像素的3D距离
  ├─ 物体级 (Object-level)
  │   ├─ 2D检测框: 图像中的物体位置
  │   ├─ 3D边界框: 物体在3D空间中的范围
│   ├─ 6DoF位姿: 物体的精确位置和朝向
│   └─ 3D网格: 物体的完整几何形状
  ├─ 关系级 (Relation-level)
  │   ├─ 空间关系: "在...上面", "在...旁边"
  │   ├─ 功能关系: "用于...", "属于..."
│   └─ 交互关系: "可抓取", "可推动"
  └─ 场景级 (Scene-level)
      ├─ 场景类别: "厨房", "办公室"
      ├─ 活动区域: "操作台面", "行走区域"
      └─ 状态描述: "整洁", "杂乱"
```

### 1.2 机器人场景标注的特殊挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| **视角受限** | 机器人通常只有1-3个相机，存在大量遮挡 | 物体不完整观测 |
| **动态环境** | 物体被移动、操作过程中状态变化 | 标注时序一致性 |
| **未知物体** | 开放世界，训练时未见过的物体 | 需要开放词汇检测 |
| **实时性要求** | 标注需支持实时决策 | 计算资源约束 |
| **多模态融合** | 视觉+深度+触觉联合标注 | 跨模态对齐 |

## §2 — 物体检测与分割标注

### 2.1 2D物体检测标注

**边界框 (Bounding Box) 标注：**

```yaml
bbox_2d:
  type: struct
  fields:
    x_min: float  # 左上角x (像素)
    y_min: float  # 左上角y (像素)
    x_max: float  # 右下角x (像素)
    y_max: float  # 右下角y (像素)
    # 或等价表示:
    # center_x, center_y, width, height
  format: "xyxy"  # 或 "xywh", "cxcywh"
  coordinate_system: "image_pixel"  # 左上角原点，y向下
```

**边界框格式对比：**

| 格式 | 表示 | 优点 | 缺点 |
|------|------|------|------|
| xyxy | [x1, y1, x2, y2] | 直观、无需计算 | 需保证x2>x1, y2>y1 |
| xywh | [x, y, w, h] | 便于平移变换 | 宽高可能为负 |
| **cxcywh** | **[cx, cy, w, h]** | **中心点便于定位** | **需转换** |
| normalized | 值域[0,1] | 分辨率无关 | 需图像尺寸还原 |

**2D检测标注标准：**

| 标准 | 来源 | 格式 | 特点 |
|------|------|------|------|
| **COCO** | Microsoft | JSON | 实例分割、关键点、全景分割 |
| **PASCAL VOC** | Oxford | XML | 简单边界框 |
| **YOLO** | Ultralytics | txt | 每图一个文件，归一化坐标 |
| **OpenImages** | Google | CSV | 层级标签、关系标注 |

### 2.2 2D实例分割标注

实例分割在语义分割基础上区分同类不同实例，是机器人抓取规划的关键输入。

```yaml
instance_mask:
  type: struct
  fields:
    mask_rle: string  # Run-Length Encoding 压缩掩码
    # 或
    mask_polygon: list[list[float]]  # 多边形顶点 [[x1,y1], [x2,y2], ...]
    instance_id: int
    category_id: int
    category_name: string
    confidence: float  # 模型预测时可用
```

**掩码表示对比：**

| 表示法 | 存储效率 | 精度 | 编辑友好 | 适用场景 |
|--------|----------|------|----------|----------|
| 像素掩码 (H×W) | 低 | 最高 | 差 | 模型输入/输出 |
| **RLE编码** | **高** | **无损** | **差** | **COCO标准存储** |
| 多边形 | 中 | 取决于顶点数 | 好 | 人工标注工具 |
| 距离变换 | 中 | 高 | 中 | 边缘细化任务 |

### 2.3 3D物体检测与分割

**3D边界框表示：**

```yaml
bbox_3d:
  type: struct
  fields:
    center: float32[3]       # [x, y, z] 米
    size: float32[3]         # [length, width, height] 米
    orientation: float32[4]   # 四元数 (x, y, z, w)
    # 或欧拉角: yaw, pitch, roll
    coordinate_frame: "world"  # 或 "camera", "robot_base"
```

**3D表示方法对比：**

| 方法 | 输入 | 输出 | 精度 | 计算成本 | 代表工作 |
|------|------|------|------|----------|----------|
| 基于深度 | RGB-D | 3D bbox | 中 | 低 | PointRCNN, VoteNet |
| 基于点云 | PointCloud | 3D bbox | 高 | 中 | PointNet++, VoxelNet |
| 基于NeRF | 多视角RGB | 3D mesh | 很高 | 高 | NeRF, Instant-NGP |
| 基于单目 | RGB | 3D bbox | 中 | 低 | Mono3D, FCOS3D |
| **Foundation Model** | **RGB** | **3D bbox + 开放词汇** | **中高** | **中** | **SAM, OWL-ViT** |

### 2.4 6DoF物体位姿标注

6DoF (6 Degrees of Freedom) 位姿标注是机器人操作的核心——知道物体在哪里、朝向如何，才能规划抓取。

```yaml
object_pose_6dof:
  type: struct
  fields:
    position: float32[3]      # [x, y, z] 米
    orientation: float32[4]   # 四元数
    # 或 rotation_matrix: float32[3, 3]
    reference_frame: string    # 坐标系名称
    object_model: string       # 物体CAD模型ID
    symmetry_type: string      # "none", "z_180", "z_any", "sphere"
```

**对称性处理：**

| 对称类型 | 描述 | 对抓取的影响 |
|----------|------|--------------|
| 无对称 | 唯一位姿 | 需精确估计 |
| Z轴180° | 绕垂直轴旋转180°等效 | 杯状物体 |
| Z轴任意 | 绕垂直轴任意旋转等效 | 圆柱、球体 (顶抓) |
| 完全对称 | 任意旋转等效 | 球体 |

**对称性在损失函数中的处理：**
```python
# 考虑对称性的位姿损失
def pose_loss_with_symmetry(pred_pose, gt_pose, symmetry_type):
    if symmetry_type == "z_180":
        # 尝试两种可能的GT位姿，取较小误差
        alt_pose = rotate_z(gt_pose, 180)
        loss = min(pose_distance(pred_pose, gt_pose),
                   pose_distance(pred_pose, alt_pose))
    elif symmetry_type == "z_any":
        # 仅比较z轴距离和xy平面位置
        loss = cylindrical_distance(pred_pose, gt_pose)
    return loss
```

## §3 — 物体关系标注

### 3.1 空间关系标注

空间关系是机器人任务规划的基础——理解"杯子在桌子上"才能执行"拿杯子"。

**空间关系类型：**

```yaml
spatial_relation:
  type: struct
  fields:
    subject: string      # 主体物体ID
    object: string       # 客体物体ID
    relation_type: enum  # 见下表
    confidence: float    # 关系置信度
    reference_frame: string  # 相对哪个坐标系
```

**空间关系分类：**

| 类别 | 关系 | 机器人语义 | 示例 |
|------|------|----------|------|
| **拓扑** | on, in, under | 支撑/包含关系 | "杯子在托盘上" |
| **方向** | left, right, front, back | 相对位置 | "杯子在盘子左边" |
| **距离** | near, far, touching | 接近程度 | "杯子靠近水壶" |
| **对齐** | aligned_with, facing | 姿态关系 | "把手朝向右侧" |
| **包含** | inside, outside | 空间占据 | "勺子在杯子里" |

**空间关系计算 (基于3D bbox)：**
```python
def compute_spatial_relations(obj_a, obj_b, threshold=0.05):
    relations = []

    # "on" 关系: A底部接近B顶部，且水平投影重叠
    if (abs(obj_a.bbox.min_z - obj_b.bbox.max_z) < threshold and
        horizontal_overlap(obj_a, obj_b) > 0.5):
        relations.append(("on", obj_a.id, obj_b.id))

    # "in" 关系: A中心在B bbox内
    if point_in_bbox(obj_a.center, obj_b.bbox):
        relations.append(("in", obj_a.id, obj_b.id))

    # "near" 关系: 中心距离小于阈值
    if distance(obj_a.center, obj_b.center) < NEAR_THRESHOLD:
        relations.append(("near", obj_a.id, obj_b.id))

    return relations
```

### 3.2 功能关系标注

功能关系描述物体的用途和 affordance，对任务规划至关重要。

| 关系类型 | 描述 | 示例 |
|----------|------|------|
| **used_for** | 物体A用于执行动作B | "杯子 used_for 喝水" |
| **part_of** | 物体A是物体B的部件 | "把手 part_of 杯子" |
| **graspable_by** | 物体可被某类夹爪抓取 | "杯子 graspable_by 平行夹爪" |
| **pourable_from** | 液体可从A倒入B | "水壶 pourable_from 杯子" |
| **stackable_on** | 物体A可堆叠在B上 | "盘子 stackable_on 盘子" |

**Affordance 标注：**

Affordance (功能可供性) 标注每个物体的可操作区域：

```yaml
affordance_annotation:
  object_id: "mug_001"
  affordances:
    - type: "grasp"
      region:  # 3D点云或2D掩码
        mask_3d: [...]
      gripper_type: "parallel_jaw"
      approach_direction: [0, 0, -1]  # 从上方接近
    - type: "contain"
      region:
        mask_3d: [...]
      capacity_liters: 0.3
```

## §4 — 场景状态变化标注

### 4.1 状态变化类型

机器人操作会导致场景状态变化，标注这些变化对理解操作效果至关重要。

| 变化类型 | 描述 | 标注方法 |
|----------|------|----------|
| **位置变化** | 物体被移动 | 前后位姿对比 |
| **姿态变化** | 物体被旋转/翻转 | 前后朝向对比 |
| **存在变化** | 物体出现/消失 | 帧间物体列表对比 |
| **关系变化** | 物体间关系改变 | 前后关系图对比 |
| **属性变化** | 物体内在属性改变 | 视觉特征变化检测 |
| **拓扑变化** | 接触/连接关系改变 | 接触图变化 |

### 4.2 状态变化检测方法

```python
class SceneStateChangeDetector:
    def detect_changes(self, state_before, state_after):
        changes = []

        # 1. 物体存在性变化
        appeared = set(state_after.objects) - set(state_before.objects)
        disappeared = set(state_before.objects) - set(state_after.objects)

        # 2. 物体位姿变化
        for obj_id in set(state_before.objects) & set(state_after.objects):
            pose_before = state_before.get_pose(obj_id)
            pose_after = state_after.get_pose(obj_id)
            if pose_distance(pose_before, pose_after) > POSE_THRESHOLD:
                changes.append({
                    "type": "pose_change",
                    "object": obj_id,
                    "delta": pose_after - pose_before
                })

        # 3. 关系变化
        relations_before = state_before.get_relations()
        relations_after = state_after.get_relations()
        new_relations = relations_after - relations_before
        lost_relations = relations_before - relations_after

        return changes
```

### 4.3 操作效果标注

对于每个操作片段，标注操作前后的场景状态：

```yaml
operation_effect:
  operation: "pick_up"
  object: "mug_001"
  before_state:
    object_pose: {position: [0.5, 0.2, 0.03], orientation: [...]}
    supporting_surface: "table_001"
    relations: ["on(mug_001, table_001)", "near(mug_001, plate_001)"]
  after_state:
    object_pose: {position: [0.5, 0.2, 0.25], orientation: [...]}
    supporting_surface: null
    relations: ["held_by(gripper, mug_001)", "near(mug_001, plate_001)"]
  success: true
```

## §5 — 数据集标准详解

### 5.1 COCO (Common Objects in Context)

COCO是计算机视觉领域最广泛使用的目标检测与分割数据集标准。

**COCO格式核心结构：**
```json
{
  "images": [
    {"id": 1, "file_name": "000001.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "segmentation": {"counts": [...], "size": [480, 640]},
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "person"}
  ]
}
```

**COCO与机器人场景标注的适配：**

| COCO特性 | 机器人场景适配 | 方法 |
|----------|--------------|------|
| 80类预定义 | 需要开放词汇 | 结合CLIP/OWL-ViT |
| 2D bbox | 需要3D信息 | 结合深度估计投影 |
| 静态图像 | 需要时序 | 扩展为视频标注格式 |
| 人工标注 | 需要自动化 | 半自动标注流程 |

### 5.2 Visual Genome

Visual Genome专注于图像中的结构化关系表示，是场景图 (Scene Graph) 标注的先驱。

**Visual Genome场景图结构：**
```json
{
  "image_id": 1,
  "objects": [
    {"object_id": 1, "names": ["cup"], "synsets": ["cup.n.01"]},
    {"object_id": 2, "names": ["table"], "synsets": ["table.n.01"]}
  ],
  "relationships": [
    {
      "relationship_id": 1,
      "predicate": "on",
      "subject_id": 1,
      "object_id": 2,
      "synsets": ["on.r.01"]
    }
  ],
  "attributes": [
    {"object_id": 1, "attribute": "red"}
  ]
}
```

**Visual Genome对机器人场景的价值：**
- 丰富的关系标注可用于训练关系理解模型
- WordNet synset 链接支持语义推理
- 但需注意：VG标注噪声较大，需清洗后使用

### 5.3 Scene Graph标准

场景图是图像/场景的结构化表示，节点为物体，边为关系。

**场景图Schema：**
```yaml
scene_graph:
  nodes:
    - id: "obj_001"
      category: "cup"
      attributes: ["red", "ceramic"]
      bbox_2d: [x, y, w, h]
      bbox_3d: {center: [x,y,z], size: [l,w,h], orientation: [...]}
      mask: "RLE_encoded_mask"
    - id: "obj_002"
      category: "table"
      # ...
  edges:
    - source: "obj_001"
      target: "obj_002"
      relation: "on"
      confidence: 0.95
    - source: "obj_001"
      target: "obj_002"
      relation: "near"
      confidence: 0.98
```

**场景图与机器人任务规划：**

场景图可直接转化为PDDL (Planning Domain Definition Language) 的规划问题：

```pddl
; 从场景图生成的PDDL
(define (problem scene_001)
  (:objects cup_001 table_001)
  (:init
    (on cup_001 table_001)
    (graspable cup_001)
    (clear table_001))
  (:goal
    (holding cup_001)))
```

### 5.4 机器人专用场景数据集

| 数据集 | 规模 | 标注内容 | 特点 |
|--------|------|----------|------|
| **ObjectNet3D** | 100类, 201物 | 2D框+3D位姿 | 日常物体，CAD模型对齐 |
| **NOCS** | 6类, 430物 | 归一化物体坐标 | 类别级6DoF位姿 |
| **HOPE** | 28类 | 6DoF位姿+ affordance | 家庭物体操作 |
| **YCB-Video** | 21类 | 6DoF位姿+分割 | 机器人操作基准 |
| **Replica** | 18场景 | 语义分割+深度 | 室内重建 |
| **AI2-THOR** | 120场景 | 完整交互标注 | 仿真环境，可交互 |

## §6 — 半自动化场景标注工具

### 6.1 交互式分割工具

| 工具 | 核心功能 | 自动化程度 | 适用场景 |
|------|----------|------------|----------|
| **SAM (Segment Anything)** | 点/框提示分割 | 高 | 快速获取高质量掩码 |
| **LabelMe** | 多边形标注 | 低 | 精确边界标注 |
| **CVAT** | 视频跟踪+插值 | 中 | 大规模视频标注 |
| **LabelStudio** | 多模态标注平台 | 中 | 团队协作 |
| **RoboTurk Annotation** | 机器人场景专用 | 中 | 遥操作数据标注 |

**SAM + 机器人场景标注流程：**
```python
# 半自动标注流程
import cv2
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

def auto_annotate_object(image, bbox_hint):
    """给定检测框，SAM自动优化分割掩码"""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=bbox_hint,
        multimask_output=True
    )
    # 选择最佳掩码
    best_mask = masks[scores.argmax()]
    return best_mask

# 人机协作: 人工提供粗略框，SAM精化掩码
# 人工审核 -> 必要时修正 -> 确认
```

### 6.2 3D场景重建与标注

**基于NeRF/3DGS的自动标注：**

1. **多视角图像采集**：围绕场景拍摄/录制视频
2. **3D重建**：使用Instant-NGP或3D Gaussian Splatting重建
3. **交互式标注**：在3D空间中标注物体，自动投影到各视角
4. **稠密标注传播**：3D标注自动传播到所有2D视图

**优势：**
- 一次3D标注，多视角自动获得2D标注
- 遮挡区域可通过其他视角补全
- 支持自由视角新视图合成

### 6.3 开放词汇检测与标注

传统检测器受限于预定义类别，开放词汇检测器可识别任意文本描述的物体。

| 模型 | 方法 | 特点 |
|------|------|------|
| **OWL-ViT** | 视觉-语言对齐 | 零样本检测，实时 |
| **GLIP** | 语言引导定位 | 短语级理解 |
| **Grounding DINO** | DINO + 语言 | 高精度，多种粒度 |
| **SAM + CLIP** | 分割+分类 | 任意类别分割 |

**开放词汇标注流程：**
```python
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

def open_vocabulary_detect(image, text_queries):
    """用自然语言描述检测任意物体"""
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # 解析结果: bbox + 类别 + 置信度
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.2, target_sizes=target_sizes
    )
    return results

# 示例: 检测机器人场景中任意相关物体
detect(image, ["mug", "cup", "bottle", "container", "utensil"])
```

## §7 — 最佳实践与常见陷阱

### 7.1 标注质量控制

| 检查项 | 方法 | 阈值 |
|--------|------|------|
| 边界框与掩码一致性 | IoU(bbox_from_mask, bbox_annotation) > 0.9 | 0.9 |
| 遮挡处理 | 被遮挡物体仍标注 (visibility字段) | visibility > 0.2 |
| 时序一致性 | 跟踪ID跨帧连续 | 无跳变 |
| 类别平衡 | 每类样本数检查 | 最小100样本/类 |
| 3D-2D投影一致性 | 重投影误差 < 5像素 | 5px |

### 7.2 常见陷阱

| 陷阱 | 描述 | 后果 | 规避方法 |
|------|------|------|----------|
| **坐标系混淆** | 未明确标注坐标系原点与轴向 | 3D位姿理解错误 | 所有位姿必须含`frame_id` |
| **尺度歧义** | 深度/尺寸单位未标注 | 物理量计算错误 | 统一SI单位，字段名含单位 |
| **对称性忽略** | 旋转对称物体标注单一朝向 | 抓取姿态评估偏差 | 标注对称类型，损失函数适配 |
| **遮挡不标注** | 完全忽略被遮挡物体 | 训练数据不完整 | 标注可见度比例，区分全/部分可见 |
| **静态假设** | 假设场景不变 | 动态物体漏标 | 时序标注，帧间物体跟踪 |
| **类别粒度不一致** | 同类物体有时细分有时粗分 | 模型混淆 | 定义明确的类别层级 |

### 7.3 与DVAS项目的关联

DVAS项目的场景标注模块设计：

1. **分层架构**：
   - 底层：SAM自动生成候选掩码
   - 中层：人工审核与修正
   - 高层：关系标注与场景图生成

2. **多模态融合**：
   - 视觉检测 (2D bbox + mask)
   - 深度估计 (3D位置)
   - 点云配准 (6DoF位姿)
   - 融合输出统一场景图

3. **开放世界支持**：
   - 基础类别：使用预训练检测器 (COCO 80类)
   - 新类别：开放词汇检测 (OWL-ViT) + 人工确认
   - 未知物体："object"通用类别 + 几何特征

**DVAS场景标注流水线：**
```
输入: RGB视频 + 深度图 + 相机参数
  ├── 帧级处理
  │   ├── SAM生成候选分割
  │   ├── 开放词汇分类 (CLIP/OWL-ViT)
│   ├── 深度投影 → 3D点云
│   └── 6DoF位姿估计 (已知模型) / 3D bbox (未知模型)
  ├── 时序关联
  │   ├── 多目标跟踪 (DeepSORT/ByteTrack)
│   └── 实例ID一致性维护
  ├── 关系推理
  │   ├── 空间关系计算 (基于3D bbox)
│   ├── 功能关系查询 (知识库)
│   └── 场景图构建
  └── 输出: 结构化场景图 (JSON)
      ├── 物体列表 (含2D/3D/位姿)
      ├── 关系列表
      └── 时序变化记录
```

---

*References:*
- COCO: Lin et al., "Microsoft COCO: Common Objects in Context", 2014
- Visual Genome: Krishna et al., "Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations", 2017
- SAM: Kirillov et al., "Segment Anything", 2023
- OWL-ViT: Minderer et al., "Scaling Open-Vocabulary Object Detection", 2022
- YCB-Video: Xiang et al., "PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation", 2018
- NOCS: Wang et al., "Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation", 2019

*Related: [01-schema-design](11-schema-design.md) | [02-action-annotation](12-action-annotation.md) | [04-physics-annotation](14-physics-annotation.md)*
