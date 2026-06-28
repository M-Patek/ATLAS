---
id: data-formats
title: "数据格式对照"
status: complete
complexity: medium
related:
  - "./01-ego-collection.md"
  - "./02-umi-systems.md"
  - "./03-sim2real.md"
  - "./04-teleoperation.md"
prerequisites:
  - "数据序列化"
  - "文件系统"
  - "ROS/ROS2基础"
last_validated: 2026-06-27
---

# 数据格式对照

## §0 — One-liner

数据格式对照文档详细梳理机器人学习中常用的数据存储格式，为DVAS项目的数据管理和转换提供标准化参考。

## §1 — ROS/ROS2 Bag格式详解

### 1.1 ROS Bag概述

ROS Bag是ROS (Robot Operating System) 的标准数据记录格式，用于存储传感器数据、控制指令和时间戳信息。

| 特性 | ROS1 Bag | ROS2 Bag |
|------|----------|----------|
| 文件扩展名 | `.bag` | `.db3` (SQLite3) |
| 存储格式 | 自定义二进制 | SQLite3数据库 |
| 压缩 | 支持 (bz2/lz4) | 支持 (zstd) |
| 时间戳 | rostime | rclcpp::Time |
| 消息序列化 | 自定义 | CDR (Common Data Representation) |
| 工具链 | rosbag, rqt_bag | ros2 bag, rqt_bag |

### 1.2 ROS2 Bag内部结构

```
rosbag2_YYYY_MM_DD-HH_MM_SS/
├── metadata.yaml          # 元数据信息
├── ros2bag.db3            # SQLite3数据库 (消息存储)
└── ros2bag.db3-shm        # 共享内存文件 (可选)
```

**metadata.yaml示例**：
```yaml
rosbag2_bagfile_information:
  version: 5
  storage_identifier: sqlite3
  duration:
    nanoseconds: 120000000000  # 120秒
  starting_time:
    nanoseconds_since_epoch: 1700000000000000000
  ending_time:
    nanoseconds_since_epoch: 1700000120000000000
  topics_with_message_count:
    - topic_metadata:
        name: /camera/color/image_raw
        type: sensor_msgs/Image
        serialization_format: cdr
      message_count: 3600
    - topic_metadata:
        name: /joint_states
        type: sensor_msgs/JointState
        serialization_format: cdr
      message_count: 12000
    - topic_metadata:
        name: /tf
        type: tf2_msgs/TFMessage
        serialization_format: cdr
      message_count: 12000
```

### 1.3 ROS2 Bag数据表结构

```sql
-- messages表：存储所有消息
CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id    INTEGER NOT NULL,
    timestamp   INTEGER NOT NULL,  -- 纳秒级时间戳
    data        BLOB NOT NULL       -- 序列化消息数据
);

-- topics表：存储话题信息
CREATE TABLE topics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    type            TEXT NOT NULL,
    serialization_format TEXT NOT NULL,
    offered_qos_profiles TEXT
);
```

### 1.4 ROS Bag操作命令

```bash
# 录制Bag
ros2 bag record /camera/image_raw /joint_states /tf -o my_bag

# 回放Bag
ros2 bag play my_bag --rate 1.0 --topics /camera/image_raw

# 查看Bag信息
ros2 bag info my_bag

# 压缩Bag
ros2 bag reindex my_bag
ros2 bag convert -i my_bag -o output_bag --compression zstd
```

### 1.5 ROS Bag优缺点

| 优点 | 缺点 |
|------|------|
| 与ROS生态无缝集成 | 依赖ROS环境 |
| 时间戳自动管理 | 文件体积大 |
| 支持多种消息类型 | 二进制格式，不易直接读取 |
| 压缩选项丰富 | 跨版本兼容性差 |
| 成熟的工具链 | 不适合长期存储 |

## §2 — HDF5格式

### 2.1 HDF5概述

HDF5 (Hierarchical Data Format 5) 是一种用于存储和组织大量科学数据的文件格式。

| 特性 | 说明 |
|------|------|
| 文件扩展名 | `.h5`, `.hdf5` |
| 数据结构 | 层次化（类似文件系统） |
| 数据类型 | 支持多种数值类型、字符串、复合类型 |
| 压缩 | 支持GZIP、LZF等多种压缩算法 |
| 分块存储 | 支持分块(chunking)存储，高效随机访问 |
| 跨平台 | 支持Windows、Linux、macOS |

### 2.2 HDF5在机器人数据中的应用

**典型HDF5结构（机器人操作数据）**：

```
data.hdf5
├── attrs/                          # 全局属性
│   ├── dataset_name = "robot manipulation"
│   ├── version = "1.0"
│   └── date_created = "2024-01-15"
│
├── data/
│   ├── episode_000/
│   │   ├── observations/
│   │   │   ├── images/              # (T, H, W, C) uint8
│   │   │   ├── depth/               # (T, H, W) float32
│   │   │   ├── joint_positions/     # (T, N) float32
│   │   │   └── gripper_states/      # (T, 1) float32
│   │   ├── actions/                 # (T, M) float32
│   │   ├── rewards/                 # (T, 1) float32
│   │   └── timestamps/              # (T, 1) float64
│   │
│   ├── episode_001/
│   │   └── ...
│   │
│   └── episode_002/
│       └── ...
│
└── metadata/
    ├── camera_intrinsics/           # (3, 3) float32
    ├── camera_extrinsics/           # (4, 4) float32
    └── robot_urdf/                  # 字符串
```

### 2.3 Python操作示例

```python
import h5py
import numpy as np

# 创建HDF5文件
with h5py.File('robot_data.hdf5', 'w') as f:
    # 创建组
    episode = f.create_group('episode_000')

    # 存储图像数据 (使用压缩)
    images = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
    episode.create_dataset(
        'images',
        data=images,
        compression='gzip',
        compression_opts=4,
        chunks=(1, 480, 640, 3)  # 按帧分块
    )

    # 存储关节角度
    joint_positions = np.random.randn(100, 7).astype(np.float32)
    episode.create_dataset('joint_positions', data=joint_positions)

    # 存储元数据
    episode.attrs['task'] = 'pick_and_place'
    episode.attrs['success'] = True
    episode.attrs['duration'] = 10.5

# 读取HDF5文件
with h5py.File('robot_data.hdf5', 'r') as f:
    images = f['episode_000/images'][:]
    task = f['episode_000'].attrs['task']
    print(f"Task: {task}, Images shape: {images.shape}")
```

### 2.4 HDF5优缺点

| 优点 | 缺点 |
|------|------|
| 高效的数值数据存储 | 不适合存储文本/JSON数据 |
| 支持大文件 (>2GB) | 文件损坏后难以修复 |
| 随机访问性能优异 | 需要专用库读取 |
| 压缩率高 | 跨语言支持有限 |
| 层次化组织 | 版本兼容性问题 |

## §3 — NPZ格式

### 3.1 NPZ概述

NPZ (NumPy Zip) 是NumPy提供的一种简单数据存储格式，基于ZIP压缩。

| 特性 | 说明 |
|------|------|
| 文件扩展名 | `.npz` |
| 底层格式 | ZIP压缩的.npy文件 |
| 数据类型 | NumPy数组 |
| 压缩 | ZIP压缩 (DEFLATE) |
| 依赖 | 仅需NumPy |

### 3.2 NPZ在机器人数据中的应用

```python
import numpy as np

# 创建NPZ文件
data = {
    'images': np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8),
    'actions': np.random.randn(100, 7).astype(np.float32),
    'rewards': np.random.randn(100).astype(np.float32),
    'metadata': np.array({
        'task': 'pick_and_place',
        'robot': 'franka',
        'fps': 30
    }, dtype=object)  # 使用object存储字典
}

np.savez_compressed('episode.npz', **data)

# 读取NPZ文件
loaded = np.load('episode.npz', allow_pickle=True)
images = loaded['images']
actions = loaded['actions']
metadata = loaded['metadata'].item()  # 恢复字典
```

### 3.3 NPZ优缺点

| 优点 | 缺点 |
|------|------|
| 使用简单 | 不支持增量写入 |
| 依赖少（仅NumPy） | 不支持层次化结构 |
| 压缩率适中 | 大文件性能差 |
| 跨平台 | 不适合流式读取 |
| 适合小规模数据 | 元数据支持有限 |

## §4 — JSON格式

### 4.1 JSON在机器人数据中的应用

JSON (JavaScript Object Notation) 适合存储结构化元数据和配置信息。

**典型JSON结构**：
```json
{
  "dataset_info": {
    "name": "robot_manipulation_v1",
    "version": "1.0.0",
    "created_at": "2024-01-15T10:30:00Z",
    "total_episodes": 1000,
    "total_frames": 500000
  },
  "episodes": [
    {
      "episode_id": "ep_0001",
      "task": "pick_and_place",
      "success": true,
      "duration_seconds": 12.5,
      "num_frames": 375,
      "robot_config": {
        "type": "franka_emika_panda",
        "gripper": "franka_hand"
      },
      "camera_configs": [
        {
          "name": "wrist_camera",
          "resolution": [480, 640],
          "fps": 30,
          "intrinsics": {
            "fx": 600.0, "fy": 600.0,
            "cx": 320.0, "cy": 240.0
          }
        }
      ],
      "file_paths": {
        "video": "ep_0001/video.mp4",
        "actions": "ep_0001/actions.npy",
        "metadata": "ep_0001/meta.json"
      }
    }
  ]
}
```

### 4.2 JSON优缺点

| 优点 | 缺点 |
|------|------|
| 人类可读 | 不适合存储大规模数值数据 |
| 广泛支持 | 无压缩，文件体积大 |
| 层次化结构 | 解析速度较慢 |
| 自描述性 | 数值精度损失 |
| 版本控制友好 | 二进制数据需Base64编码 |

## §5 — 各范式数据格式对比

### 5.1 数据格式特性对比

| 特性 | ROS Bag | HDF5 | NPZ | JSON |
|------|---------|------|-----|------|
| 数值数据存储 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| 文本数据存储 | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 时间戳管理 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 随机访问 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| 流式读取 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 压缩效率 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 跨语言支持 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 工具链成熟度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 大文件支持 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 版本兼容性 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 5.2 各范式推荐格式

| 采集范式 | 推荐格式 | 原因 |
|----------|----------|------|
| Ego-centric | HDF5 + JSON | 视频+IMU+元数据分离存储 |
| UMI | NPZ / HDF5 | 数值数据为主，结构简单 |
| Sim2Real | HDF5 / NPZ | 大规模仿真数据，高效存储 |
| 遥操作 | ROS2 Bag | 与ROS2控制栈无缝集成 |
| 混合数据 | HDF5 + JSON | 灵活组合不同数据类型 |

## §6 — 数据转换工具链

### 6.1 常用转换工具

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| `rosbag2` | ROS Bag读写、转换 | ROS生态数据管理 |
| `h5py` | HDF5读写 | Python数值数据存储 |
| `numpy` | NPZ读写 | 小规模数据快速存储 |
| `json` | JSON读写 | 元数据、配置信息 |
| `cv2` | 视频读写 | 图像序列处理 |
| `imageio` | 视频/GIF读写 | 多格式图像处理 |

### 6.2 格式转换示例

**ROS Bag → HDF5**：
```python
import h5py
from rosbag import Bag

with Bag('input.bag', 'r') as bag:
    with h5py.File('output.hdf5', 'w') as h5f:
        # 创建数据集
        images = []
        for topic, msg, t in bag.read_messages(topics=['/camera/image_raw']):
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            images.append(img)

        h5f.create_dataset('images', data=np.array(images), compression='gzip')
```

**HDF5 → NPZ**：
```python
import h5py
import numpy as np

with h5py.File('input.hdf5', 'r') as h5f:
    data = {
        'images': h5f['images'][:],
        'actions': h5f['actions'][:],
        'rewards': h5f['rewards'][:]
    }
    np.savez_compressed('output.npz', **data)
```

**视频帧 → HDF5**：
```python
import cv2
import h5py

with h5py.File('video_data.hdf5', 'w') as h5f:
    # 创建可扩展数据集
    max_frames = 10000
    images_ds = h5f.create_dataset(
        'images',
        shape=(max_frames, 480, 640, 3),
        maxshape=(None, 480, 640, 3),
        dtype='uint8',
        chunks=(1, 480, 640, 3),
        compression='gzip'
    )

    cap = cv2.VideoCapture('input.mp4')
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        images_ds[frame_idx] = frame
        frame_idx += 1

    # 调整数据集大小
    images_ds.resize((frame_idx, 480, 640, 3))
```

### 6.3 DVAS推荐数据存储规范

```
dvas_dataset/
├── dataset_metadata.json       # 数据集元数据
├── episodes/
│   ├── episode_0001/
│   │   ├── data.hdf5          # 主要数据 (图像、动作、状态)
│   │   └── metadata.json      # 片段元数据
│   ├── episode_0002/
│   │   └── ...
│   └── ...
├── calibration/
│   ├── camera_intrinsics.json
│   └── hand_eye_transform.json
└── README.md
```

**dataset_metadata.json模板**：
```json
{
  "dataset_name": "dvas_manipulation_v1",
  "version": "1.0.0",
  "description": "DVAS manipulation dataset",
  "collection_method": "teleoperation",
  "robot_platform": "franka_panda",
  "sensors": {
    "cameras": [
      {
        "name": "wrist_camera",
        "type": "RGB",
        "resolution": [480, 640],
        "fps": 30,
        "intrinsics": "calibration/camera_intrinsics.json"
      }
    ],
    "imu": {
      "model": "BMI270",
      "frequency_hz": 200
    }
  },
  "statistics": {
    "total_episodes": 1000,
    "total_frames": 500000,
    "total_duration_hours": 4.5,
    "success_rate": 0.85
  }
}
```

## §7 — 与DVAS项目的关联

### 7.1 数据格式在DVAS Pipeline中的角色

```
┌─────────────────────────────────────────────────────┐
│              DVAS 数据Pipeline                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  采集层                                              │
│  ├── Ego: 原始视频 + IMU → HDF5/ROS Bag            │
│  ├── UMI: 图像 + 动作 → NPZ                        │
│  ├── Sim2Real: 仿真数据 → HDF5                     │
│  └── 遥操作: ROS2 Bag → HDF5                       │
│                                                     │
│  转换层                                              │
│  ├── 格式标准化 (统一为HDF5)                        │
│  ├── 时间戳对齐                                     │
│  ├── 数据清洗                                       │
│  └── 元数据提取                                     │
│                                                     │
│  存储层                                              │
│  ├── 原始数据: 对象存储 (S3/MinIO)                  │
│  ├── 处理数据: HDF5数据集                           │
│  └── 元数据: JSON + 关系数据库                     │
│                                                     │
│  训练层                                              │
│  ├── DataLoader读取HDF5                            │
│  ├── 在线数据增强                                   │
│  └── 批量训练                                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 7.2 DVAS数据管理建议

1. **原始数据保留**：始终以原始格式保存原始数据（ROS Bag/原始视频）
2. **标准化中间格式**：使用HDF5作为标准中间格式
3. **元数据完整**：每个数据片段附带完整的JSON元数据
4. **版本控制**：数据集版本化管理，记录变更历史
5. **数据校验**：MD5校验和确保数据完整性
6. **访问接口**：提供统一的Python API访问不同格式数据

---

*Related: [01-ego-collection](31-ego-collection.md) | [02-umi-systems](32-umi-systems.md) | [03-sim2real](33-sim2real.md) | [04-teleoperation](34-teleoperation.md)*
