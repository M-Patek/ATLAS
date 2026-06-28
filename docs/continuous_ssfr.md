# Continuous SSFR - 连续状态SSFR

> 完全移除离散网格的SSFR实现

## 核心变化

| 方面 | 离散SSFR | 连续SSFR |
|------|---------|---------|
| **位置** | `Tuple[int, int]` | `Tuple[float, float]` |
| **场数据** | numpy 2D array | `ContinuousField` (稀疏采样+插值) |
| **邻居** | 4/8连通 | 16方向连续采样 |
| **距离** | 网格路径积分 | 连续路径积分 |
| **空间尺寸** | 固定网格 (40×20) | 无边界限制 |

## 架构

```
Physical Kitchen → ContinuousKitchenAdapter → ContinuousSSFR → ActionDecoder → Physical Kitchen
                         ↑
              ContinuousField (稀疏采样 + kNN插值)
```

## 核心组件

### 1. ContinuousField（连续场）

用稀疏采样点 + 空间索引 + 插值替代离散网格。

```python
from atlas.spaces.continuous import ContinuousField

field = ContinuousField(default_value=0.0)

# 添加采样点
field.add_sample((0.0, 0.0), 1.0)
field.add_sample((1.0, 0.0), 2.0)

# 查询任意位置（k近邻插值）
value = field.query((0.5, 0.5))  # 2.5

# 批量查询
values = field.query_batch([(0.5, 0.5), (0.5, 0.0)])

# 区域查询
samples = field.get_samples_in_region((0.5, 0.5), radius=1.0)
```

**特性：**
- **稀疏采样**：只在需要的位置存储
- **空间索引**：加速最近邻查询
- **kNN插值**：反距离加权
- **LRU缓存**：加速重复查询

### 2. ContinuousCognitiveSpace（连续认知空间）

```python
from atlas.spaces.continuous import ContinuousRicciSpace

space = ContinuousRicciSpace(
    curvature_scale=1.0,
    familiarity_decay=0.1,
)

# 距离计算（连续路径积分）
dist = space.compute_distance((0.0, 0.0), (1.0, 0.0))

# 启发式
h = space.get_heuristic((0.0, 0.0), (1.0, 0.0))

# 更新（连续位置）
space.update_from_observation((0.5, 0.5), {
    'obstacles': [(0.3, 0.3)],
    'goal_position': (2.0, 2.0),
})

# 预测（16方向采样）
prediction = space.predict_next_state((0.5, 0.5), observation)
```

**实现的空间：**

| 空间 | 特性 | 场 |
|------|------|-----|
| `ContinuousEuclideanSpace` | 基线 | 无 |
| `ContinuousRicciSpace` | 曲率感知 | uncertainty, curvature, familiarity |
| `ContinuousFisherSpace` | 信息几何 | belief, confidence |
| `ContinuousWassersteinSpace` | 最优传输 | cost, mass |

### 3. ContinuousSSFR（连续SSFR）

```python
from atlas.spaces.continuous_ssfr import ContinuousSSFR

ssfr = ContinuousSSFR(
    space_names=['ricci', 'fisher', 'wasserstein'],
    max_structures=50,
    evolution_interval=20,
)

# 感知（连续位置）
hypotheses = ssfr.perceive(
    position=(1.0, 2.0),
    observation={...},
    active_space_name='ricci'
)

# 竞争
winner = ssfr.compete(observation, actual)

# 演化
new_structures = ssfr.evolve()
```

### 4. 厨房集成

```python
from experiments.test_continuous_ssfr import (
    ContinuousPhysicalSSFR, ContinuousSSFRTaskPlanner
)

# 创建
kitchen = create_demo_kitchen()
physical_ssfr = ContinuousPhysicalSSFR(kitchen)
planner = ContinuousSSFRTaskPlanner(physical_ssfr)

# 分配任务
planner.assign_task(robot_id, 'make_coffee')

# 执行
for _ in range(100):
    kitchen.step()
    result = planner.step(robot_id)
```

## 与离散SSFR的对比

| 特性 | 离散SSFR | 连续SSFR |
|------|---------|---------|
| 内存占用 | O(width × height) | O(num_samples) |
| 精度 | 网格分辨率限制 | 任意精度 |
| 边界 | 固定 | 无限制 |
| 插值 | 线性/最近邻 | kNN反距离加权 |
| 邻居搜索 | 4/8连通 | 16方向采样 |
| 适用场景 | 网格世界 | 物理世界 |

## 测试

```bash
# 运行测试
python experiments/test_continuous_ssfr.py
```

**测试结果：** 4/4 通过

| 测试 | 描述 | 状态 |
|------|------|------|
| Continuous Field | 稀疏采样 + kNN插值 | ✅ PASS |
| Continuous Space | Ricci空间连续版本 | ✅ PASS |
| Continuous SSFR | 感知、竞争、演化 | ✅ PASS |
| Kitchen Integration | 物理厨房集成 | ✅ PASS |

## 下一步

1. **性能优化** - 空间索引优化（R-tree, KD-tree）
2. **更多空间** - 将更多离散空间转为连续版本
3. **混合模式** - 离散+连续混合使用
4. **GPU加速** - 场的GPU并行计算
