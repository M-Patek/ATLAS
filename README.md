# ATLAS: Atlas of Technologies for Learning Autonomous Systems

**可插拔认知空间框架** | [English](#english)

```
架构设计: 认知空间作为核心抽象
┌─────────────────────────────────────────────────────────────┐
│  Solver Layer (A*, Dijkstra, etc.)                          │
│  - 在空间中求解测地线                                          │
│  - 与具体空间类型解耦                                          │
├─────────────────────────────────────────────────────────────┤
│  Space Layer (CognitiveSpace)                                 │
│  - Ricci Space: 信息几何，曲率 = -Δ log(uncertainty)         │
│  - Conformal Space: 共形变换 g' = Ω² × g                     │
│  - Fisher Space: 信息几何，g = 1/confidence                  │
│  - Wasserstein Space: 最优传输成本                           │
│  - Finsler Space: 非对称度量                                 │
│  - Continuous Space: 连续坐标 (Tuple[float, float])         │
├─────────────────────────────────────────────────────────────┤
│  SSFR Layer (Structure Self-Discovery & Reuse)               │
│  - StructureHypothesis: 结构假设（可验证、可竞争、可演化）     │
│  - StructurePool: 结构竞争池（竞争、选择、演化）              │
│  - MultiSpaceRepresentation: 多空间联合表示                     │
├─────────────────────────────────────────────────────────────┤
│  Kitchen Layer (Physical Environment)                         │
│  - ContinuousPhysicalSSFR: 物理世界连续SSFR                    │
│  - ContinuousSSFRTaskPlanner: 任务规划器                     │
│  - pymunk-based 2D physics simulation                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 设计理念

**核心洞察**: 认知不是状态的表示，而是空间中的运动。

- **传统AI**: 感知 → 状态估计 → 规划 → 执行
- **ATLAS**: 感知 → **更新空间** → **在空间中求解** → 执行

空间是认知的"场"，规划是场中的"测地线"。

---

## 快速开始

### 1. 基础使用

```python
from atlas.core import Experiment, GeodesicSolver
from atlas.core.registry import create_space

# 创建空间
space = create_space("ricci", width=40, height=20, curvature_scale=2.0)

# 创建求解器
solver = GeodesicSolver(space)

# 求解路径
result = solver.solve(start=(5, 10), goal=(35, 10), obstacles={(20, 10)})

if result.success:
    print(f"Path found: {len(result.path)} steps")
```

### 2. 连续空间 SSFR

```python
from atlas.spaces.continuous_ssfr import ContinuousSSFR

# 创建连续空间 SSFR（无需离散网格）
ssfr = ContinuousSSFR(
    space_names=['ricci', 'fisher', 'wasserstein', 'conformal']
)

# 感知：使用连续坐标 (float, float)
hypotheses = ssfr.perceive(
    position=(1.0, 2.0),
    observation={
        'position': (1.0, 2.0),
        'goal_position': (5.0, 5.0),
        'obstacles': [(2.0, 2.0)],
        'uncertainty': 0.3,
    }
)

# 竞争
winner = ssfr.compete(observation, actual)

# 演化
new_structures = ssfr.evolve()
```

### 3. 物理厨房集成

```python
from experiments.tests.test_continuous_ssfr import (
    ContinuousPhysicalSSFR, ContinuousSSFRTaskPlanner
)
from atlas.kitchen import create_demo_kitchen

# 创建物理厨房
kitchen = create_demo_kitchen()
robot_id = list(kitchen.robots.keys())[0]

# 创建连续SSFR（直接物理坐标）
physical_ssfr = ContinuousPhysicalSSFR(kitchen)
planner = ContinuousSSFRTaskPlanner(physical_ssfr)

# 分配任务
planner.assign_task(robot_id, 'make_coffee')

# 执行
for _ in range(100):
    kitchen.step()
    result = planner.step(robot_id)
```

### 4. 对比实验

```python
from atlas.core import Experiment

# 创建实验
experiment = Experiment("space_comparison")

# 注册多个空间
experiment.register_space("ricci", create_space("ricci", 40, 20))
experiment.register_space("conformal", create_space("conformal", 40, 20))
experiment.register_space("euclidean", create_space("euclidean", 40, 20))

# 添加场景
experiment.add_scenario({
    'start': (5, 10),
    'goal': (35, 10),
    'obstacles': {(20, y) for y in range(5, 15)},
})

# 运行实验
results = experiment.run(num_trials=10)
print(experiment.get_summary())
```

---

## 可用认知空间

### 基础空间

| 空间 | 核心数学 | 适用场景 |
|------|----------|----------|
| `euclidean` | 欧氏距离 | 基线对照 |
| `ricci` | R = -Δ log(u) | 探索-利用平衡 |
| `conformal` | g' = Ω² × g | 目标导向导航 |
| `fisher` | g = 1/confidence | 统计学习 |
| `wasserstein` | 传输成本 | 资源分配 |
| `finsler` | 非对称度量 | 习惯建模 |

### 连续空间 (NEW)

| 空间 | 特性 | 适用场景 |
|------|------|----------|
| `continuous_ricci` | 稀疏采样 + kNN插值 | 物理世界导航 |
| `continuous_fisher` | 连续置信度场 | 连续状态估计 |
| `continuous_wasserstein` | 连续成本场 | 连续资源分配 |
| `continuous_euclidean` | 基线 | 连续坐标基线 |

### 复合空间

| 空间 | 组合方式 | 适用场景 |
|------|----------|----------|
| `product` | d² = Σ wᵢ × dᵢ² | 多目标平衡 |
| `hierarchical` | 全局+局部 | 大场景规划 |
| `mixed` | 条件切换 | 场景自适应 |

### 时序空间

| 空间 | 特性 | 适用场景 |
|------|------|----------|
| `temporal` | 历史+预测 | 动态环境 |
| `predictive_ricci` | 曲率预测 | 预测性导航 |

---

## 项目结构

```
atlas/
├── core/                      # 核心框架
│   ├── space.py              # CognitiveSpace 抽象
│   ├── solver.py             # GeodesicSolver
│   ├── experiment.py         # Experiment 框架
│   ├── registry.py           # 空间注册表
│   ├── replanning.py         # D* Lite 增量规划
│   └── ssfr_enhanced.py     # 增强版 SSFR
├── spaces/                    # 基础空间实现
│   ├── euclidean.py          # 欧氏空间（离散网格）
│   ├── ricci.py              # Ricci流空间
│   ├── conformal.py          # 共形空间
│   ├── fisher.py             # Fisher信息几何
│   ├── wasserstein.py        # Wasserstein空间
│   ├── finsler.py            # Finsler空间
│   ├── continuous.py         # 连续空间基类（NEW）
│   ├── continuous_ssfr.py    # 连续SSFR核心（NEW）
│   ├── continuous_optimized.py # 优化版连续SSFR（NEW）
│   ├── temporal.py           # 时序空间
│   ├── composite.py          # 复合空间
│   ├── grid3d.py             # 3D网格空间
│   └── solver3d.py           # 3D求解器
├── kitchen/                   # 物理厨房环境
│   ├── __init__.py
│   └── controller.py         # 厨房控制器
├── learning/                  # 学习模块
│   ├── bayesian_optimizer.py
│   ├── meta_learner.py
│   ├── neural_space.py
│   └── trainer.py
├── visualization/             # 可视化
│   ├── space_visualizer.py
│   ├── path_animator.py
│   └── comparison_plots.py
└── research/                  # 理论研究
    ├── ssfr_information_geometry.py
    ├── ssfr_continuous.py
    ├── ssfr_hierarchical.py
    └── ab_testing.py

experiments/
├── tests/                     # 测试脚本（16个）
│   ├── test_continuous_ssfr.py
│   ├── test_ssfr_enhanced.py
│   ├── test_structure_reuse_v2.py
│   └── ...
├── demos/                     # 演示脚本（6个）
│   ├── demo_physical_kitchen.py
│   └── ...
├── benchmarks/                # 基准测试（3个）
│   ├── benchmark_continuous_ssfr.py
│   └── ...
└── research/                  # 研究分析（2个）
    ├── meta_ssfr_atlas.py
    └── meta_ssfr_enhanced.py

tests/                         # pytest 测试
├── test_core.py
└── test_spaces.py
```

---

## 增强版 SSFR

### 核心改进

| 特性 | 传统 SSFR | 增强版 SSFR | 连续 SSFR (NEW) |
|------|-----------|-------------|-----------------|
| **位置表示** | 离散网格 (int, int) | 离散网格 | 连续坐标 (float, float) |
| **场数据** | numpy 2D array | numpy 2D array | 稀疏采样 + kNN插值 |
| **距离计算** | 网格路径 | 网格路径 | 连续路径积分 |
| **边界** | 固定网格 | 固定网格 | 无限制 |
| **结构定义** | 聚类结果 | 可验证假设 | 可验证假设 |
| **表示方式** | 单空间 | 多空间联合 | 多空间联合 |
| **存储方式** | 静态存储 | 动态竞争 | 动态竞争 |

### 核心组件

```python
# StructureHypothesis: 结构假设
hypothesis = StructureHypothesis(
    id="hyp_001",
    name="corridor_structure",
    representations={
        "ricci": {"fields": {...}, "params": {...}},
        "fisher": {"fields": {...}, "params": {...}},
    },
    context={"scene_type": "corridor"}
)

# StructurePool: 结构竞争池
pool = StructurePool(max_structures=100)
pool.add(hypothesis)
winner, results = pool.compete(observation, actual)

# MultiSpaceRepresentation: 多空间联合表示
multi = MultiSpaceRepresentation([ricci_space, fisher_space])
representations = multi.encode(observation)
consistent = multi.find_consistent_structure(representations, observation)
```

---

## 开发新的认知空间

```python
from atlas.core.space import CognitiveSpace, register_space

@register_space("my_space")
class MySpace(CognitiveSpace):
    def compute_distance(self, pos1, pos2):
        # 实现距离计算
        pass

    def get_heuristic(self, pos, goal):
        # 实现启发式
        pass

    def update_from_observation(self, position, observation):
        # 实现更新逻辑
        pass
```

空间会自动注册，可以通过 `create_space("my_space", ...)` 使用。

---

## 核心概念

### 可更新 (Updatable)
空间必须能根据观测更新内部状态。

```python
space.update_from_observation(
    position=(10.0, 10.0),  # 连续坐标
    observation={'obstacles': [(20.0, 10.0)], 'goal_position': (35.0, 10.0)}
)
```

### 可规划 (Plannable)
空间必须提供距离度量和启发式函数。

```python
distance = space.compute_distance((0, 0), (10, 10))
heuristic = space.get_heuristic((5, 5), (10, 10))
```

### 可插拔 (Pluggable)
任何实现 CognitiveSpace 接口的类都可以被框架使用。

---

## 理论背景

### Ricci 空间
基于信息几何，使用 Ricci 曲率:
```
R(x) = -Δ log(uncertainty)
g_ij = (1 + |R|)² δ_ij
```

### 共形空间
基于动态度量变换:
```
g'_ij = Ω(x)² × g_ij
```
其中 Ω 由 attractors 和 repellers 决定。

### Fisher 空间
基于统计流形的信息度量:
```
g_ij = E[∂_i log p · ∂_j log p]
```

### 增强版 SSFR
基于结构竞争和演化:
```
Structure = Hypothesis × Representation × Competition
Fitness = Accuracy / Cost
Evolution = Mutation + Crossover + Selection
```

---

## 引用

```bibtex
@software{atlas2024,
  title={ATLAS: Pluggable Cognitive Architecture Framework},
  author={ATLAS Contributors},
  year={2024},
  url={https://github.com/yourusername/atlas}
}
```

---

## License

MIT
