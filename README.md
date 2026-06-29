# ATLAS: Atlas of Technologies for Learning Autonomous Systems

**可插拔认知空间框架**

**架构设计：认知空间作为核心抽象**

| 层级 | 组件 | 功能 |
|------|------|------|
| 任务层 | 任务规划、SSFR | 目标分解 → 子任务序列 → 空间选择；结构自发现与复用 |
| 求解器层 | A*, D* Lite, Dijkstra | 与具体空间类型解耦；测地线 = 认知空间中的最优路径 |
| 空间层 | 认知空间（可插拔） | 抽象接口：compute_distance, get_heuristic, update；支持离散/连续/复合实现 |
| 环境层 | 物理厨房、网格世界、连续世界 | pymunk 2D 物理模拟；离散状态空间；物理坐标直接输入 |

---

## 设计理念

**核心洞察**: 认知不是状态的表示，而是空间中的运动。

- **传统AI**: 感知 → 状态估计 → 规划 → 执行
- **ATLAS**: 感知 → **更新空间** → **在空间中求解** → 执行

空间是认知的"场"，规划是场中的"测地线"。

**关键设计**: 空间是可插拔的。你可以：
1. 使用内置空间（Euclidean, Ricci, Fisher, Conformal, Wasserstein, Finsler）
2. 开发自己的空间（实现 `CognitiveSpace` 接口）
3. 组合多个空间（Product, Hierarchical, Mixed）
4. 使用连续坐标替代离散网格（Continuous Space）

---

## 快速开始

### 1. 基础使用（离散网格）

```python
from src.core import Experiment, GeodesicSolver
from src.core.registry import create_space

# 创建空间（以 Euclidean 为例）
space = create_space("euclidean", width=40, height=20)

# 创建求解器
solver = GeodesicSolver(space)

# 求解路径
result = solver.solve(start=(5, 10), goal=(35, 10), obstacles={(20, 10)})

if result.success:
    print(f"Path found: {len(result.path)} steps")
```

### 2. 基础使用（连续坐标）

```python
from src.spaces.continuous import ContinuousRicciSpace

# 创建连续空间（无网格限制）
space = ContinuousRicciSpace(curvature_scale=2.0)

# 更新空间状态
space.update_from_observation(
    position=(1.0, 2.0),
    observation={
        'obstacles': [(2.0, 2.0)],
        'goal_position': (5.0, 5.0),
    }
)

# 计算距离（连续路径积分）
dist = space.compute_distance((0.0, 0.0), (5.0, 5.0))

# 预测下一步
prediction = space.predict_next_state(
    position=(1.0, 2.0),
    observation={'goal_position': (5.0, 5.0), 'step_size': 0.5}
)
```

### 3. 连续空间 SSFR

```python
from src.spaces.continuous_ssfr import ContinuousSSFR

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

### 4. 物理厨房集成

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

### 5. 对比实验

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

### 离散空间（网格世界）

| 空间 | 核心数学 | 适用场景 |
|------|----------|----------|
| `euclidean` | 欧氏距离 | 基线对照 |
| `ricci` | R = -Δ log(u) | 探索-利用平衡 |
| `conformal` | g' = Ω² × g | 目标导向导航 |
| `fisher` | g = 1/confidence | 统计学习 |
| `wasserstein` | 传输成本 | 资源分配 |
| `finsler` | 非对称度量 | 习惯建模 |

### 连续空间（物理世界）

| 空间 | 特性 | 适用场景 |
|------|------|----------|
| `continuous_euclidean` | 基线 | 连续坐标基线 |
| `continuous_ricci` | 稀疏采样 + kNN插值 | 物理世界导航 |
| `continuous_fisher` | 连续置信度场 | 连续状态估计 |
| `continuous_wasserstein` | 连续成本场 | 连续资源分配 |

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

### 子系统（文档）

| 编号 | 子系统 | 代码 | 说明 |
|------|--------|------|------|
| [01](docs/subsystems/01-core.md) | Core | `src/core/` | 空间抽象、注册表、求解器 |
| [02](docs/subsystems/02-ssfr.md) | SSFR | `src/core/ssfr*.py` | 结构自发现与复用 |
| [03](docs/subsystems/03-discrete-spaces.md) | Discrete Spaces | `src/spaces/` | 离散网格空间 |
| [04](docs/subsystems/04-continuous-spaces.md) | Continuous Spaces | `src/spaces/continuous*.py` | 连续坐标空间 |
| [05](docs/subsystems/05-environment.md) | Environment | `src/kitchen/`, `src/visualization/` | 物理环境+可视化 |
| [06](docs/subsystems/06-learning.md) | Learning | `src/learning/` | 学习模块 |
| [07](docs/subsystems/07-research.md) | Research | `src/research/` | 研究工具 |

完整导航见 [docs/INDEX.md](docs/INDEX.md)

### 代码结构

```
src/
├── core/                      # [01] 核心框架
│   ├── space.py              # CognitiveSpace 抽象
│   ├── solver.py             # GeodesicSolver
│   ├── experiment.py         # Experiment 框架
│   ├── registry.py           # 空间注册表
│   ├── replanning.py         # D* Lite 增量规划
│   └── ssfr_enhanced.py     # [02] 增强版 SSFR
├── spaces/                    # [03][04] 空间实现
│   ├── euclidean.py          # 欧氏空间
│   ├── ricci.py              # Ricci空间
│   ├── conformal.py          # 共形空间
│   ├── fisher.py             # Fisher空间
│   ├── wasserstein.py        # Wasserstein空间
│   ├── finsler.py            # Finsler空间
│   ├── continuous.py         # [04] 连续空间基类
│   ├── continuous_ssfr.py    # [04] 连续SSFR
│   ├── temporal.py           # 时序空间
│   ├── composite.py          # 复合空间
│   ├── grid3d.py             # 3D网格空间
│   └── solver3d.py           # 3D求解器
├── kitchen/                   # [05] 物理厨房
│   └── controller.py
├── learning/                  # [06] 学习模块
│   ├── bayesian_optimizer.py
│   ├── meta_learner.py
│   └── neural_space.py
├── visualization/             # [05] 可视化
│   ├── space_visualizer.py
│   ├── path_animator.py
│   └── comparison_plots.py
└── research/                  # [07] 研究工具
    ├── ab_testing.py
    ├── multi_agent_ssfr.py
    ├── consensus.py
    └── neural_gradient.py

experiments/
├── tests/                     # 测试脚本
├── demos/                     # 演示脚本
└── benchmarks/                # 基准测试

tests/                         # pytest 测试
├── test_core.py
└── test_spaces.py
```

---

## SSFR: 结构自发现与复用

### 核心概念

SSFR (Structure Self-Discovery and Reuse) 是 ATLAS 的核心认知机制：

```
感知 → 竞争 → 演化
  ↓      ↓      ↓
生成假设  验证假设  优化结构池
```

### 三种实现对比

| 特性 | 离散 SSFR | 增强版 SSFR | 连续 SSFR |
|------|-----------|-------------|-----------|
| **位置表示** | 离散网格 (int, int) | 离散网格 | 连续坐标 (float, float) |
| **场数据** | numpy 2D array | numpy 2D array | 稀疏采样 + kNN插值 |
| **距离计算** | 网格路径 | 网格路径 | 连续路径积分 |
| **边界** | 固定网格 | 固定网格 | 无限制 |
| **结构定义** | 聚类结果 | 可验证假设 | 可验证假设 |
| **表示方式** | 单空间 | 多空间联合 | 多空间联合 |

### 核心组件

```python
# StructureHypothesis: 结构假设
hypothesis = StructureHypothesis(
    id="hyp_001",
    name="corridor_structure",
    representations={
        "euclidean": {"fields": {...}, "params": {...}},
        "fisher": {"fields": {...}, "params": {...}},
    },
    context={"scene_type": "corridor"}
)

# StructurePool: 结构竞争池
pool = StructurePool(max_structures=100)
pool.add(hypothesis)
winner, results = pool.compete(observation, actual)

# MultiSpaceRepresentation: 多空间联合表示
multi = MultiSpaceRepresentation([euclidean_space, fisher_space])
representations = multi.encode(observation)
consistent = multi.find_consistent_structure(representations, observation)
```

---

## 开发新的认知空间

```python
from src.core.space import CognitiveSpace, register_space

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

### 信息几何与认知空间

认知空间可以看作统计流形上的度量空间：

```
g_ij = E[∂_i log p · ∂_j log p]
```

不同空间对应不同的几何结构：
- **Euclidean**: 平坦度量，基线对照
- **Ricci**: R = -Δ log(u)，曲率反映不确定性
- **Conformal**: g' = Ω² × g，动态调整度量
- **Fisher**: 信息矩阵，统计学习
- **Wasserstein**: 最优传输，资源分配
- **Finsler**: 非对称度量，习惯建模

### SSFR 与信息几何的关系

SSFR ⊂ 信息几何（子集关系）

SSFR 是信息几何在"结构压缩-预测系统"上的一个受限实现：
- 结构 = 统计流形上的点
- 结构发现 = 最大似然估计
- 稳定性 = Fisher 信息
- 价值 = 信息增益 / 成本

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
