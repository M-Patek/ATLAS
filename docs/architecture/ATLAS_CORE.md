# ATLAS Core: 可插拔认知架构框架

## 架构重构目标

将原有的 Phase 1-6 重构为**可插拔的认知空间实验平台**。

**核心理念**:
- **空间(Space)** 是核心抽象
- **World Model** 负责更新空间
- **Solver** 负责在空间中求解
- **所有组件可插拔、可替换、可对比**

---

## 新架构 (v0.2)

```
atlas/
├── core/                    # 核心框架 (NEW)
│   ├── space.py            # CognitiveSpace 抽象基类
│   ├── world_model.py      # WorldModel 抽象
│   ├── solver.py           # GeodesicSolver 及变体
│   ├── experiment.py       # Experiment & AblationStudy
│   └── registry.py         # 空间注册表
├── spaces/                  # 可插拔空间实现
│   ├── ricci.py           # Ricci 空间 (Phase 5)
│   ├── euclidean.py       # 欧氏基线
│   ├── conformal.py       # 共形空间 (Phase 2/3)
│   ├── fisher.py          # Fisher 信息几何
│   ├── wasserstein.py     # 最优传输
│   └── finsler.py         # Finsler 非对称
├── examples/               # 示例代码
│   └── core_example.py    # 使用示例
└── ...                    # 旧架构保留
```

---

## 快速开始

### 1. 创建和对比不同空间

```python
from atlas.core import Experiment, GeodesicSolver
from atlas.core.registry import create_space

# 创建实验
experiment = Experiment("space_comparison")

# 注册多个空间
experiment.register_space("ricci", create_space("ricci", 40, 20, curvature_scale=2.0))
experiment.register_space("fisher", create_space("fisher", 40, 20))
experiment.register_space("euclidean", create_space("euclidean", 40, 20))

# 添加测试场景
experiment.add_scenario({
    'start': (5, 10),
    'goal': (35, 10),
    'obstacles': {(20, y) for y in range(3, 17) if y != 10},
})

# 运行实验
results = experiment.run(num_trials=10)

# 查看结果
print(experiment.get_summary())
```

### 2. 消融研究

```python
from atlas.core import AblationStudy

# 定义基线配置
base_config = {
    'space_type': 'ricci',
    'width': 40,
    'height': 20,
    'curvature_scale': 1.0,
}

study = AblationStudy(base_config)

# 添加变体
study.add_variation("no_curvature", {'curvature_scale': 0.0})
study.add_variation("high_curvature", {'curvature_scale': 3.0})

# 运行研究
results = study.run(test_fn=my_test_function, num_runs=10)
print(study.analyze(results))
```

---

## 核心概念

### CognitiveSpace (认知空间)

所有空间必须实现的接口:

```python
class CognitiveSpace(ABC):
    def compute_distance(self, pos1, pos2) -> float:
        """计算认知距离"""
        pass

    def get_heuristic(self, pos, goal) -> float:
        """返回启发式估计（必须可接受）"""
        pass

    def update_from_observation(self, position, observation):
        """根据观测更新空间"""
        pass
```

### 内置空间实现

| 空间 | 核心思想 | 参数 |
|------|----------|------|
| `euclidean` | 欧氏距离基线 | 无 |
| `ricci` | Ricci 曲率 = -Δ log(uncertainty) | `curvature_scale` |
| `conformal` | 共形因子变形 | `base_scale` |
| `fisher` | Fisher 信息 = 1/confidence | `temperature` |
| `wasserstein` | 传输成本 | `base_cost` |
| `finsler` | 非对称度量 | `asymmetry` |

---

## 与旧架构的关系

### 向后兼容

旧架构 (`cognitive_arch_v2`) 仍然可用:

```python
# 旧方式 (仍然有效)
from atlas.cognitive_arch_v2 import Phase5PureRicciSystem

# 新方式 (推荐)
from atlas.core.registry import create_space
space = create_space("ricci", 40, 20)
```

### Phase 映射

| 旧 Phase | 新空间实现 |
|----------|------------|
| Phase 1 (Ricci Attention) | `ricci` 空间 + 特定初始化 |
| Phase 2 (Dynamic Metric) | `conformal` 空间 |
| Phase 3 (Hierarchical) | `ricci` + 压缩 |
| Phase 5 (Pure Ricci) | `ricci` 空间完整版 |
| Phase 6 (Pluggable) | **核心框架本身** |

---

## 扩展: 添加新空间

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

空间会自动注册到全局注册表，可以通过 `create_space("my_space", ...)` 使用。

---

## 运行示例

```bash
cd D:\Github\ATLAS
python -m atlas.examples.core_example
```

---

## 下一步

1. **运行对比实验** 测试不同空间的性能
2. **设计混合架构** 组合多个空间
3. **添加新空间** 探索更多数学结构
4. **元学习** 让系统自动选择最优空间
