# ATLAS Architecture Roadmap

> 深度架构规划 v1.0 — 从实验室原型到通用认知框架

---

## 1. 架构总览

### 1.1 长期愿景

ATLAS = **A**daptive **T**opological **L**earning for **A**utonomous **S**ystems

最终形态：一个能自动发现、学习、组合认知空间的元架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Meta-Learning Layer                           │
│  - 自动选择最优空间类型                                           │
│  - 学习空间参数 (curvature_scale, asymmetry, ...)                │
│  - 发现新的空间组合方式                                           │
├─────────────────────────────────────────────────────────────────┤
│                    Composition Layer                             │
│  - 空间融合: ProductSpace(SpaceA, SpaceB)                        │
│  - 空间层次: HierarchicalSpace(global, local)                    │
│  - 空间切换: MixedSpace([space1, space2], context_fn)            │
├─────────────────────────────────────────────────────────────────┤
│                    Cognitive Spaces                              │
│  - 基础: Euclidean, Ricci, Conformal, Fisher, Wasserstein       │
│  - 时序: TemporalSpace (sequential field evolution)              │
│  - 概率: ProbabilisticSpace (stochastic metrics)                 │
│  - 神经: NeuralSpace (learned representations)                   │
├─────────────────────────────────────────────────────────────────┤
│                    Core Framework                                │
│  - Space (abstraction)                                           │
│  - Solver (Geodesic, Replanning, Multi-objective)                │
│  - WorldModel (update, predict, compress)                        │
│  - Registry, Experiment, Visualization                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心层演进

### 2.1 当前状态

| 组件 | 状态 | 局限 |
|------|------|------|
| Space | ✅ 稳定 | 仅2D网格 |
| Solver | ✅ 可用 | 仅A*单次求解 |
| WorldModel | ⚠️ 简单 | 无预测、无压缩 |
| Experiment | ⚠️ 基础 | 无统计检验、无可视化 |

### 2.2 演进路线图

#### Phase A: 实时适应能力 (1-2周)

**目标**: 从"单次规划"到"持续适应"

```python
# 当前
result = solver.solve(start, goal)  # 一次性

# 目标
navigator = AdaptiveNavigator(space, solver)
for step in environment:
    observation = sense()
    action = navigator.step(observation)  # 内部自动更新+重新规划
    execute(action)
```

**技术方案**:
- D* Lite 增量路径规划
- 触发式重新规划（阈值触发、事件触发）
- 预测性更新（根据 velocity 预测障碍物位置）

**新增组件**:
```
atlas/core/
├── replanning.py          # D*Lite, LPA*
├── prediction.py          # 场演化预测
└── compression.py         # 空间压缩
```

#### Phase B: 空间组合系统 (2-3周)

**目标**: 多个空间可以组合、切换、融合

```python
# 1. 空间直接积 (并行组合)
combined = ProductSpace(space_a, space_b, weights=[0.7, 0.3])
# distance² = 0.7 * d_a² + 0.3 * d_b²

# 2. 空间层次 (多尺度)
hierarchical = HierarchicalSpace(
    global_space=CoarseRicciSpace(100, 100),
    local_space=DetailedConformalSpace(10, 10)
)

# 3. 上下文切换
adaptive = MixedSpace([
    (exploration_space, lambda ctx: ctx['uncertainty'] > 0.5),
    (navigation_space, lambda ctx: ctx['goal_visible']),
])
```

**数学基础**:
- 乘积流形: M × N
- 共形和: g = Σ ωᵢgᵢ
- 层次映射: π: M → M₀ (投影到粗粒度空间)

#### Phase C: 时序空间 (3-4周)

**目标**: 空间不仅是空间的，也是时间的

```python
class TemporalSpace(CognitiveSpace):
    """
    4D 时空认知空间

    核心思想: 现在的观测影响过去空间的解释
    （类似信号处理中的平滑）
    """
    def __init__(self, ...):
        self.history = CircularBuffer(maxlen=1000)
        self.field_evolution = GaussianProcess()  # 预测场演化

    def update_from_observation(self, position, observation, timestamp):
        # 不仅更新当前，还平滑历史
        self.history.add((position, observation, timestamp))
        self._smooth_past_observations()
        self._predict_future_field()
```

**应用**:
- 动态障碍物轨迹预测
- 周期性模式发现
- 长期记忆形成

---

## 3. 空间类型扩展

### 3.1 即将实现

| 空间类型 | 核心概念 | 应用场景 |
|----------|----------|----------|
| `ProbabilisticSpace` | 距离是分布而非数值 | 不确定性导航 |
| `SpectralSpace` | 基于图拉普拉斯频谱 | 全局结构发现 |
| `NeuralSpace` | 神经网络编码的隐空间 | 端到端学习 |
| `SocialSpace` | 其他智能体的影响场 | 多智能体导航 |

### 3.2 ProbabilisticSpace 详细设计

```python
@register_space("probabilistic")
class ProbabilisticSpace(CognitiveSpace):
    """
    概率认知空间

    不是计算确定距离 d(p1, p2)，而是距离分布 P(D|observations)
    """
    def compute_distance_distribution(self, pos1, pos2) -> Distribution:
        """返回距离的后验分布"""
        # 基于 uncertainty 场建模方差
        mean = self.expected_distance(pos1, pos2)
        variance = self._path_uncertainty(pos1, pos2)
        return Gaussian(mean, variance)

    def plan_with_uncertainty(self, start, goal, confidence=0.9):
        """
        置信度路径规划

        返回满足 P(cost < budget) > confidence 的路径
        """
```

### 3.3 NeuralSpace 详细设计

```python
@register_space("neural")
class NeuralSpace(CognitiveSpace):
    """
    神经认知空间

    用神经网络编码空间结构，可端到端学习
    """
    def __init__(self, encoder: nn.Module, metric_network: nn.Module):
        self.encoder = encoder  # 观测 -> 隐向量
        self.metric_net = metric_network  # (z1, z2) -> distance

    def compute_distance(self, obs1, obs2):
        # 观测可以是原始传感器数据
        z1 = self.encoder(obs1)
        z2 = self.encoder(obs2)
        return self.metric_net(z1, z2)
```

---

## 4. 学习系统集成

### 4.1 三层次学习

```
Level 1: Parameter Learning (空间内部)
    - 学习 Ricci 的 curvature_scale
    - 学习 Conformal 的 attractor 强度
    - 方法: 贝叶斯优化、梯度下降

Level 2: Space Selection (空间之间)
    - 根据任务类型自动选择空间
    - 方法: 元学习 (MAML), 自动机

Level 3: Architecture Discovery (架构层面)
    - 发现新的空间组合方式
    - 方法: 神经架构搜索 (NAS), 遗传算法
```

### 4.2 元学习器设计

```python
class MetaLearner:
    """
    元学习器: 学习如何为任务选择空间
    """
    def __init__(self):
        self.space_library = [...]  # 所有可用空间
        self.task_embedding = TaskEmbeddingNetwork()
        self.space_selector = SpaceSelectionPolicy()

    def meta_train(self, task_distribution):
        """
        在任务分布上元训练
        """
        for task in task_distribution:
            # 少量梯度步适应新任务
            space = self.select_space(task)
            adapted_space = self.adapt(space, task, n_steps=5)

            # 元梯度更新选择策略
            loss = evaluate(adapted_space, task)
            self.meta_update(loss)

    def select_for_new_task(self, task_description) -> CognitiveSpace:
        """
        为新任务选择并配置空间
        """
        embedding = self.task_embedding(task_description)
        space_type = self.space_selector(embedding)
        params = self.optimize_params(space_type, task)
        return create_space(space_type, **params)
```

---

## 5. 多智能体扩展

### 5.1 共享空间

```python
class SharedCognitiveSpace(CognitiveSpace):
    """
    多智能体共享的认知空间

    每个智能体维护局部估计，通过通信融合
    """
    def __init__(self, agent_id, communication_range):
        self.agent_id = agent_id
        self.local_estimate = ...
        self.communication_graph = ...

    def fuse_observations(self, neighbor_estimates):
        """
        分布式融合 (类似卡尔曼滤波共识)
        """
```

### 5.2 社会空间

```python
class SocialSpace(CognitiveSpace):
    """
    建模其他智能体的意图/能力

    g_social(x, y) = f(其他智能体的预测轨迹)
    """
    def update_from_other_agent(self, agent_id, observed_trajectory):
        # 预测其他智能体的目标
        predicted_goal = predict_intention(observed_trajectory)
        # 在其预测路径上增加 repeller
        self.add_dynamic_repeller(predicted_goal, time_horizon=5)
```

---

## 6. 真实世界接口

### 6.1 ROS2 集成

```python
# atlas/interfaces/ros2_interface.py
class ROS2CognitiveNavigator(Node):
    """
    ROS2 节点：将 ATLAS 空间连接到真实机器人
    """
    def __init__(self):
        self.space = create_space("ricci", ...)
        self.solver = GeodesicSolver(self.space)

        # 订阅
        self.create_subscription(LaserScan, '/scan', self.on_scan)
        self.create_subscription(Odometry, '/odom', self.on_odom)

        # 发布
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel')

    def on_scan(self, msg):
        # 激光雷达 -> 障碍物 -> 更新空间
        obstacles = self.process_scan(msg)
        self.space.update_from_observation(self.position, {'obstacles': obstacles})
        self.replan_if_needed()
```

### 6.2 Gazebo/Isaac Sim

```python
class SimulatedEnvironment:
    """
    适配到仿真环境
    """
    def step(self, action):
        # 执行动作
        self.robot.set_velocity(action)
        self.sim.step()

        # 获取观测
        observation = self.get_observation()

        # ATLAS 处理
        self.navigator.update(observation)
        path = self.navigator.get_path()

        return self.compute_reward(path), observation
```

---

## 7. 认知科学对接

### 7.1 与认知理论的映射

| ATLAS 概念 | 认知理论 | 研究方向 |
|------------|----------|----------|
| Cognitive Space | 认知地图 (Tolman) | 生物可塑性验证 |
| Ricci Curvature | 新奇检测 (Sansom) | 神经相关性研究 |
| Conformal Factor | 注意力分配 (Kahneman) | 眼动追踪验证 |
| Fisher Metric | 统计学习 (Tenenbaum) | 人类行为对比 |
| Space Update | 贝叶斯大脑 (Friston) | 主动推断形式化 |

### 7.2 人类实验设计

```python
class HumanStudyFramework:
    """
    对比 ATLAS 预测与人类行为的框架
    """
    def run_navigation_experiment(self, environment, task):
        # 1. 记录人类轨迹
        human_trajectory = self.record_human(environment, task)

        # 2. ATLAS 预测
        atlas_trajectory = self.predict_with_atlas(environment, task)

        # 3. 对比分析
        similarity = trajectory_similarity(human_trajectory, atlas_trajectory)
        field_correlation = compare_attention_fields(
            self.reconstruct_human_attention(human_trajectory),
            self.atlas_space.get_visualization_fields()
        )

        return {
            'trajectory_iou': similarity,
            'attention_correlation': field_correlation,
            'decision_agreement': self.compare_decisions(...)
        }
```

---

## 8. 实现路线图

### 第一阶段：核心强化 (Month 1)

**目标**: 让框架"可用"于研究

- [x] 基础空间框架
- [x] 基本实验系统
- [ ] D* Lite 重新规划
- [ ] 可视化工具 (matplotlib/plotly)
- [ ] 参数优化工具
- [ ] 完备文档

### 第二阶段：组合能力 (Month 2)

**目标**: 支持复杂任务

- [ ] ProductSpace 实现
- [ ] HierarchicalSpace 实现
- [ ] TemporalSpace 实现
- [ ] 空间切换机制
- [ ] 动态障碍物支持
- [ ] 多目标规划 (Pareto 前沿)

### 第三阶段：学习能力 (Month 3)

**目标**: 自适应空间

- [ ] 贝叶斯参数优化
- [ ] 元学习原型
- [ ] NeuralSpace (PyTorch 集成)
- [ ] 从演示学习 (Learning from Demonstration)
- [ ] 在线适应

### 第四阶段：应用拓展 (Month 4-6)

**目标**: 真实场景

- [ ] ROS2 接口
- [ ] Gazebo/Isaac Sim 集成
- [ ] 多智能体系统
- [ ] 人类行为数据集对比
- [ ] 论文发表

---

## 9. 技术栈决策

### 9.1 依赖管理

```
核心 (必须):
  - numpy (数值计算)
  - heapq (A* 内建)

性能 (可选):
  - numba (JIT 加速 compute_distance)
  - scipy (稀疏矩阵、优化)

学习 (可选):
  - torch (NeuralSpace)
  - optuna (贝叶斯优化)

可视化 (可选):
  - matplotlib (2D 场可视化)
  - plotly (交互式)
  - pygame (实时可视化)

仿真 (可选):
  - rclpy (ROS2)
  - gymnasium (强化学习环境)
```

### 9.2 设计原则

1. **渐进式复杂性**: 用户可以先只用 Euclidean，再逐步引入复杂空间
2. **向后兼容**: 空间 API 稳定，新功能通过扩展实现
3. **性能隔离**: 复杂功能（如 NeuralSpace）不拖累核心
4. **可验证性**: 每个空间都有数学性质的单元测试

---

## 10. 下一步行动

**你需要决定优先方向：**

**A. 实时适应性** — 让当前 2D 系统能动态重新规划
- 适合：如果你想在简单环境测试自适应行为

**B. 空间组合** — 实现 Product、Hierarchical、Mixed 空间
- 适合：如果你想测试不同空间的组合效果

**C. 3D/连续空间** — 扩展到更真实的维度
- 适合：如果你想对接真实机器人或仿真器

**D. 可视化工具** — 先看到空间内部如何工作
- 适合：如果你想调试、展示、发表

**E. 其他** — 告诉我具体方向

---

*ATLAS Architecture Roadmap v1.0 — 2026.06.27*
