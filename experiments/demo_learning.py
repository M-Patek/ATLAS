"""
演示: Learning Integration

展示贝叶斯优化、元学习和神经空间
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from atlas.learning import (
    BayesianOptimizer, SpaceOptimizer,
    MetaLearner, TaskEmbedding,
    NeuralSpace, SpatialEncoder, MetricNetwork,
    MetaTrainingEnvironment
)
from atlas.core import GeodesicSolver
from atlas.core.registry import create_space

print("=" * 70)
print("ATLAS: Learning Integration Demo")
print("=" * 70)
print()

# ============================================================================
# 1. Bayesian Optimization Demo
# ============================================================================

print("=" * 70)
print("1. Bayesian Optimization: Optimize Ricci Curvature Scale")
print("=" * 70)
print()


def evaluate_space_performance(space):
    """评估空间性能"""
    start = (5, 10)
    goal = (35, 10)
    obstacles = {(20, y) for y in range(5, 16) if y != 10}

    try:
        space.update_from_observation(start, {'obstacles': list(obstacles)})
        solver = GeodesicSolver(space)
        result = solver.solve(start, goal, obstacles)

        if result.success:
            # 综合评分：步数少 + 成本低 + 时间短
            score = (100 / (len(result.path) + 1)) + \
                   (10 / (result.cost + 1)) + \
                   (1 / (result.time_ms + 1))
            return score
    except:
        pass

    return 0.0


optimizer = SpaceOptimizer(
    space_type="ricci",
    param_bounds={'curvature_scale': (0.1, 3.0)}
)

print("Optimizing curvature_scale for navigation task...")
result = optimizer.optimize_for_task(
    evaluate_space_performance,
    base_space_params={'width': 40, 'height': 20},
    n_iterations=15
)

print(f"\nOptimization complete!")
print(f"  Best curvature_scale: {result.best_params['curvature_scale']:.3f}")
print(f"  Best score: {result.best_score:.2f}")
print(f"  Total trials: {len(result.all_trials)}")
print()

# 对比优化前后的性能
print("Comparing before/after optimization:")

# 优化前（默认值）
space_default = create_space("ricci", width=40, height=20, curvature_scale=1.0)
score_default = evaluate_space_performance(space_default)

# 优化后
space_optimized = create_space("ricci", width=40, height=20,
                               curvature_scale=result.best_params['curvature_scale'])
score_optimized = evaluate_space_performance(space_optimized)

print(f"  Default (1.0):    score = {score_default:.2f}")
print(f"  Optimized ({result.best_params['curvature_scale']:.2f}): score = {score_optimized:.2f}")
print(f"  Improvement: {((score_optimized - score_default) / score_default * 100):+.1f}%")
print()

# ============================================================================
# 2. Meta-Learning Demo
# ============================================================================

print("=" * 70)
print("2. Meta-Learning: Automatic Space Selection")
print("=" * 70)
print()

meta = MetaLearner(
    space_library=['euclidean', 'ricci', 'conformal', 'fisher'],
    embedding_dim=8
)

# 定义不同任务
tasks = [
    {
        'name': 'Open Field Navigation',
        'obstacle_density': 0.1,
        'dynamic_ratio': 0.0,
        'goal_distance': 30,
        'required_precision': 0.5,
        'exploration_importance': 0.2
    },
    {
        'name': 'Dense Maze',
        'obstacle_density': 0.6,
        'dynamic_ratio': 0.1,
        'goal_distance': 50,
        'required_precision': 0.8,
        'exploration_importance': 0.5
    },
    {
        'name': 'Dynamic Environment',
        'obstacle_density': 0.3,
        'dynamic_ratio': 0.5,
        'goal_distance': 40,
        'required_precision': 0.6,
        'exploration_importance': 0.7
    }
]

print("Task-to-Space recommendations:")
print()
print(f"{'Task':<25} {'Recommended':<15} {'Confidence':<12} {'Alternatives'}")
print("-" * 70)

for task in tasks:
    recommendation = meta.get_space_recommendation(task)

    # 前3推荐
    top3 = [r['space'] for r in recommendation['all_recommendations'][:3]]
    alternatives = ", ".join(top3[1:])

    print(f"{task['name']:<25} {recommendation['recommended']:<15} "
          f"{recommendation['confidence']:.2f}        {alternatives}")

print()

# 模拟在线学习
print("Simulating online learning...")
test_task = tasks[1]  # Dense Maze

for i in range(5):
    selected, metadata = meta.select_space_for_task(test_task, explore_prob=0.2)

    # 模拟性能反馈
    if selected == 'conformal':
        performance = 0.9  # 好的选择
    elif selected == 'ricci':
        performance = 0.7
    else:
        performance = 0.5

    # 报告反馈（在线更新）
    meta.report_performance(test_task, selected, performance)

print(f"After 5 iterations, policy updated with average performance improvements")
print()

# ============================================================================
# 3. Neural Space Demo
# ============================================================================

print("=" * 70)
print("3. NeuralSpace: Learning Spatial Representations")
print("=" * 70)
print()

# 创建神经空间
neural_space = NeuralSpace(
    width=30, height=20,
    observation_shape=(20,),
    embedding_dim=16,
    learning_rate=0.001
)

print("Creating neural space with:")
print(f"  Encoder: {neural_space.encoder.input_dim} -> {neural_space.embedding_dim} dimensions")
print(f"  Metric network: {neural_space.metric.embedding_dim * 2} -> 1 (distance)")
print()

# 模拟训练数据
training_data = []
width, height = 30, 20

print("Generating synthetic training data...")
for _ in range(100):
    # 随机位置
    pos1 = (np.random.randint(0, width), np.random.randint(0, height))
    pos2 = (np.random.randint(0, width), np.random.randint(0, height))

    # 真实距离（欧氏）
    true_dist = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    # 模拟观测
    obs1 = np.random.randn(20)
    obs2 = np.random.randn(20)

    # 更新空间缓存
    neural_space.update_from_observation(pos1, {
        'lidar': obs1,
        'position': pos1
    })
    neural_space.update_from_observation(pos2, {
        'lidar': obs2,
        'position': pos2
    })

print(f"Collected {len(neural_space.position_embeddings)} position embeddings")

# 测试距离计算
test_pos1 = (5, 5)
test_pos2 = (15, 10)

if test_pos1 in neural_space.position_embeddings and \
   test_pos2 in neural_space.position_embeddings:
    neural_dist = neural_space.compute_distance(test_pos1, test_pos2)
    euclidean_dist = np.sqrt((5-15)**2 + (5-10)**2)

    print(f"\nDistance comparison:")
    print(f"  Neural distance:    {neural_dist:.2f}")
    print(f"  Euclidean distance: {euclidean_dist:.2f}")
    print(f"  Ratio: {neural_dist / euclidean_dist:.2f}")
else:
    print(f"\nTest positions not cached")

print()

# ============================================================================
# 4. Meta-Training Environment Demo
# ============================================================================

print("=" * 70)
print("4. Meta-Training Environment: Curriculum Learning")
print("=" * 70)
print()

env = MetaTrainingEnvironment(
    width_range=(20, 40),
    height_range=(20, 30),
    obstacle_density_range=(0.1, 0.5)
)

# 创建课程
curriculum = env.create_curriculum(n_stages=4)

print("Curriculum stages:")
for i, stage in enumerate(curriculum.stages):
    print(f"  Stage {i+1}: {stage.name}, max episodes={stage.max_episodes}")
print()

# 模拟课程学习
print("Simulating curriculum learning...")
for stage_idx in range(len(curriculum.stages)):
    stage = curriculum.current_stage
    print(f"\n  {stage.name}")

    # 生成环境并评估
    for episode in range(5):  # 模拟5个episode
        task = env.generate_task(difficulty=stage.difficulty)

        # 选择空间
        rec = meta.get_space_recommendation(task)
        space_type = rec['recommended']

        # 模拟成功率（高难度任务成功率低）
        success_prob = 0.9 - stage.difficulty * 0.3
        success = np.random.random() < success_prob

        curriculum.report_episode(success)

    print(f"    Stage completed, success rate: {np.mean(list(curriculum.success_history)):.2%}")

    if curriculum.current_stage_idx > stage_idx:
        print(f"    -> Advanced to next stage!")

print()

# ============================================================================
# 5. Full Learning Pipeline
# ============================================================================

print("=" * 70)
print("5. Complete Learning Pipeline")
print("=" * 70)
print()

print("Scenario: Agent deployed to new environment type")
print()

# 步骤1: 元学习选择空间
new_task = {
    'name': 'Unknown Environment',
    'obstacle_density': 0.35,
    'dynamic_ratio': 0.2,
    'goal_distance': 35,
    'exploration_importance': 0.4
}

recommendation = meta.get_space_recommendation(new_task)
selected_type = recommendation['recommended']
print(f"Step 1: Meta-learner selects '{selected_type}' space")
print(f"        (confidence: {recommendation['confidence']:.2f})")

# 步骤2: 贝叶斯优化参数
print(f"\nStep 2: Optimizing parameters for {selected_type}...")

if selected_type == 'ricci':
    opt = SpaceOptimizer(selected_type, {'curvature_scale': (0.5, 2.5)})
else:
    opt = SpaceOptimizer(selected_type, {})  # 无参数可优化

best_params = {'width': 40, 'height': 20}
if hasattr(opt, 'best_params') and opt.best_params:
    best_params.update(opt.best_params)

print(f"        Optimal params: {best_params}")

# 步骤3: 部署训练好的空间
final_space = create_space(selected_type, **best_params)
print(f"\nStep 3: Deploying optimized {selected_type} space")

# 测试性能
start = (5, 10)
goal = (35, 10)
obstacles = {(20, y) for y in range(5, 16) if y != 10}

final_space.update_from_observation(start, {
    'obstacles': list(obstacles),
    'goal_position': goal
})

solver = GeodesicSolver(final_space)
result = solver.solve(start, goal, obstacles)

if result.success:
    print(f"\n        Path found: {len(result.path)} steps")
    print(f"        Cost: {result.cost:.2f}")
else:
    print(f"\n        Path not found (unexpected!)")

print()
print("=" * 70)
print("Learning Integration Demo completed")
print("=" * 70)
