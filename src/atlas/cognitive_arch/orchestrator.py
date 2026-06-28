"""
Orchestrator - 主控制器

组装所有模块，实现依赖注入
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interfaces import (
    Action, Percept, ActionResult,
    IObjectEnvironment, ICognitiveLayer, IExtendedCognitiveAgent,
    ITask, ITaskPlanner, IEpisodicMemory
)
from environment.hierarchical_env import HierarchicalEnvironment
from core.spatial_dynamics import (
    CognitiveOrchestrator, FastLayer, SlowLayer,
    PredictiveField, Consequence
)
from memory.episodic_memory import EpisodicMemory, MemoryManager
from task.planner import TaskPlanner, TaskFactory


class CognitiveSystem(IExtendedCognitiveAgent):
    """
    认知系统

    整合所有模块的统一智能体，通过构造函数注入依赖
    """

    def __init__(self,
                 env: IObjectEnvironment,
                 cognitive_layer: ICognitiveLayer,
                 task_planner: ITaskPlanner,
                 memory: Optional[MemoryManager] = None):
        """
        依赖注入构造函数

        Args:
            env: 环境接口实现
            cognitive_layer: 认知层（快/慢层协调器）
            task_planner: 任务规划器
            memory: 可选的记忆管理器
        """
        self._env = env
        self._cognitive = cognitive_layer
        self._planner = task_planner
        self._memory = memory

        # 状态
        self._current_percept: Optional[Percept] = None
        self._step_count: int = 0
        self._episode_count: int = 0
        self._total_reward: float = 0.0

    # -------------------------------------------------------------------------
    # IExtendedCognitiveAgent 实现
    # -------------------------------------------------------------------------

    def perceive(self) -> Percept:
        """感知环境"""
        self._current_percept = self._env.get_percept()
        return self._current_percept

    def think(self, percept: Percept) -> Tuple[Action, Dict]:
        """
        思考决策 - 空间动力学版本

        流程：
        1. 检查任务子目标
        2. 如果任务需要导航，将目标转化为预测场中的效价吸引子
        3. 让空间动力学（快层/慢层）在预测场上决策
        """
        # 1. 任务层 - 获取当前子目标
        agent_state = {
            'position': percept.position,
            'env': self._env,
            'step': self._step_count
        }

        task_action = self._planner.get_next_action(percept, agent_state)
        task_state = self._planner.get_task_state()

        # 2. 如果任务提供了明确的导航动作（如INTERACT），直接执行
        if task_action is not None:
            # 但对于移动的L/R/U/D，让空间动力学处理
            if task_action not in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
                return task_action, {
                    'source': 'task_planner',
                    'layer': 'task',
                    'task_state': task_state
                }

        # 3. 获取任务目标位置（如果存在）
        goal_position = self._get_task_goal_position(task_state)

        # 4. 空间动力学决策
        # 预测场会自动使用效价梯度和不确定性来决策
        context = {
            'step_count': self._step_count,
            'task_hint': task_state.get('subgoal_type'),
            'goal_position': goal_position  # 会被传播为效价梯度
        }
        action, meta = self._cognitive.process(percept, context)

        # 调试输出
        if self._step_count % 50 == 0:
            print(f"    [Debug] goal_pos={goal_position}, action={action.name}, source={meta.get('source')}")

        return action, {
            **meta,
            'task_state': task_state
        }

    def _get_task_goal_position(self, task_state: Dict) -> Optional[Tuple[float, float]]:
        """从任务状态中提取目标位置"""
        if not task_state.get('active'):
            return None

        subgoal_type = task_state.get('subgoal_type')
        task_name = task_state.get('task_name', '')

        # 直接从当前任务获取子目标
        current_task = self._planner.get_current_task()
        if current_task and hasattr(current_task, 'get_current_subgoal'):
            subgoal = current_task.get_current_subgoal()
            if subgoal and hasattr(subgoal, '_target'):
                target = subgoal._target

                if subgoal_type == 'goto':
                    # 获取目标房间中心
                    if target in self._env._rooms:
                        return self._env._rooms[target].center()

                elif subgoal_type in ['find', 'pickup']:
                    # 获取目标物体位置
                    for obj in self._env._objects.values():
                        if obj.id == target or obj.name == target:
                            return (obj.x, obj.y)

        return None

    def act(self, action: Action) -> ActionResult:
        """执行动作"""
        result = self._env.step(action)
        self._total_reward += result.reward
        return result

    def learn(self, experience: Dict[str, Any]) -> None:
        """学习"""
        # 认知层学习
        self._cognitive.learn(experience)

        # 情节记忆学习
        if self._memory and self._current_percept:
            self._memory.record_step(
                state_key=experience.get('state_key', ''),
                percept=self._current_percept,
                action=experience.get('action'),
                context={'subgoal_type': experience.get('subgoal_type')},
                result=experience.get('result', {}),
                reward=experience.get('reward', 0.0)
            )

    def get_cognitive_state(self) -> Dict[str, Any]:
        """获取认知状态"""
        return {
            'step_count': self._step_count,
            'total_reward': self._total_reward,
            'cognitive': self._cognitive.get_state(),
            'task': self._planner.get_task_state(),
            'memory': self._memory.episodic.get_stats() if self._memory else None,
            'position': self._env.agent_position,
            'room': (self._current_percept.metadata.get('room')
                     if self._current_percept and self._current_percept.metadata
                     else None)
        }

    def assign_task(self, task: ITask) -> None:
        """分配任务"""
        self._planner.assign_task(task)
        if self._memory:
            self._memory.episodic.start_episode(task.name)

    def get_episodic_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return self._memory.episodic.get_stats() if self._memory else {}

    # -------------------------------------------------------------------------
    # 高级接口
    # -------------------------------------------------------------------------

    def step(self) -> Tuple[Action, ActionResult, Dict]:
        """
        完整的一步：感知 -> 思考 -> 执行 -> 学习

        Returns:
            (动作, 执行结果, 决策元数据)
        """
        # 感知（前一状态）
        old_percept = self._current_percept or self.perceive()

        # 思考
        action, meta = self.think(old_percept)

        # 执行
        result = self.act(action)

        # 新的感知
        new_percept = result.percept
        self._current_percept = new_percept

        # 计算 delta_perception（空间动力学核心！）
        delta_perception = new_percept.features - old_percept.features

        # 构建经验
        experience = {
            'state_key': f"{int(old_percept.position[0])},{int(old_percept.position[1])}",
            'position': old_percept.position,
            'action': action,
            'percept': old_percept,
            'delta_perception': delta_perception,  # 空间动力学核心数据
            'result': {
                'new_position': result.percept.position,
                'success': result.reward > 0,
            },
            'reward': result.reward,
            'prediction_error': abs(result.reward - meta.get('expected_value', 0)),
            'subgoal_type': meta.get('task_state', {}).get('subgoal_type')
        }

        # 学习
        self.learn(experience)

        self._step_count += 1

        return action, result, meta


# ============================================================================
# 工厂：负责组装系统
# ============================================================================

class SystemFactory:
    """
    系统工厂

    负责创建和组装认知系统各组件
    """

    @staticmethod
    def create_standard_system(seed: int = 42) -> CognitiveSystem:
        """
        创建标准配置的认知系统 - 空间动力学核心版

        组装流程：
        1. 创建环境
        2. 创建空间动力学认知层（共享预测场）
        3. 创建记忆系统
        4. 创建任务规划器
        5. 组装成系统
        """
        # 1. 环境
        env = HierarchicalEnvironment(width=100, height=100, seed=seed)

        # 2. 空间动力学认知层
        # 核心是共享的预测场，快层和慢层都在上面操作
        cognitive = CognitiveOrchestrator(
            perception_dim=24,  # 环境与感知的维度
            field_width=100,
            field_height=100,
            fast_layer=FastLayer(uncertainty_threshold=0.5),  # 提高阈值以更快自动化
            slow_layer=SlowLayer(num_simulations=20, depth=3)  # 减少模拟步数提高速度
        )

        # 3. 记忆系统（情节记忆辅助经验检索）
        episodic = EpisodicMemory(
            max_episodes=100,
            grid_size=5
        )
        memory = MemoryManager(episodic=episodic)

        # 4. 任务规划器
        planner = TaskPlanner()

        # 5. 组装
        system = CognitiveSystem(
            env=env,
            cognitive_layer=cognitive,
            task_planner=planner,
            memory=memory
        )

        return system


# ============================================================================
# 演示
# ============================================================================

def run_decoupled_demo():
    """
    运解耦架构演示
    """
    print("=" * 70)
    print("解耦认知架构演示")
    print("=" * 70)
    print()
    print("架构:")
    print("  - 接口层: 定义契约边界")
    print("  - 环境层: HierarchicalEnvironment")
    print("  - 认知层: FastLayer + SlowLayer")
    print("  - 记忆层: EpisodicMemory")
    print("  - 任务层: TaskPlanner")
    print("  - 组装: SystemFactory (依赖注入)")
    print()

    # 创建系统
    print("【系统组装】")
    system = SystemFactory.create_standard_system(seed=42)
    print("  CognitiveSystem created")
    print(f"  Components: {type(system).__name__}")
    print()

    # 阶段1: 无任务探索
    print("【阶段1】无任务探索 (100步)")
    print("-" * 50)

    for i in range(100):
        action, result, meta = system.step()

        if i in [30, 60, 80]:  # 添加关键步数调试
            state = system.get_cognitive_state()
            print(f"  Step {i:3d}: pos={state['position']}, "
                  f"room={state['room']}, "
                  f"action={action.name:8s}, "
                  f"source={meta['source']:15s}, "
                  f"reward={state['total_reward']:.1f}")

        if i % 25 == 0 and i not in [30, 60, 80]:
            state = system.get_cognitive_state()
            print(f"  Step {i:3d}: pos={state['position']}, "
                  f"room={state['room']}, "
                  f"action={action.name:8s}, "
                  f"source={meta['source']:15s}, "
                  f"reward={state['total_reward']:.1f}")

    print()

    # 阶段2: 分配任务
    task = TaskFactory.create_fetch_water()
    print(f"【阶段2】分配任务: {task.name}")
    print(f"  子目标: {len([t for t in task._subgoals])}个")
    # 调试：显示物体位置
    env = system._env
    print(f"  Bedroom中的物体:")
    for obj in env._objects.values():
        if obj.room_id == "bedroom":
            print(f"    - {obj.name}({obj.id}): ({obj.x:.1f}, {obj.y:.1f})")
    print()

    system.assign_task(task)

    initial_reward = system._total_reward
    task_start_step = system._step_count

    for i in range(100):
        action, result, meta = system.step()

        if i % 20 == 0 or task.is_completed:
            state = system.get_cognitive_state()
            task_state = state['task']
            print(f"  Step {system._step_count:3d}: pos={state['position']}, room={state['room']}, "
                  f"task={task_state.get('progress', 'N/A'):6s}, "
                  f"source={meta['source']:15s}, "
                  f"action={action.name:8s}")

        if task.is_completed:
            print()
            print(f"  [任务完成] Step {system._step_count}")
            break

    task_steps = system._step_count - task_start_step
    task_reward = system._total_reward - initial_reward

    # 阶段3: 重复任务
    print()
    print(f"【阶段3】重复任务 (情节记忆加速)")

    system._env.reset()
    task2 = TaskFactory.create_fetch_water()
    system.assign_task(task2)

    task2_start = system._step_count
    memory_uses = 0

    for i in range(200):
        action, result, meta = system.step()

        if meta.get('source') == 'episodic_memory':
            memory_uses += 1

        if i % 30 == 0 or task2.is_completed:
            state = system.get_cognitive_state()
            print(f"  Step {system._step_count:3d}: "
                  f"progress={state['task'].get('progress', 'N/A'):6s}, "
                  f"source={meta['source']:15s}")

        if task2.is_completed:
            print()
            print(f"  [任务完成] Step {system._step_count}")
            break

    task2_steps = system._step_count - task2_start

    # 总结
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  任务1步数: {task_steps} (探索+学习)")
    print(f"  任务2步数: {task2_steps} (记忆加速)")
    print(f"  记忆使用率: {memory_uses}/{task2_steps} ({100*memory_uses/max(1,task2_steps):.1f}%)")
    print(f"  总经验数: {system.get_episodic_memory_stats().get('num_experiences', 0)}")
    print()

    return system


if __name__ == "__main__":
    run_decoupled_demo()
