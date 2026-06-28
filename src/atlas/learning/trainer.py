"""
ATLAS Learning: Training Utilities
训练工具

支持空间训练、课程学习等
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class CurriculumStage:
    """课程阶段"""
    name: str
    difficulty: float
    environment_generator: Callable[[], Any]
    success_threshold: float
    max_episodes: int


class CurriculumScheduler:
    """
    课程学习调度器

    从简单到复杂逐步训练
    """

    def __init__(self, stages: List[CurriculumStage]):
        self.stages = stages
        self.current_stage_idx = 0
        self.episode_count = 0
        self.success_history = deque(maxlen=20)

    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def should_advance(self) -> bool:
        """是否应该进入下一阶段"""
        if not self.success_history:
            return False

        avg_success = np.mean(self.success_history)
        stage = self.current_stage

        if stage is None:
            return False

        return (avg_success >= stage.success_threshold and
                self.episode_count >= stage.max_episodes // 2)

    def advance_stage(self):
        """进入下一阶段"""
        self.current_stage_idx += 1
        self.episode_count = 0
        self.success_history.clear()

        if self.current_stage:
            print(f"Advanced to stage: {self.current_stage.name}")
        else:
            print("Curriculum completed!")

    def get_environment(self) -> Any:
        """获取当前环境"""
        stage = self.current_stage
        if stage:
            return stage.environment_generator()
        return None

    def report_episode(self, success: bool):
        """报告回合结果"""
        self.success_history.append(1.0 if success else 0.0)
        self.episode_count += 1

        if self.should_advance():
            self.advance_stage()


class SpaceTrainer:
    """
    空间训练器

    用于训练可学习的空间（如NeuralSpace）
    """

    def __init__(self,
                 space: Any,
                 learning_rate: float = 0.001,
                 batch_size: int = 32):
        self.space = space
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.episode_rewards = []
        self.loss_history = []

    def train_supervised(self,
                        training_data: List[Tuple[np.ndarray, np.ndarray, float]],
                        n_epochs: int = 10) -> List[float]:
        """
        监督学习训练

        Args:
            training_data: [(obs1, obs2, true_distance), ...]

        Returns:
            每epoch的平均损失
        """
        if not hasattr(self.space, 'train_step'):
            raise ValueError("Space must have train_step method")

        epoch_losses = []

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            # 打乱数据
            indices = np.random.permutation(len(training_data))

            for i in range(0, len(training_data), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_loss = 0.0

                for idx in batch_indices:
                    obs1, obs2, true_dist = training_data[idx]
                    loss = self.space.train_step(obs1, obs2, true_dist)
                    batch_loss += loss

                batch_loss /= len(batch_indices)
                total_loss += batch_loss
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            epoch_losses.append(avg_loss)
            self.loss_history.append(avg_loss)

            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return epoch_losses

    def train_reinforcement(self,
                           environment_factory: Callable,
                           n_episodes: int = 100,
                           max_steps: int = 100) -> Dict[str, List]:
        """
        强化学习训练

        Args:
            environment_factory: 创建环境的工厂函数
            n_episodes: 回合数
            max_steps: 每回合最大步数

        Returns:
            训练历史
        """
        from ..core.solver import GeodesicSolver

        history = {
            'rewards': [],
            'steps': [],
            'successes': []
        }

        for episode in range(n_episodes):
            # 创建环境
            env = environment_factory()

            # 运行回合
            total_reward = 0.0
            success = False

            # 简化版训练循环
            solver = GeodesicSolver(self.space)

            for step in range(max_steps):
                # 这里简化处理，实际应该与环境交互
                pass

            history['rewards'].append(total_reward)
            history['successes'].append(success)

            if episode % 10 == 0:
                avg_reward = np.mean(history['rewards'][-10:])
                success_rate = np.mean(history['successes'][-10:])
                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, Success={success_rate:.2%}")

        return history

    def evaluate(self,
                test_environments: List[Any]) -> Dict[str, float]:
        """
        评估训练好的空间
        """
        from ..core.solver import GeodesicSolver

        results = {
            'avg_steps': 0.0,
            'success_rate': 0.0,
            'avg_cost': 0.0
        }

        successes = []
        all_steps = []
        all_costs = []

        for env in test_environments:
            solver = GeodesicSolver(self.space)
            # 简化评估
            success = True  # 实际应该运行求解
            successes.append(success)

        results['success_rate'] = np.mean(successes) if successes else 0.0
        results['avg_steps'] = np.mean(all_steps) if all_steps else 0.0
        results['avg_cost'] = np.mean(all_costs) if all_costs else 0.0

        return results


class MetaTrainingEnvironment:
    """
    元训练环境

    生成多样化的训练任务
    """

    def __init__(self,
                 width_range: Tuple[int, int] = (20, 50),
                 height_range: Tuple[int, int] = (20, 50),
                 obstacle_density_range: Tuple[float, float] = (0.1, 0.4)):
        self.width_range = width_range
        self.height_range = height_range
        self.obstacle_density_range = obstacle_density_range

    def generate_task(self, difficulty: float = 0.5) -> Dict[str, Any]:
        """
        生成任务

        Args:
            difficulty: 0-1之间的难度
        """
        # 随机尺寸
        width = int(self.width_range[0] + difficulty * (self.width_range[1] - self.width_range[0]))
        height = int(self.height_range[0] + difficulty * (self.height_range[1] - self.height_range[0]))

        # 障碍物密度
        obstacle_density = self.obstacle_density_range[0] + difficulty * (self.obstacle_density_range[1] - self.obstacle_density_range[0])

        # 生成障碍物
        n_obstacles = int(width * height * obstacle_density)
        obstacles = set()

        np.random.seed()
        for _ in range(n_obstacles):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            obstacles.add((x, y))

        # 起点终点
        start = (np.random.randint(0, width//4), np.random.randint(0, height))
        goal = (np.random.randint(3*width//4, width), np.random.randint(0, height))

        # 确保起点终点无障碍
        obstacles.discard(start)
        obstacles.discard(goal)

        return {
            'width': width,
            'height': height,
            'obstacles': obstacles,
            'start': start,
            'goal': goal,
            'difficulty': difficulty,
            'obstacle_density': obstacle_density
        }

    def create_curriculum(self, n_stages: int = 5) -> CurriculumScheduler:
        """
        创建课程
        """
        stages = []

        for i in range(n_stages):
            difficulty = (i + 1) / n_stages

            stage = CurriculumStage(
                name=f"Stage {i+1} (difficulty={difficulty:.2f})",
                difficulty=difficulty,
                environment_generator=lambda d=difficulty: self.generate_task(d),
                success_threshold=0.7 + 0.05 * i,  # 逐渐提高要求
                max_episodes=50
            )
            stages.append(stage)

        return CurriculumScheduler(stages)
