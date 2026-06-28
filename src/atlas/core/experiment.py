"""
ATLAS Core: Experiment Framework
实验框架

支持可插拔的对比实验设计
"""

import time
import copy
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .space import CognitiveSpace, SpaceMetrics
from .world_model import WorldModel
from .solver import GeodesicSolver, SolverResult


@dataclass
class TrialResult:
    """单次试验结果"""
    success: bool = False
    steps: int = 0
    path_cost: float = 0.0
    nodes_expanded: int = 0
    planning_time_ms: float = 0.0
    total_time_ms: float = 0.0
    path: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionResult:
    """某个实验条件的结果统计"""
    condition_name: str
    space_name: str
    solver_name: str = "default"
    trials: int = 0
    successes: int = 0
    mean_steps: float = 0.0
    std_steps: float = 0.0
    mean_cost: float = 0.0
    mean_time_ms: float = 0.0
    mean_nodes_expanded: float = 0.0
    trial_results: List[TrialResult] = field(default_factory=list)

    def add_trial(self, result: TrialResult):
        """添加单次试验结果"""
        self.trials += 1
        self.trial_results.append(result)

        if result.success:
            self.successes += 1

        # 重新计算统计
        if self.trial_results:
            self.mean_steps = np.mean([t.steps for t in self.trial_results])
            self.std_steps = np.std([t.steps for t in self.trial_results])
            self.mean_cost = np.mean([t.path_cost for t in self.trial_results])
            self.mean_time_ms = np.mean([t.total_time_ms for t in self.trial_results])
            self.mean_nodes_expanded = np.mean([t.nodes_expanded for t in self.trial_results])

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials > 0 else 0.0

    def __repr__(self) -> str:
        return (f"{self.condition_name}: {self.successes}/{self.trials} "
                f"({self.success_rate:.0%}), steps={self.mean_steps:.1f}±{self.std_steps:.1f}")


class Experiment:
    """
    可插拔对比实验

    核心设计:
    - 可以同时测试多个空间
    - 可以同时测试多个求解器
    - 通过配置灵活组合
    """

    def __init__(self, name: str = "experiment"):
        self.name = name
        self.spaces: Dict[str, CognitiveSpace] = {}
        self.solvers: Dict[str, GeodesicSolver] = {}
        self.results: Dict[str, ConditionResult] = {}
        self.scenarios: List[Dict] = []

    def register_space(self, name: str, space: CognitiveSpace) -> 'Experiment':
        """注册空间"""
        self.spaces[name] = space
        return self

    def register_solver(self, name: str, solver: GeodesicSolver) -> 'Experiment':
        """注册求解器"""
        self.solvers[name] = solver
        return self

    def add_scenario(self, scenario: Dict) -> 'Experiment':
        """添加测试场景"""
        self.scenarios.append(scenario)
        return self

    def run(self, num_trials: int = 1,
           progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, ConditionResult]:
        """
        运行实验

        Args:
            num_trials: 每个条件的试验次数
            progress_callback: 进度回调函数 (condition_name, current, total)

        Returns:
            Dict[condition_name, ConditionResult]
        """
        total_conditions = len(self.spaces) * len(self.solvers) * len(self.scenarios)
        current = 0

        for space_name, space in self.spaces.items():
            for solver_name, base_solver in self.solvers.items():
                for scenario_idx, scenario in enumerate(self.scenarios):
                    condition_name = f"{space_name}_{solver_name}_scenario{scenario_idx}"

                    if progress_callback:
                        progress_callback(condition_name, current, total_conditions)

                    result = self._run_condition(
                        condition_name, space_name, space,
                        solver_name, base_solver, scenario, num_trials
                    )
                    self.results[condition_name] = result
                    current += 1

        return self.results

    def _run_condition(self, condition_name: str,
                      space_name: str, space_template: CognitiveSpace,
                      solver_name: str, solver_template: GeodesicSolver,
                      scenario: Dict, num_trials: int) -> ConditionResult:
        """运行单个条件的实验"""
        result = ConditionResult(condition_name, space_name, solver_name)

        for trial in range(num_trials):
            # 深拷贝空间（确保每个试验独立）
            space = copy.deepcopy(space_template)

            # 重新创建求解器（绑定到新空间）
            solver = GeodesicSolver(space,
                                   use_diagonal=solver_template.use_diagonal,
                                   max_iterations=solver_template.max_iterations)

            # 运行试验
            trial_result = self._run_trial(space, solver, scenario, trial)
            result.add_trial(trial_result)

        return result

    def _run_trial(self, space: CognitiveSpace,
                  solver: GeodesicSolver,
                  scenario: Dict,
                  seed: int) -> TrialResult:
        """运行单次试验"""
        import random
        import numpy as np
        np.random.seed(seed)
        random.seed(seed)

        start_time = time.time()

        # 初始化空间（如果有初始化数据）
        if 'initial_uncertainty' in scenario:
            if hasattr(space, 'uncertainty'):
                space.uncertainty = scenario['initial_uncertainty'].copy()
                if hasattr(space, '_recalculate_curvature'):
                    space._recalculate_curvature()

        # 执行观测序列
        for obs in scenario.get('observations', []):
            space.update_from_observation(obs['position'], obs['data'])

        # 规划路径
        start = scenario['start']
        goal = scenario['goal']
        obstacles = scenario.get('obstacles', set())

        solver_result = solver.solve(start, goal, obstacles)

        total_time = (time.time() - start_time) * 1000

        return TrialResult(
            success=solver_result.success,
            steps=len(solver_result.path) if solver_result.path else 0,
            path_cost=solver_result.cost,
            nodes_expanded=solver_result.nodes_expanded,
            planning_time_ms=solver_result.time_ms,
            total_time_ms=total_time,
            path=solver_result.path,
            metadata={'scenario': scenario}
        )

    def get_summary(self) -> str:
        """获取实验摘要"""
        lines = [f"Experiment: {self.name}", "=" * 70, ""]

        # 按空间分组
        by_space = defaultdict(list)
        for name, result in self.results.items():
            by_space[result.space_name].append(result)

        for space_name, results in by_space.items():
            lines.append(f"\nSpace: {space_name}")
            lines.append("-" * 50)
            for r in results:
                lines.append(f"  {r.solver_name}: {r}")

        return "\n".join(lines)

    def compare_spaces(self, metric: str = 'success_rate') -> List[Tuple[str, float]]:
        """
        比较所有空间的某个指标

        Args:
            metric: 'success_rate', 'mean_steps', 'mean_cost', 'mean_time_ms'

        Returns:
            [(space_name, value), ...] 按值排序
        """
        space_values = defaultdict(list)

        for result in self.results.values():
            value = getattr(result, metric, 0.0)
            space_values[result.space_name].append(value)

        # 平均每个空间的多个结果
        averaged = {
            name: np.mean(values)
            for name, values in space_values.items()
        }

        # 排序
        return sorted(averaged.items(), key=lambda x: x[1], reverse=True)


class AblationStudy:
    """
    消融研究

    系统地移除或替换组件，分析每个组件的贡献
    """

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.variations: List[Tuple[str, Dict[str, Any]]] = []

    def add_variation(self, name: str, changes: Dict[str, Any]) -> 'AblationStudy':
        """
        添加一个变体配置

        Args:
            name: 变体名称
            changes: 相对于 base_config 的修改
        """
        self.variations.append((name, changes))
        return self

    def run(self, test_fn: Callable[[Dict[str, Any]], float],
           num_runs: int = 10) -> Dict[str, List[float]]:
        """
        运行消融研究

        Args:
            test_fn: 测试函数，接收配置，返回性能分数
            num_runs: 每个配置的测试次数

        Returns:
            Dict[variation_name, [score1, score2, ...]]
        """
        results = {}

        # 测试基线
        baseline_scores = [test_fn(self.base_config) for _ in range(num_runs)]
        results['baseline'] = baseline_scores

        # 测试每个变体
        for name, changes in self.variations:
            config = copy.deepcopy(self.base_config)
            config.update(changes)

            scores = [test_fn(config) for _ in range(num_runs)]
            results[name] = scores

        return results

    def analyze(self, results: Dict[str, List[float]]) -> str:
        """分析消融研究结果"""
        lines = ["Ablation Study Results", "=" * 50]

        baseline_mean = np.mean(results['baseline'])
        lines.append(f"\nBaseline: {baseline_mean:.3f} ± {np.std(results['baseline']):.3f}")

        for name, scores in results.items():
            if name == 'baseline':
                continue

            mean = np.mean(scores)
            std = np.std(scores)
            delta = mean - baseline_mean
            delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0

            lines.append(f"\n{name}:")
            lines.append(f"  Score: {mean:.3f} ± {std:.3f}")
            lines.append(f"  Delta: {delta:+.3f} ({delta_pct:+.1f}%)")

        return "\n".join(lines)
