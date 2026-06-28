"""
ATLAS Research: A/B Testing Framework
A/B测试研究框架

严格对比不同认知空间性能的实验框架
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
from pathlib import Path


class NumpyJSONEncoder(json.JSONEncoder):
    """支持numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class TestScenario:
    """标准测试场景"""
    name: str
    width: int
    height: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: set
    difficulty: str  # 'easy', 'medium', 'hard'
    description: str

    # 可选的动态元素
    moving_obstacles: Optional[Dict[str, List[Tuple[int, int]]]] = None


def create_standard_scenarios() -> Dict[str, TestScenario]:
    """创建标准测试场景套件"""
    scenarios = {}

    # 场景1: 开放空间（基线）
    scenarios['open_field'] = TestScenario(
        name='open_field',
        width=40, height=20,
        start=(5, 10), goal=(35, 10),
        obstacles=set(),
        difficulty='easy',
        description='无障碍开放空间，测试基本导航能力'
    )

    # 场景2: 简单墙壁（缺口测试）
    obstacles = {(20, y) for y in range(5, 16) if y != 10}
    scenarios['narrow_gap'] = TestScenario(
        name='narrow_gap',
        width=40, height=20,
        start=(5, 10), goal=(35, 10),
        obstacles=obstacles,
        difficulty='easy',
        description='中间有缺口墙，测试缺口检测'
    )

    # 场景3: 迷宫（中等难度）
    maze_obs = set()
    # 垂直墙
    for y in range(3, 18):
        if y not in [7, 13]:
            maze_obs.add((12, y))
            maze_obs.add((28, y))
    # 水平墙
    for x in range(15, 26):
        if x not in [18, 22]:
            maze_obs.add((x, 5))
            maze_obs.add((x, 15))
    scenarios['maze'] = TestScenario(
        name='maze',
        width=40, height=20,
        start=(5, 10), goal=(35, 10),
        obstacles=maze_obs,
        difficulty='medium',
        description='迷宫结构，测试长距离规划'
    )

    # 场景4: 密集障碍（困难）
    np.random.seed(42)
    dense_obs = set()
    for _ in range(80):
        x = np.random.randint(8, 35)
        y = np.random.randint(3, 17)
        dense_obs.add((x, y))
    # 确保起点终点通畅
    for pos in [(5, 10), (35, 10), (6, 10), (34, 10)]:
        dense_obs.discard(pos)
    scenarios['dense_obstacles'] = TestScenario(
        name='dense_obstacles',
        width=40, height=20,
        start=(5, 10), goal=(35, 10),
        obstacles=dense_obs,
        difficulty='hard',
        description='密集随机障碍，测试鲁棒性'
    )

    # 场景5: 动态障碍
    moving = {
        'patrol1': [(15 + i % 5, 10) for i in range(20)],
        'patrol2': [(25, 6 + i % 8) for i in range(20)]
    }
    scenarios['dynamic'] = TestScenario(
        name='dynamic',
        width=40, height=20,
        start=(5, 10), goal=(35, 10),
        obstacles={(15, 10)},
        difficulty='hard',
        description='动态障碍物，测试适应能力',
        moving_obstacles=moving
    )

    # 场景6: 强障碍长距离
    scenarios['corridor'] = TestScenario(
        name='corridor',
        width=40, height=20,
        start=(5, 10), goal=(35, 10),
        obstacles={(20, y) for y in range(20) if y != 10},
        difficulty='medium',
        description='垂直墙走廊，测试障碍绕行'
    )

    return scenarios


@dataclass
class ExperimentResult:
    """实验结果"""
    space_name: str
    scenario_name: str
    trial: int

    # 核心指标
    success: bool
    path_length: int
    path_cost: float
    planning_time_ms: float
    nodes_expanded: int

    # 路径质量
    path_smoothness: float  # 曲率变化
    path_efficiency: float  # 欧氏距离/实际路径长度

    # 鲁棒性
    replan_count: int = 0
    deviation_count: int = 0

    # 元数据
    timestamp: float = field(default_factory=time.time)


class StatisticalTest:
    """统计检验工具"""

    @staticmethod
    def t_test(group_a: List[float], group_b: List[float]) -> Dict[str, float]:
        """
        独立样本t检验

        Returns:
            {'t_statistic': float, 'p_value': float, 'significant': bool}
        """
        from scipy import stats

        if len(group_a) < 2 or len(group_b) < 2:
            return {'t_statistic': 0, 'p_value': 1.0, 'significant': False}

        t_stat, p_value = stats.ttest_ind(group_a, group_b)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_a': float(np.mean(group_a)),
            'mean_b': float(np.mean(group_b)),
            'std_a': float(np.std(group_a)),
            'std_b': float(np.std(group_b))
        }

    @staticmethod
    def anova(groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        单因素ANOVA

        Returns:
            {'f_statistic': float, 'p_value': float, 'significant': bool}
        """
        from scipy import stats

        values = [g for g in groups.values() if len(g) >= 2]
        if len(values) < 2:
            return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}

        f_stat, p_value = stats.f_oneway(*values)

        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'group_means': {k: float(np.mean(v)) for k, v in groups.items()}
        }

    @staticmethod
    def cohen_d(group_a: List[float], group_b: List[float]) -> float:
        """计算Cohen's d效应量"""
        if len(group_a) == 0 or len(group_b) == 0:
            return 0.0

        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        std_a, std_b = np.std(group_a, ddof=1), np.std(group_b, ddof=1)

        # 合并标准差
        n_a, n_b = len(group_a), len(group_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std


class ABTestExperiment:
    """
    A/B测试实验框架

    严格对比两种或多种认知空间
    """

    def __init__(self,
                 space_factories: Dict[str, Callable],
                 scenarios: Optional[Dict[str, TestScenario]] = None,
                 n_trials: int = 30):
        """
        Args:
            space_factories: {space_name: lambda: create_space(...), ...}
            scenarios: 测试场景，默认使用标准套件
            n_trials: 每个条件的重复次数
        """
        self.space_factories = space_factories
        self.scenarios = scenarios or create_standard_scenarios()
        self.n_trials = n_trials
        self.results: List[ExperimentResult] = []

    def run(self, verbose: bool = True) -> 'ABTestExperiment':
        """运行完整实验"""
        from ..core.solver import GeodesicSolver

        total_runs = len(self.space_factories) * len(self.scenarios) * self.n_trials
        current = 0

        for space_name, factory in self.space_factories.items():
            for scenario_name, scenario in self.scenarios.items():
                for trial in range(self.n_trials):
                    current += 1

                    if verbose and current % 10 == 0:
                        print(f"Progress: {current}/{total_runs} "
                              f"({100*current/total_runs:.1f}%)")

                    # 创建空间
                    space = factory()

                    # 设置场景
                    for obs in scenario.obstacles:
                        space.update_from_observation(
                            scenario.start,
                            {'obstacles': [obs]}
                        )

                    # 规划
                    solver = GeodesicSolver(space)
                    start_time = time.time()
                    result = solver.solve(
                        scenario.start,
                        scenario.goal,
                        scenario.obstacles
                    )
                    planning_time = (time.time() - start_time) * 1000

                    # 计算路径质量
                    smoothness = self._compute_smoothness(result.path) if result.path else float('inf')
                    efficiency = self._compute_efficiency(result.path, scenario) if result.path else 0.0

                    # 记录结果
                    exp_result = ExperimentResult(
                        space_name=space_name,
                        scenario_name=scenario_name,
                        trial=trial,
                        success=result.success,
                        path_length=len(result.path) if result.path else 0,
                        path_cost=result.cost,
                        planning_time_ms=planning_time,
                        nodes_expanded=result.nodes_expanded,
                        path_smoothness=smoothness,
                        path_efficiency=efficiency
                    )

                    self.results.append(exp_result)

        return self

    def _compute_smoothness(self, path: List[Tuple[int, int]]) -> float:
        """计算路径平滑度（平均转向角）"""
        if len(path) < 3:
            return 0.0

        angles = []
        for i in range(1, len(path) - 1):
            p0, p1, p2 = np.array(path[i-1]), np.array(path[i]), np.array(path[i+1])
            v1 = p0 - p1
            v2 = p2 - p1

            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(abs(np.sin(angle)))

        return float(np.mean(angles))

    def _compute_efficiency(self, path: List[Tuple[int, int]], scenario: TestScenario) -> float:
        """计算路径效率（欧氏距离/实际距离）"""
        if not path or len(path) < 2:
            return 0.0

        euclidean_dist = np.sqrt(
            (scenario.goal[0] - scenario.start[0])**2 +
            (scenario.goal[1] - scenario.start[1])**2
        )

        actual_dist = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            actual_dist += np.sqrt(dx**2 + dy**2)

        return euclidean_dist / actual_dist if actual_dist > 0 else 0.0

    def analyze(self) -> Dict[str, Any]:
        """分析实验结果"""
        analysis = {
            'spaces': list(self.space_factories.keys()),
            'scenarios': list(self.scenarios.keys()),
            'n_trials': self.n_trials,
            'metrics': {}
        }

        # 按空间和场景分组
        grouped = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            for metric in ['success', 'path_length', 'path_cost',
                          'planning_time_ms', 'path_efficiency']:
                value = getattr(r, metric)
                if metric == 'success' or not np.isnan(value):
                    grouped[metric][r.space_name].append(value)

        # 统计分析
        stats = StatisticalTest()
        space_names = list(self.space_factories.keys())

        for metric, space_data in grouped.items():
            analysis['metrics'][metric] = {
                'means': {name: float(np.mean(space_data[name]))
                         for name in space_names},
                'stds': {name: float(np.std(space_data[name]))
                        for name in space_names},
                'cis': {name: self._confidence_interval(space_data[name])
                       for name in space_names}
            }

            # 两两t检验
            if len(space_names) == 2:
                a, b = space_names
                ttest = stats.t_test(space_data[a], space_data[b])
                analysis['metrics'][metric]['ttest'] = ttest

            # ANOVA
            anova = stats.anova(space_data)
            analysis['metrics'][metric]['anova'] = anova

        return analysis

    def _confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        if len(data) < 2:
            return (0.0, 0.0)

        from scipy import stats
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        return (float(ci[0]), float(ci[1]))

    def generate_report(self, output_dir: str = 'results/ab_test'):
        """生成研究报告"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        analysis = self.analyze()

        # JSON报告（处理numpy类型）
        json_path = Path(output_dir) / 'analysis.json'
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyJSONEncoder)

        # Markdown报告
        md_path = Path(output_dir) / 'report.md'
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown(analysis))

        # LaTeX表格
        latex_path = Path(output_dir) / 'tables.tex'
        with open(latex_path, 'w') as f:
            f.write(self._generate_latex(analysis))

        print(f"Report generated in {output_dir}")
        return analysis

    def _generate_markdown(self, analysis: Dict) -> str:
        """生成Markdown报告"""
        lines = [
            "# ATLAS A/B Test Results",
            "",
            f"**Spaces tested**: {', '.join(analysis['spaces'])}",
            f"**Scenarios**: {', '.join(analysis['scenarios'])}",
            f"**Trials per condition**: {analysis['n_trials']}",
            "",
            "## Summary",
            ""
        ]

        # 按指标总结
        for metric, data in analysis['metrics'].items():
            lines.extend([
                f"### {metric}",
                "",
                "| Space | Mean | Std | 95% CI |",
                "|-------|------|-----|--------|"
            ])

            for space in analysis['spaces']:
                mean = data['means'][space]
                std = data['stds'][space]
                ci = data['cis'][space]
                lines.append(f"| {space} | {mean:.3f} | {std:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")

            # 统计显著性
            if 'ttest' in data:
                ttest = data['ttest']
                lines.extend([
                    "",
                    f"**t-test**: t={ttest['t_statistic']:.3f}, "
                    f"p={ttest['p_value']:.4f}, "
                    f"significant={'✓' if ttest['significant'] else '✗'}"
                ])

            if 'anova' in data:
                anova = data['anova']
                lines.extend([
                    "",
                    f"**ANOVA**: F={anova['f_statistic']:.3f}, "
                    f"p={anova['p_value']:.4f}"
                ])

            lines.append("")

        return '\n'.join(lines)

    def _generate_latex(self, analysis: Dict) -> str:
        """生成LaTeX表格"""
        lines = [
            "% Auto-generated A/B test results",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{A/B Test: Space Performance Comparison}",
            "\\begin{tabular}{l" + "c" * len(analysis['spaces']) + "c}",
            "\\toprule",
            "Metric & " + " & ".join(analysis['spaces']) + " & p-value \\\\",
            "\\midrule"
        ]

        for metric, data in analysis['metrics'].items():
            means = [f"{data['means'][s]:.3f}" for s in analysis['spaces']]
            p_val = data.get('ttest', {}).get('p_value', data.get('anova', {}).get('p_value', 1.0))
            sig = "*" if p_val < 0.05 else ""
            lines.append(f"{metric} & {' & '.join(means)} & {p_val:.3f}{sig} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:ab_test}",
            "\\end{table}",
            "",
            "% * p < 0.05"
        ])

        return '\n'.join(lines)


class AblationStudy:
    """
    消融研究

    系统地移除组件以测量其贡献
    """

    def __init__(self, base_space_factory: Callable,
                 component_removals: Dict[str, Callable]):
        """
        Args:
            base_space_factory: 创建完整空间的工厂
            component_removals: {name: lambda space: remove_component(space), ...}
        """
        self.base_factory = base_space_factory
        self.component_removals = component_removals

    def run(self, scenario: TestScenario, n_trials: int = 20) -> Dict[str, Any]:
        """运行消融实验"""
        results = {}

        # 基线（完整系统）
        print("Testing baseline (full system)...")
        baseline_exp = ABTestExperiment(
            {'full': self.base_factory},
            {'scenario': scenario},
            n_trials=n_trials
        )
        baseline_exp.run(verbose=False)
        baseline_results = [r for r in baseline_exp.results]

        # 各种消融
        for name, removal_fn in self.component_removals.items():
            print(f"Testing without {name}...")

            def ablated_factory(fn=removal_fn):
                space = self.base_factory()
                return fn(space)

            ablated_exp = ABTestExperiment(
                {'ablated': ablated_factory},
                {'scenario': scenario},
                n_trials=n_trials
            )
            ablated_exp.run(verbose=False)

            # 对比基线
            baseline_success = np.mean([r.success for r in baseline_results])
            ablated_success = np.mean([r.success for r in ablated_exp.results])

            results[name] = {
                'baseline_success': baseline_success,
                'ablated_success': ablated_success,
                'degradation': baseline_success - ablated_success,
                'relative_impact': (baseline_success - ablated_success) / baseline_success if baseline_success > 0 else 0
            }

        return results
