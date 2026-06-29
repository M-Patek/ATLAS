"""
Multi-Agent SSFR Framework
多智能体协同SSFR框架

理论基础：
- 多智能体SSFR位于乘积流形 M = M₁ × M₂ × ... × M_N 上
- 每个智能体有自己的流形M_i
- 联合流形是笛卡尔积

核心概念：
1. ProductManifold: 乘积流形实现
2. MultiAgentSSFR: 多智能体SSFR协调器
3. SharedStructurePool: 共享结构池
4. ConsensusProtocol: 一致性协议
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
import copy
import time

from ..core.space import CognitiveSpace
from ..core.registry import create_space
from ..core.ssfr_enhanced import SSFREnhanced, StructureHypothesis, StructurePool


# ============================================================================
# 1. 乘积流形 Product Manifold
# ============================================================================

class ProductManifold:
    """
    多智能体的乘积流形

    M = M₁ × M₂ × ... × M_N

    属性:
    - agent_spaces: 每个智能体的认知空间集合
    - metric_type: 联合度量类型 ('euclidean', 'finsler', 'wasserstein')

    数学:
    - 联合距离 = 各智能体距离的组合
      d_joint(state1, state2) = sqrt(Σ w_i * d_i²)
    - 联合Fisher信息 = 直和
      I_joint = I₁ ⊕ I₂ ⊕ ... ⊕ I_N
    """

    def __init__(self,
                 agent_spaces: List[Dict[str, CognitiveSpace]],
                 metric_type: str = 'euclidean',
                 agent_weights: Optional[List[float]] = None):
        """
        初始化乘积流形

        Args:
            agent_spaces: 每个智能体的空间字典列表 [{space_name: space}, ...]
            metric_type: 联合度量类型
            agent_weights: 智能体权重（如果不指定则平均分配）
        """
        self.num_agents = len(agent_spaces)
        self.agent_spaces = agent_spaces  # List[Dict[str, CognitiveSpace]]
        self.metric_type = metric_type

        # 智能体权重
        if agent_weights is None:
            self.agent_weights = [1.0 / self.num_agents] * self.num_agents
        else:
            total = sum(agent_weights)
            self.agent_weights = [w / total for w in agent_weights]

        # 验证所有智能体具有相同的空间结构
        self._validate_spaces()

        # 联合Fisher信息矩阵（分块对角）
        self._fisher_matrix = None
        self._fisher_dirty = True

    def _validate_spaces(self):
        """验证所有智能体具有相同的空间结构"""
        if self.num_agents == 0:
            return

        base_spaces = set(self.agent_spaces[0].keys())
        for i, spaces in enumerate(self.agent_spaces[1:], 1):
            if set(spaces.keys()) != base_spaces:
                raise ValueError(
                    f"Agent {i} has different spaces than agent 0. "
                    f"Expected {base_spaces}, got {set(spaces.keys())}"
                )

    def compute_distance(self,
                         state1: List[Tuple[int, int]],
                         state2: List[Tuple[int, int]],
                         space_name: Optional[str] = None) -> float:
        """
        计算两个联合状态间的距离

        Args:
            state1: 联合状态1 [(x1, y1), (x2, y2), ...]
            state2: 联合状态2 [(x1, y1), (x2, y2), ...]
            space_name: 如果指定，只使用该空间计算距离

        Returns:
            联合距离（非负浮点数）
        """
        if len(state1) != self.num_agents or len(state2) != self.num_agents:
            raise ValueError(
                f"State dimension mismatch. Expected {self.num_agents}, "
                f"got {len(state1)} and {len(state2)}"
            )

        distances_squared = []

        for i, (pos1, pos2, weight) in enumerate(zip(state1, state2, self.agent_weights)):
            agent_spaces = self.agent_spaces[i]

            if space_name and space_name in agent_spaces:
                # 使用指定空间
                d = agent_spaces[space_name].compute_distance(pos1, pos2)
                distances_squared.append(weight * d ** 2)
            elif not space_name:
                # 使用所有空间的加权平均
                agent_distances = []
                for space in agent_spaces.values():
                    try:
                        agent_distances.append(space.compute_distance(pos1, pos2))
                    except Exception:
                        pass
                if agent_distances:
                    d = np.mean(agent_distances)
                    distances_squared.append(weight * d ** 2)

        if not distances_squared:
            return float('inf')

        # 欧氏组合
        if self.metric_type == 'euclidean':
            return np.sqrt(sum(distances_squared))
        # 最大值（Chebyshev-like）
        elif self.metric_type == 'max':
            return np.sqrt(max(distances_squared))
        # 平均值
        elif self.metric_type == 'mean':
            return np.mean([np.sqrt(d) for d in distances_squared])
        else:
            return np.sqrt(sum(distances_squared))

    def get_heuristic(self,
                      joint_state: List[Tuple[int, int]],
                      joint_goal: List[Tuple[int, int]],
                      space_name: Optional[str] = None) -> float:
        """
        获取启发式估计（从联合状态到联合目标）

        Args:
            joint_state: 当前联合状态
            joint_goal: 目标联合状态
            space_name: 指定使用的空间

        Returns:
            启发式估计值
        """
        # 启发式必须是可接受的（admissible）
        # 使用各智能体启发式的加权和（确保可接受）
        heuristics = []

        for i, (state, goal, weight) in enumerate(
            zip(joint_state, joint_goal, self.agent_weights)
        ):
            agent_spaces = self.agent_spaces[i]

            if space_name and space_name in agent_spaces:
                h = agent_spaces[space_name].get_heuristic(state, goal)
                heuristics.append(weight * h)
            elif not space_name:
                # 取最小启发式（最乐观的估计）
                agent_heuristics = []
                for space in agent_spaces.values():
                    try:
                        agent_heuristics.append(space.get_heuristic(state, goal))
                    except Exception:
                        pass
                if agent_heuristics:
                    h = min(agent_heuristics)
                    heuristics.append(weight * h)

        return sum(heuristics) if heuristics else 0.0

    def get_fisher_information(self, agent_id: Optional[int] = None) -> np.ndarray:
        """
        获取联合Fisher信息矩阵（分块对角）

        M_fisher = diag(I₁, I₂, ..., I_N)

        Args:
            agent_id: 如果指定，只返回该智能体的Fisher信息

        Returns:
            Fisher信息矩阵（或其近似）
        """
        if agent_id is not None:
            return self._get_agent_fisher(agent_id)

        # 构建分块对角矩阵
        blocks = []
        for i in range(self.num_agents):
            fisher_i = self._get_agent_fisher(i)
            blocks.append(fisher_i)

        # 分块对角矩阵
        total_dim = sum(f.shape[0] for f in blocks)
        joint_fisher = np.zeros((total_dim, total_dim))

        idx = 0
        for block in blocks:
            n = block.shape[0]
            joint_fisher[idx:idx+n, idx:idx+n] = block
            idx += n

        return joint_fisher

    def _get_agent_fisher(self, agent_id: int) -> np.ndarray:
        """获取单个智能体的Fisher信息矩阵近似"""
        agent_spaces = self.agent_spaces[agent_id]

        # 近似：取各空间Fisher信息的平均
        fishers = []
        for space in agent_spaces.values():
            if hasattr(space, 'fisher_information'):
                fishers.append(space.fisher_information)
            elif hasattr(space, 'curvature'):
                # 用曲率场近似
                curv = np.mean(space.curvature)
                fishers.append(np.array([[1.0 + abs(curv)]]))

        if fishers:
            return np.mean(fishers, axis=0)
        return np.eye(2)  # 默认2D单位矩阵

    def update_from_observation(self,
                                agent_id: int,
                                position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """
        根据观测更新特定智能体的空间

        Args:
            agent_id: 智能体ID
            position: 观测位置
            observation: 观测数据（包含 'agent_observations' 字段）
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        agent_spaces = self.agent_spaces[agent_id]

        # 更新该智能体的所有空间
        for space in agent_spaces.values():
            try:
                space.update_from_observation(position, observation)
            except Exception:
                pass

        self._fisher_dirty = True

    def get_joint_state_representation(self,
                                       joint_observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取联合状态的表示

        Args:
            joint_observation: 联合观测 {'agent_0': {...}, 'agent_1': {...}}

        Returns:
            联合表示字典
        """
        representations = {}

        for i in range(self.num_agents):
            agent_obs = joint_observation.get(f'agent_{i}', {})
            agent_reps = {}

            for name, space in self.agent_spaces[i].items():
                try:
                    fields = space.get_visualization_fields()
                    agent_reps[name] = {
                        'fields': fields,
                        'statistics': space.get_statistics(),
                    }
                except Exception:
                    pass

            representations[f'agent_{i}'] = agent_reps

        return representations

    def get_statistics(self) -> Dict[str, Any]:
        """获取乘积流形统计信息"""
        return {
            'num_agents': self.num_agents,
            'metric_type': self.metric_type,
            'agent_weights': self.agent_weights,
            'space_names': list(self.agent_spaces[0].keys()) if self.agent_spaces else [],
        }


# ============================================================================
# 2. 共享结构池 Shared Structure Pool
# ============================================================================

@dataclass
class SharedStructure:
    """共享结构（带智能体来源信息）"""
    structure: StructureHypothesis
    source_agents: Set[int] = field(default_factory=set)
    consensus_round: int = 0
    local_versions: Dict[int, 'SharedStructure'] = field(default_factory=dict)


class SharedStructurePool:
    """
    多智能体共享结构池

    功能：
    1. 存储全局一致的结构
    2. 跟踪每个结构的来源智能体
    3. 支持结构的合并与分歧检测
    """

    def __init__(self, max_structures: int = 100):
        self.structures: Dict[str, SharedStructure] = {}
        self.max_structures = max_structures
        self.consensus_history: List[Dict] = []

    def add(self,
            hypothesis: StructureHypothesis,
            source_agent: int) -> SharedStructure:
        """添加结构到共享池"""
        shared = SharedStructure(
            structure=hypothesis,
            source_agents={source_agent},
            local_versions={source_agent: hypothesis}
        )

        # 检查是否已存在相似结构
        existing = self._find_similar(hypothesis)
        if existing:
            # 合并来源信息
            existing.source_agents.add(source_agent)
            existing.local_versions[source_agent] = hypothesis
            return existing

        self.structures[hypothesis.id] = shared

        # 如果超过限制，淘汰最老的结构
        if len(self.structures) > self.max_structures:
            self._eliminate_oldest()

        return shared

    def _find_similar(self, hypothesis: StructureHypothesis) -> Optional[SharedStructure]:
        """查找相似的结构"""
        for shared in self.structures.values():
            # 基于空间类型和特征相似度比较
            if self._are_similar(shared.structure, hypothesis):
                return shared
        return None

    def _are_similar(self, s1: StructureHypothesis, s2: StructureHypothesis) -> bool:
        """判断两个结构是否相似"""
        # 比较空间类型
        type1 = s1.context.get('space_type', '')
        type2 = s2.context.get('space_type', '')
        if type1 != type2:
            return False

        # 比较特征
        f1 = s1.context.get('features', {})
        f2 = s2.context.get('features', {})

        # 简单的特征相似度
        u1 = f1.get('uncertainty_pattern', {}) if f1 else {}
        u2 = f2.get('uncertainty_pattern', {}) if f2 else {}
        if u1 and u2:
            mean1 = u1.get('mean', 0) if isinstance(u1, dict) else 0
            mean2 = u2.get('mean', 0) if isinstance(u2, dict) else 0
            if abs(mean1 - mean2) < 0.2:
                return True

        return False

    def _eliminate_oldest(self) -> None:
        """淘汰最老的结构（基于创建时间）"""
        if not self.structures:
            return

        oldest_id = min(
            self.structures.keys(),
            key=lambda sid: self.structures[sid].structure.created_at
        )
        del self.structures[oldest_id]

    def get_consensus_structures(self,
                                 min_agents: int = 2) -> List[SharedStructure]:
        """
        获取达成共识的结构

        Args:
            min_agents: 需要至少多少智能体认可

        Returns:
            达成共识的结构列表
        """
        return [
            s for s in self.structures.values()
            if len(s.source_agents) >= min_agents
        ]

    def get_agent_structures(self, agent_id: int) -> List[SharedStructure]:
        """获取特定智能体的结构"""
        return [
            s for s in self.structures.values()
            if agent_id in s.source_agents
        ]

    def update_consensus(self,
                         round_num: int,
                         agent_structures: Dict[int, List[StructureHypothesis]]) -> Dict[str, Any]:
        """
        更新共识

        Args:
            round_num: 共识轮次
            agent_structures: {agent_id: [hypotheses]}

        Returns:
            共识统计
        """
        # 收集所有结构
        for agent_id, structures in agent_structures.items():
            for struct in structures:
                shared = self.add(struct, agent_id)
                shared.consensus_round = round_num

        # 统计
        consensus = self.get_consensus_structures(min_agents=2)

        stats = {
            'round': round_num,
            'total_structures': len(self.structures),
            'consensus_structures': len(consensus),
            'agent_contributions': {
                aid: len(struc_list)
                for aid, struc_list in agent_structures.items()
            }
        }

        self.consensus_history.append(stats)
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        consensus = self.get_consensus_structures(min_agents=2)
        return {
            'total_structures': len(self.structures),
            'consensus_structures': len(consensus),
            'consensus_rate': len(consensus) / max(1, len(self.structures)),
            'total_rounds': len(self.consensus_history),
        }


# ============================================================================
# 3. 多智能体SSFR
# ============================================================================

class MultiAgentSSFR:
    """
    多智能体SSFR协调器

    核心循环:
    1. Local Step:   每个智能体根据本地观测更新
    2. Consensus Step: 智能体交换结构，达成共同结构池
    3. Dissemination Step: 共同结构分发回每个智能体

    通信模式:
    - 全连接: 每个智能体与其他所有智能体通信
    - 邻居: 智能体只与邻居通信（通过邻接矩阵定义）
    """

    def __init__(self,
                 num_agents: int,
                 space_names: List[str],
                 width: int = 40,
                 height: int = 20,
                 communication_mode: str = 'full',
                 adjacency_matrix: Optional[np.ndarray] = None,
                 consensus_interval: int = 5):
        """
        初始化多智能体SSFR

        Args:
            num_agents: 智能体数量
            space_names: 空间类型列表
            width: 环境宽度
            height: 环境高度
            communication_mode: 通信模式 ('full', 'neighbor', 'sparse')
            adjacency_matrix: 邻接矩阵（用于邻居通信）
            consensus_interval: 一致性步骤间隔
        """
        self.num_agents = num_agents
        self.space_names = space_names
        self.width = width
        self.height = height
        self.communication_mode = communication_mode
        self.consensus_interval = consensus_interval

        # 初始化每个智能体的SSFR
        self.agents: List[SSFREnhanced] = [
            SSFREnhanced(width, height, space_names.copy())
            for _ in range(num_agents)
        ]

        # 共享结构池
        self.common_pool = SharedStructurePool()

        # 乘积流形
        agent_spaces = [agent.spaces for agent in self.agents]
        self.product_manifold = ProductManifold(agent_spaces)

        # 通信图
        if adjacency_matrix is None:
            # 默认全连接
            self.adjacency_matrix = np.ones((num_agents, num_agents)) - np.eye(num_agents)
        else:
            self.adjacency_matrix = adjacency_matrix

        # 时间步
        self.step_count = 0
        self.consensus_count = 0

        # 统计
        self.local_step_history: List[Dict] = []
        self.consensus_history: List[Dict] = []
        self.communication_stats = {
            'bytes_sent': 0,
            'messages_sent': 0,
            'consensus_rounds': 0,
        }

    def local_step(self,
                   agent_id: int,
                   observation: Dict[str, Any],
                   active_space: Optional[str] = None) -> List[StructureHypothesis]:
        """
        单步：个体感知

        Args:
            agent_id: 智能体ID
            observation: 本地观测
            active_space: 活跃空间名称

        Returns:
            生成的假设列表
        """
        agent = self.agents[agent_id]
        position = observation.get('position', (0, 0))

        # 执行感知
        hypotheses = agent.perceive(position, observation, active_space)

        # 记录
        self.local_step_history.append({
            'step': self.step_count,
            'agent_id': agent_id,
            'num_hypotheses': len(hypotheses),
            'active_space': active_space,
        })

        return hypotheses

    def consensus_step(self) -> Dict[str, Any]:
        """
        一致性步骤：个体结构 → 共同结构

        基于邻接矩阵进行分布式一致性更新

        Returns:
            一致性结果统计
        """
        start_time = time.time()

        # 收集每个智能体的结构
        agent_structures: Dict[int, List[StructureHypothesis]] = {}

        for agent_id in range(self.num_agents):
            # 获取该智能体的最佳结构
            best = self.agents[agent_id].get_best_structures(n=5)
            agent_structures[agent_id] = best

        # 更新共享池
        stats = self.common_pool.update_consensus(
            self.consensus_count, agent_structures
        )

        # 计算通信开销（模拟）
        bytes_sent = self._estimate_communication_cost(agent_structures)
        self.communication_stats['bytes_sent'] += bytes_sent
        self.communication_stats['messages_sent'] += self.num_agents
        self.communication_stats['consensus_rounds'] += 1

        elapsed = time.time() - start_time

        self.consensus_history.append({
            'round': self.consensus_count,
            'elapsed_ms': elapsed * 1000,
            'stats': stats,
            'bytes_sent': bytes_sent,
        })

        self.consensus_count += 1

        return {
            **stats,
            'elapsed_ms': elapsed * 1000,
            'bytes_sent': bytes_sent,
        }

    def _estimate_communication_cost(
        self,
        agent_structures: Dict[int, List[StructureHypothesis]]
    ) -> int:
        """估算通信开销（字节数）"""
        # 估算每个结构的大小
        bytes_per_structure = 200  # 假设每个结构约200字节

        total_structures = sum(len(s) for s in agent_structures.values())

        if self.communication_mode == 'full':
            # 全连接：每个智能体向所有其他智能体发送
            messages = self.num_agents * (self.num_agents - 1)
        elif self.communication_mode == 'neighbor':
            # 邻居：只向邻居发送
            messages = int(np.sum(self.adjacency_matrix))
        else:
            # 稀疏：假设10%的连接
            messages = max(1, self.num_agents // 10)

        return total_structures * bytes_per_structure * messages

    def dissemination_step(self) -> Dict[int, List[StructureHypothesis]]:
        """
        分发步骤：共同结构 → 个体结构

        将达成共识的结构分发给每个智能体

        Returns:
            {agent_id: [disseminated_structures]}
        """
        disseminated: Dict[int, List[StructureHypothesis]] = {
            i: [] for i in range(self.num_agents)
        }

        # 获取共识结构
        consensus = self.common_pool.get_consensus_structures(min_agents=2)

        for shared in consensus:
            # 将共识结构添加到每个智能体的池中
            for agent_id in range(self.num_agents):
                # 创建深拷贝
                local_copy = copy.deepcopy(shared.structure)
                local_copy.id = f"{local_copy.id}_a{agent_id}"
                local_copy.name = f"consensus_{shared.consensus_round}"

                # 添加到智能体的池
                self.agents[agent_id].structure_pool.add(local_copy)
                disseminated[agent_id].append(local_copy)

        return disseminated

    def step(self,
             agent_observations: Dict[int, Dict[str, Any]],
             agent_actives: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
        """
        执行完整的多智能体步骤

        Args:
            agent_observations: {agent_id: observation}
            agent_actives: {agent_id: active_space_name}

        Returns:
            步骤结果统计
        """
        agent_actives = agent_actives or {}

        # 1. Local Step: 每个智能体感知
        local_results = {}
        for agent_id, obs in agent_observations.items():
            active = agent_actives.get(agent_id)
            hyps = self.local_step(agent_id, obs, active)
            local_results[agent_id] = hyps

        # 2. 定期执行 Consensus + Dissemination
        consensus_result = None
        disseminated = None

        if self.step_count % self.consensus_interval == 0:
            # Consensus
            consensus_result = self.consensus_step()

            # Dissemination
            disseminated = self.dissemination_step()

        self.step_count += 1

        return {
            'step': self.step_count,
            'local_results': {
                aid: len(hyps) for aid, hyps in local_results.items()
            },
            'consensus': consensus_result,
            'disseminated': {
                aid: len(hyps) for aid, hyps in (disseminated or {}).items()
            },
        }

    def run_competition(self,
                        agent_id: int,
                        observation: Dict[str, Any],
                        actual: Dict[str, Any]) -> Optional[StructureHypothesis]:
        """
        在指定智能体上运行竞争

        Args:
            agent_id: 智能体ID
            observation: 观测
            actual: 实际结果

        Returns:
            胜者结构
        """
        return self.agents[agent_id].compete(observation, actual)

    def evolve_structures(self, agent_id: int) -> List[StructureHypothesis]:
        """在指定智能体上演化结构"""
        return self.agents[agent_id].evolve()

    def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        # 每个智能体的统计
        agent_stats = [
            agent.get_statistics()
            for agent in self.agents
        ]

        return {
            'step_count': self.step_count,
            'num_agents': self.num_agents,
            'consensus_count': self.consensus_count,
            'common_pool': self.common_pool.get_statistics(),
            'communication': self.communication_stats,
            'agent_stats': agent_stats,
            'product_manifold': self.product_manifold.get_statistics(),
        }

    def get_consensus_structures(self, min_agents: int = 2) -> List[SharedStructure]:
        """获取当前达成共识的结构"""
        return self.common_pool.get_consensus_structures(min_agents)


# ============================================================================
# 4. 效用函数
# ============================================================================

def create_multi_agent_ssfr(
    num_agents: int,
    space_names: List[str] = None,
    width: int = 40,
    height: int = 20,
    communication_mode: str = 'full',
    consensus_interval: int = 5
) -> MultiAgentSSFR:
    """
    创建多智能体SSFR的工厂函数

    Args:
        num_agents: 智能体数量
        space_names: 空间类型列表（默认：['ricci', 'fisher', 'wasserstein']）
        width: 环境宽度
        height: 环境高度
        communication_mode: 通信模式
        consensus_interval: 一致性间隔

    Returns:
        配置好的MultiAgentSSFR实例
    """
    space_names = space_names or ['ricci', 'fisher', 'wasserstein']

    return MultiAgentSSFR(
        num_agents=num_agents,
        space_names=space_names,
        width=width,
        height=height,
        communication_mode=communication_mode,
        consensus_interval=consensus_interval
    )


def compute_joint_path_cost(
    product_manifold: ProductManifold,
    joint_path: List[List[Tuple[int, int]]]
) -> float:
    """
    计算联合路径的总成本

    Args:
        product_manifold: 乘积流形
        joint_path: 联合路径 [[(x1,y1), ...], [(x2,y2), ...], ...]

    Returns:
        总成本
    """
    if not joint_path or len(joint_path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(joint_path) - 1):
        total += product_manifold.compute_distance(
            joint_path[i], joint_path[i+1]
        )

    return total


# ============================================================================
# 5. 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Agent SSFR Framework Demo")
    print("=" * 70)

    # 创建多智能体SSFR
    ma_ssfr = create_multi_agent_ssfr(
        num_agents=3,
        space_names=['ricci', 'fisher'],
        width=20,
        height=20,
        communication_mode='full',
        consensus_interval=3
    )

    print(f"\nInitialized {ma_ssfr.num_agents} agents")
    print(f"Spaces: {ma_ssfr.space_names}")
    print(f"Consensus interval: {ma_ssfr.consensus_interval}")

    # 模拟多智能体探索
    print("\n--- Simulating Multi-Agent Exploration ---")

    np.random.seed(42)

    for step in range(10):
        # 生成每个智能体的观测
        agent_obs = {}
        for i in range(ma_ssfr.num_agents):
            pos = (
                np.random.randint(0, ma_ssfr.width),
                np.random.randint(0, ma_ssfr.height)
            )
            agent_obs[i] = {
                'position': pos,
                'goal_position': (ma_ssfr.width - 1, ma_ssfr.height - 1),
                'obstacles': set(),
            }

        # 执行步骤
        result = ma_ssfr.step(agent_obs)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Local hypotheses: {result['local_results']}")
            if result['consensus']:
                print(f"  Consensus structures: {result['consensus']['consensus_structures']}")
                print(f"  Bytes sent: {result['consensus']['bytes_sent']}")

    # 最终统计
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)

    stats = ma_ssfr.get_global_statistics()
    print(f"Total steps: {stats['step_count']}")
    print(f"Consensus rounds: {stats['consensus_count']}")
    print(f"Common pool structures: {stats['common_pool']['total_structures']}")
    print(f"Consensus rate: {stats['common_pool']['consensus_rate']:.1%}")
    print(f"Total communication: {stats['communication']['bytes_sent']} bytes")
    print(f"Messages sent: {stats['communication']['messages_sent']}")

    # 乘积流形测试
    print("\n" + "=" * 70)
    print("Product Manifold Test")
    print("=" * 70)

    state1 = [(0, 0), (1, 1), (2, 2)]
    state2 = [(5, 5), (6, 6), (7, 7)]

    dist = ma_ssfr.product_manifold.compute_distance(state1, state2)
    print(f"\nJoint distance between {state1} and {state2}:")
    print(f"  d_joint = {dist:.3f}")

    heuristic = ma_ssfr.product_manifold.get_heuristic(state1, state2)
    print(f"  h_joint = {heuristic:.3f}")

    print("\n" + "=" * 70)
    print("Key Concepts:")
    print("  - Product Manifold: M = M₁ × M₂ × ... × M_N")
    print("  - Joint distance: d_joint = sqrt(Σ w_i * d_i²)")
    print("  - Consensus: Structures agreed upon by multiple agents")
    print("  - Local step: Individual perception")
    print("  - Consensus step: Sharing and merging structures")
    print("  - Dissemination step: Broadcasting consensus structures")
    print("=" * 70)
