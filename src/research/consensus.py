"""
Consensus Protocol for Multi-Agent SSFR
多智能体SSFR一致性协议

实现分布式一致性算法：
1. Raft简化版：领导者选举和日志复制
2. Weighted Voting：基于智能体权重的投票
3. Gossip Protocol：流言协议（异步通信）
4. Federated Averaging：联邦平均
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import time


# ============================================================================
# 1. 基础数据结构
# ============================================================================

@dataclass
class ConsensusMessage:
    """一致性协议消息"""
    sender_id: int
    message_type: str  # 'proposal', 'vote', 'commit', 'heartbeat'
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    term: int = 0  # Raft term


@dataclass
class AgentState:
    """智能体在一致性协议中的状态"""
    agent_id: int
    weight: float = 1.0  # 投票权重
    is_leader: bool = False
    is_active: bool = True
    last_seen: float = field(default_factory=time.time)
    structures_hash: str = ""  # 本地结构的哈希（用于比较）


# ============================================================================
# 2. 基础一致性协议
# ============================================================================

class ConsensusProtocol:
    """
    基础一致性协议

    抽象基类，定义一致性协议接口
    """

    def __init__(self, num_agents: int, agent_weights: Optional[List[float]] = None):
        self.num_agents = num_agents
        self.agent_weights = agent_weights or [1.0] * num_agents
        self.agent_states: Dict[int, AgentState] = {
            i: AgentState(agent_id=i, weight=self.agent_weights[i])
            for i in range(num_agents)
        }
        self.message_log: List[ConsensusMessage] = []
        self.round = 0

    def propose(self,
                agent_id: int,
                structures: List[Any]) -> ConsensusMessage:
        """提议结构"""
        raise NotImplementedError

    def vote(self,
             agent_id: int,
             proposal: ConsensusMessage) -> ConsensusMessage:
        """对提议投票"""
        raise NotImplementedError

    def commit(self,
               proposals: List[ConsensusMessage],
               votes: List[ConsensusMessage]) -> Tuple[List[Any], Dict[str, Any]]:
        """提交达成共识的结构"""
        raise NotImplementedError

    def run_consensus_round(self,
                            agent_structures: Dict[int, List[Any]]) -> Tuple[List[Any], Dict[str, Any]]:
        """运行一轮一致性协议"""
        raise NotImplementedError


# ============================================================================
# 3. Weighted Voting Consensus
# ============================================================================

class WeightedVotingConsensus(ConsensusProtocol):
    """
    加权投票一致性

    算法:
    1. 每个智能体提出自己的最佳结构
    2. 所有智能体对所有提议投票（基于相似度）
    3. 得票权重超过阈值的结构被接受

    适用场景:
    - 结构相似度可计算
    - 智能体有不同可信度权重
    """

    def __init__(self,
                 num_agents: int,
                 agent_weights: Optional[List[float]] = None,
                 threshold: float = 0.5,
                 similarity_fn: Optional[Callable[[Any, Any], float]] = None):
        super().__init__(num_agents, agent_weights)
        self.threshold = threshold  # 通过阈值（权重比例）
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, s1: Any, s2: Any) -> float:
        """默认相似度函数"""
        # 基于ID的简单比较
        if hasattr(s1, 'id') and hasattr(s2, 'id'):
            return 1.0 if s1.id == s2.id else 0.0
        return 1.0 if s1 == s2 else 0.0

    def propose(self,
                agent_id: int,
                structures: List[Any]) -> ConsensusMessage:
        """提议结构"""
        return ConsensusMessage(
            sender_id=agent_id,
            message_type='proposal',
            payload={'structures': structures},
            term=self.round
        )

    def vote(self,
             agent_id: int,
             proposal: ConsensusMessage) -> ConsensusMessage:
        """
        对提议投票

        智能体比较提议的结构与自己的结构，
        对相似的结构投赞成票
        """
        # 这里简化处理：总是投赞成票
        # 实际实现中应该基于相似度计算
        return ConsensusMessage(
            sender_id=agent_id,
            message_type='vote',
            payload={
                'proposal_sender': proposal.sender_id,
                'vote': 'yes',
                'weight': self.agent_weights[agent_id]
            },
            term=self.round
        )

    def commit(self,
               proposals: List[ConsensusMessage],
               votes: List[ConsensusMessage]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        提交达成共识的结构

        统计每个结构的得票权重，超过阈值的被接受
        """
        # 收集所有提议的结构
        all_structures: List[Tuple[Any, int]] = []  # (structure, proposer_id)
        for prop in proposals:
            structures = prop.payload.get('structures', [])
            for s in structures:
                all_structures.append((s, prop.sender_id))

        # 统计每个结构的得票
        structure_votes: Dict[str, Tuple[Any, float]] = {}  # id -> (structure, total_weight)

        for vote in votes:
            if vote.payload.get('vote') == 'yes':
                voter_weight = vote.payload.get('weight', 1.0)
                proposal_sender = vote.payload.get('proposal_sender')

                # 找到该投票者对应的提议结构
                for struct, proposer in all_structures:
                    if proposer == proposal_sender:
                        struct_id = getattr(struct, 'id', str(struct))
                        if struct_id in structure_votes:
                            structure_votes[struct_id] = (
                                struct,
                                structure_votes[struct_id][1] + voter_weight
                            )
                        else:
                            structure_votes[struct_id] = (struct, voter_weight)

        # 计算总权重
        total_weight = sum(self.agent_weights)

        # 选择超过阈值的结构
        consensus_structures = []
        for struct_id, (struct, weight) in structure_votes.items():
            if weight / total_weight >= self.threshold:
                consensus_structures.append(struct)

        stats = {
            'proposed': len(all_structures),
            'votes_cast': len(votes),
            'consensus_reached': len(consensus_structures),
            'threshold': self.threshold,
        }

        return consensus_structures, stats

    def run_consensus_round(self,
                            agent_structures: Dict[int, List[Any]]) -> Tuple[List[Any], Dict[str, Any]]:
        """运行一轮加权投票一致性"""
        self.round += 1

        # 1. 收集提议
        proposals = []
        for agent_id, structures in agent_structures.items():
            prop = self.propose(agent_id, structures)
            proposals.append(prop)

        # 2. 收集投票
        votes = []
        for agent_id in agent_structures.keys():
            for prop in proposals:
                vote = self.vote(agent_id, prop)
                votes.append(vote)

        # 3. 提交
        consensus, stats = self.commit(proposals, votes)

        return consensus, stats


# ============================================================================
# 4. Raft简化版
# ============================================================================

class RaftConsensus(ConsensusProtocol):
    """
    Raft一致性算法简化版

    角色:
    - Leader: 领导者，负责接收客户端请求并复制日志
    - Follower: 跟随者，接收并存储领导者复制的日志
    - Candidate: 候选者，发起选举

    简化:
    - 不处理日志持久化
    - 不处理成员变更
    - 固定领导者（选举只进行一次）
    """

    def __init__(self,
                 num_agents: int,
                 agent_weights: Optional[List[float]] = None,
                 election_timeout: float = 1.0):
        super().__init__(num_agents, agent_weights)
        self.election_timeout = election_timeout
        self.leader_id: Optional[int] = None
        self.term = 0
        self.voted_for: Optional[int] = None

        # 日志条目: (term, structures)
        self.log: List[Tuple[int, List[Any]]] = []

    def elect_leader(self) -> int:
        """
        选举领导者

        简化版：选择权重最高的智能体作为领导者
        """
        max_weight = max(self.agent_weights)
        leader = self.agent_weights.index(max_weight)

        self.leader_id = leader
        self.agent_states[leader].is_leader = True

        for i, state in self.agent_states.items():
            state.is_leader = (i == leader)

        return leader

    def propose(self,
                agent_id: int,
                structures: List[Any]) -> ConsensusMessage:
        """领导者提议结构（日志复制）"""
        if agent_id != self.leader_id:
            # 非领导者转发给领导者
            return ConsensusMessage(
                sender_id=agent_id,
                message_type='forward',
                payload={'structures': structures, 'target': self.leader_id},
                term=self.term
            )

        self.term += 1
        self.log.append((self.term, structures))

        return ConsensusMessage(
            sender_id=agent_id,
            message_type='append_entries',
            payload={'term': self.term, 'structures': structures, 'log_index': len(self.log) - 1},
            term=self.term
        )

    def vote(self,
             agent_id: int,
             proposal: ConsensusMessage) -> ConsensusMessage:
        """对领导者提议投票（日志复制确认）"""
        if proposal.message_type == 'append_entries':
            # 检查term
            if proposal.term >= self.term:
                self.term = proposal.term
                return ConsensusMessage(
                    sender_id=agent_id,
                    message_type='ack',
                    payload={'log_index': proposal.payload.get('log_index'), 'success': True},
                    term=self.term
                )

        return ConsensusMessage(
            sender_id=agent_id,
            message_type='nack',
            payload={'reason': 'term_mismatch'},
            term=self.term
        )

    def commit(self,
               proposals: List[ConsensusMessage],
               votes: List[ConsensusMessage]) -> Tuple[List[Any], Dict[str, Any]]:
        """提交日志条目"""
        # 统计ack
        acks = [v for v in votes if v.message_type == 'ack']

        # 如果超过半数确认，则提交
        if len(acks) > self.num_agents / 2:
            # 获取最新的日志条目
            if self.log:
                latest_term, latest_structures = self.log[-1]
                return latest_structures, {
                    'committed': True,
                    'term': latest_term,
                    'acks': len(acks),
                }

        return [], {
            'committed': False,
            'acks': len(acks),
            'needed': self.num_agents / 2,
        }

    def run_consensus_round(self,
                            agent_structures: Dict[int, List[Any]]) -> Tuple[List[Any], Dict[str, Any]]:
        """运行一轮Raft一致性"""
        # 如果没有领导者，先选举
        if self.leader_id is None:
            self.elect_leader()

        # 领导者收集所有结构
        leader_structures = []
        for agent_id, structures in agent_structures.items():
            leader_structures.extend(structures)

        # 领导者提议
        proposal = self.propose(self.leader_id, leader_structures)

        # 所有智能体投票
        votes = []
        for agent_id in agent_structures.keys():
            vote = self.vote(agent_id, proposal)
            votes.append(vote)

        # 提交
        consensus, stats = self.commit([proposal], votes)

        return consensus, stats


# ============================================================================
# 5. Gossip Protocol
# ============================================================================

class GossipConsensus(ConsensusProtocol):
    """
    流言协议（异步一致性）

    算法:
    1. 每个智能体定期随机选择k个邻居交换结构
    2. 接收方合并接收到的结构
    3. 经过多轮传播，结构在全网收敛

    适用场景:
    - 大规模网络
    - 容忍延迟
    - 不需要强一致性
    """

    def __init__(self,
                 num_agents: int,
                 agent_weights: Optional[List[float]] = None,
                 fanout: int = 2,
                 rounds: int = 5):
        super().__init__(num_agents, agent_weights)
        self.fanout = fanout  # 每轮传播的邻居数
        self.rounds = rounds  # 传播轮数

        # 每个智能体的本地结构集合
        self.local_structures: Dict[int, Set[str]] = {
            i: set() for i in range(num_agents)
        }

    def propose(self,
                agent_id: int,
                structures: List[Any]) -> List[ConsensusMessage]:
        """
        流言传播

        随机选择fanout个邻居发送结构
        """
        messages = []

        # 更新本地结构
        for s in structures:
            struct_id = getattr(s, 'id', str(s))
            self.local_structures[agent_id].add(struct_id)

        # 随机选择邻居
        neighbors = list(range(self.num_agents))
        neighbors.remove(agent_id)
        selected = np.random.choice(
            neighbors,
            size=min(self.fanout, len(neighbors)),
            replace=False
        )

        for neighbor in selected:
            messages.append(ConsensusMessage(
                sender_id=agent_id,
                message_type='gossip',
                payload={
                    'structures': structures,
                    'recipient': neighbor,
                },
                term=self.round
            ))

        return messages

    def vote(self, *args, **kwargs) -> ConsensusMessage:
        """Gossip不需要投票"""
        pass

    def commit(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """Gossip不需要显式提交"""
        pass

    def run_consensus_round(self,
                            agent_structures: Dict[int, List[Any]]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        运行多轮流言传播

        返回在所有智能体中都存在的结构
        """
        # 多轮传播
        for round_num in range(self.rounds):
            self.round = round_num

            # 每个智能体发送流言
            all_messages = []
            for agent_id, structures in agent_structures.items():
                messages = self.propose(agent_id, structures)
                all_messages.extend(messages)

            # 处理接收到的流言（合并结构）
            for msg in all_messages:
                recipient = msg.payload.get('recipient')
                received_structures = msg.payload.get('structures', [])

                for s in received_structures:
                    struct_id = getattr(s, 'id', str(s))
                    self.local_structures[recipient].add(struct_id)

                    # 添加到该智能体的结构列表
                    if recipient in agent_structures:
                        # 检查是否已存在
                        existing_ids = {
                            getattr(x, 'id', str(x))
                            for x in agent_structures[recipient]
                        }
                        if struct_id not in existing_ids:
                            agent_structures[recipient].append(s)

        # 找出在所有智能体中都存在的结构（收敛的结构）
        if self.local_structures:
            common_ids = set.intersection(*self.local_structures.values())

            # 从原始结构中找到对应的完整结构
            consensus = []
            for agent_structs in agent_structures.values():
                for s in agent_structs:
                    struct_id = getattr(s, 'id', str(s))
                    if struct_id in common_ids:
                        consensus.append(s)
                        common_ids.remove(struct_id)

            return consensus, {
                'rounds': self.rounds,
                'fanout': self.fanout,
                'converged_structures': len(consensus),
            }

        return [], {'rounds': self.rounds, 'converged_structures': 0}


# ============================================================================
# 6. Federated Averaging
# ============================================================================

class FederatedAveragingConsensus(ConsensusProtocol):
    """
    联邦平均一致性

    算法:
    1. 中央服务器收集各智能体的结构参数
    2. 按权重平均参数
    3. 将平均后的参数分发回各智能体

    适用场景:
    - 结构有连续参数可平均
    - 隐私保护（不共享原始数据）
    """

    def __init__(self,
                 num_agents: int,
                 agent_weights: Optional[List[float]] = None,
                 server_id: int = 0):
        super().__init__(num_agents, agent_weights)
        self.server_id = server_id

    def extract_parameters(self, structure: Any) -> np.ndarray:
        """从结构中提取可平均的参数"""
        if hasattr(structure, 'representations'):
            # 从SSFR结构中提取参数
            params = []
            for rep in structure.representations.values():
                if isinstance(rep, dict) and 'params' in rep:
                    params.extend(rep['params'].values())
            return np.array(params) if params else np.array([0.0])
        return np.array([0.0])

    def set_parameters(self, structure: Any, params: np.ndarray) -> Any:
        """将参数设置回结构"""
        # 创建副本并更新参数
        new_structure = copy.deepcopy(structure)
        if hasattr(new_structure, 'representations'):
            idx = 0
            for rep in new_structure.representations.values():
                if isinstance(rep, dict) and 'params' in rep:
                    for key in rep['params']:
                        if idx < len(params):
                            rep['params'][key] = params[idx]
                            idx += 1
        return new_structure

    def run_consensus_round(self,
                            agent_structures: Dict[int, List[Any]]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        运行一轮联邦平均

        对每个空间类型的结构分别平均
        """
        # 按空间类型分组
        by_space_type: Dict[str, List[Tuple[Any, int]]] = defaultdict(list)

        for agent_id, structures in agent_structures.items():
            for s in structures:
                space_type = getattr(s, 'context', {}).get('space_type', 'unknown')
                by_space_type[space_type].append((s, agent_id))

        # 对每个类型进行联邦平均
        averaged_structures = []

        for space_type, struct_list in by_space_type.items():
            if not struct_list:
                continue

            # 提取所有参数
            all_params = []
            weights = []

            for s, agent_id in struct_list:
                params = self.extract_parameters(s)
                all_params.append(params)
                weights.append(self.agent_weights[agent_id])

            # 对齐参数维度
            max_dim = max(len(p) for p in all_params)
            aligned_params = []
            for p in all_params:
                if len(p) < max_dim:
                    p = np.pad(p, (0, max_dim - len(p)))
                aligned_params.append(p[:max_dim])

            # 加权平均
            weights = np.array(weights)
            weights = weights / weights.sum()

            stacked = np.stack(aligned_params)
            averaged = np.average(stacked, axis=0, weights=weights)

            # 创建新的平均结构（基于第一个结构）
            if struct_list:
                base_structure = copy.deepcopy(struct_list[0][0])
                new_structure = self.set_parameters(base_structure, averaged)
                new_structure.id = f"fed_avg_{space_type}_{self.round}"
                new_structure.name = f"federated_{space_type}"
                averaged_structures.append(new_structure)

        stats = {
            'space_types': list(by_space_type.keys()),
            'averaged_structures': len(averaged_structures),
            'total_source_structures': sum(len(s) for s in agent_structures.values()),
        }

        self.round += 1
        return averaged_structures, stats


# ============================================================================
# 7. 一致性协议工厂
# ============================================================================

def create_consensus_protocol(
    protocol_type: str,
    num_agents: int,
    agent_weights: Optional[List[float]] = None,
    **kwargs
) -> ConsensusProtocol:
    """
    创建一致性协议的工厂函数

    Args:
        protocol_type: 'weighted_voting', 'raft', 'gossip', 'federated'
        num_agents: 智能体数量
        agent_weights: 智能体权重
        **kwargs: 协议特定参数

    Returns:
        配置好的一致性协议实例
    """
    if protocol_type == 'weighted_voting':
        return WeightedVotingConsensus(
            num_agents, agent_weights,
            threshold=kwargs.get('threshold', 0.5),
            similarity_fn=kwargs.get('similarity_fn')
        )
    elif protocol_type == 'raft':
        return RaftConsensus(
            num_agents, agent_weights,
            election_timeout=kwargs.get('election_timeout', 1.0)
        )
    elif protocol_type == 'gossip':
        return GossipConsensus(
            num_agents, agent_weights,
            fanout=kwargs.get('fanout', 2),
            rounds=kwargs.get('rounds', 5)
        )
    elif protocol_type == 'federated':
        return FederatedAveragingConsensus(
            num_agents, agent_weights,
            server_id=kwargs.get('server_id', 0)
        )
    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")


def consensus_update(
    agent_structures: Dict[int, List[Any]],
    adjacency_matrix: np.ndarray,
    protocol_type: str = 'weighted_voting',
    agent_weights: Optional[List[float]] = None,
    **kwargs
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    基于邻接矩阵的结构一致性更新

    这是主要的对外接口函数。

    Args:
        agent_structures: {agent_id: [structures]}
        adjacency_matrix: 邻接矩阵（定义通信拓扑）
        protocol_type: 一致性协议类型
        agent_weights: 智能体权重
        **kwargs: 额外参数

    Returns:
        (consensus_structures, statistics)
    """
    num_agents = len(agent_structures)

    # 根据邻接矩阵调整智能体权重（邻居多的权重高）
    if agent_weights is None:
        agent_weights = [1.0] * num_agents

    # 创建协议
    protocol = create_consensus_protocol(
        protocol_type, num_agents, agent_weights, **kwargs
    )

    # 运行一致性
    consensus, stats = protocol.run_consensus_round(agent_structures)

    # 添加拓扑信息
    stats['topology'] = 'full' if np.all(adjacency_matrix == 1 - np.eye(num_agents)) else 'custom'
    stats['protocol'] = protocol_type

    return consensus, stats


# ============================================================================
# 8. 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Consensus Protocol for Multi-Agent SSFR")
    print("=" * 70)

    # 模拟结构
    class MockStructure:
        def __init__(self, sid, space_type='ricci'):
            self.id = sid
            self.context = {'space_type': space_type}
            self.representations = {
                space_type: {'params': {'alpha': np.random.random(), 'beta': np.random.random()}}
            }

    num_agents = 5

    # 生成模拟结构
    agent_structures = {
        i: [MockStructure(f"agent{i}_struct{j}", 'ricci' if j % 2 == 0 else 'fisher')
            for j in range(3)]
        for i in range(num_agents)
    }

    print(f"\nGenerated structures for {num_agents} agents")
    for aid, structs in agent_structures.items():
        print(f"  Agent {aid}: {len(structs)} structures")

    # 测试各种一致性协议
    protocols = ['weighted_voting', 'raft', 'gossip', 'federated']

    for protocol_type in protocols:
        print(f"\n{'=' * 70}")
        print(f"Testing: {protocol_type.upper()}")
        print("=" * 70)

        # 创建邻接矩阵（全连接）
        adjacency = np.ones((num_agents, num_agents)) - np.eye(num_agents)

        # 运行一致性
        consensus, stats = consensus_update(
            copy.deepcopy(agent_structures),
            adjacency,
            protocol_type=protocol_type,
            threshold=0.3 if protocol_type == 'weighted_voting' else None,
            rounds=3 if protocol_type == 'gossip' else None
        )

        print(f"\nResults:")
        print(f"  Consensus structures: {len(consensus)}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # 测试不同拓扑
    print("\n" + "=" * 70)
    print("Testing Different Topologies")
    print("=" * 70)

    topologies = {
        'full': np.ones((num_agents, num_agents)) - np.eye(num_agents),
        'ring': np.eye(num_agents, k=1) + np.eye(num_agents, k=-1) +
                np.eye(num_agents, k=num_agents-1) + np.eye(num_agents, k=-(num_agents-1)),
        'star': np.array([[0 if i != 0 and j != 0 else 1
                          for j in range(num_agents)] for i in range(num_agents)]) - np.eye(num_agents),
    }

    for topo_name, adjacency in topologies.items():
        print(f"\nTopology: {topo_name}")

        consensus, stats = consensus_update(
            copy.deepcopy(agent_structures),
            adjacency,
            protocol_type='weighted_voting',
            threshold=0.3
        )

        print(f"  Consensus structures: {len(consensus)}")

    print("\n" + "=" * 70)
    print("Key Concepts:")
    print("  - Weighted Voting: Vote based on structure similarity")
    print("  - Raft: Leader-based consensus with log replication")
    print("  - Gossip: Probabilistic propagation for large networks")
    print("  - Federated: Parameter averaging for continuous structures")
    print("=" * 70)
