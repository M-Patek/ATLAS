"""
ATLAS Research: Research Tools
研究工具

支持严格的A/B测试和消融研究
"""

from .ab_testing import (
    ABTestExperiment,
    AblationStudy,
    TestScenario,
    create_standard_scenarios,
    StatisticalTest,
)

from .multi_agent_ssfr import (
    ProductManifold,
    MultiAgentSSFR,
    SharedStructurePool,
    SharedStructure,
    create_multi_agent_ssfr,
    compute_joint_path_cost,
)

from .consensus import (
    ConsensusProtocol,
    WeightedVotingConsensus,
    RaftConsensus,
    GossipConsensus,
    FederatedAveragingConsensus,
    create_consensus_protocol,
    consensus_update,
)

from .neural_gradient import (
    KroneckerFactors,
    LayerWiseNaturalGradient,
    NeuralNaturalGradient,
    AmortizedNaturalGradient,
    compute_fisher_information_matrix,
    natural_gradient_step,
)

__all__ = [
    # A/B Testing
    "ABTestExperiment",
    "AblationStudy",
    "TestScenario",
    "create_standard_scenarios",
    "StatisticalTest",
    # Multi-Agent SSFR
    "ProductManifold",
    "MultiAgentSSFR",
    "SharedStructurePool",
    "SharedStructure",
    "create_multi_agent_ssfr",
    "compute_joint_path_cost",
    # Consensus Protocols
    "ConsensusProtocol",
    "WeightedVotingConsensus",
    "RaftConsensus",
    "GossipConsensus",
    "FederatedAveragingConsensus",
    "create_consensus_protocol",
    "consensus_update",
    # Neural Natural Gradient
    "KroneckerFactors",
    "LayerWiseNaturalGradient",
    "NeuralNaturalGradient",
    "AmortizedNaturalGradient",
    "compute_fisher_information_matrix",
    "natural_gradient_step",
]
