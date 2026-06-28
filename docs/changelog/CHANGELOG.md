---
id: changelog
title: ATLAS Changelog
status: stable
last_validated: 2026-06-27
tags: [changelog, history]
---

# ATLAS Changelog

All notable changes to ATLAS are documented in this file.

## 2026-06-28 — Round 2: Remove Obsolete Discrete Kitchen & Duplicate Tests

- **Type**: T5 (Cross-cutting Change)
- **Goal**: Remove obsolete discrete kitchen integration and duplicate test files
- **Done**:
  - **Deleted obsolete discrete kitchen files**:
    - `experiments/demos/demo_ssfr_kitchen.py` — 被 `test_continuous_ssfr.py` 中的 `ContinuousPhysicalSSFR` 替代
    - `experiments/tests/test_ssfr_kitchen.py` — 离散网格适配器已过时
  - **Deleted duplicate test files**:
    - `test_structure_reuse.py` → 功能与 `test_structure_reuse_v2.py` 重叠
    - `test_reuse_rate.py` → 与 `test_structure_reuse_v2.py` 重叠
  - **Deleted obsolete analysis files**:
    - `test_first_update.py` — 一次性初始化开销测试
    - `test_spike_analysis.py` — 一次性性能尖峰分析
    - `test_astar_limits.py` — 极限测试，已被压力测试覆盖
- **Files**: `experiments/*`
- **Validation**: V3 — 文件数从148→141，实验目录结构清晰
- **Left for next time**: 离散空间 (euclidean/ricci/fisher等) 的逐步迁移

---

## 2026-06-28 — Deep Repository Cleanup & Reorganization

- **Type**: T5 (Cross-cutting Change)
- **Goal**: Deep clean and reorganize repository structure
- **Done**:
  - **Deleted `__pycache__`**: 11 directories removed
  - **Deleted unused modules**:
    - `src/atlas/cognitive_arch/` (2314 lines, zero references)
    - `src/atlas/predictive_system/` (3477 lines, zero references)
  - **Cleaned imports**: Removed unused `exploration`, `navigation`, `integration` imports from `spaces/__init__.py` and `__init__.py`
  - **Deleted empty files**: `benchmark_output.txt`, `benchmark_result.txt`
  - **Reorganized `experiments/`**:
    - `tests/` — 22 test files
    - `demos/` — 6 demo files
    - `benchmarks/` — 3 benchmark files
    - `research/` — 2 research files
  - **Deleted duplicate files**:
    - `test_ultra_complex_benchmark.py` → merged into `test_complex_benchmark.py`
    - `test_extreme_stress.py` → merged into `test_complex_benchmark.py`
    - `test_pool_pressure.py` → merged into `test_pressure_benchmark.py`
    - `test_realistic_stress.py` → merged into `test_pressure_benchmark.py`
  - **Created `tests/`**: Proper pytest structure with `test_core.py` and `test_spaces.py`
  - **Updated `.gitignore`**: Added benchmark output files
- **Files**: `src/atlas/__init__.py`, `src/atlas/spaces/__init__.py`, `experiments/*`, `tests/*`
- **Validation**: V3 — File count reduced from 216 to 148, all imports verified
- **Left for next time**: Remove remaining unused `exploration/`, `navigation/`, `integration/` modules if confirmed unused

---

## 2026-06-28 — Continuous SSFR: Remove Discrete Grid

- **Type**: T5 (Architecture Refactor)
- **Goal**: Refactor SSFR to use continuous coordinates instead of discrete grid
- **Done**:
  - **ContinuousField**: Sparse sampling + kNN interpolation
    - Replaces numpy 2D array with sparse sample points
    - Spatial index for fast nearest-neighbor queries
    - kNN inverse-distance-weighted interpolation
    - LRU cache for repeated queries
  - **ContinuousCognitiveSpace**: Base class for continuous spaces
    - Position: Tuple[int, int] → Tuple[float, float]
    - Neighbors: 4/8-connected → 16-direction continuous sampling
    - Distance: grid path integral → continuous path integral
    - No fixed grid boundaries
  - **Continuous Spaces Implemented**:
    - ContinuousEuclideanSpace: baseline
    - ContinuousRicciSpace: curvature, uncertainty, familiarity fields
    - ContinuousFisherSpace: belief, confidence fields
    - ContinuousWassersteinSpace: cost, mass fields
  - **ContinuousSSFR**: SSFR core without grid
    - StructurePool with continuous positions
    - Perceive/Compete/Evolve with float coordinates
    - Structure reuse based on Euclidean distance
  - **Kitchen Integration**: ContinuousPhysicalSSFR
    - Direct physical position encoding (no grid conversion)
    - Continuous space validity computation
    - Task planning with continuous navigation
- **Files**:
  - `src/atlas/spaces/continuous.py` — Continuous spaces
  - `src/atlas/spaces/continuous_ssfr.py` — Continuous SSFR core
  - `experiments/test_continuous_ssfr.py` — 4 integration tests
  - `docs/continuous_ssfr.md` — Documentation
- **Validation**: V3 — 4/4 tests pass, continuous field query works, kitchen integration works
- **Next**: Performance optimization (R-tree, KD-tree), GPU acceleration

---

## 2026-06-27 — SSFR-Information Geometry: Corrected Relationship

- **Type**: T5 (Theoretical Correction)
- **Goal**: Correct the SSFR-Information Geometry relationship from "equivalence" to "subset"
- **Done**:
  - **Identified flaw**: Previous "proof" was a homomorphism, not an isomorphism
  - **Missing elements**: No invertible mapping, no measure consistency, ignored compression constraints
  - **Corrected relationship**: SSFR ⊂ Information Geometry (subset, not equivalence)
  - **4 Theorems**:
    1. SSFR is a subset of Information Geometry
    2. Information Geometry = limit of SSFR as lambda -> 0
    3. Constrained optimality: theta*_SSFR = argmax_{C(theta)<=B} L(theta)
    4. Value = Rate-Distortion optimization
  - **Key insight**: "Compression is intelligence" is a constraint ON information geometry, not a consequence
- **Files**:
  - `src/atlas/research/ssfr_information_geometry_v2.py` — Corrected proof
  - `docs/research/ssfr_information_geometry_v2.md` — Corrected theory document
- **Validation**: V3 — Numerical demonstration of compression constraint effects
- **Next**: Continuous extension, hierarchical structures, multi-agent

---

## 2026-06-27 — SSFR Continuous Extension & Hierarchical Structures

- **Type**: T5 (Theoretical Extension)
- **Goal**: Extend SSFR to continuous spaces and hierarchical structures
- **Done**:
  - **Continuous SSFR**: Infinite-dimensional statistical manifold
    - RKHS (Reproducing Kernel Hilbert Space) for function representation
    - Support points = finite representation of infinite-dimensional functions
    - Fisher information operator (infinite-dimensional version)
    - Value formula: V(S) = I(f) / C(f) (same as discrete)
  - **Hierarchical SSFR**: Fiber bundle and meta-patterns
    - 3-level hierarchy: raw patterns -> meta-patterns -> meta-meta-patterns
    - Fiber bundle: E = (Base, Fiber, Projection)
    - Cross-level similarity computation
    - Hierarchical information geometry (submanifolds at each level)
  - **Unified framework**: Discrete ⊂ Continuous ⊂ Hierarchical
- **Files**:
  - `src/atlas/research/ssfr_continuous.py` — Continuous extension
  - `src/atlas/research/ssfr_hierarchical.py` — Hierarchical structures
  - `docs/research/ssfr_continuous_hierarchical.md` — Theory document
- **Validation**: V3 — Numerical demonstrations of both extensions
- **Next**: Multi-agent (product manifold), deep learning integration

---

## 2026-06-27 — A/B Test Validation: Continuous & Hierarchical SSFR

- **Type**: T5 (Experimental Validation)
- **Goal**: Validate continuous and hierarchical SSFR with rigorous A/B testing
- **Done**:
  - **Continuous SSFR (GP version)**:
    - Gaussian Process regression for function approximation
    - Automatic bandwidth selection
    - Predictive uncertainty estimation
    - Results vs Discrete SSFR (polynomial fit):
      - Sine: +93.8% MSE improvement
      - Exponential: +37.0% MSE improvement
      - Piecewise: +39.0% MSE improvement
      - Noisy sine: +72.8% MSE improvement
  - **Hierarchical SSFR**:
    - 3-level hierarchy demonstrated
    - Meta-pattern stability > raw pattern stability (0.3175 vs 0.0693)
    - Cross-level similarity computation working
    - Fiber bundle structure validated
  - **Key finding**: Continuous SSFR (GP) outperforms discrete for non-linear patterns
- **Files**:
  - `src/atlas/research/ssfr_continuous_gp.py` — Gaussian Process SSFR
  - `experiments/ab_test_ssfr_extensions.py` — A/B test validation
- **Validation**: V3 — Numerical experiments across 5 test functions
- **Next**: Multi-agent (product manifold), deep learning integration

---

## 2026-06-27 — A/B Testing Research Framework ✓

- **Type**: T5 (Research Infrastructure)
- **Goal**: Build rigorous A/B testing framework for comparing cognitive spaces
- **Done**:
  - **Standardized Test Scenarios**: 6 scenarios (open field, narrow gap, maze, dense obstacles, dynamic, long distance)
  - **ABTestExperiment**: Automated experiment runner with configurable trials
  - **Statistical Testing**: t-test, ANOVA, Cohen's d effect size
  - **AblationStudy**: Component removal to measure contribution
  - **Report Generation**: Markdown and LaTeX output for papers
  - **Key Metrics**: success rate, path efficiency, planning time, path smoothness
- **Files**:
  - `src/atlas/research/ab_testing.py` — Complete research framework
  - `experiments/ab_test_quick.py` — Quick validation
  - `experiments/ab_test_spaces.py` — Full research script
- **Validation**: V3 — Successfully compared 3+ spaces with statistical tests
- **Next**: Run large-scale A/B experiments (30+ trials) to generate paper results

---

## 2026-06-27 — Learning Integration (Phase D) ✓

- **Type**: T4 (Framework Enhancement)
- **Goal**: Add learning capabilities to optimize and select cognitive spaces
- **Done**:
  - **BayesianOptimizer**: Gaussian Process-based parameter optimization
    - EI/UCB acquisition functions
    - Space parameter optimization (SpaceOptimizer)
    - Multi-objective Pareto optimization support
  - **MetaLearner**: Learning to select space type for tasks
    - TaskEmbedding: Encode task descriptions to vectors
    - SpaceSelectionPolicy: Learnable policy for space selection
    - Online learning from performance feedback
  - **NeuralSpace**: End-to-end learned spatial representations
    - SpatialEncoder: MLP encoder for observations
    - MetricNetwork: Learnable distance function
    - Supervised and contrastive learning variants
  - **Training Utilities**: Curriculum learning and meta-training
    - CurriculumScheduler: Progressive difficulty training
    - MetaTrainingEnvironment: Task distribution generator
  - **Tests**: test_learning.py with 7 test cases
  - **Demo**: demo_learning.py showing complete pipeline
- **Files**:
  - `src/atlas/learning/` — Complete learning module
  - `experiments/test_learning.py` — Unit tests
  - `experiments/demo_learning.py` — Full pipeline demo
- **Validation**: V3 — 16 spaces total, all tests pass
- **Next**: Phase E Real-World Interface (3D, ROS2, Gazebo)

---

## 2026-06-27 — TemporalSpace System (Phase C) ✓

- **Type**: T4 (Framework Enhancement)
- **Goal**: Add time dimension to cognitive spaces with prediction capabilities
- **Done**:
  - **CircularBuffer**: Fixed-size temporal history storage
  - **FieldPredictor**: Gaussian Process regression for field evolution
    - Linear trend fallback when sklearn unavailable
    - Uncertainty estimation with GP
  - **PeriodicityDetector**: Pattern detection via FFT and autocorrelation
  - **TemporalSpace**: Time-aware wrapper for any base space
    - Automatic history recording on each update
    - Field prediction at future timestamps
    - Dynamic obstacle trajectory prediction
  - **PredictiveRicciSpace**: Specialized Ricci with curvature forecasting
  - **Tests**: test_temporal.py with 6 test cases
  - **Demo**: demo_temporal.py showing all temporal features
- **Files**:
  - `src/atlas/spaces/temporal.py` — All temporal components
  - `experiments/test_temporal.py` — Unit tests
  - `experiments/demo_temporal.py` — Interactive demo
- **Validation**: V3 — 14 spaces total, all tests pass
- **Next**: Phase D Learning Integration (Bayesian optimization, meta-learning, NeuralSpace)

---

## 2026-06-27 — Space Composition System (Phase B) ✓

- **Type**: T4 (Framework Enhancement)
- **Goal**: Enable composition of multiple cognitive spaces
- **Done**:
  - **ProductSpace**: Parallel composition with weighted distance combination
    - d² = Σ wᵢ × dᵢ² (euclidean mode)
    - d = Σ wᵢ × dᵢ (manhattan mode)
  - **HierarchicalSpace**: Multi-scale global-local hierarchy
    - Coarse global space for long distances
    - Fine local space for short distances
    - Automatic transition based on threshold
  - **MixedSpace**: Context-aware dynamic switching
    - Condition-based space selection
    - Hard/soft blending modes
  - **Helper functions**: create_exploration_navigation_balance(), create_adaptive_exploration_space()
  - **Tests**: test_composite_spaces.py with 5 test cases
  - **Demo**: demo_composition.py showing all 3 composition types
- **Files**:
  - `src/atlas/spaces/composite.py` — All 3 composite space types
  - `experiments/test_composite_spaces.py` — Unit tests
  - `experiments/demo_composition.py` — Interactive demo
- **Validation**: V3 — 12 spaces total, all tests pass
- **Next**: Phase C TemporalSpace or Phase D Learning Integration

---

## 2026-06-27 — Phase Removal + Real-time Adaptation Framework

- **Type**: T4 (Framework Change)
- **Goal**: Complete transformation from documentation-centric to code library
- **Done**:
  - **Core**: Removed all "Phase" terminology from codebase
  - **Core**: Implemented D* Lite incremental solver (`replanning.py`)
  - **Core**: Implemented `AdaptiveNavigator` for closed-loop navigation
  - **Visualization**: Created `visualization/` module with 4 components
    - `SpaceVisualizer`: Field visualization with overlays
    - `PathAnimator`: Animation generation for paths
    - `RealtimeMonitor`: Runtime monitoring and loop detection
    - `ComparisonPlotter`: Performance comparison plots
  - **Spaces**: Verified @register_space decorator works across all spaces
  - **Experiments**: Created `test_dstar_lite.py` and `demo_visualization.py`
  - **Cleanup**: Deleted `cognitive_arch_v2/` and phase-named files
- **Files**:
  - `src/atlas/core/replanning.py` — D* Lite + AdaptiveNavigator
  - `src/atlas/visualization/` — Complete visualization toolkit
  - `experiments/test_dstar_lite.py` — D* Lite tests
  - `experiments/demo_visualization.py` — Visualization demo
- **Validation**: V3 — All tests pass, 3 PNG outputs generated
- **Left for next time**:
  - Phase B: Space composition (ProductSpace, HierarchicalSpace)
  - Phase C: TemporalSpace with field evolution prediction

---

## 2026-06-27 — Repository Reorganization to Standard Architecture

- **Type**: T2
- **Goal**: Align repository structure with canonical architecture template
- **Done**:
  - Created `CLAUDE.md` as alias to `AGENTS.md`
  - Created `docs/architecture/` with positioning, top-level-design, constitution
  - Created `docs/adr/` with README and template
  - Created `docs/operations/` with cheatsheet and testing docs
  - Created `docs/changelog/` for tracking changes
  - Migrated content from 01-05 folders to `docs/subsystems/` structure
  - Updated `docs/INDEX.md` with new navigation structure
- **Files**: `CLAUDE.md`, `docs/architecture/*`, `docs/subsystems/*`, `docs/INDEX.md`
- **Validation**: V1 — structure validated with `check_doc_consistency.py`
- **Left for next time**: Create automation scripts in `scripts/` directory

---

## 2026-06-27 — 3D Grid Space + Large-Scale A/B Testing ✓

- **Type**: T5 (Research Infrastructure + 3D Extension)
- **Goal**: Extend ATLAS to 3D grids and run rigorous large-scale A/B experiments
- **Done**:
  - **3D Grid Space**: `src/atlas/spaces/grid3d.py`
    - `EuclideanSpace3D`: 3D baseline with z-axis support
    - `RicciSpace3D`: Ricci flow extended to 3D (6/26 connectivity)
    - `ConformalSpace3D`: Conformal transformation in 3D
  - **3D Solver**: `src/atlas/spaces/solver3d.py`
    - A* search with 6-neighbor and 26-neighbor connectivity
    - Multi-goal solving support
  - **Large-Scale A/B Testing**: `experiments/ab_test_research.py`
    - 5 spaces × 6 scenarios × 30 trials = 900 runs (2D)
    - 3 spaces × 3 scenarios × 30 trials = 270 runs (3D)
    - **Ablation Study**: Systematic component removal to measure contribution
    - **LaTeX Report Generation**: Publication-ready tables
    - **Statistical Testing**: ANOVA, t-test, effect sizes
- **Key Findings** (30-trial results):
  - Euclidean: fastest planning (0.68ms), lowest path cost (32.3)
  - Ricci: 6x slower (4.15ms) but same path cost
  - Fisher: significantly higher path cost (3151) - needs calibration
  - 3D Ricci: same pattern as 2D, 17x slower than 3D Euclidean
- **Files**:
  - `src/atlas/spaces/grid3d.py` — 3D cognitive spaces
  - `src/atlas/spaces/solver3d.py` — 3D A* solver
  - `experiments/ab_test_research.py` — Complete research framework
- **Validation**: V3 — 1170 total runs completed successfully
- **Next**: Use results for paper writing, calibrate Fisher space

---

## 2026-06-27 — SSFR理论形式化：与信息几何等价 ✓

- **Type**: T5 (Theoretical Foundation)
- **Goal**: Prove SSFR is equivalent to information geometry on statistical manifold
- **Done**:
  - **Formalized SSFR**: StableStructure with Fisher information, value, stability
  - **5 Correspondences Proved**:
    1. Structure ↔ Point on statistical manifold
    2. Structure discovery ↔ Maximum likelihood estimation
    3. Stability ↔ Fisher information
    4. Value ↔ Information gain / Cost
    5. Navigation ↔ Natural gradient flow
  - **Equivalence Theorem**: SSFR ≡ Information Geometry
  - **Numerical Verification**: Fisher trace ∝ stability, value ∝ info gain
  - **A/B Test**: SSFR vs Traditional (94.5% cost reduction)
- **Files**:
  - `src/atlas/research/ssfr_information_geometry.py` — Formal proof with code
  - `docs/research/ssfr_information_geometry.md` — Complete theory document
  - `experiments/ab_test_stable_structure.py` — A/B test validation
- **Validation**: V3 — Numerical verification + A/B test
- **Next**: Continuous extension, hierarchical structures, multi-agent

---

## 2026-06-28 — SSFR Kitchen Integration (Application)

- **Type**: T5 (Cross-cutting Application)
- **Goal**: Apply SSFR to physical kitchen environment for real robot task execution
- **Done**:
  - **KitchenSpaceAdapter**: Continuous → discrete mapping for SSFR
    - World-to-grid coordinate conversion (0.5m resolution)
    - Obstacle extraction from physical objects
    - Observation encoding with uncertainty estimation
    - Action decoding to physical robot actions
  - **PhysicalSSFR**: SSFR wrapper for physical environments
    - Automatic grid dimension calculation
    - Perception with physical state encoding
    - Competition with actual position feedback
    - Space validity computation per robot
  - **SSFRTaskPlanner**: Task execution with SSFR guidance
    - Task assignment and step execution
    - Space-aware navigation (Ricci/Fisher/Wasserstein strategies)
    - SSFR perception at each step
    - Real-time structure competition and evolution
  - **Integration Tests**: 7/7 passing
    - Space adapter, Physical SSFR, Task planner
    - Navigation, Structure evolution, Space comparison, End-to-end
  - **Demos**: 4 demonstration scenarios
    - SSFR perception, Task planning, Structure evolution, Space comparison
- **Files**:
  - `experiments/demo_ssfr_kitchen.py` — Integration demos
  - `experiments/test_ssfr_kitchen.py` — 7 integration tests
  - `docs/ssfr_kitchen_integration.md` — Integration documentation
- **Validation**: V3 — 7/7 tests pass, SSFR perceives physical environment, structures evolve
- **Next**: Structure reuse across tasks, online learning for space selection, multi-robot

---

## Earlier Changes

- Initial repository structure with 5-layer organization
- Foundation layer (01): VLM, World Model, VLA
- Annotation layer (02): Schema, action/scene/physics annotation
- Perception layer (03): Stereo+IMU, depth, SLAM, hand pose, reconstruction
- Data ecosystem layer (04): Ego, UMI, Sim2Real, teleop, hardware
- Integration layer (05): Pipeline patterns, quality gates
