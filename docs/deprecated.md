---
id: deprecated
title: Deprecated Files and Legacy Structure
status: stable
last_validated: 2026-06-28
tags: [deprecated, legacy]
---

# Deprecated

This document tracks deprecated paths and migration notes.

## Legacy Directory Structure (Pre-2026-06-27)

The following directory structure has been reorganized to align with the standard architecture template:

```
docs/
├── 01-foundation/          → Migrated to docs/subsystems/ (01-04)
├── 02-annotation/          → Migrated to docs/subsystems/ (11-14)
├── 03-perception/          → Migrated to docs/subsystems/ (21-26)
├── 04-data-ecosystem/      → Migrated to docs/subsystems/ (31-36)
└── 05-integration/         → Migrated to docs/subsystems/ (41-43)
```

### New Structure

```
docs/
├── architecture/           # Project-level architecture docs
│   ├── constitution.md
│   ├── positioning.md
│   ├── ROADMAP_v1.0.md
│   └── top-level-design.md
├── adr/                    # Architecture Decision Records
│   ├── 0000-template.md
│   └── README.md
├── subsystems/             # Per-subsystem specifications
│   ├── 01-*.md            # Foundation (Models)
│   ├── 11-*.md            # Annotation
│   ├── 21-*.md            # Perception
│   ├── 31-*.md            # Data Ecosystem
│   └── 41-*.md            # Integration
├── operations/             # Runbooks and guides
│   ├── cheatsheet.md
│   └── testing.md
├── changelog/              # Change history
│   └── CHANGELOG.md
├── research/               # Research notes and experiments
│   ├── ssfr_continuous_hierarchical.md
│   └── ssfr_information_geometry.md
└── _machine/               # Machine-readable state files
    ├── bugs.yaml
    ├── status.yaml
    └── tech-debt.yaml
```

## Removed Files (2026-06-28 Cleanup)

### Old Experiment Files
The following historical experiment files have been removed. They are preserved in git history if needed:

| File | Reason | Replacement |
|------|--------|-------------|
| `experiments/ssfr_agent_v2.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_agent_v3.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_agent_v4.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_rearchitecture_v1.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_rearchitecture_v2.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_integration.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_representation_selection.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ssfr_analysis.py` | Historical analysis | See `docs/research/` |
| `experiments/ssfr_applicability_domain.py` | Historical analysis | See `docs/research/` |
| `experiments/ssfr_problem_solving.py` | Superseded by enhanced SSFR | `src/atlas/core/ssfr_enhanced.py` |
| `experiments/ab_test_quick.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/ab_test_research.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/ab_test_spaces.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/ab_test_ssfr_extensions.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/ab_test_stable_structure.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_3d_navigation.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_3d_simple.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_multi_mechanism.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_observable.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_ssfr.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_ssfr_v2.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/benchmark_unified.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/meta_ssfr_complex_test.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/meta_ssfr_integration.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |
| `experiments/meta_ssfr_test.py` | Superseded by unified test | `experiments/test_ssfr_enhanced.py` |

### Removed Directories
- `outputs/` - Generated visualization outputs (regenerated on demand)
- `results/` - Test results (regenerated on demand)

## Migration Notes

- All topic documents now include required YAML frontmatter
- Internal references updated from `../01-foundation/` to subsystem paths
- Index files retained in legacy locations for reference (will be removed)
- Core architecture files remain in `src/atlas/core/`
- Space implementations remain in `src/atlas/spaces/`

## Current Active Files

### Core Architecture
- `src/atlas/core/space.py` - CognitiveSpace ABC
- `src/atlas/core/registry.py` - SpaceRegistry
- `src/atlas/core/world_model.py` - WorldModel
- `src/atlas/core/solver.py` - GeodesicSolver
- `src/atlas/core/replanning.py` - D* Lite
- `src/atlas/core/experiment.py` - Experiment framework
- `src/atlas/core/ssfr_enhanced.py` - Enhanced SSFR (NEW)

### Space Implementations
- `src/atlas/spaces/euclidean.py` - Baseline
- `src/atlas/spaces/ricci.py` - Information geometry
- `src/atlas/spaces/fisher.py` - Statistical manifold
- `src/atlas/spaces/wasserstein.py` - Optimal transport
- `src/atlas/spaces/finsler.py` - Asymmetric metric
- `src/atlas/spaces/conformal.py` - Dynamic metric
- `src/atlas/spaces/temporal.py` - Time-aware space
- `src/atlas/spaces/composite.py` - Product/Hierarchical/Mixed
- `src/atlas/spaces/grid3d.py` - 3D grid
- `src/atlas/spaces/solver3d.py` - 3D solver

### Active Experiments
- `experiments/compare_spaces.py` - Space comparison
- `experiments/test_space_updates.py` - Space update verification
- `experiments/test_composite_spaces.py` - Composite space tests
- `experiments/test_dstar_lite.py` - D* Lite tests
- `experiments/test_temporal.py` - Temporal space tests
- `experiments/test_learning.py` - Learning tests
- `experiments/test_ssfr_enhanced.py` - Enhanced SSFR tests (NEW)
- `experiments/meta_ssfr_atlas.py` - Meta-SSFR integration
- `experiments/demo_visualization.py` - Visualization demo
- `experiments/demo_composition.py` - Composition demo
- `experiments/demo_learning.py` - Learning demo
- `experiments/demo_temporal.py` - Temporal demo

## See Also

- [docs/INDEX.md](INDEX.md) — New navigation hub
- [docs/architecture/](architecture/) — Architecture documentation
- [docs/changelog/CHANGELOG.md](changelog/CHANGELOG.md) — Change history
