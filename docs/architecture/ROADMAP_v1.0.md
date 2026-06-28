---
id: roadmap
title: ATLAS Roadmap
status: stable
last_validated: 2026-06-28
tags: [roadmap, planning]
---

# ATLAS Roadmap

## Current State (2026-06-28)

| Component | Status | Files |
|-----------|--------|-------|
| Core Framework | ✅ Stable | `core/*.py` |
| Discrete Spaces | ✅ Stable | `spaces/euclidean.py`, `ricci.py`, etc. |
| Continuous Spaces | ✅ Stable | `spaces/continuous.py`, `continuous_ssfr.py` |
| Composite Spaces | ✅ Stable | `spaces/composite.py` |
| Temporal Spaces | ✅ Stable | `spaces/temporal.py` |
| SSFR | ✅ Stable | `core/ssfr_enhanced.py` |
| Kitchen | ✅ Stable | `kitchen/*.py` |
| Learning | ✅ Beta | `learning/*.py` |
| Visualization | ✅ Beta | `visualization/*.py` |

## Completed Milestones

| Date | Milestone |
|------|-----------|
| 2026-06-28 | Deep cleanup: removed 75 files, reorganized structure |
| 2026-06-28 | Continuous SSFR: removed discrete grid dependency |
| 2026-06-28 | SSFR Kitchen Integration: physical robot tasks |
| 2026-06-27 | Space Composition: Product, Hierarchical, Mixed |
| 2026-06-27 | Temporal Space: time-aware with prediction |
| 2026-06-27 | Learning Module: Bayesian optimization, meta-learning |
| 2026-06-27 | 3D Grid Space: Euclidean, Ricci, Conformal in 3D |
| 2026-06-27 | SSFR-Information Geometry: corrected subset relationship |
| 2026-06-27 | A/B Testing Framework: statistical validation |

## Next Steps

| Priority | Task | Status |
|----------|------|--------|
| 1 | Multi-agent: product manifold for coordination | Planned |
| 2 | GPU acceleration for field queries | Planned |
| 3 | Real robot validation (ROS2) | Planned |
| 4 | Paper writing | Planned |
