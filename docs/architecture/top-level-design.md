---
id: top-level-design
title: ATLAS Top-level Architecture
status: stable
last_validated: 2026-06-28
tags: [architecture, mental-model]
---

# ATLAS — Top-level Architecture

## Core Abstraction

```
┌─────────────────────────────────────────────────────────────┐
│                    Task Layer                               │
│  - Task decomposition                                       │
│  - Space selection                                          │
│  - SSFR: Structure discovery & reuse                        │
├─────────────────────────────────────────────────────────────┤
│                    Solver Layer                               │
│  - A*, D* Lite, Dijkstra                                   │
│  - Decoupled from space type                                │
├─────────────────────────────────────────────────────────────┤
│                    Space Layer (Pluggable)                  │
│  - Abstract: CognitiveSpace interface                       │
│  - Discrete: Euclidean, Ricci, Fisher, Conformal...      │
│  - Continuous: ContinuousRicci, ContinuousFisher...        │
│  - Composite: Product, Hierarchical, Mixed               │
├─────────────────────────────────────────────────────────────┤
│                    Environment Layer                          │
│  - Physical: pymunk 2D kitchen                            │
│  - Grid: Discrete state space                             │
│  - Continuous: Physical coordinates                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principle

**Space is pluggable.** Any implementation of `CognitiveSpace` can be:
- Registered via `@register_space`
- Used by any `GeodesicSolver`
- Compared in `Experiment` framework
- Composed in `ProductSpace`, `HierarchicalSpace`, `MixedSpace`

## Data Flow

```
Observation → update space → compute distance → solve path → execute action
     ↑                                                    ↓
     └───────────────── actual result ←────────────────────┘
```
