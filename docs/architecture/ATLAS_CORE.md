---
id: atlas-core
title: ATLAS Core Framework
description: Pluggable cognitive space framework with SSFR
status: stable
last_validated: 2026-06-28
tags: [core, architecture, spaces]
---

# ATLAS Core

## Overview

ATLAS is a **pluggable cognitive space framework**. The core abstraction is `CognitiveSpace` — any implementation can be registered and used by the solver.

## Quick Start

```python
from atlas.core.registry import create_space
from atlas.core.solver import GeodesicSolver

# Create any space
space = create_space("euclidean", width=40, height=20)

# Create solver (same for all spaces)
solver = GeodesicSolver(space)

# Solve
result = solver.solve(start=(5, 10), goal=(35, 10))
```

## Core Components

### CognitiveSpace (Abstract Base)

```python
class CognitiveSpace(ABC):
    def compute_distance(self, pos1, pos2) -> float: ...
    def get_heuristic(self, pos, goal) -> float: ...
    def update_from_observation(self, position, observation): ...
```

### Registry

```python
from atlas.core.space import register_space

@register_space("my_space")
class MySpace(CognitiveSpace):
    ...

# Auto-registered, use anywhere
space = create_space("my_space", ...)
```

### Solver

```python
solver = GeodesicSolver(space)
result = solver.solve(start, goal, obstacles)
# result.path, result.cost, result.space_name
```

### Experiment

```python
from atlas.core import Experiment

experiment = Experiment("comparison")
experiment.register_space("euclidean", create_space("euclidean", 40, 20))
experiment.register_space("ricci", create_space("ricci", 40, 20))
experiment.add_scenario({...})
results = experiment.run(num_trials=10)
```

## Available Spaces

| Space | Type | Key Feature |
|-------|------|-------------|
| `euclidean` | Discrete | Baseline |
| `ricci` | Discrete | Curvature from uncertainty |
| `fisher` | Discrete | Information geometry |
| `conformal` | Discrete | Dynamic metric |
| `wasserstein` | Discrete | Optimal transport |
| `finsler` | Discrete | Asymmetric metric |
| `continuous_euclidean` | Continuous | Baseline, no grid |
| `continuous_ricci` | Continuous | Sparse sampling + kNN |
| `continuous_fisher` | Continuous | Continuous belief field |
| `product` | Composite | Weighted combination |
| `hierarchical` | Composite | Multi-scale |
| `mixed` | Composite | Context-aware switching |
| `temporal` | Temporal | History + prediction |

## SSFR Integration

```python
from atlas.core.ssfr_enhanced import SSFREnhanced

ssfr = SSFREnhanced(space_names=['ricci', 'fisher'])
result = ssfr.step(position, observation, actual)
```

## File Structure

```
src/atlas/core/
├── space.py          # CognitiveSpace ABC
├── registry.py       # @register_space
├── solver.py         # GeodesicSolver
├── replanning.py     # D* Lite
├── experiment.py     # Experiment framework
├── path_planning.py  # A*, greedy
└── ssfr_enhanced.py  # SSFR core
```
