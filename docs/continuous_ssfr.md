# Continuous SSFR

> Continuous-state SSFR with sparse sampling + kNN interpolation

## Key Difference

| Aspect | Discrete SSFR | Continuous SSFR |
|--------|-------------|-----------------|
| Position | `Tuple[int, int]` | `Tuple[float, float]` |
| Field | numpy 2D array | `ContinuousField` (sparse samples) |
| Neighbors | 4/8-connected | 16-direction sampling |
| Distance | Grid path | Continuous path integral |
| Boundary | Fixed grid | Unbounded |

## Quick Start

```python
from atlas.spaces.continuous import ContinuousRicciSpace
from atlas.spaces.continuous_ssfr import ContinuousSSFR

# Create continuous space
space = ContinuousRicciSpace(curvature_scale=1.0)

# Update with continuous coordinates
space.update_from_observation((0.5, 0.5), {
    'obstacles': [(0.3, 0.3)],
    'goal_position': (2.0, 2.0),
})

# Create SSFR
ssfr = ContinuousSSFR(space_names=['ricci', 'fisher'])

# Perceive with continuous coordinates
hypotheses = ssfr.perceive(
    position=(1.0, 2.0),
    observation={...}
)
```

## Components

### ContinuousField

```python
from atlas.spaces.continuous import ContinuousField

field = ContinuousField(default_value=0.0)
field.add_sample((0.0, 0.0), 1.0)
value = field.query((0.5, 0.5))  # kNN interpolation
```

### Continuous Spaces

| Space | Fields |
|-------|--------|
| `ContinuousEuclideanSpace` | None (baseline) |
| `ContinuousRicciSpace` | uncertainty, curvature, familiarity |
| `ContinuousFisherSpace` | belief, confidence |
| `ContinuousWassersteinSpace` | cost, mass |

### ContinuousSSFR

```python
ssfr = ContinuousSSFR(
    space_names=['ricci', 'fisher', 'wasserstein'],
    max_structures=50,
    evolution_interval=20,
)

ssfr.perceive(position, observation)
ssfr.compete(observation, actual)
ssfr.evolve()
```

## Kitchen Integration

```python
from experiments.tests.test_continuous_ssfr import (
    ContinuousPhysicalSSFR, ContinuousSSFRTaskPlanner
)

physical_ssfr = ContinuousPhysicalSSFR(kitchen)
planner = ContinuousSSFRTaskPlanner(physical_ssfr)
planner.assign_task(robot_id, 'make_coffee')
```

## Tests

```bash
python experiments/tests/test_continuous_ssfr.py
```

| Test | Status |
|------|--------|
| Continuous Field | ✅ PASS |
| Continuous Space | ✅ PASS |
| Continuous SSFR | ✅ PASS |
| Kitchen Integration | ✅ PASS |
