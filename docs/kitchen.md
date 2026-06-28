# ATLAS Physical Kitchen

基于 pymunk 的2D物理模拟厨房环境。

## Architecture

```
Physical Kitchen → ContinuousPhysicalSSFR → ContinuousSSFRTaskPlanner → Action
```

## Quick Start

```python
from atlas.kitchen import create_demo_kitchen
from experiments.tests.test_continuous_ssfr import (
    ContinuousPhysicalSSFR, ContinuousSSFRTaskPlanner
)

# Create kitchen
kitchen = create_demo_kitchen()
robot_id = list(kitchen.robots.keys())[0]

# Create continuous SSFR (direct physical coordinates)
physical_ssfr = ContinuousPhysicalSSFR(kitchen)
planner = ContinuousSSFRTaskPlanner(physical_ssfr)

# Assign task
planner.assign_task(robot_id, 'make_coffee')

# Run
for _ in range(100):
    kitchen.step()
    result = planner.step(robot_id)
```

## Components

| Component | Description |
|-----------|-------------|
| `Kitchen` | pymunk 2D physics environment |
| `Robot` | Physical body with sensors and actuators |
| `ContinuousPhysicalSSFR` | SSFR with continuous coordinates |
| `ContinuousSSFRTaskPlanner` | Task planning with SSFR guidance |

## Tests

```bash
python experiments/tests/test_physical_kitchen.py
python experiments/tests/test_continuous_ssfr.py
```

| Test | Status |
|------|--------|
| Physics | ✅ PASS |
| Robot movement | ✅ PASS |
| Continuous SSFR | ✅ PASS |
| Kitchen integration | ✅ PASS |
