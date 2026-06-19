# Dataset Comparison

## Ego-Centric Datasets

| Dataset | Size | Modalities | Tasks | Annotations |
|---------|------|------------|-------|-------------|
| **Ego4D** | 3,670h | RGB, Audio, IMU, GPS | 16 scenarios | NL queries, point tracks |
| **EPIC-Kitchens** | 100h | RGB, Audio | Cooking | Actions, objects, narration |
| **EGTEA Gaze+** | 28h | RGB, Gaze | Cooking | Gaze, actions, hands |
| **HOI4D** | 2,400 clips | RGB-D | Object interaction | 4D annotation, contact |

## Robot Manipulation Datasets

| Dataset | Robot | Scenes | Demos | Action Space |
|---------|-------|--------|-------|--------------|
| **Open X-Embodiment** | 22 robots | Varied | 1M+ | Joint/EE pose |
| **ALOHA** | 2x ViperX | Static | 500+ | Joint pos |
| **RoboTurk** | 3 robots | Varied | 100K+ | EE pose |
| **Bridge** | WidowX | Kitchen | 10K+ | Joint pos |
| **RH20T** | 6 robots | Varied | 110K+ | EE pose |

## Simulated Datasets

| Simulator | Physics | Rendering | Domain Randomization | Parallel |
|-----------|---------|-----------|---------------------|----------|
| **Isaac Gym** | GPU | RTX | ✅ | ✅ 1000s |
| **MuJoCo** | CPU | Basic | ✅ | ❌ |
| **PyBullet** | CPU | Basic | ✅ | ✅ 100s |
| **SAPIEN** | CPU/GPU | RTX | ✅ | ✅ |
| **NVIDIA Omniverse** | GPU | RTX/Path | ✅ | ✅ |

## Dataset Selection Guide

| Goal | Recommended Dataset |
|------|---------------------|
| General manipulation | Open X-Embodiment |
| Bimanual tasks | ALOHA, RH20T |
| Ego-centric understanding | Ego4D, EPIC-Kitchens |
| Hand-object interaction | HOI4D, DexYCB |
| Sim2Real validation | Bridge, RoboTurk |

## Key Metrics

When comparing datasets:
- **Diversity**: Object types, scenes, lighting
- **Scale**: Hours, trajectories, unique tasks
- **Annotation quality**: Automatic vs human-verified
- **Action space**: Joint control vs end-effector
- **Reproducibility**: Hardware availability, cost
