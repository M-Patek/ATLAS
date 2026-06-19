# ATLAS Documentation Index

## Layer Overview

ATLAS organizes embodied AI data infrastructure into five layers:

| Layer | Focus | Key Topics |
|-------|-------|------------|
| [01-foundation](01-foundation/) | Model layer | VLM底座, World Model, VLA端到端 |
| [02-annotation](02-annotation/) | Annotation layer | Schema设计, 动作/场景/物理标注标准 |
| [03-perception](03-perception/) | Perception layer | 双目+IMU, 深度估计, SLAM, 手势, 重建 |
| [04-data-ecosystem](04-data-ecosystem/) | Data ecosystem | Ego-centric, UMI, Sim2Real, 遥操作, 硬件 |
| [05-integration](05-integration/) | Integration layer | Pipeline模式, 质量门控, 系统设计 |

## Reading Paths

### Path 1: I want to choose data collection hardware
Start → [04-data-ecosystem/index.md](04-data-ecosystem/index.md) → [05-hardware-matrix.md](04-data-ecosystem/05-hardware-matrix.md)

### Path 2: I need to implement depth estimation
Start → [03-perception/index.md](03-perception/index.md) → [02-depth-estimation.md](03-perception/02-depth-estimation.md) (FoundationStereo)

### Path 3: I'm designing an annotation pipeline
Start → [02-annotation/index.md](02-annotation/index.md) → [01-schema-design.md](02-annotation/01-schema-design.md)

### Path 4: I want the full system view
Start → [05-integration/index.md](05-integration/index.md) → [01-pipeline-patterns.md](05-integration/01-pipeline-patterns.md)

## Cross-Layer Connections

### Data Flow (Bottom-Up)
```
04-data-ecosystem (采集)
    ↓
03-perception (感知处理)
    ↓
02-annotation (结构化标注)
    ↓
01-foundation (模型训练)
    ↓
05-integration (系统部署)
```

### Decision Dependencies (Top-Down)
```
01-foundation (模型需求)
    ↓
02-annotation (标注策略)
    ↓
03-perception (感知精度要求)
    ↓
04-data-ecosystem (采集方案)
```

## Content Status

See [_machine/status.yaml](_machine/status.yaml) for detailed completion status.

Quick status:
- **01-foundation**: In progress (VLM complete, VLA/World Model draft)
- **02-annotation**: Not started
- **03-perception**: Not started (FoundationStereo prioritized)
- **04-data-ecosystem**: Not started
- **05-integration**: Not started

## Meta Documents

- [00-meta/glossary.md](00-meta/glossary.md) — 术语表
- [00-meta/how-to-use.md](00-meta/how-to-use.md) — 使用指南

---

*Last updated: 2026-06-20*
