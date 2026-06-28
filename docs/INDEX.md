---
id: docs-index
title: ATLAS Documentation Index
status: stable
last_validated: 2026-06-27
tags: [index, navigation]
---

# docs/INDEX.md — Navigation Hub

> **Token budget**: ~500 tokens to read this file.
> Purpose: Route to the right document. Do not store domain knowledge here.

---

## NAVIGATE — Where to Go

```
NEED project scope / non-goals?
└── READ: docs/architecture/positioning.md

NEED architecture overview?
└── READ: docs/architecture/top-level-design.md

NEED design rules / invariants?
└── READ: docs/architecture/constitution.md

NEED why a decision was made?
└── READ: docs/adr/NNNN-*.md

NEED subsystem details?
└── READ: docs/subsystems/NN-*.md

NEED recent changes?
└── READ: docs/changelog/CHANGELOG.md

NEED tech debt status?
└── RUN: python scripts/generate_reports.py tech-debt

NEED status/doc consistency?
└── RUN: python scripts/check_doc_consistency.py

NEED ops commands?
└── READ: docs/operations/cheatsheet.md
```

---

## SUBSYSTEMS — Quick Index

### 01 — Foundation (Models)

| ID | Name | Status | Doc |
|--|--|--|--|
| 01 | VLM底座模型 | stable | [subsystems/01-vlm.md](subsystems/01-vlm.md) |
| 02 | World Model | stable | [subsystems/02-world-model.md](subsystems/02-world-model.md) |
| 03 | VLA端到端 | stable | [subsystems/03-vla.md](subsystems/03-vla.md) |
| 04 | 架构模式 | stable | [subsystems/04-architecture-patterns.md](subsystems/04-architecture-patterns.md) |

### 11 — Annotation

| ID | Name | Status | Doc |
|--|--|--|--|
| 11 | Schema设计 | stable | [subsystems/11-schema-design.md](subsystems/11-schema-design.md) |
| 12 | 动作标注 | stable | [subsystems/12-action-annotation.md](subsystems/12-action-annotation.md) |
| 13 | 场景标注 | stable | [subsystems/13-scene-annotation.md](subsystems/13-scene-annotation.md) |
| 14 | 物理标注 | stable | [subsystems/14-physics-annotation.md](subsystems/14-physics-annotation.md) |

### 21 — Perception

| ID | Name | Status | Doc |
|--|--|--|--|
| 21 | 双目+IMU | stable | [subsystems/21-stereo-imu.md](subsystems/21-stereo-imu.md) |
| 22 | 深度估计 | stable | [subsystems/22-depth-estimation.md](subsystems/22-depth-estimation.md) |
| 23 | SLAM | stable | [subsystems/23-slam.md](subsystems/23-slam.md) |
| 24 | 手势识别 | stable | [subsystems/24-hand-pose.md](subsystems/24-hand-pose.md) |
| 25 | 3D重建 | stable | [subsystems/25-3d-reconstruction.md](subsystems/25-3d-reconstruction.md) |
| 26 | 传感器融合 | stable | [subsystems/26-sensor-fusion.md](subsystems/26-sensor-fusion.md) |

### 31 — Data Ecosystem

| ID | Name | Status | Doc |
|--|--|--|--|
| 31 | Ego采集 | stable | [subsystems/31-ego-collection.md](subsystems/31-ego-collection.md) |
| 32 | UMI系统 | stable | [subsystems/32-umi-systems.md](subsystems/32-umi-systems.md) |
| 33 | Sim2Real | stable | [subsystems/33-sim2real.md](subsystems/33-sim2real.md) |
| 34 | 遥操作 | stable | [subsystems/34-teleoperation.md](subsystems/34-teleoperation.md) |
| 35 | 硬件矩阵 | stable | [subsystems/35-hardware-matrix.md](subsystems/35-hardware-matrix.md) |
| 36 | 数据格式 | stable | [subsystems/36-data-formats.md](subsystems/36-data-formats.md) |

### 41 — Integration

| ID | Name | Status | Doc |
|--|--|--|--|
| 41 | Pipeline模式 | stable | [subsystems/41-pipeline-patterns.md](subsystems/41-pipeline-patterns.md) |
| 42 | 质量门控 | stable | [subsystems/42-quality-gates.md](subsystems/42-quality-gates.md) |
| 43 | 系统设计 | stable | [subsystems/43-system-design.md](subsystems/43-system-design.md) |

**Status values**: `draft`, `stable`, `active-dev`, `experimental`, `deprecated`

---

## READING PATHS

### Path 1: I want to choose data collection hardware
Start → [35-hardware-matrix](subsystems/35-hardware-matrix.md)

### Path 2: I need to implement depth estimation
Start → [22-depth-estimation](subsystems/22-depth-estimation.md)

### Path 3: I'm designing an annotation pipeline
Start → [11-schema-design](subsystems/11-schema-design.md)

### Path 4: I want the full system view
Start → [41-pipeline-patterns](subsystems/41-pipeline-patterns.md)

---

## DATA FLOW (Bottom-Up)

```
31-36 Data Ecosystem (采集)
    ↓
21-26 Perception (感知处理)
    ↓
11-14 Annotation (结构化标注)
    ↓
01-04 Foundation (模型训练)
    ↓
41-43 Integration (系统部署)
```

---

## TOKENS — Budget Guide

| Step | File | ~Tokens |
|--|--|--|
| 1 | `AGENTS.md` (auto-loaded) | 1000 |
| 2 | `docs/INDEX.md` (this) | 400 |
| 3 | Status via tool | 100 |
| 4 | `docs/subsystems/NN-*.md` | 1500-2500 |
| 5 | Source files | varies |

**Total overhead: ~3000 tokens**

IF task spans 3+ subsystems: **SPAWN Explore sub-agent**

---

## EXIT — Session End Checklist

Per `AGENTS.md`:

| Change Type | Check |
|--|--|
| T1 (Docs/Typo) | Nothing |
| T2+ (Code) | Run `scripts/check_doc_anchors.py` |
| T3+ (Bug/Feature) | Update `docs/_machine/status.yaml` |
| T2+ (Code) | Update `docs/changelog/CHANGELOG.md` |
| Removal | Update `docs/deprecated.md` |

---

## EXTERNAL — Key Files

| Path | Purpose |
|--|--|
| `AGENTS.md` (root) | Boot protocol |
| `CLAUDE.md` (root) | Thin `@AGENTS.md` import |
| `llms.txt` (root) | Machine-readable index |
| `scripts/check_doc_consistency.py` | Drift detector |
| `scripts/check_known_gaps.py` | Gap validator |

---

*Updated: 2026-06-27*
