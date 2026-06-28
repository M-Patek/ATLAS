---
id: top-level-design
title: ATLAS Top-level Architecture
status: stable
last_validated: 2026-06-27
tags: [architecture, diagram, mental-model]
---

# ATLAS — Top-level Architecture

## 1. Full System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        ATLAS Knowledge Base                      │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ 05-Integration│ ← System Integration & Deployment            │
│  └───────┬──────┘                                               │
│          │                                                       │
│  ┌───────▼──────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 01-Foundation│───→│ 02-Annotation│───→│ 03-Perception│       │
│  │  (Models)    │    │  (Schema)    │    │ (Algorithms) │       │
│  └───────┬──────┘    └──────────────┘    └───────┬──────┘       │
│          │                                       │               │
│          └───────────────┬───────────────────────┘               │
│                          │                                       │
│                   ┌──────▼──────┐                                │
│                   │ 04-Data-Eco │ ← Data Infrastructure          │
│                   │  (Hardware) │                                │
│                   └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Core Mental Models

### 2.1 Layer Dependency Model

**Data Flow (Bottom-Up)**
```
04-Data-Ecosystem (采集层)
    ↓ 原始数据流
03-Perception (感知处理层)
    ↓ 结构化感知输出
02-Annotation (标注结构化层)
    ↓ 训练数据
01-Foundation (模型训练层)
    ↓ 模型权重
05-Integration (系统部署层)
```

**Decision Flow (Top-Down)**
```
01-Foundation (模型需求确定)
    ↓ 决定了需要什么样的数据
02-Annotation (标注策略制定)
    ↓ 决定了需要什么感知输出
03-Perception (感知精度要求)
    ↓ 决定了采集配置
04-Data-Ecosystem (采集方案落地)
```

### 2.2 Five-Layer Architecture

| Layer | Core Question | Key Decisions |
|-------|---------------|---------------|
| **01-Foundation** | 用什么模型？ | VLM底座、World Model、VLA选型 |
| **02-Annotation** | 怎么标注数据？ | Schema设计、动作/场景/物理标注标准 |
| **03-Perception** | 感知如何实现？ | 双目+IMU、深度估计、SLAM、手势识别 |
| **04-Data-Ecosystem** | 如何采集数据？ | Ego/UMI/Sim2Real/遥操作、硬件矩阵 |
| **05-Integration** | 如何整合系统？ | Pipeline模式、质量门控、部署架构 |

### 2.3 Decision Matrix Pattern

每个技术选型都遵循统一的评估框架：
- **Trade-offs**: 精度 vs 速度 vs 成本
- **Dependencies**: 上游输入要求和下游输出承诺
- **Ecosystem**: 开源工具链和硬件支持

## 3. Run Mode Matrix

| Dimension | Options | Effect on System |
|---|---|---|
| **Collection Mode** | `ego-centric` / `umi` / `sim2real` / `teleop` | 决定硬件配置、空间约束、数据格式 |
| **Perception Stack** | `stereo-only` / `stereo+imu` / `rgbd` | 影响深度精度、SLAM方法、标定复杂度 |
| **Model Family** | `vlm-first` / `vla-end-to-end` / `world-model` | 决定标注策略、推理架构、训练数据需求 |

## 4. Control Flow vs Data Flow Separation

| Category | Components | Job |
|---|---|---|
| **Control** | Pipeline orchestration, Quality gates | 决定"何时做什么" |
| **Data** | Raw sensors → Perception → Annotations | 决定"信息如何流动" |
| **Cross-cutting** | Schema definitions, Model interfaces | 连接控制与数据的一致性契约 |

## 5. Subsystem Connectivity

```
01-Foundation (VLM/VLA) ←──── schema ────→ 02-Annotation
        ↑                                      ↑
        └──── inference requirements ──────────┘

03-Perception ─── sensor specs ───→ 04-Data-Ecosystem
        ←──── hardware constraints ───┘

05-Integration ←── interfaces from all layers
```
