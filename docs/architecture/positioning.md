---
id: positioning
title: ATLAS Project Scope and Positioning
status: stable
last_validated: 2026-06-28
tags: [scope, positioning]
---

# ATLAS — Positioning

> **Atlas of Technologies for Learning Autonomous Systems**
> 可插拔认知空间框架

## In Scope

1. **Cognitive Space Framework**
   - 可插拔认知空间抽象 (`CognitiveSpace`)
   - 空间注册与发现机制 (`@register_space`)
   - 求解器与空间解耦 (`GeodesicSolver`)

2. **Space Implementations**
   - 基础空间: Euclidean, Ricci, Fisher, Conformal, Wasserstein, Finsler
   - 连续空间: 稀疏采样 + kNN插值，无网格限制
   - 复合空间: Product, Hierarchical, Mixed
   - 时序空间: Temporal, PredictiveRicci

3. **SSFR (Structure Self-Discovery & Reuse)**
   - 结构假设生成与验证
   - 结构竞争池 (StructurePool)
   - 结构演化机制

4. **Physical Environment**
   - pymunk-based 2D physics kitchen
   - ContinuousPhysicalSSFR integration

## Out of Scope (Non-Goals)

1. **Production Robotics** - 研究框架，非生产系统
2. **3D Simulation** - 仅2D物理环境
3. **Real-time Guarantees** - Python原型，非实时

## Target Audience

- **AI Researchers** - 研究认知空间与结构发现
- **Robotics Students** - 学习可插拔架构设计
- **Hobbyists** - 探索物理模拟中的认知架构
