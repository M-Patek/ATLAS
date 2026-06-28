---
id: positioning
title: ATLAS Project Scope and Positioning
status: stable
last_validated: 2026-06-27
tags: [scope, non-goals, positioning]
---

# ATLAS — Positioning

> **Atlas of Technologies for Learning Autonomous Systems**
> 具身智能数据基础设施学习图谱

## In Scope

1. **Embodied AI Data Infrastructure**
   - 数据收集范式（Ego-centric, UMI, Sim2Real, Teleoperation）
   - 感知算法栈（深度估计、SLAM、手势识别）
   - 标注标准与Schema设计
   - 模型选型指南（VLM, World Model, VLA）

2. **Component Integration**
   - 数据流架构设计
   - 质量门控策略
   - 系统级决策依赖关系

3. **Hardware-Software Mapping**
   - 采集硬件矩阵
   - 感知设备选型
   - 成本-性能权衡分析

## Out of Scope (Non-Goals)

1. **End-to-End Implementation Code** - 聚焦架构决策与设计，而非完整代码实现
2. **Real-time System Optimization** - 不涉及运行时优化细节
3. **Production Deployment Operations** - 不包括生产环境运维指南
4. **Deep Algorithm Tutorials** - 不替代原始论文和官方文档

## Target Audience

- **Robotics Engineers** - 寻找感知算法选型参考
- **Data Infrastructure Teams** - 设计数据采集与标注Pipeline
- **AI Researchers** - 了解VLA/World Model技术现状
- **Technical Leads** - 系统架构决策参考

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Coverage completeness | >90% | Status tracking via `docs/_machine/status.yaml` |
| Cross-reference accuracy | 100% | All internal links validated |
| Knowledge freshness | <6 months | Regular review of fast-moving topics |
| Decision support | Qualitative | User can make informed architectural choices |
