---
id: adr-0002
title: Research Repository Architecture
deciders: ATLAS Team
consulted: Architecture Guidelines, Claude Code
date: 2026-06-29
status: accepted
last_validated: 2026-06-29
supersedes: null
tags: [adr, architecture, research]
---

# 0002. Research Repository Architecture

## Context

ATLAS 是一个**研究型仓库**，与生产型软件有本质区别：

| 维度 | Production | Research |
|------|------------|----------|
| **Goal** | 稳定运行，服务用户 | 探索假设，验证理论 |
| **Code churn** | 低，谨慎变更 | 高，快速迭代 |
| **Documentation** | 精确，版本锁定 | 概念性，定期重写 |
| **Testing** | 全面覆盖，CI/CD | 核心功能，示例驱动 |
| **Dependencies** | 严格锁定 | 灵活，最新可用 |

需要一套适合研究型仓库的架构文档和管理方法。

## Decision

采用适应性简化版的 Architecture 文档规范：

### 1. 保留核心文档结构
```
docs/
├── INDEX.md              # 导航中心
├── architecture/         # 设计约束
│   ├── positioning.md    # Scope/Non-goals
│   ├── constitution.md   # Ironclad rules
│   └── top-level-design.md
├── subsystems/           # 子系统文档
│   ├── NN-template.md
│   ├── 10-core-space.md
│   └── ...
├── adr/                  # 架构决策
│   └── NNNN-*.md
└── changelog/
    └── CHANGELOG.md
```

### 2. 简化工具链

| Production | Research (ATLAS) | Rationale |
|------------|------------------|-----------|
| Full status.yaml tracking | Manual status in docs | Research moves fast, automation overhead not worth it |
| strict doc-consistency checking | Basic frontmatter check | Core structure only |
| E2E test requirements | Unit + Integration sufficient | Physics sim is E2E enough |
| Full gap tracking | TODO scanning | Informative, not blocking |

### 3. 子系统编号方案

采用语义化编号而非层级编号：
```
10 — Core Framework      # 核心框架
20 — Cognitive Spaces    # 认知空间实现
30 — Extensions          # 扩展模块
40 — Research Tools      # 研究工具
```

对比原始 VLA 架构的层级编号（01-04 Foundation, 11-14 Annotation...），语义化编号更适合 ATLAS 的扁平架构。

### 4. Exit Protocol 适配

保留 T-level 分类，但简化验证级别：
- V0: None (docs only)
- V1: Unit tests
- V2: Integration tests + basic benchmark
- V3: Full benchmark + manual review

去掉了 production 需要的 E2E 测试要求。

## Consequences

### Positive

- 文档结构与业界最佳实践对齐
- 适合快速迭代的轻量级流程
- 保留了架构决策的可追溯性
- 为潜在的"毕业"到 production 做好准备

### Negative

- 仍需要维护文档（对于研究代码来说可能是负担）
- 协作时需要团队成员理解这套规范

### Neutral

- 工具链不完整（没有生成 status.yaml 的自动化）
- 可以根据研究阶段动态调整严格程度

## Alternatives Considered

### Free-form documentation (README only)

**Rejected**: Research code also needs structure, especially when:
- Collaborating with other researchers
- Returning to code after 6 months
- Explaining the approach to reviewers

### Full production-grade architecture

**Rejected**: Overkill for research. Would slow down experimentation with:
- Strict CI/CD requirements
- Mandatory E2E tests
- Comprehensive doc coverage requirements

## References

- [docs/architecture/positioning.md](../architecture/positioning.md)
- [docs/architecture/constitution.md](../architecture/constitution.md)
- [Architecture Guidelines (external)](https://github.com/example/architecture)
