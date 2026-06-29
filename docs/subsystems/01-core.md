---
id: 01-core
title: 01 — Core
code_anchors:
  - src/core/space.py:CognitiveSpace
  - src/core/registry.py:create_space
  - src/core/solver.py:GeodesicSolver
---

# 01 — Core

ATLAS 核心框架：认知空间抽象、注册表、求解器。

## §0

可插拔认知架构的基础契约。

## §1

- **CognitiveSpace**: 空间抽象基类
- **Registry**: 空间注册与发现
- **GeodesicSolver**: A*测地线求解

## §2

| Anchor | Purpose |
|--------|---------|
| `space.py:CognitiveSpace` | 抽象基类，所有空间必须继承 |
| `registry.py:create_space` | 工厂函数，动态创建空间 |
| `solver.py:GeodesicSolver` | A*求解器 |

## §5

| Aspect | Status |
|--------|--------|
| Core abstraction | Stable |
| Registration | Stable |
| Solver | Stable |

---
