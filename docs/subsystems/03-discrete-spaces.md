---
id: 03-discrete-spaces
title: 03 — Discrete Spaces
code_anchors:
  - src/spaces/ricci.py:RicciSpace
  - src/spaces/fisher.py:FisherSpace
  - src/spaces/conformal.py:ConformalSpace
  - src/spaces/composite.py:ProductSpace
---

# 03 — Discrete Spaces

离散网格认知空间：Ricci、Fisher、Conformal、Euclidean、Finsler、Wasserstein及复合空间。

## §0

网格世界中的认知空间实现。

## §1

| Space | Key Feature |
|-------|-------------|
| Ricci | 曲率 = 不确定度Laplacian |
| Fisher | 1/置信度 度量 |
| Conformal | 共形因子调整 |
| Euclidean | 基线欧氏距离 |
| Finsler | 非对称度量 |
| Wasserstein | 最优传输成本 |
| Composite | 加权组合 |

## §2

| Anchor | Purpose |
|--------|---------|
| `create_space('ricci', w, h)` | Ricci空间 |
| `create_space('fisher', w, h)` | Fisher空间 |
| `create_space('product', ...)` | 乘积空间 |

## §5

| Aspect | Status |
|--------|--------|
| Ricci | Stable |
| Fisher | Stable |
| Conformal | Stable |
| Composite | Stable |

---
