---
id: 04-continuous-spaces
title: 04 — Continuous Spaces
code_anchors:
  - src/spaces/continuous.py:ContinuousCognitiveSpace
  - src/spaces/continuous.py:ContinuousField
  - src/spaces/continuous_ssfr.py:ContinuousSSFR
---

# 04 — Continuous Spaces

**ATLAS独特贡献**：连续坐标认知空间，无网格限制。

## §0

稀疏采样 + kNN插值，支持物理坐标直接输入。

## §1

- **ContinuousField**: 稀疏采样场
- **kNN Interpolation**: 反距离加权
- **Path Integration**: 连续路径积分

## §2

| Anchor | Purpose |
|--------|---------|
| `continuous.py:ContinuousField` | 连续场容器 |
| `continuous.py:ContinuousRicciSpace` | 连续Ricci |
| `continuous_ssfr.py:ContinuousSSFR` | 连续SSFR |

## §5

| Aspect | Status |
|--------|--------|
| ContinuousField | Stable |
| Ricci continuous | Stable |
| SSFR continuous | Stable |

---
