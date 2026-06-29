---
id: adr-0001
title: Flat Package Structure Decision
deciders: ATLAS Team
consulted: Claude Code, Architecture Guidelines
date: 2026-06-29
status: accepted
last_validated: 2026-06-29
supersedes: null
tags: [adr, architecture, packaging]
---

# 0001. Flat Package Structure

## Context

原始 ATLAS 结构使用嵌套的 `src/atlas/` 包结构：
```
src/
└── atlas/
    ├── core/
    ├── spaces/
    ├── kitchen/
```

这种结构存在以下问题：
1. **冗余**: 仓库名已经是 ATLAS，`src/atlas` 重复命名
2. **导入冗长**: `from atlas.core import ...` 比 `from src.core import ...` 更容易混淆
3. **研究型仓库不需要发布**: 作为研究框架，不需要 PyPI 发布，扁平结构更简单

## Decision

采用扁平包结构：
```
src/
├── core/         # 原来是 src/atlas/core/
├── spaces/       # 原来是 src/atlas/spaces/
├── kitchen/      # 原来是 src/atlas/kitchen/
└── ...
```

并更新 `pyproject.toml`：
```toml
[tool.setuptools.packages.find]
where = ["src"]
```

## Consequences

### Positive

- 导入路径更直观: `from src.core import ...`
- 目录结构更扁平，减少导航深度
- 与研究型代码风格一致（快速原型）
- 避免命名空间包复杂性

### Negative

- 与 Python 包命名惯例不完全一致（通常用项目名作为包名）
- 如果未来要发布到 PyPI，需要重构回标准结构

### Neutral

- 测试需要更新 import 路径
- 文档中的所有代码示例需要更新

## Alternatives Considered

### Keep nested structure (src/atlas/)

**Rejected**: 增加了不必要的嵌套层级，对于研究型仓库没有实际收益。

### Use flat structure with package renaming (atlas_core/)

**Rejected**: 虽然解决了命名重复问题，但仍然增加了记忆负担。`src` 明确指出代码位置，足够清晰。

## References

- [New docs/subsystems/10-core-space.md](../subsystems/10-core-space.md)
- [Updated docs/INDEX.md](../INDEX.md)
- [Migration commit](https://github.com/atlas/research)
