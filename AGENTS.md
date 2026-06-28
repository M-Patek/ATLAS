# AGENTS.md — ATLAS Agent Boot Protocol

> For Claude Code, Cursor, Copilot, Gemini CLI, and compatible AI agents.
> This file is auto-loaded on session start. Do not rename.
>
> **ATLAS** = Atlas of Technologies for Learning Autonomous Systems

---

## PROTOCOL

### START — Session Initialization

```
READ docs/INDEX.md
├── Quick Map → route to functional module
├── Available Spaces → note target space type
└── Experiment Framework → if running comparisons

IF touching space implementations:
    READ src/atlas/spaces/NN-*.py FOR EACH target:
    ├── compute_distance implementation
    ├── get_heuristic implementation  
    └── update_from_observation logic

IF task spans 3+ spaces:
    SPAWN Explore sub-agent
```

### EXEC — During Work

| Trigger | Action |
|---------|--------|
| Add new space | Update `src/atlas/spaces/__init__.py`, register with `@register_space` |
| Add new experiment | Place in `experiments/`, import in docs |
| Modify cross-space content | Note in CHANGELOG which spaces affected |
| Performance regression | Run `experiments/compare_spaces.py` to verify |

### EXIT — Session Termination

**Step 1: Classify**

| Change Pattern | Type | Checklist | Validate |
|---------------|------|-----------|----------|
| `.md`, `.txt`, typo | T1 | Nothing | V0 — skip |
| Single space update | T2 | CHANGELOG + test | V1 — consistency |
| New space implementation | T3 | CHANGELOG + STATUS + compare test | V2 |
| Framework change | T4 | CHANGELOG + STATUS + all spaces test | V3 |
| Cross-cutting change | T5 | FULL + all spaces update | V3 |

**Step 2: Execute Checklist**

| Checklist | Do |
|-----------|-----|
| LIGHT (T1) | Nothing |
| STANDARD (T2) | CHANGELOG + run tests |
| FULL (T3/T4) | CHANGELOG + STATUS + space tests |
| FULL+ (T5) | FULL + all spaces review + registry update |

**Step 3: CHANGELOG Entry**

Template:
```markdown
### Session — <summary>

- **Type**: T<N>
- **Goal**: <why>
- **Done**:
  - <change 1>
  - <change 2>
- **Files**: <paths>
- **Validation**: V<N> — <evidence>
- **Left for next time**: <if any>
```

---

## REFERENCE

### Navigation

| Need | Location |
|------|----------|
| Space implementations | `src/atlas/spaces/` |
| Core abstractions | `src/atlas/core/` |
| Experiment framework | `src/atlas/core/experiment.py` |
| Space registry | `src/atlas/core/registry.py` |
| Recent changes | `docs/changelog/CHANGELOG.md` |

### Facts

- **Type**: Cognitive Architecture Framework
- **Domain**: Embodied AI Navigation & Exploration
- **Architecture**: Pluggable Cognitive Spaces
- **Core Abstraction**: CognitiveSpace interface

### Forbidden

| Action | Why |
|--------|-----|
| Edit accepted ADRs | Immutable; supersede with new ADR |
| Hardcode space-specific logic in solver | Breaks pluggability |
| Skip CHANGELOG for new spaces | Documentation gap |
| Use "Phase" naming | Deprecated terminology |
| Skip validation tests for new space | No verification |

---

*ATLAS Agent Protocol v0.3* — Pluggable Architecture
