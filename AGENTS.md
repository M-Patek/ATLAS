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
├── NAVIGATE → route to subsystem
├── SUBSYSTEMS → note target ID
└── READING PATHS → if specific task

IF touching subsystem(s):
    READ docs/subsystems/NN-*.md FOR EACH ID:
    ├── frontmatter.code_anchors → entry points
    ├── frontmatter.agent_hints → known gotchas
    ├── §0 or §1 → concepts
    └── §2+ → ONLY IF task requires depth

    APPLY agent_hints protocol (see EXEC section)

IF task spans 3+ subsystems:
    SPAWN Explore sub-agent

IF implementation task:
    READ code_anchors in relevant subsystem docs
    VERIFY code matches doc description
```

### EXEC — During Work

**Hints Protocol**: After reading subsystem doc, check `frontmatter.agent_hints`:
- `WARNING: <message>` → Display the message, then continue with root AGENTS.md rules
- `OVERRIDE: <rule> → <description>` → Apply the override instead of root rule for this subsystem

| Trigger | Action |
|---------|--------|
| Add new space | Update `src/spaces/__init__.py`, register with `@register_space` |
| Add new experiment | Place in `experiments/`, reference in docs |
| Modify cross-space content | Note in CHANGELOG which spaces affected, run benchmarks |
| Performance regression | Run `experiments/benchmarks/perf_benchmark.py` to verify |
| Doc vs code conflict | Trust code, update doc |

### EXIT — Session Termination

**Step 1: Classify**

| Change Pattern | Type | Checklist | Validate |
|---------------|------|-----------|----------|
| `.md`, `.txt`, typo, format | T1 | Nothing | V0 — skip |
| Rename/extract within space | T2 | CHANGELOG + test | V1 — unit tests |
| Fix bug, edge case | T3 | CHANGELOG + doc update + test | V2 — integration tests |
| New space implementation | T4 | CHANGELOG + subsystem doc + test | V2 — integration tests |
| Core framework change | T5 | CHANGELOG + doc update + all tests | V3 — full benchmarks |
| Space interface change | T6 | FULL + migration guide + ADR | V3 — full review |

**Step 2: Execute Checklist**

| Checklist | Do |
|-----------|-----|
| LIGHT (T1) | Nothing |
| STANDARD (T2) | CHANGELOG + run tests |
| FULL (T3/T4) | CHANGELOG + subsystem doc `last_validated` update + tests |
| FULL+ (T5) | FULL + all spaces test + INDEX.md update |
| FULL++ (T6) | FULL+ + ADR + migration guide |

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

**Step 4: Documentation Check** (for T3+)

```bash
# Run doc checks
python scripts/check_doc_consistency.py
python scripts/check_doc_anchors.py
```

---

## REFERENCE

### Navigation

| Need | Location |
|------|----------|
| Subsystem routing | `docs/INDEX.md` |
| Subsystem details | `docs/subsystems/NN-*.md` |
| Architecture decisions | `docs/adr/NNNN-*.md` |
| Space implementations | `src/spaces/` |
| Core abstractions | `src/core/` |
| Experiment framework | `src/core/experiment.py` |
| Space registry | `src/core/registry.py` |
| Recent changes | `docs/changelog/CHANGELOG.md` |
| Design rules | `docs/architecture/constitution.md` |

### Facts

- **Type**: Research Framework (Cognitive Architecture)
- **Domain**: Embodied AI Navigation & Exploration
- **Architecture**: Pluggable Cognitive Spaces
- **Core Abstraction**: `CognitiveSpace` interface
- **Package Structure**: Flat (`src/core/`, `src/spaces/`)
- **Doc Validation**: V0=skip / V1=unit / V2=integration / V3=benchmarks

### Forbidden

| Action | Why |
|--------|-----|
| Edit accepted ADRs | Immutable; supersede with new ADR |
| Hardcode space-specific logic in solver | Breaks pluggability |
| Skip CHANGELOG for new spaces | Documentation gap |
| Use "Phase" naming | Deprecated terminology |
| Skip validation tests for new space | No verification |
| Add absolute links in docs | Breaks forks and moves |
| Mark doc `stable` without validation | Misleading status |

### Subsystem Index (1-7)

| Num | ID | Name | Code | Doc |
|-----|----|------|------|-----|
| 1 | 01-core | Core | `src/core/` | [01-core.md](docs/subsystems/01-core.md) |
| 2 | 02-ssfr | SSFR | `src/core/ssfr*.py` | [02-ssfr.md](docs/subsystems/02-ssfr.md) |
| 3 | 03-discrete-spaces | Discrete Spaces | `src/spaces/` | [03-discrete-spaces.md](docs/subsystems/03-discrete-spaces.md) |
| 4 | 04-continuous-spaces | Continuous Spaces | `src/spaces/continuous*.py` | [04-continuous-spaces.md](docs/subsystems/04-continuous-spaces.md) |
| 5 | 05-environment | Environment | `src/kitchen/`, `src/visualization/` | [05-environment.md](docs/subsystems/05-environment.md) |
| 6 | 06-learning | Learning | `src/learning/` | [06-learning.md](docs/subsystems/06-learning.md) |
| 7 | 07-research | Research | `src/research/` | [07-research.md](docs/subsystems/07-research.md) |

---

*ATLAS Agent Protocol v0.4* — Research Architecture
