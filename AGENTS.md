# ATLAS Agent Protocol

Thin import of DVAS `@AGENTS.md` protocol adapted for knowledge base.

## Quick Start

1. **Read**: `llms.txt` → `docs/INDEX.md` → target subsystem
2. **Navigate**: `docs/<NN>-<layer>/index.md` for layer overview
3. **Deep dive**: `docs/<layer>/<NN>-<topic>.md` for specific topic
4. **Status**: `docs/_machine/status.yaml`

## Directory Convention

```
atlas/
├── AGENTS.md              # This file
├── llms.txt               # Quick navigation for LLMs
├── docs/
│   ├── INDEX.md           # Doc entry point
│   ├── _machine/          # Status & planning
│   ├── 01-foundation/     # Layer 1: Models
│   ├── 02-annotation/     # Layer 2: Annotation
│   ├── 03-perception/     # Layer 3: Algorithms
│   ├── 04-data-ecosystem/ # Layer 4: Data & Hardware
│   └── 05-integration/    # Layer 5: System Integration
└── references/            # Papers, datasets, tools
```

## File Naming

- **Folders**: `NN-name/` — zero-padded number for ordering
- **Files**: `NN-topic-name.md` — kebab-case, machine-friendly
- **Index**: Each layer has `index.md` as entry point

## Cross-Reference Format

Use relative paths for Agent traceability:

```markdown
See [VLA architecture](../01-foundation/03-vla.md#architecture)
Input from [Ego collection](../04-data-ecosystem/01-ego-collection.md)
```

## Document Structure

Every topic document includes YAML frontmatter:

```yaml
---
id: unique-topic-id
title: "Human-readable title"
status: draft | in-progress | complete
complexity: low | medium | high
related:
  - "../other-layer/topic.md"
prerequisites:
  - "Concept name"
---
```

## Agent Hints

When processing ATLAS documents:

- **STATUS MATTERS**: Check `status:` field before citing
- **FOLLOW RELATED**: Use `related:` links for cross-layer understanding
- **LAYER FLOW**: 04-data → 03-perception → 02-annotation → 01-foundation → 05-integration

## Layer Responsibilities

| Layer | Contents | When to Use |
|-------|----------|-------------|
| 01-foundation | VLM, World Model, VLA | Model selection, architecture decisions |
| 02-annotation | Schema design, standards | Building annotation pipeline |
| 03-perception | Stereo+IMU, Depth, SLAM, Hand Pose | Implementing perception stack |
| 04-data-ecosystem | Ego, UMI, Sim2Real, Teleop, Hardware | Choosing data collection setup |
| 05-integration | Pipeline patterns, quality gates | System-level design |

---

*ATLAS Agent Protocol v0.1*
