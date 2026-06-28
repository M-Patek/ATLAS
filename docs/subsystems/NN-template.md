---
id: subsystem-template
title: Subsystem Template
status: draft
last_validated: 2026-06-27
tags: [template, subsystem]
code_anchors:
  - "src/path/to/entry_point.py"
agent_hints:
  - "WARNING: This is a template file - do not modify directly"
  - "Copy this file to create new subsystem documentation"
---

# {{SUBSYSTEM_NAME}}

> Brief description of what this subsystem does.

## Overview

### Purpose
What problem does this subsystem solve?

### Scope
- **In scope**: {{what is covered}}
- **Out of scope**: {{what is not covered}}

### Key Concepts

1. **Concept 1**
   - Description and significance

2. **Concept 2**
   - Description and significance

## Architecture

```
[Component Diagram]
```

### Entry Points
- `code_anchor:1` — Primary interface

## Dependencies

| Upstream | Interface | Purpose |
|----------|-----------|---------|
| {{upstream}} | {{interface}} | {{purpose}} |

| Downstream | Interface | Purpose |
|------------|-----------|---------|
| {{downstream}} | {{interface}} | {{purpose}} |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| {{param}} | {{default}} | {{description}} |

## Operations

### Common Tasks

```bash
# Task 1
command --option

# Task 2
command --option
```

### Monitoring

What to check when things go wrong:
1. Check {{metric/log}}
2. Verify {{connection}}
3. Review {{dashboard}}

## References

- [Upstream system](link.md)
- [Related ADR](../adr/NNNN-decision.md)
- External: [link](https://example.com)

---

*Status: {{status}} | Last updated: {{date}}*
