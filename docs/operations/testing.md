---
id: ops-testing
title: ATLAS Testing Guide
status: draft
last_validated: 2026-06-27
tags: [testing, validation, quality]
---

# ATLAS Testing Guide

> Testing strategies for validating document accuracy and link integrity.

## Document Consistency Tests

### Link Validation

All internal links are validated by `scripts/check_doc_consistency.py`:
- Relative paths must resolve
- Anchors (`#section`) must exist
- No circular references

### Frontmatter Validation

Every `.md` file under `docs/` must include:
```yaml
---
id: unique-identifier
title: "Human-readable title"
status: draft|stable|deprecated
last_validated: YYYY-MM-DD
tags: [tag1, tag2]
---
```

## Cross-Reference Tests

### Code Anchor Validation

When subsystem documents reference code files:
- Line numbers are checked against actual files
- Function/class names are verified to exist
- Warnings issued for outdated anchors

## Status Consistency Tests

### status.yaml Synchronization

- All subsystem docs must have entry in `status.yaml`
- Status in YAML must match document frontmatter
- Completeness percentages must be calculable
