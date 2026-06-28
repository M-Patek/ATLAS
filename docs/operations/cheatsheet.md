---
id: ops-cheatsheet
title: ATLAS Operations Cheatsheet
status: stable
last_validated: 2026-06-27
tags: [operations, commands, quick-reference]
---

# ATLAS Operations Cheatsheet

## Documentation Validation

```bash
# Check document consistency
python scripts/check_doc_consistency.py

# Check documentation anchors (after code changes)
python scripts/check_doc_anchors.py

# Check for known gaps
python scripts/check_known_gaps.py
```

## Status Reports

```bash
# Generate tech debt report
python scripts/generate_reports.py tech-debt

# Get subsystem status
python scripts/get_subsystem_status.py
```

## Changelog Maintenance

```bash
# Promote changelog entries (dry run)
python scripts/promote_changelog.py --dry-run

# Promote changelog entries (apply)
python scripts/promote_changelog.py
```

## Common Editing Tasks

### Adding a New Topic

1. Choose appropriate layer (01-05)
2. Create file: `docs/subsystems/NN-topic-name.md`
3. Copy frontmatter from template
4. Update `docs/_machine/status.yaml`
5. Run validation: `python scripts/check_doc_consistency.py`

### Updating Status

1. Edit `docs/_machine/status.yaml`
2. Update `last_updated` timestamp
3. Run validation

### Cross-Referencing

Always use relative paths:
```markdown
See [VLA architecture](../subsystems/03-vla.md)
Input from [Ego collection](../subsystems/31-ego-collection.md)
```
