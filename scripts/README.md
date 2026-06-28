# Scripts

This directory contains automation and validation scripts for ATLAS.

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `check_doc_consistency.py` | Validate frontmatter and links | `python scripts/check_doc_consistency.py` |
| `check_doc_anchors.py` | Check documentation anchor references | `python scripts/check_doc_anchors.py` |
| `check_known_gaps.py` | Find missing docs vs status.yaml | `python scripts/check_known_gaps.py` |
| `generate_reports.py` | Generate status reports | `python scripts/generate_reports.py tech-debt` |

## Development

When adding new scripts:
1. Use Python 3 with type hints
2. Add docstring with purpose
3. Return exit code 0 on success, 1 on failure
4. Update this README
