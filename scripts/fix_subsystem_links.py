#!/usr/bin/env python3
"""
Fix broken links in subsystem files after migration.
"""

import re
from pathlib import Path

# Files and their link replacements
REPLACEMENTS = {
    '04-architecture-patterns.md': {
        '../05-integration/01-pipeline-patterns.md': '41-pipeline-patterns.md',
        '../05-integration/03-system-design.md': '43-system-design.md',
        '../05-integration/index.md': '../INDEX.md',
    },
    '31-ego-collection.md': {
        '../03-perception/01-stereo-imu.md': '21-stereo-imu.md',
    },
    '33-sim2real.md': {
        '../01-foundation/03-vla.md': '03-vla.md',
    },
    '41-pipeline-patterns.md': {
        '../04-data-ecosystem/index.md': '../INDEX.md',
        '../04-data-ecosystem/': '',  # Remove prefix
    },
    '42-quality-gates.md': {
        '../02-annotation/index.md': '../INDEX.md',
        '../02-annotation/': '',
    },
    '43-system-design.md': {
        '../04-data-ecosystem/index.md': '../INDEX.md',
        '../04-data-ecosystem/': '',
    },
    '21-stereo-imu.md': {
        '](index.md)': '](../INDEX.md)',
    },
    '26-sensor-fusion.md': {
        '](index.md)': '](../INDEX.md)',
    },
}


def fix_file(filepath: Path, replacements: dict):
    """Fix links in a single file."""
    content = filepath.read_text(encoding='utf-8')
    original = content

    for old, new in replacements.items():
        content = content.replace(old, new)

    if content != original:
        filepath.write_text(content, encoding='utf-8')
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes: {filepath}")


def main():
    """Main entry point."""
    subsystems_dir = Path('docs/subsystems')

    if not subsystems_dir.exists():
        print("Error: docs/subsystems/ directory not found")
        return

    for filename, replacements in REPLACEMENTS.items():
        filepath = subsystems_dir / filename
        if filepath.exists():
            fix_file(filepath, replacements)
        else:
            print(f"Warning: File not found: {filepath}")

    print("\nLink fixing complete!")


if __name__ == '__main__':
    main()
