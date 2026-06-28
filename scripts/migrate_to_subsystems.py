#!/usr/bin/env python3
"""
Migrate docs from 01-05 layer structure to subsystems/ directory.
Updates frontmatter and internal references.
"""

import re
import shutil
from pathlib import Path

# Mapping: source file -> target file in subsystems/
MIGRATION_MAP = {
    # 01-foundation
    'docs/01-foundation/01-vlm.md': 'docs/subsystems/01-vlm.md',
    'docs/01-foundation/02-world-model.md': 'docs/subsystems/02-world-model.md',
    'docs/01-foundation/03-vla.md': 'docs/subsystems/03-vla.md',
    'docs/01-foundation/04-architecture-patterns.md': 'docs/subsystems/04-architecture-patterns.md',

    # 02-annotation
    'docs/02-annotation/01-schema-design.md': 'docs/subsystems/11-schema-design.md',
    'docs/02-annotation/02-action-annotation.md': 'docs/subsystems/12-action-annotation.md',
    'docs/02-annotation/03-scene-annotation.md': 'docs/subsystems/13-scene-annotation.md',
    'docs/02-annotation/04-physics-annotation.md': 'docs/subsystems/14-physics-annotation.md',

    # 03-perception
    'docs/03-perception/01-stereo-imu.md': 'docs/subsystems/21-stereo-imu.md',
    'docs/03-perception/02-depth-estimation.md': 'docs/subsystems/22-depth-estimation.md',
    'docs/03-perception/03-slam.md': 'docs/subsystems/23-slam.md',
    'docs/03-perception/04-hand-pose.md': 'docs/subsystems/24-hand-pose.md',
    'docs/03-perception/05-3d-reconstruction.md': 'docs/subsystems/25-3d-reconstruction.md',
    'docs/03-perception/06-sensor-fusion.md': 'docs/subsystems/26-sensor-fusion.md',

    # 04-data-ecosystem
    'docs/04-data-ecosystem/01-ego-collection.md': 'docs/subsystems/31-ego-collection.md',
    'docs/04-data-ecosystem/02-umi-systems.md': 'docs/subsystems/32-umi-systems.md',
    'docs/04-data-ecosystem/03-sim2real.md': 'docs/subsystems/33-sim2real.md',
    'docs/04-data-ecosystem/04-teleoperation.md': 'docs/subsystems/34-teleoperation.md',
    'docs/04-data-ecosystem/05-hardware-matrix.md': 'docs/subsystems/35-hardware-matrix.md',
    'docs/04-data-ecosystem/06-data-formats.md': 'docs/subsystems/36-data-formats.md',

    # 05-integration
    'docs/05-integration/01-pipeline-patterns.md': 'docs/subsystems/41-pipeline-patterns.md',
    'docs/05-integration/02-quality-gates.md': 'docs/subsystems/42-quality-gates.md',
    'docs/05-integration/03-system-design.md': 'docs/subsystems/43-system-design.md',
}

# Reference update mapping (old path -> new path)
REF_UPDATES = {
    '../01-foundation/': '../subsystems/',
    '../02-annotation/': '../subsystems/',
    '../03-perception/': '../subsystems/',
    '../04-data-ecosystem/': '../subsystems/',
    '../05-integration/': '../subsystems/',
    '01-vlm.md': '01-vlm.md',
    '02-world-model.md': '02-world-model.md',
    '03-vla.md': '03-vla.md',
    '04-architecture-patterns.md': '04-architecture-patterns.md',
    '01-schema-design.md': '11-schema-design.md',
    '02-action-annotation.md': '12-action-annotation.md',
    '03-scene-annotation.md': '13-scene-annotation.md',
    '04-physics-annotation.md': '14-physics-annotation.md',
    '01-stereo-imu.md': '21-stereo-imu.md',
    '02-depth-estimation.md': '22-depth-estimation.md',
    '03-slam.md': '23-slam.md',
    '04-hand-pose.md': '24-hand-pose.md',
    '05-3d-reconstruction.md': '25-3d-reconstruction.md',
    '06-sensor-fusion.md': '26-sensor-fusion.md',
    '01-ego-collection.md': '31-ego-collection.md',
    '02-umi-systems.md': '32-umi-systems.md',
    '03-sim2real.md': '33-sim2real.md',
    '04-teleoperation.md': '34-teleoperation.md',
    '05-hardware-matrix.md': '35-hardware-matrix.md',
    '06-data-formats.md': '36-data-formats.md',
    '01-pipeline-patterns.md': '41-pipeline-patterns.md',
    '02-quality-gates.md': '42-quality-gates.md',
    '03-system-design.md': '43-system-design.md',
}


def update_references(content: str) -> str:
    """Update internal references in content."""
    for old, new in REF_UPDATES.items():
        # Update markdown links [text](old)
        content = content.replace(f']({old})', f']({new})')
        content = content.replace(f'"{old}"', f'"{new}"')
    return content


def add_last_validated(frontmatter: str) -> str:
    """Add last_validated field if missing."""
    if 'last_validated:' not in frontmatter:
        # Add after status line
        frontmatter = frontmatter.rstrip() + '\nlast_validated: 2026-06-27'
    return frontmatter


def migrate_file(src: Path, dst: Path):
    """Migrate a single file."""
    print(f"Migrating: {src} -> {dst}")

    content = src.read_text(encoding='utf-8')

    # Extract and potentially update frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2]

            # Add last_validated if missing
            frontmatter = add_last_validated(frontmatter)

            # Reconstruct
            content = f'---{frontmatter}\n---{body}'

    # Update references
    content = update_references(content)

    # Ensure directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Write to destination
    dst.write_text(content, encoding='utf-8')


def main():
    """Main entry point."""
    base_path = Path('docs').parent if not Path('docs').exists() else Path('.')

    if not (base_path / 'docs').exists():
        print("Error: docs/ directory not found")
        return

    for src_rel, dst_rel in MIGRATION_MAP.items():
        src = base_path / src_rel
        dst = base_path / dst_rel

        if src.exists():
            migrate_file(src, dst)
        else:
            print(f"Warning: Source not found: {src}")

    print("\nMigration complete!")
    print("Next steps:")
    print("1. Review migrated files")
    print("2. Update docs/INDEX.md")
    print("3. Run: python scripts/check_doc_consistency.py")


if __name__ == '__main__':
    main()
