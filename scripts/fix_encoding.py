#!/usr/bin/env python3
"""
Fix encoding issues in subsystem files by removing BOM and rewriting.
"""

from pathlib import Path

BOM_FILES = [
    '04-architecture-patterns.md',
    '21-stereo-imu.md',
    '26-sensor-fusion.md',
    '31-ego-collection.md',
    '33-sim2real.md',
    '41-pipeline-patterns.md',
    '42-quality-gates.md',
    '43-system-design.md',
]


def fix_encoding(filepath: Path):
    """Fix file encoding by removing BOM if present."""
    content = filepath.read_bytes()

    # Remove UTF-8 BOM if present
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
        filepath.write_bytes(content)
        print(f"Fixed BOM: {filepath}")
    else:
        print(f"No BOM: {filepath}")


def main():
    """Main entry point."""
    subsystems_dir = Path('docs/subsystems')

    for filename in BOM_FILES:
        filepath = subsystems_dir / filename
        if filepath.exists():
            fix_encoding(filepath)
        else:
            print(f"Not found: {filepath}")

    print("\nEncoding fix complete!")


if __name__ == '__main__':
    main()
