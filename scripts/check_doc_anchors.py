#!/usr/bin/env python3
"""
Check for documentation anchors drift.
Run after code changes to ensure doc references are still valid.
"""

import os
import re
import sys
from pathlib import Path

def check_anchors(filepath: Path) -> list:
    """Check anchor references in a file."""
    issues = []
    content = filepath.read_text(encoding='utf-8')

    # Find all anchor links like [text](file.md#anchor)
    anchor_pattern = r'\[([^\]]+)\]\(([^)]+)#([^)]+)\)'

    for match in re.finditer(anchor_pattern, content):
        link_text, target_file, anchor = match.groups()
        target_path = filepath.parent / target_file

        if target_path.exists():
            target_content = target_path.read_text(encoding='utf-8')
            # Check if anchor exists (heading with that id)
            anchor_heading = f'id: {anchor}'
            anchor_heading2 = f'#{anchor}'

            if anchor_heading not in target_content and anchor_heading2 not in target_content:
                # Check for markdown heading with that text
                heading_pattern = rf'^#+ .*{re.escape(anchor.replace("-", " "))}'
                if not re.search(heading_pattern, target_content, re.MULTILINE | re.IGNORECASE):
                    issues.append(f"{filepath}: Anchor '#{anchor}' not found in {target_file}")

    return issues

def main():
    """Main entry point."""
    docs_dir = Path('docs')
    if not docs_dir.exists():
        print("Error: docs/ directory not found")
        sys.exit(1)

    all_issues = []

    for md_file in docs_dir.rglob('*.md'):
        issues = check_anchors(md_file)
        all_issues.extend(issues)

    if all_issues:
        print(f"Found {len(all_issues)} anchor issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("✓ All anchors valid!")
        sys.exit(0)

if __name__ == '__main__':
    main()
