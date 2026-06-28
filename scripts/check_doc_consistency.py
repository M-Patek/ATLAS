#!/usr/bin/env python3
"""
Check documentation consistency:
- All .md files have required frontmatter
- Internal links are valid
- No broken cross-references
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

REQUIRED_FIELDS = {'id', 'title', 'status', 'last_validated'}
VALID_STATUSES = {'draft', 'review', 'stable', 'deprecated', 'complete', 'in-progress', 'not-started', 'proposed', 'template'}

def extract_frontmatter(content: str) -> Tuple[dict, str]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith('---'):
        return {}, content

    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    # Simple YAML parsing (sufficient for our needs)
    frontmatter = {}
    for line in frontmatter_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            frontmatter[key.strip()] = value.strip().strip('"\'')

    return frontmatter, body

def check_file(filepath: Path) -> List[str]:
    """Check a single file for consistency issues."""
    errors = []
    content = filepath.read_text(encoding='utf-8')

    frontmatter, body = extract_frontmatter(content)

    # Check required fields
    missing = REQUIRED_FIELDS - set(frontmatter.keys())
    if missing:
        errors.append(f"{filepath}: Missing frontmatter fields: {missing}")

    # Check status validity
    if 'status' in frontmatter:
        if frontmatter['status'] not in VALID_STATUSES:
            errors.append(f"{filepath}: Invalid status '{frontmatter['status']}'")

    # Check for internal links (basic validation)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    for match in re.finditer(link_pattern, content):
        link_text, link_target = match.groups()
        if not link_target.startswith(('http://', 'https://', '#')):
            # Relative link - check if it exists
            target_path = filepath.parent / link_target.split('#')[0]
            if not target_path.exists() and not str(link_target).startswith('mailto:'):
                errors.append(f"{filepath}: Broken link to '{link_target}'")

    return errors

def main():
    """Main entry point."""
    docs_dir = Path('docs')
    if not docs_dir.exists():
        print("Error: docs/ directory not found")
        sys.exit(1)

    all_errors = []
    md_files = list(docs_dir.rglob('*.md'))

    print(f"Checking {len(md_files)} markdown files...")

    for md_file in md_files:
        errors = check_file(md_file)
        all_errors.extend(errors)

    if all_errors:
        print(f"\nFound {len(all_errors)} issues:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\n✓ All checks passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
