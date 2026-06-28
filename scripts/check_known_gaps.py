#!/usr/bin/env python3
"""
Check for known gaps and inconsistencies in documentation.
"""

import yaml
import sys
from pathlib import Path

def check_status_yaml():
    """Check status.yaml for consistency."""
    issues = []
    status_file = Path('docs/_machine/status.yaml')

    if not status_file.exists():
        return ["status.yaml not found"]

    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            status = yaml.safe_load(f)

        # Check overall structure
        if 'layers' not in status:
            issues.append("status.yaml: Missing 'layers' key")

        # Check for completeness
        if 'overall_completeness' not in status:
            issues.append("status.yaml: Missing 'overall_completeness'")

    except yaml.YAMLError as e:
        issues.append(f"status.yaml: YAML parse error - {e}")

    return issues

def check_subsystem_docs():
    """Check that subsystem docs exist."""
    issues = []
    subsystems_dir = Path('docs/subsystems')

    if not subsystems_dir.exists():
        return ["docs/subsystems/ directory not found"]

    # Check for expected files based on status.yaml
    status_file = Path('docs/_machine/status.yaml')
    if status_file.exists():
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = yaml.safe_load(f)

            for layer_id, layer_info in status.get('layers', {}).items():
                for topic_id, topic_info in layer_info.get('topics', {}).items():
                    expected_file = topic_info.get('file', '')
                    if expected_file:
                        expected_path = subsystems_dir / expected_file
                        if not expected_path.exists():
                            issues.append(f"Missing subsystem doc: {expected_path}")

        except Exception as e:
            issues.append(f"Error checking subsystem docs: {e}")

    return issues

def main():
    """Main entry point."""
    print("Checking for known gaps...")

    all_issues = []
    all_issues.extend(check_status_yaml())
    all_issues.extend(check_subsystem_docs())

    if all_issues:
        print(f"\nFound {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("\n✓ No gaps found!")
        sys.exit(0)

if __name__ == '__main__':
    main()
