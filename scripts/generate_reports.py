#!/usr/bin/env python3
"""
Generate reports on documentation status.
"""

import yaml
import sys
from pathlib import Path

def generate_tech_debt_report():
    """Generate tech debt report from status.yaml."""
    status_file = Path('docs/_machine/status.yaml')

    if not status_file.exists():
        print("Error: status.yaml not found")
        return

    with open(status_file, 'r', encoding='utf-8') as f:
        status = yaml.safe_load(f)

    print("=" * 70)
    print("ATLAS Tech Debt Report")
    print("=" * 70)
    print()

    # Overall stats
    total_topics = 0
    complete_topics = 0

    for layer_id, layer_info in status.get('layers', {}).items():
        layer_complete = 0
        layer_total = 0

        for topic_id, topic_info in layer_info.get('topics', {}).items():
            total_topics += 1
            layer_total += 1
            if topic_info.get('status') == 'complete':
                complete_topics += 1
                layer_complete += 1

        percentage = (layer_complete / layer_total * 100) if layer_total > 0 else 0
        print(f"{layer_id}: {layer_complete}/{layer_total} ({percentage:.1f}%)")

    print()
    print(f"Overall: {complete_topics}/{total_topics} ({complete_topics/total_topics*100:.1f}%)")
    print()

    # Tech debt items
    tech_debt = status.get('tech_debt', [])
    if tech_debt:
        print("Active Tech Debt Items:")
        for item in tech_debt:
            print(f"  - {item}")
    else:
        print("No tech debt items recorded.")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_reports.py [report-type]")
        print("  report-type: tech-debt")
        sys.exit(1)

    report_type = sys.argv[1]

    if report_type == 'tech-debt':
        generate_tech_debt_report()
    else:
        print(f"Unknown report type: {report_type}")
        sys.exit(1)

if __name__ == '__main__':
    main()
