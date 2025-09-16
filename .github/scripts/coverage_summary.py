#!/usr/bin/env python3
"""Generate coverage summary from XML files and check thresholds."""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_coverage_from_xml(xml_file: str) -> float:
    """Extract coverage percentage from XML file.

    Args:
        xml_file: Path to coverage XML file

    Returns:
        Coverage percentage (0.0-100.0)
    """
    try:
        if not Path(xml_file).exists():
            return 0.0

        tree = ET.parse(xml_file)
        root = tree.getroot()
        line_rate = float(root.get("line-rate", 0))
        return line_rate * 100
    except Exception:
        return 0.0


def main():
    """Generate coverage summary and check thresholds."""
    parser = argparse.ArgumentParser(description="Generate coverage summary report")
    parser.add_argument("--unit-file", help="Unit test coverage XML file")
    parser.add_argument(
        "--unit-threshold", type=float, default=80, help="Unit test coverage threshold"
    )
    parser.add_argument("--integration-file", help="Integration test coverage XML file")
    parser.add_argument(
        "--integration-threshold",
        type=float,
        default=60,
        help="Integration test coverage threshold",
    )

    args = parser.parse_args()

    print("📈 Coverage Summary Report:")
    print("==================================")

    unit_pass = True
    integration_pass = True

    # Check unit test coverage
    if args.unit_file:
        unit_cov = get_coverage_from_xml(args.unit_file)
        print(f"🧪 Unit Test Coverage: {unit_cov:.1f}% (target: {args.unit_threshold}%)")

        if unit_cov >= args.unit_threshold:
            print("✅ Unit coverage meets threshold")
        else:
            print("❌ Unit coverage below threshold")
            unit_pass = False

        # Set environment variable for GitHub Actions
        print(f"UNIT_COVERAGE_PASS={'true' if unit_pass else 'false'}")

    # Check integration test coverage
    if args.integration_file:
        integration_cov = get_coverage_from_xml(args.integration_file)
        target = args.integration_threshold
        print(f"🔄 Integration Test Coverage: {integration_cov:.1f}% (target: {target}%)")

        if integration_cov >= args.integration_threshold:
            print("✅ Integration coverage meets threshold")
        else:
            print("❌ Integration coverage below threshold")
            integration_pass = False

        # Set environment variable for GitHub Actions
        print(f"INTEGRATION_COVERAGE_PASS={'true' if integration_pass else 'false'}")

    print("==================================")

    # Exit with success if both pass (or if files don't exist)
    sys.exit(0 if (unit_pass and integration_pass) else 1)


if __name__ == "__main__":
    main()
