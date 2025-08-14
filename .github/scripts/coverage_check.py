#!/usr/bin/env python3
"""Check coverage against quality gate threshold."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check_coverage(
    coverage_file: str, threshold: float, format: str = "auto"
) -> tuple[float, bool]:
    """Check if coverage meets the threshold.

    Args:
        coverage_file: Path to coverage file (JSON or XML)
        threshold: Required coverage percentage
        format: File format ('json', 'xml', or 'auto')

    Returns:
        Tuple of (coverage_percentage, meets_threshold)
    """
    coverage = 0.0

    try:
        file_path = Path(coverage_file)
        if not file_path.exists():
            print(f"❌ Coverage file not found: {coverage_file}")
            return 0.0, False

        # Auto-detect format
        if format == "auto":
            format = "json" if coverage_file.endswith(".json") else "xml"

        if format == "json":
            # Read coverage from JSON
            with open(coverage_file, "r") as f:
                data = json.load(f)
                coverage = data.get("totals", {}).get("percent_covered", 0)
        else:
            # Read coverage from XML
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            line_rate = float(root.get("line-rate", 0))
            coverage = line_rate * 100

    except (json.JSONDecodeError, ET.ParseError) as e:
        print(f"❌ Error parsing coverage file: {e}")
        return 0.0, False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 0.0, False

    # Check against threshold
    meets_threshold = coverage >= threshold

    # Print results
    print(f"📊 Final Coverage: {coverage:.1f}%")
    print(f"🎯 Required Threshold: {threshold}%")

    if meets_threshold:
        print(f"✅ SUCCESS: Coverage {coverage:.1f}% meets threshold {threshold}%")
        print("::notice title=Coverage Success::Coverage target achieved!")
    else:
        gap = threshold - coverage
        print(f"❌ FAILURE: Coverage {coverage:.1f}% is {gap:.1f}% below threshold {threshold}%")
        error_msg = f"Coverage {coverage:.1f}% is below required {threshold}%"
        print(f"::error title=Coverage Failure::{error_msg}")

    return coverage, meets_threshold


def generate_badge(
    coverage: float, output_file: str = "coverage-badge.json", test_type: str = "Coverage"
):
    """Generate a coverage badge JSON file.

    Args:
        coverage: Coverage percentage
        output_file: Output file path for badge data
        test_type: Type of tests (e.g., "Unit Tests", "Integration Tests")
    """
    # Determine badge color
    if coverage >= 80:
        color = "brightgreen"
    elif coverage >= 60:
        color = "yellow"
    else:
        color = "red"

    # Create badge data
    badge_data = {
        "schemaVersion": 1,
        "label": f"{test_type.lower().replace(' ', '-')}-coverage",
        "message": f"{coverage:.1f}%",
        "color": color,
    }

    # Write badge file
    with open(output_file, "w") as f:
        json.dump(badge_data, f, indent=2)

    print(f"🏷️ Generated {test_type} coverage badge: {coverage:.1f}% ({color})")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check coverage against quality gate threshold")
    parser.add_argument("coverage_file", help="Path to coverage file (JSON or XML)")
    parser.add_argument("threshold", type=float, help="Required coverage percentage")
    parser.add_argument("--badge", help="Generate badge JSON file")
    parser.add_argument(
        "--type", default="Coverage", help="Type of tests (e.g., 'Unit Tests', 'Integration Tests')"
    )
    parser.add_argument(
        "--format", choices=["json", "xml", "auto"], default="auto", help="Coverage file format"
    )

    args = parser.parse_args()

    # Check coverage
    coverage, meets_threshold = check_coverage(args.coverage_file, args.threshold, args.format)

    # Print results
    print(f"📊 {args.type} Coverage: {coverage:.1f}%")
    print(f"🎯 Threshold: {args.threshold}%")

    if meets_threshold:
        print(f"✅ {args.type} coverage meets threshold")
    else:
        print(f"❌ {args.type} coverage below threshold ({coverage:.1f}% < {args.threshold}%)")

    # Generate badge if requested
    if args.badge:
        generate_badge(coverage, args.badge, args.type)

    # Exit with appropriate code
    sys.exit(0 if meets_threshold else 1)
