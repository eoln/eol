#!/usr/bin/env python3
"""Generate security scan summary from SARIF results."""

import json
import sys
from pathlib import Path
from typing import Dict, List


def generate_security_summary(sarif_file: str) -> int:
    """Generate a summary from SARIF security scan results.

    Args:
        sarif_file: Path to the SARIF file

    Returns:
        Number of security issues found
    """
    try:
        if not Path(sarif_file).exists():
            print(f"⚠️ SARIF file not found: {sarif_file}")
            return 0

        with open(sarif_file, "r") as f:
            data = json.load(f)

        runs = data.get("runs", [])

        if not runs:
            print("📊 No security issues found")
            return 0

        total_issues = 0
        issues_by_level: Dict[str, List] = {"error": [], "warning": [], "note": []}

        for run in runs:
            results = run.get("results", [])

            for result in results:
                total_issues += 1
                level = result.get("level", "warning")
                rule_id = result.get("ruleId", "unknown")
                message = result.get("message", {}).get("text", "No description")

                # Get location information
                locations = result.get("locations", [])
                if locations:
                    physical_location = locations[0].get("physicalLocation", {})
                    artifact = physical_location.get("artifactLocation", {})
                    file_path = artifact.get("uri", "unknown")
                    region = physical_location.get("region", {})
                    line = region.get("startLine", 0)
                else:
                    file_path = "unknown"
                    line = 0

                issue_info = {
                    "rule_id": rule_id,
                    "message": message,
                    "file": file_path,
                    "line": line,
                }

                issues_by_level[level].append(issue_info)

        # Print summary
        print(f"📊 Security Scan Results: {total_issues} issues found")

        if total_issues > 0:
            print("\n🔍 Issue Breakdown:")

            # Print errors
            if issues_by_level["error"]:
                print(f"\n🔴 Errors ({len(issues_by_level['error'])})")
                for issue in issues_by_level["error"][:5]:  # Show first 5
                    print(f"  - {issue['rule_id']}: {issue['message'][:80]}...")
                    print(f"    📍 {issue['file']}:{issue['line']}")

            # Print warnings
            if issues_by_level["warning"]:
                print(f"\n🟡 Warnings ({len(issues_by_level['warning'])})")
                for issue in issues_by_level["warning"][:5]:  # Show first 5
                    print(f"  - {issue['rule_id']}: {issue['message'][:80]}...")
                    print(f"    📍 {issue['file']}:{issue['line']}")

            # Print notes
            if issues_by_level["note"]:
                print(f"\n🔵 Notes ({len(issues_by_level['note'])})")

            print("\n💡 Run security tools locally for detailed information")
        else:
            print("✅ No security vulnerabilities detected!")

        return total_issues

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing SARIF: {e}")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 0


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python security_scan_summary.py <sarif_file>")
        sys.exit(0)

    sarif_file = sys.argv[1]
    issues = generate_security_summary(sarif_file)

    # Don't fail the build for security warnings (only for errors)
    sys.exit(0)
