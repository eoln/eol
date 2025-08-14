#!/usr/bin/env python3
"""Generate test summary from JUnit XML results."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def generate_test_summary(xml_file: str, test_type: str = "Tests") -> int:
    """Generate a summary from JUnit XML test results.

    Args:
        xml_file: Path to the JUnit XML file
        test_type: Type of tests (e.g., "Integration Tests", "Unit Tests")

    Returns:
        0 on success, 1 if there are failures/errors
    """
    try:
        if not Path(xml_file).exists():
            print(f"⚠️ {test_type}: Results file not found")
            return 0

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract test statistics
        tests = int(root.get("tests", 0))
        failures = int(root.get("failures", 0))
        errors = int(root.get("errors", 0))
        skipped = int(root.get("skipped", 0))
        time = float(root.get("time", 0))

        # Calculate passed tests
        passed = tests - failures - errors - skipped

        # Generate summary
        if tests == 0:
            print(f"⚠️ {test_type}: No tests found")
            return 0

        pass_rate = (passed / tests) * 100 if tests > 0 else 0

        print(f"📊 {test_type}: {passed}/{tests} passed ({pass_rate:.1f}%)")

        if time > 0:
            print(f"⏱️ Execution time: {time:.2f}s")

        if skipped > 0:
            print(f"⏭️ Skipped: {skipped} tests")

        if failures > 0 or errors > 0:
            print(f"❌ {failures} failures, {errors} errors")

            # Print failed test details
            for testcase in root.findall(".//testcase"):
                failure = testcase.find("failure")
                error = testcase.find("error")
                if failure is not None or error is not None:
                    classname = testcase.get("classname", "")
                    name = testcase.get("name", "")
                    print(f"  ❌ {classname}::{name}")

            return 1
        else:
            print(f"✅ All {test_type.lower()} passed!")
            return 0

    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_summary.py <xml_file> [test_type]")
        sys.exit(1)

    xml_file = sys.argv[1]
    test_type = sys.argv[2] if len(sys.argv) > 2 else "Tests"

    exit_code = generate_test_summary(xml_file, test_type)
    sys.exit(exit_code)
