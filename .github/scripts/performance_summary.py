#!/usr/bin/env python3
"""Generate performance test summary from benchmark results."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def generate_performance_summary(json_file: str) -> int:
    """Generate a summary from performance benchmark results.

    Args:
        json_file: Path to the benchmark JSON file

    Returns:
        0 on success, 1 on failure
    """
    try:
        if not Path(json_file).exists():
            print(f"⚠️ Performance results file not found: {json_file}")
            return 0

        with open(json_file, "r") as f:
            data = json.load(f)

        benchmarks = data.get("benchmarks", [])

        if not benchmarks:
            print("⚠️ No benchmark results found")
            return 0

        print("📊 Performance Test Results:")
        print("=" * 50)

        # Group benchmarks by category (if available)
        categories: Dict[str, List[Any]] = {}
        for benchmark in benchmarks:
            group = benchmark.get("group", "default")
            if group not in categories:
                categories[group] = []
            categories[group].append(benchmark)

        # Print results by category
        for category, tests in categories.items():
            if category != "default":
                print(f"\n📁 {category}")

            for benchmark in tests:
                name = benchmark["name"]
                stats = benchmark.get("stats", {})

                # Extract statistics
                mean = stats.get("mean", 0)
                stddev = stats.get("stddev", 0)
                min_time = stats.get("min", 0)
                max_time = stats.get("max", 0)
                iterations = stats.get("iterations", 1)

                # Format output
                print(f"\n⚡ {name}")
                print(f"   Mean: {format_time(mean)} ± {format_time(stddev)}")
                print(f"   Range: {format_time(min_time)} - {format_time(max_time)}")
                print(f"   Iterations: {iterations}")

                # Check for performance regressions (if baseline available)
                extra = benchmark.get("extra", {})
                if "baseline" in extra:
                    baseline = extra["baseline"]
                    regression = ((mean - baseline) / baseline) * 100
                    if regression > 10:
                        print(f"   ⚠️ Regression: {regression:.1f}% slower than baseline")
                    elif regression < -10:
                        print(f"   ✅ Improvement: {abs(regression):.1f}% faster than baseline")

        print("\n" + "=" * 50)
        print("✅ Performance tests completed successfully")
        return 0

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python performance_summary.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    exit_code = generate_performance_summary(json_file)
    sys.exit(exit_code)
