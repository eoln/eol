#!/usr/bin/env python3
"""
Update performance badges in README based on performance test results.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def load_performance_results(file_path):
    """Load performance results from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def extract_metrics(results):
    """Extract key metrics from performance results."""
    # Default values
    metrics = {
        "indexing_speed": 0,
        "search_latency_ms": 0,
        "cache_hit_rate": 0,
        "chunks_per_second": 0,
    }

    # Try to extract from pytest-benchmark format
    if "benchmarks" in results:
        for benchmark in results["benchmarks"]:
            name = benchmark.get("name", "")
            if "indexing" in name.lower():
                # Convert to docs/s from the benchmark stats
                metrics["indexing_speed"] = benchmark.get("stats", {}).get("mean", 0)
            elif "search" in name.lower():
                # Convert to ms
                metrics["search_latency_ms"] = benchmark.get("stats", {}).get("mean", 0) * 1000
            elif "cache" in name.lower():
                metrics["cache_hit_rate"] = benchmark.get("extra_info", {}).get("hit_rate", 0)

    # Alternative: direct metrics format (for custom output)
    else:
        metrics.update({k: v for k, v in results.items() if k in metrics})

    return metrics


def get_badge_color(metric_name, value):
    """Determine badge color based on metric performance."""
    thresholds = {
        "indexing_speed": {"good": 10, "warning": 5},
        "search_latency_ms": {"good": 100, "warning": 200},
        "cache_hit_rate": {"good": 31, "warning": 20},
        "chunks_per_second": {"good": 40, "warning": 20},
    }

    if metric_name not in thresholds:
        return "blue"

    threshold = thresholds[metric_name]

    if metric_name == "search_latency_ms":
        # Lower is better for latency
        if value <= threshold["good"]:
            return "success"
        elif value <= threshold["warning"]:
            return "yellow"
        else:
            return "critical"
    else:
        # Higher is better for other metrics
        if value >= threshold["good"]:
            return "success"
        elif value >= threshold["warning"]:
            return "yellow"
        else:
            return "critical"


def format_metric_value(metric_name, value):
    """Format metric value for display."""
    if value == 0 or value == "N/A":
        return "N/A"

    if metric_name == "indexing_speed":
        return f"{value:.1f} docs/s"
    elif metric_name == "search_latency_ms":
        return f"{value:.0f}ms"
    elif metric_name == "cache_hit_rate":
        return f"{value:.1f}%"
    elif metric_name == "chunks_per_second":
        return f"{value:.0f} chunks/s"
    else:
        return str(value)


def generate_badge_url(label, message, color):
    """Generate shields.io badge URL."""
    # URL encode the message
    message_encoded = message.replace(" ", "_").replace("/", "%2F").replace("%", "%25")
    return f"https://img.shields.io/badge/{label}-{message_encoded}-{color}"


def generate_performance_section(metrics):
    """Generate the performance section for README."""
    indexing_value = metrics.get("indexing_speed", 0)
    search_value = metrics.get("search_latency_ms", 0)
    cache_value = metrics.get("cache_hit_rate", 0)
    chunks_value = metrics.get("chunks_per_second", 0)

    # Generate badge URLs
    indexing_badge = generate_badge_url(
        "Indexing",
        format_metric_value("indexing_speed", indexing_value),
        get_badge_color("indexing_speed", indexing_value),
    )

    search_badge = generate_badge_url(
        "Search",
        format_metric_value("search_latency_ms", search_value),
        get_badge_color("search_latency_ms", search_value),
    )

    cache_badge = generate_badge_url(
        "Cache_Hit",
        format_metric_value("cache_hit_rate", cache_value),
        get_badge_color("cache_hit_rate", cache_value),
    )

    # Generate status indicators
    indexing_status = "✅" if indexing_value >= 10 else "⚠️" if indexing_value >= 5 else "❌"
    search_status = "✅" if search_value <= 100 else "⚠️" if search_value <= 200 else "❌"
    cache_status = "✅" if cache_value >= 31 else "⚠️" if cache_value >= 20 else "❌"

    # Format values for table
    idx_val = format_metric_value("indexing_speed", indexing_value)
    srch_val = format_metric_value("search_latency_ms", search_value)
    cache_val = format_metric_value("cache_hit_rate", cache_value)
    chunk_val = format_metric_value("chunks_per_second", chunks_value)
    chunk_status = "✅" if chunks_value >= 40 else "⚠️"

    return f"""## 🚀 Performance Metrics

![Indexing Speed]({indexing_badge})
![Search Latency]({search_badge})
![Cache Hit Rate]({cache_badge})

<details>
<summary>📊 Detailed Performance Metrics</summary>

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Indexing Speed** | {idx_val} | >10 docs/s | {indexing_status} |
| **Search Latency** | {srch_val} | <100ms | {search_status} |
| **Cache Hit Rate** | {cache_val} | >31% | {cache_status} |
| **Chunks/sec** | {chunk_val} | >40 chunks/s | {chunk_status} |

*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC via CI/CD*

</details>"""


def update_readme(readme_path, performance_section):
    """Update README with new performance section."""
    with open(readme_path, "r") as f:
        content = f.read()

    # Check if performance section exists
    if "## 🚀 Performance Metrics" in content:
        # Replace existing section
        pattern = r"## 🚀 Performance Metrics.*?(?=\n## |\Z)"
        content = re.sub(pattern, performance_section, content, flags=re.DOTALL)
    else:
        # Add after the main badges section (usually after the first set of badges)
        # Find the end of the badges section
        badges_pattern = r"(!\[.*?\]\(.*?\)\n+)+"
        match = re.search(badges_pattern, content)
        if match:
            insert_pos = match.end()
            content = (
                content[:insert_pos] + "\n" + performance_section + "\n" + content[insert_pos:]
            )
        else:
            # If no badges found, add after the first heading
            pattern = r"(# .*?\n+.*?\n+)"
            content = re.sub(pattern, r"\1" + performance_section + "\n\n", content, count=1)

    with open(readme_path, "w") as f:
        f.write(content)


def main():
    """Main function."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: update-performance-badges.py <performance-results.json> [README.md]")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    readme_file = Path(sys.argv[2] if len(sys.argv) > 2 else "README.md")

    # Load and process results
    results = load_performance_results(results_file)
    metrics = extract_metrics(results)

    # Generate performance section
    performance_section = generate_performance_section(metrics)

    # Update README
    if readme_file.exists():
        update_readme(readme_file, performance_section)
        print(f"✅ Updated {readme_file} with performance metrics")
    else:
        print(f"❌ README file not found: {readme_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
