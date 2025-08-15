#!/bin/bash
# Update performance badges in README.md based on test results

set -e

# Default values
INDEXING_SPEED="N/A"
SEARCH_LATENCY="N/A"
CACHE_HIT_RATE="N/A"
CHUNK_SPEED="N/A"

# Parse performance results if available
if [ -f "performance-results.json" ]; then
    # Extract metrics from JSON (requires jq)
    if command -v jq &> /dev/null; then
        INDEXING_SPEED=$(jq -r '.indexing_speed // "N/A"' performance-results.json)
        SEARCH_LATENCY=$(jq -r '.search_latency_ms // "N/A"' performance-results.json)
        CACHE_HIT_RATE=$(jq -r '.cache_hit_rate // "N/A"' performance-results.json)
        CHUNK_SPEED=$(jq -r '.chunks_per_second // "N/A"' performance-results.json)
    fi
fi

# Determine badge colors based on thresholds
get_indexing_color() {
    if [ "$1" != "N/A" ] && [ $(echo "$1 >= 10" | bc -l) -eq 1 ]; then
        echo "success"
    elif [ "$1" != "N/A" ] && [ $(echo "$1 >= 5" | bc -l) -eq 1 ]; then
        echo "yellow"
    else
        echo "critical"
    fi
}

get_latency_color() {
    if [ "$1" != "N/A" ] && [ $(echo "$1 <= 100" | bc -l) -eq 1 ]; then
        echo "success"
    elif [ "$1" != "N/A" ] && [ $(echo "$1 <= 200" | bc -l) -eq 1 ]; then
        echo "yellow"
    else
        echo "critical"
    fi
}

get_cache_color() {
    if [ "$1" != "N/A" ] && [ $(echo "$1 >= 31" | bc -l) -eq 1 ]; then
        echo "success"
    elif [ "$1" != "N/A" ] && [ $(echo "$1 >= 20" | bc -l) -eq 1 ]; then
        echo "yellow"
    else
        echo "critical"
    fi
}

# Format values for display
if [ "$INDEXING_SPEED" != "N/A" ]; then
    INDEXING_DISPLAY="${INDEXING_SPEED}_docs%2Fs"
    INDEXING_COLOR=$(get_indexing_color "$INDEXING_SPEED")
else
    INDEXING_DISPLAY="N%2FA"
    INDEXING_COLOR="lightgrey"
fi

if [ "$SEARCH_LATENCY" != "N/A" ]; then
    SEARCH_DISPLAY="${SEARCH_LATENCY}ms"
    SEARCH_COLOR=$(get_latency_color "$SEARCH_LATENCY")
else
    SEARCH_DISPLAY="N%2FA"
    SEARCH_COLOR="lightgrey"
fi

if [ "$CACHE_HIT_RATE" != "N/A" ]; then
    CACHE_DISPLAY="${CACHE_HIT_RATE}%25"
    CACHE_COLOR=$(get_cache_color "$CACHE_HIT_RATE")
else
    CACHE_DISPLAY="N%2FA"
    CACHE_COLOR="lightgrey"
fi

# Update README.md with new badge URLs
README_FILE="${1:-README.md}"

if [ -f "$README_FILE" ]; then
    # Update indexing badge
    sed -i.bak "s|Indexing-[^-]*-[^)]*|Indexing-${INDEXING_DISPLAY}-${INDEXING_COLOR}|g" "$README_FILE"

    # Update search badge
    sed -i.bak "s|Search-[^-]*-[^)]*|Search-${SEARCH_DISPLAY}-${SEARCH_COLOR}|g" "$README_FILE"

    # Update cache badge
    sed -i.bak "s|Cache_Hit-[^-]*-[^)]*|Cache_Hit-${CACHE_DISPLAY}-${CACHE_COLOR}|g" "$README_FILE"

    # Also update the table values if they exist
    if [ "$INDEXING_SPEED" != "N/A" ]; then
        sed -i.bak "s|Indexing Speed | Files/sec | [0-9.]* |Indexing Speed | Files/sec | ${INDEXING_SPEED} |g" "$README_FILE"
    fi

    if [ "$SEARCH_LATENCY" != "N/A" ]; then
        sed -i.bak "s|Query Latency (P50) | Milliseconds | [0-9]* |Query Latency (P50) | Milliseconds | ${SEARCH_LATENCY} |g" "$README_FILE"
    fi

    if [ "$CACHE_HIT_RATE" != "N/A" ]; then
        sed -i.bak "s|Hit Rate | Percentage | [0-9.]*% |Hit Rate | Percentage | ${CACHE_HIT_RATE}% |g" "$README_FILE"
    fi

    # Clean up backup files
    rm -f "${README_FILE}.bak"

    echo "✅ Updated performance metrics in $README_FILE"
else
    echo "❌ README file not found: $README_FILE"
    exit 1
fi
