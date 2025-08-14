# CI/CD Scripts

This directory contains Python scripts used by GitHub Actions workflows to generate summaries, check thresholds, and process test results.

## Why External Scripts?

Instead of embedding complex Python code directly in YAML workflow files, we use external scripts for:

1. **Maintainability**: Easier to read, test, and debug
2. **Reusability**: Same scripts can be used across multiple workflows
3. **Reliability**: Avoid YAML escaping issues and syntax errors
4. **Testability**: Scripts can be tested locally before deployment
5. **Version Control**: Better diff tracking and code review

## Available Scripts

### test_summary.py
Generates a summary from JUnit XML test results.

```bash
python test_summary.py <xml_file> [test_type]
```

**Features:**
- Shows pass/fail statistics
- Lists failed tests with details
- Calculates pass rate percentage
- Displays execution time

### coverage_check.py
Checks if code coverage meets the required threshold.

```bash
python coverage_check.py <coverage_file> <threshold> [--badge <output_file>]
```

**Features:**
- Supports both JSON and XML coverage formats
- Generates GitHub Actions annotations
- Creates coverage badge JSON
- Returns appropriate exit codes

### performance_summary.py
Generates performance test summary from benchmark results.

```bash
python performance_summary.py <json_file>
```

**Features:**
- Formats timing in appropriate units
- Groups benchmarks by category
- Detects performance regressions
- Shows statistical information

### security_scan_summary.py
Processes SARIF security scan results.

```bash
python security_scan_summary.py <sarif_file>
```

**Features:**
- Categorizes issues by severity
- Shows file locations
- Limits output for readability
- Non-blocking for warnings

### dependency_parser.py
Extracts dependencies from pyproject.toml.

```bash
python dependency_parser.py <pyproject.toml> [output_file]
```

**Features:**
- Parses project dependencies
- Includes optional dependencies
- Outputs to file or stdout
- Compatible with security scanners

## Usage in Workflows

### Before (Embedded Python)
```yaml
- name: Check coverage
  run: |
    python -c "import json; data = json.load(open('coverage.json')); coverage = data.get('totals', {}).get('percent_covered', 0); print(f'Coverage: {coverage:.1f}%')"
```

### After (External Script)
```yaml
- name: Check coverage
  run: |
    python ${{ github.workspace }}/.github/scripts/coverage_check.py \
      coverage.json \
      80
```

## Local Testing

Test scripts locally before committing:

```bash
# Generate test data
pytest --junit-xml=test-results.xml
pytest --cov=mypackage --cov-report=json

# Test scripts
python .github/scripts/test_summary.py test-results.xml
python .github/scripts/coverage_check.py coverage.json 80
```

## Adding New Scripts

When adding new scripts:

1. Follow the existing pattern with clear argument parsing
2. Include comprehensive error handling
3. Use appropriate exit codes (0 for success, non-zero for failure)
4. Add documentation in this README
5. Make the script executable: `chmod +x script.py`
6. Test locally before using in workflows

## Best Practices

1. **Error Handling**: Always catch and handle exceptions gracefully
2. **Exit Codes**: Use consistent exit codes across scripts
3. **Output Format**: Use emoji and formatting for clear CI logs
4. **Fallbacks**: Provide sensible defaults and fallbacks
5. **Documentation**: Include usage examples in script docstrings

## Dependencies

Scripts use only Python standard library modules when possible. If additional dependencies are needed:

```python
try:
    import tomli  # For Python < 3.11
except ImportError:
    import tomllib as tomli  # Python 3.11+
```

## Troubleshooting

### Script Not Found
Ensure the script path is correct:
```yaml
python ${{ github.workspace }}/.github/scripts/script.py
```

### Permission Denied
Make scripts executable:
```bash
chmod +x .github/scripts/*.py
```

### Import Errors
Use try/except for optional imports and provide fallbacks.

### YAML Escaping Issues
This is exactly why we use external scripts! No more escaping nightmares.
