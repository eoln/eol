# Quality Commands

Quick quality validation commands that complement the PRP workflow for the EOL RAG Framework.

## Overview

These commands provide rapid quality checks during development, supplementing the comprehensive PRP validation process.

## Available Commands

| Command | Purpose | Usage |
|---------|---------|-------|
| `quick-check` | Fast validation before commits | `/quality:quick-check` |
| `full-validation` | Complete quality gate simulation | `/quality:full-validation` |
| `performance-check` | RAG performance benchmarks | `/quality:performance-check` |

## Integration with PRP Workflow

These quality commands are used throughout the PRP execution process:

1. **During Development**: Use `quick-check` frequently
2. **Phase Completion**: Run `full-validation` after each phase
3. **Before PR**: Execute complete validation suite
4. **Performance Tuning**: Use `performance-check` for optimization

## Quick Reference

```bash
# Before committing
/quality:quick-check

# After implementing feature
/quality:full-validation

# Performance optimization
/quality:performance-check
```

See individual command files for detailed usage and options.
