# prp-review - Review and Update Product Requirements Prompts

Reviews existing PRPs for accuracy, completeness, and alignment with current codebase patterns.

## Command Overview

**Purpose**: Review and update PRPs to maintain their relevance and accuracy
**Target**: Existing PRP documents in `.claude/plans/`
**Output**: Updated PRP with current patterns and improved confidence score

## Usage

```bash
/prp:review [plan-file]
```

### Examples
```bash
# Review a draft PRP
/prp:review .claude/plans/draft/prp-semantic-caching.md

# Review a ready PRP before execution
/prp:review .claude/plans/ready/prp-vector-search.md

# Review completed PRP for lessons learned
/prp:review .claude/plans/completed/prp-indexing.md
```

## Review Process

### 1. Accuracy Verification
- Verify code examples still exist and are current
- Check if referenced patterns are still best practice
- Validate that dependencies are up to date
- Ensure performance targets remain relevant

### 2. Completeness Check
- All phases properly defined
- Success metrics clearly stated
- Risk mitigation strategies present
- Testing approach comprehensive

### 3. Pattern Currency
```python
# Verify patterns are current
- Check if async patterns match current codebase
- Validate Redis operations against latest patterns
- Ensure type hints follow current standards
- Confirm testing patterns are up to date
```

### 4. Confidence Score Update
Recalculate confidence based on:
- Pattern match with current codebase (40%)
- Completeness of implementation plan (30%)
- Clarity of success metrics (20%)
- Risk mitigation quality (10%)

## Review Checklist

### Architecture Review
- [ ] Component structure aligns with project
- [ ] Dependencies correctly identified
- [ ] Integration points well defined
- [ ] Scalability considerations addressed

### Implementation Review
- [ ] Task breakdown is logical
- [ ] Phase ordering makes sense
- [ ] Time estimates realistic
- [ ] Prerequisites clearly stated

### Quality Review
- [ ] Testing strategy comprehensive
- [ ] Performance targets achievable
- [ ] Documentation requirements clear
- [ ] Security considerations included

### Context Review
- [ ] References to `.claude/context/` accurate
- [ ] Code examples from actual codebase
- [ ] Patterns follow established conventions
- [ ] RAG-specific considerations included

## Update Actions

### Pattern Updates
```markdown
## Updated Patterns Found

### Previous Pattern
```python
# Old implementation
async def old_pattern():
    pass
```

### Current Best Practice
```python
# New implementation from current codebase
async def new_pattern():
    # Better approach found in packages/eol-rag-context/
    pass
```
```

### Dependency Updates
- Remove deprecated dependencies
- Add newly required packages
- Update version constraints
- Note breaking changes

### Performance Target Adjustments
Based on current benchmarks:
- Update realistic targets
- Add new metrics if needed
- Remove obsolete measurements

## Review Output Format

```markdown
# PRP Review Report

## Summary
- **PRP**: [Name]
- **Original Confidence**: X/10
- **Updated Confidence**: Y/10
- **Status**: [Current/Needs Update/Obsolete]

## Changes Made
1. Updated pattern examples from current codebase
2. Adjusted performance targets based on benchmarks
3. Added new risk considerations
4. Improved test coverage approach

## Recommendations
- [ ] Update implementation approach for Phase 2
- [ ] Add integration tests for new patterns
- [ ] Consider alternative caching strategy
- [ ] Review security implications

## Lessons from Implementation (if completed)
- What worked well
- What could be improved
- Patterns to promote to context/
- Anti-patterns discovered
```

## Integration with Planning Workflow

### When to Review PRPs

1. **Before Execution**: Review ready PRPs to ensure accuracy
2. **During Execution**: Update if significant changes discovered
3. **After Completion**: Extract lessons and update patterns
4. **Periodic Review**: Monthly review of draft PRPs
5. **Context Changes**: When major patterns change in codebase

### State Transitions
- **draft → ready**: Requires review pass
- **ready → pending**: Final review recommended
- **completed → context**: Extract patterns for reuse

## Success Criteria

### Good PRP Review
- ✅ All patterns verified against current code
- ✅ Performance targets validated
- ✅ Dependencies current
- ✅ Clear improvement suggestions
- ✅ Confidence score justified

### Review Red Flags
- ❌ Code examples don't exist
- ❌ Patterns contradict current standards
- ❌ Unrealistic performance targets
- ❌ Missing critical dependencies
- ❌ No clear success metrics

## Best Practices

1. **Review regularly** - Don't let PRPs get stale
2. **Update incrementally** - Small updates better than rewrites
3. **Document changes** - Track what changed and why
4. **Share learnings** - Update context/ with discoveries
5. **Validate with execution** - Test updates before marking ready

This review command ensures PRPs remain valuable, accurate blueprints for feature implementation.