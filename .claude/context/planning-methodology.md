# Planning Methodology

This methodology is implemented through the PRP (Product Requirements Prompt) commands in [`.claude/commands/prp/`](../commands/prp/).

## Structured Plan Format

### Plan Header

```markdown
# [Plan Title]

## Overview
Brief description of what the plan accomplishes

**Created**: YYYY-MM-DD
**Status**: Draft/Ready/Pending/Completed
**Priority**: High/Medium/Low
**Estimated Duration**: X days/weeks
**Scope**: Specific boundaries
```

### Checkbox Format

Use checkboxes for trackable tasks:

- `[ ]` - Not started
- `[~]` - In progress
- `[x]` - Completed
- `[!]` - Blocked

## Phase-Based Development

### Phase Structure

```markdown
## Phase 1: [Phase Name]

### 1.1 [Sub-task]
- [ ] **Action item**
  - Details
  - Expected outcome

### 1.2 [Sub-task]
- [ ] **Action item**
```

### Phase Guidelines

1. Keep phases focused on single objective
2. Order phases by dependencies
3. Include validation at phase boundaries
4. Document phase completion criteria

## Git Branch Strategy

### Branch Lifecycle

1. **Plan Creation**: Keep in draft/
2. **Plan Ready**: Validate and move to ready/
3. **Execution Start**:
   - Create feature branch: `feat/plan-name`
   - Move plan to pending/
4. **During Execution**:
   - Update checkboxes in real-time
   - Commit incrementally
   - Push regularly
5. **Completion**:
   - Create PR from feature branch
   - Move plan to completed/
   - Extract lessons learned

### Branch Naming

- Feature: `feat/description`
- Fix: `fix/description`
- Refactor: `refactor/description`
- Docs: `docs/description`

## Success Metrics

### Defining Success

Each plan must include:

```markdown
## Success Metrics
- [ ] Metric 1: Specific, measurable outcome
- [ ] Metric 2: Performance target
- [ ] Metric 3: Quality threshold
```

### Examples

- Coverage increased to >80%
- Latency reduced to <100ms
- All tests passing
- Documentation complete

## Risk Mitigation

### Risk Assessment Template

```markdown
## Risks and Mitigations

### Risk 1: [Description]
- **Probability**: High/Medium/Low
- **Impact**: High/Medium/Low
- **Mitigation**: Specific action to reduce risk
```

### Common Patterns

1. **Dependency Risk**: Document external dependencies
2. **Technical Risk**: Prototype uncertain approaches
3. **Timeline Risk**: Include buffer time
4. **Quality Risk**: Define acceptance criteria

## Parallel Execution

### Task Dependencies

```markdown
## Task Dependencies
- Task A: No dependencies (can start immediately)
- Task B: Depends on Task A
- Task C: No dependencies (can run parallel with A)
- Task D: Depends on B and C
```

### Parallel Work Patterns

1. Identify independent tasks
2. Mark tasks that can run concurrently
3. Use separate branches for large parallel work
4. Coordinate merge order

## Progress Tracking

### Real-Time Updates

- Update checkboxes immediately
- Never batch status updates
- Add notes for blocked items
- Document deviations from plan

### Status Reporting

```markdown
## Progress Update [Date]
- Completed: X of Y tasks
- Current focus: [Task in progress]
- Blockers: [Any issues]
- Next steps: [Upcoming tasks]
```

## Lessons Learned

### Extraction Template

```markdown
## Lessons Learned

### What Worked Well
- Success pattern 1
- Success pattern 2

### What Could Be Improved
- Issue encountered
- Suggested improvement

### Action Items
- [ ] Update context/ with new patterns
- [ ] Document anti-patterns discovered
- [ ] Create follow-up tasks
```

## Best Practices

1. Keep plans focused and scoped
2. Update status in real-time
3. Use descriptive task names
4. Include acceptance criteria
5. Document assumptions
6. Track metrics throughout
7. Extract lessons immediately
