# prp-execute - Execute Product Requirements Prompt with Tracking

Systematically executes PRPs with real-time task tracking, quality gates, and git integration for the EOL RAG Framework.

## Command Overview

**Purpose**: Execute PRP implementation with systematic tracking and validation
**Target**: Plans in `.claude/plans/` following state lifecycle
**Integration**: TodoWrite for tracking, git for version control

## Usage

```bash
/prp:execute [plan-file]
```

### Examples
```bash
# Execute a ready plan
/prp:execute .claude/plans/ready/prp-semantic-caching.md

# Execute with specific branch name
/prp:execute .claude/plans/ready/prp-vector-search.md --branch=feat/vector-search

# Dry run to preview execution
/prp:execute .claude/plans/ready/prp-embeddings.md --dry-run
```

## Execution Process

### 1. Pre-Execution Validation
```bash
# Verify plan is in ready state
ls .claude/plans/ready/prp-*.md

# Check current git status
git status

# Ensure clean working tree
git stash push -m "Pre-PRP execution stash"
```

### 2. Branch Creation
```bash
# Create dedicated feature branch
git checkout -b feat/[plan-name]

# Move plan to pending state
mv .claude/plans/ready/prp-[feature].md .claude/plans/pending/
```

### 3. Task Execution Flow

#### Phase-Based Execution
```markdown
## Phase 1: Foundation [~]
- [x] Task 1 completed
- [~] Task 2 in progress  # Current task
- [ ] Task 3 pending

## Phase 2: Implementation [ ]
- [ ] All tasks pending
```

#### Real-Time Tracking
- Update task status immediately upon completion
- Use TodoWrite for persistent tracking
- Commit after each phase completion

### 4. Quality Gates

#### After Each Phase
```bash
# Run quality checks
python -m black src/ tests/ --check
python -m flake8 src/ tests/
python -m mypy src/
python -m pytest tests/ -q
```

#### Performance Validation
```python
# Verify performance targets
assert indexing_rate > 10  # docs/sec
assert search_latency < 100  # ms
assert cache_hit_rate > 0.31  # 31%
```

### 5. Incremental Commits
```bash
# Commit after significant progress
git add -A
git commit -m "feat: implement phase 1 - foundation

- Set up package structure
- Define dataclasses and protocols
- Add type hints throughout"
```

## Execution States

### Task Status Indicators
- `[ ]` - Not started
- `[~]` - In progress (only one at a time)
- `[x]` - Completed
- `[!]` - Blocked (document reason)

### Plan State Transitions
1. **draft/** → Initial development
2. **ready/** → Validated and approved
3. **pending/** → Currently executing (with git branch)
4. **completed/** → Finished (with PR created)

## Integration with TodoWrite

### Automatic Task Loading
```python
# Load tasks from PRP
tasks = parse_prp_tasks(".claude/plans/pending/prp-feature.md")
todo_write(tasks)
```

### Progress Tracking
```python
# Update task status
mark_task_complete(task_id)
mark_task_in_progress(next_task_id)
```

## Git Integration

### Branch Management
```bash
# Feature branch naming
feat/[feature-name]  # New features
fix/[issue-name]     # Bug fixes
refactor/[area]      # Refactoring

# Keep branch updated
git fetch origin main
git rebase origin main
```

### Commit Message Format
```
[type]: [description]

- Detail 1
- Detail 2

Related: #issue
```

## Error Recovery

### Rollback Procedures
```bash
# Rollback to checkpoint
git reset --hard [commit-hash]

# Move plan back to ready
mv .claude/plans/pending/prp-*.md .claude/plans/ready/

# Clean up branch
git checkout main
git branch -D feat/[feature-name]
```

### Handling Blocked Tasks
```markdown
- [!] **Task Name** - Blocked
  - Reason: Missing dependency
  - Action: Install redis-py[vector]
  - Resolution: Run `uv add redis[vector]`
```

## Post-Execution

### Completion Checklist
- [ ] All tasks marked complete
- [ ] Tests passing with >80% coverage
- [ ] Quality gates passed
- [ ] Performance targets met
- [ ] Documentation updated

### PR Creation
```bash
# Push branch
git push -u origin feat/[feature-name]

# Create PR
gh pr create \
  --title "feat: [feature description]" \
  --body "## Summary
  Implementation of PRP for [feature]
  
  ## Changes
  - [List key changes]
  
  ## Testing
  - Coverage: XX%
  - Performance: Meets targets
  
  ## PRP Reference
  See .claude/plans/completed/prp-[feature].md"
```

### Plan Archival
```bash
# Move to completed with timestamp
mv .claude/plans/pending/prp-[feature].md \
   .claude/plans/completed/prp-[feature]-$(date +%Y%m%d).md

# Update with lessons learned
echo "## Lessons Learned" >> .claude/plans/completed/prp-[feature]-*.md
echo "- [Key learning 1]" >> .claude/plans/completed/prp-[feature]-*.md
```

## Command Options

### Execution Control
```bash
# Dry run mode
/prp:execute plan.md --dry-run

# Skip specific phases
/prp:execute plan.md --skip-phase=optimization

# Continue from checkpoint
/prp:execute plan.md --continue-from=phase-3
```

### Validation Options
```bash
# Strict mode (fail on any warning)
/prp:execute plan.md --strict

# Skip performance validation
/prp:execute plan.md --no-perf-check

# Custom quality command
/prp:execute plan.md --quality-cmd="make quality"
```

## Success Metrics

### Execution Quality
- [ ] All phases completed successfully
- [ ] No blocked tasks remaining
- [ ] Quality gates passed at each phase
- [ ] Performance targets achieved

### Process Compliance
- [ ] Git branch created and used
- [ ] Incremental commits made
- [ ] Task status updated in real-time
- [ ] PR created upon completion

## Best Practices

1. **Always execute from ready state** - Ensure plan is validated
2. **One task in progress** - Focus on single task at a time
3. **Commit frequently** - After each significant milestone
4. **Update immediately** - Mark tasks complete right away
5. **Document blockers** - Explain why tasks are blocked
6. **Test continuously** - Run tests after each phase

This execution command ensures systematic, trackable implementation of PRPs with quality gates and proper version control.