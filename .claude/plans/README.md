# Plans Directory Guide

Structured task planning with state management and real-time status tracking for the EOL Framework.

## Directory Structure & State Definitions
- **draft/**: Initial ideas and exploration
- **ready/**: Validated and approved plans
- **pending/**: Currently executing plans (with git branch)
- **completed/**: Finished plans for reference (with date prefix YYYYMMDD_)

## Workflow Requirements
1. Plans start in **draft/** for development
2. Move to **ready/** when validated
3. Move to **pending/** with dedicated git branch
4. Move to **completed/** with PR creation

**State Flow**: draft → ready → pending → completed

## File Naming Conventions
- **Draft/Ready/Pending**: `feature-name-plan.md` (descriptive kebab-case)
- **Completed**: `YYYYMMDD_feature-name-plan.md` (date prefix for archival)

## Real-Time Status Updates (CRITICAL)

### Task Status Checkboxes
- `[ ]` **Pending** - Not yet started
- `[~]` **In Progress** - Currently working (update IMMEDIATELY when starting)
- `[x]` **Completed** - Finished (update IMMEDIATELY upon completion)
- `[!]` **Blocked** - Cannot proceed (update IMMEDIATELY when blocked)

### Update Requirements
- **Update checkboxes IMMEDIATELY** when task status changes
- **Never batch checkbox updates** - update as soon as work begins or completes
- **Enable tracking systems** - immediate updates allow external tools to react
- **Maintain single source of truth** - the plan file is the authoritative status

### Update Pattern
1. Mark task as `[~]` the moment you begin work
2. Mark task as `[x]` the instant it's complete
3. Mark task as `[!]` immediately when encountering a blocker
4. Never leave stale status - accuracy is critical

## Git Integration
- Create feature branch when moving to pending/
- Branch naming: `feat/[plan-name]` or `fix/[plan-name]`
- Incremental commits during execution
- PR creation upon completion

## Best Practices
1. **One plan per feature** or major change
2. **Include success metrics** in all plans
3. **Document dependencies** clearly
4. **Extract lessons learned** upon completion
5. **Update context/** with discoveries
6. **Real-time updates** for all status changes
7. **Use TodoWrite tool** for task tracking integration