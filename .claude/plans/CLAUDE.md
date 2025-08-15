# Plan Management Rules

## State Transitions

- ALWAYS move files between state directories
- Update task status immediately (no batching)
- Mark completed tasks instantly

## Real-Time Status Updates (CRITICAL)

- **Update checkboxes IMMEDIATELY** when task status changes
- **Never batch checkbox updates** - update as soon as work begins or completes
- **Enable tracking systems** - immediate updates allow external tools to react to changes
- **Maintain single source of truth** - the plan file is the authoritative status

### Why Immediate Updates Matter

- External systems may be monitoring plan files for changes
- Real-time updates enable automated workflows and notifications
- Prevents status drift between actual work and documented progress
- Allows stakeholders to track progress without manual queries

## Task Status Within Plans

- `[ ]` Pending - Not yet started
- `[~]` In Progress - Currently working (update IMMEDIATELY when starting)
- `[x]` Completed - Finished (update IMMEDIATELY upon completion)
- `[!]` Blocked - Cannot proceed (update IMMEDIATELY when blocked)

**Update Pattern:**

1. Mark task as `[~]` the moment you begin work
2. Mark task as `[x]` the instant it's complete
3. Mark task as `[!]` immediately when encountering a blocker
4. Never leave stale status - accuracy is critical

## Directory Structure

- **draft/**: Initial ideas and exploration
- **ready/**: Validated and approved plans
- **pending/**: Currently executing plans
- **completed/**: Finished plans for reference

## Workflow Requirements

1. Plans start in draft/ for development
2. Move to ready/ when validated
3. Move to pending/ with dedicated git branch
4. Move to completed/ with PR creation

## Git Integration

- Create feature branch when moving to pending/
- Branch naming: feat/[plan-name] or fix/[plan-name]
- Incremental commits during execution
- PR creation upon completion
