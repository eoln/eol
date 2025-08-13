# Plan Management Rules

## State Transitions
- ALWAYS move files between state directories
- Update task status immediately (no batching)
- Mark completed tasks instantly

## Task Status Within Plans
- [ ] Pending
- [~] In Progress  
- [x] Completed
- [!] Blocked

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