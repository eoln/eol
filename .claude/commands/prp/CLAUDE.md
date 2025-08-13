# Product Requirements Prompt (PRP) Commands

## Overview
PRP commands implement a systematic planning methodology for creating, executing, and maintaining comprehensive implementation blueprints for feature development.

## PRP Planning Workflow
1. **Analyze** - Extract patterns and context from existing codebase
2. **Create** - Generate comprehensive implementation blueprint
3. **Review** - Review and refine the plan for accuracy
4. **Execute** - Execute plan with systematic task tracking
5. **Validate** - Validate plan completeness and results

## Command Structure
- **analyze.md**: Extract patterns from codebase for planning context
- **create.md**: Generate detailed implementation blueprints
- **review.md**: Review and update existing PRPs
- **execute.md**: Execute plans with task tracking
- **validate.md**: Validate plan feasibility and completeness

## PRP Methodology

PRP commands implement the structured planning methodology documented in [`.claude/context/planning-methodology.md`](../../context/planning-methodology.md).

### Blueprint Generation
PRP commands generate comprehensive implementation plans following the structured format:
- Architecture design with component structure
- Phase-based implementation tasks (see [Phase Structure](../../context/planning-methodology.md#phase-based-development))
- Testing strategy with coverage targets
- Performance targets and benchmarks
- Success metrics (see [Defining Success](../../context/planning-methodology.md#success-metrics))

### Systematic Execution
- Phase-based implementation ([Phase Guidelines](../../context/planning-methodology.md#phase-guidelines))
- Real-time task tracking with checkbox format
- Git branch strategy ([Branch Lifecycle](../../context/planning-methodology.md#git-branch-strategy))
- Quality gates at each phase boundary
- Performance validation against defined metrics

## Best Practices
1. Always analyze before creating PRPs
2. Use systematic task tracking
3. Validate at each phase
4. Document lessons learned
5. Update context with discoveries