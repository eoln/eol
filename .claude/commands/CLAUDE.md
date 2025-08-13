# Commands Directory Rules

## Purpose
- Store reusable command patterns for common tasks
- Provide systematic development workflows
- Standardize development operations

## Command Types
- **prp/**: Product Requirements Prompt commands for feature blueprints
- **quality/**: Quality check commands
- **deploy/**: Deployment commands (future)

## Usage Pattern
- Load command files as needed for specific workflows
- Commands should be executable snippets or clear instructions
- Include context and prerequisites for each command

## Command Structure
Each command file should include:
1. **Purpose**: What the command accomplishes
2. **Prerequisites**: Required setup or conditions
3. **Command Sequence**: Step-by-step commands
4. **Success Criteria**: How to verify success
5. **Troubleshooting**: Common issues and solutions

## Best Practices
- Keep commands focused on single workflow
- Include error handling
- Document expected output
- Provide rollback procedures where applicable
- Test commands before documenting