# Commands Directory Rules

## Overview

The `.claude/commands/` directory enables custom slash commands in Claude Code, allowing you to create reusable prompt templates and workflows that can be shared with your team or kept for personal use.

## How It Works

Claude Code automatically recognizes any Markdown file in your project's `.claude/commands/` directory as a slash command. The filename becomes the command name, making it instantly available through the slash command menu when you type `/`.

## Command Types

### Project Commands (This Directory)
- **prp/**: Product Requirements Prompt commands for feature blueprints
- **quality/**: Quality check commands for validation and testing
- **deploy/**: Deployment commands (future)

Commands in this directory are:
- Available to all team members when checked into git
- Shown as `/project:command-name` in Claude Code
- Listed with "(project)" suffix in `/help`

### Personal Commands
You can also create personal commands in `~/.claude/commands/` that are:
- Available across all your Claude Code sessions
- Private to your local environment
- Shown as `/personal:command-name`

## Command Structure

Each command file should include:
1. **Purpose**: Clear description of what the command accomplishes
2. **Prerequisites**: Required setup, environment, or conditions
3. **Command Sequence**: Step-by-step instructions or prompts
4. **Success Criteria**: How to verify successful execution
5. **Troubleshooting**: Common issues and solutions

## Advanced Features

### Parameter Support
Use `$ARGUMENTS` to pass parameters to commands:
```markdown
# Example: .claude/commands/analyze-issue.md
Please analyze and fix the GitHub issue: $ARGUMENTS
```
Usage: `/project:analyze-issue 1234`

### File References
Include file contents using the `@` prefix:
```markdown
Review the implementation in @src/utils/helpers.py
Compare @src/old-version.py with @src/new-version.py
```

### Bash Command Execution
Execute bash commands before the slash command runs using the `!` prefix:
```markdown
!git status
Analyze the current git status and suggest next steps.
```

## Namespacing

Commands are organized by namespace:
- `/project:command-name` - Project-specific commands (this directory)
- `/personal:command-name` - Personal commands from ~/.claude/commands/
- `/command-name` - Built-in Claude Code commands

## Best Practices

1. **Keep commands focused** - Single responsibility per command
2. **Include error handling** - Account for common failure modes
3. **Document prerequisites** - Clear setup requirements
4. **Provide examples** - Show expected usage and output
5. **Test before committing** - Ensure commands work for team
6. **Use descriptive names** - Make command purpose obvious
7. **Version control** - Track changes to team commands in git

## Example Command

```markdown
# .claude/commands/review-code.md
Please perform a comprehensive code review:

1. Check for code style violations
2. Identify potential bugs or edge cases
3. Suggest performance improvements
4. Verify test coverage
5. Review documentation completeness

Focus on: $ARGUMENTS

Include specific line numbers and code examples in your feedback.
```

## Integration with EOL RAG Framework

Our custom commands extend Claude Code for RAG development:
- **PRP commands**: Generate comprehensive implementation blueprints
- **Quality commands**: Validate Python code and RAG performance
- **Context-aware**: Leverage .claude/context/ knowledge base

## References

- [Claude Code Overview](https://docs.anthropic.com/en/docs/claude-code/overview)
- [Slash Commands Documentation](https://docs.anthropic.com/en/docs/claude-code/slash-commands)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Code GitHub Repository](https://github.com/anthropics/claude-code)
- [CLI Reference](https://docs.anthropic.com/en/docs/claude-code/cli-reference)

## Community Resources

- [Awesome Claude Code](https://github.com/hesreallyhim/awesome-claude-code) - Curated list of commands and workflows
- [Claude Command Suite](https://github.com/qdhenry/Claude-Command-Suite) - 119+ professional slash commands
- [Claude Code Complete Guide](https://www.siddharthbharath.com/claude-code-the-complete-guide/) - Comprehensive tutorial

---

*This directory follows Claude Code's official slash command conventions, enabling powerful custom workflows for the EOL RAG Framework development.*