# Pull Request Preparation (PRP) Commands

## Workflow
1. Run quality checks
2. Update documentation
3. Verify tests pass
4. Create PR with proper description

## Command Naming
- Use descriptive names: `check-quality.md`, `prepare-pr.md`
- Include prerequisites and expected outcomes

## Standard PR Workflow

### Pre-PR Checklist
1. [ ] All tests passing
2. [ ] Code formatted (Black)
3. [ ] Imports sorted (isort)
4. [ ] Type hints added
5. [ ] Documentation updated
6. [ ] Coverage >80%
7. [ ] No security issues

### PR Description Template
```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
- How changes were tested
- Coverage report

## Performance Impact
- Any performance implications

## Breaking Changes
- List any breaking changes
```

## Command Files
- **check-quality.md**: Run all quality checks
- **prepare-pr.md**: Create and push PR
- **update-docs.md**: Update documentation
- **verify-integration.md**: Run integration tests

## Best Practices
1. Always run quality checks before PR
2. Keep PRs focused and small
3. Write clear commit messages
4. Update tests with code changes
5. Document breaking changes