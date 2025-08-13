# Prepare Pull Request Command

## Purpose
Create and push a pull request with proper preparation and description.

## Prerequisites
- All quality checks passed (run check-quality.md first)
- Working on feature branch (not main)
- Changes committed locally
- GitHub CLI (`gh`) installed and authenticated

## Command Sequence

### 1. Ensure Branch is Up to Date
```bash
# Fetch latest changes
git fetch origin main

# Rebase on main (or merge if preferred)
git rebase origin/main

# Resolve any conflicts if they arise
# git status
# git add <resolved-files>
# git rebase --continue
```

### 2. Run Final Quality Check
```bash
# Quick quality validation
python -m black src/ tests/ --check && \
python -m pytest tests/ -q
```

### 3. Review Changes
```bash
# View all changes
git diff origin/main...HEAD

# List changed files
git diff --name-only origin/main...HEAD

# Check commit history
git log --oneline origin/main..HEAD
```

### 4. Push Branch
```bash
# Push to remote
git push -u origin $(git branch --show-current)

# If rebased, might need force push
# git push -u origin $(git branch --show-current) --force-with-lease
```

### 5. Create Pull Request
```bash
# Create PR with title and body
gh pr create \
  --title "[Type] Brief description" \
  --body "$(cat <<EOF
## Summary
Brief description of what this PR accomplishes

## Changes
- List key changes
- Include important modifications
- Note any refactoring

## Testing
- Ran unit tests: \`pytest tests/\`
- Ran integration tests: \`pytest tests/integration/\`
- Coverage: XX%

## Checklist
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Code formatted (Black)
- [ ] Type hints added
- [ ] No security issues

## Breaking Changes
None (or list them)

## Performance Impact
None (or describe impact)
EOF
)" \
  --base main
```

### 6. Alternative: Create Draft PR
```bash
# Create as draft for work in progress
gh pr create --draft \
  --title "[WIP] Feature description" \
  --body "Work in progress..." \
  --base main
```

### 7. View PR Status
```bash
# Check PR checks status
gh pr checks

# View PR in browser
gh pr view --web
```

## PR Title Conventions
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Formatting, no code change
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

## Success Criteria
- ✅ Branch up to date with main
- ✅ All tests passing locally
- ✅ PR created successfully
- ✅ CI/CD checks triggered
- ✅ PR has clear description

## Troubleshooting

### Merge Conflicts
```bash
# If rebase has conflicts
git status
# Edit conflicted files
git add <resolved-files>
git rebase --continue
```

### Push Rejected
```bash
# If push rejected after rebase
git push --force-with-lease
```

### PR Checks Failing
```bash
# View check details
gh pr checks

# View specific workflow run
gh run list --branch $(git branch --show-current)
gh run view <run-id>
```

### Update Existing PR
```bash
# Add commits and push
git add .
git commit -m "Address review feedback"
git push

# Update PR description
gh pr edit --body "Updated description..."
```

## Best Practices
1. Keep PRs small and focused
2. Write descriptive commit messages
3. Respond to review feedback promptly
4. Update PR description as needed
5. Link related issues with "Fixes #123"