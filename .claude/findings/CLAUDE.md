# Findings Directory Rules

## Purpose

Store research findings, investigation reports, and analysis results from web searches, codebase exploration, and technical investigations. These findings serve as a knowledge base for informed decision-making and pattern discovery.

## Naming Convention

All findings must follow the date-prefix naming pattern:
```
YYYYMMDD_description-of-finding.md
```

### Examples
- `20250113_redis-vector-search-benchmarks.md`
- `20250113_python-async-patterns-analysis.md`
- `20250114_rag-chunking-strategies-research.md`
- `20250115_semantic-caching-investigation.md`

## Finding Types

### Research Reports
Web searches and external documentation analysis:
- Technology evaluations
- Best practices research
- Performance benchmarks
- Security advisories

### Investigation Results
Deep dives into specific problems or patterns:
- Bug root cause analysis
- Performance bottleneck investigations
- Pattern discovery reports
- Compatibility studies

### Analysis Documents
Codebase and system analysis:
- Architecture reviews
- Dependency analysis
- Code pattern extraction
- Technical debt assessment

## Document Structure

Each finding should follow this template:

```markdown
# [Title of Finding]

**Date**: YYYY-MM-DD
**Type**: Research | Investigation | Analysis
**Status**: Draft | In Progress | Complete | Archived
**Confidence**: High | Medium | Low

## Summary
Brief overview of the finding (2-3 sentences)

## Context
Why this research/investigation was conducted

## Methodology
How the research was performed:
- Sources consulted
- Tools used
- Analysis approach

## Findings

### Key Discovery 1
Details and evidence

### Key Discovery 2
Details and evidence

## Recommendations
Actionable items based on findings

## References
- Links to sources
- Related documentation
- Code examples

## Impact
How this finding affects the project

## Next Steps
Follow-up actions required
```

## Usage Patterns

### Creating New Findings
1. Conduct research/investigation
2. Create file with date prefix: `YYYYMMDD_topic.md`
3. Follow template structure
4. Link to relevant plans or context

### Referencing Findings
In plans and documentation:
```markdown
Based on findings: [Redis Performance Analysis](../findings/20250113_redis-performance.md)
```

### Archiving Old Findings
- Findings older than 6 months: Review for relevance
- Outdated findings: Update status to "Archived"
- Superseded findings: Link to newer research

## Integration with Planning

### Before Creating PRPs
- Check findings/ for existing research
- Conduct new research if needed
- Document findings before planning

### During Plan Execution
- Reference relevant findings
- Update findings with new discoveries
- Create findings for blockers/issues

### After Completion
- Document lessons learned as findings
- Update findings with real-world results
- Archive superseded research

## Quality Standards

### Good Finding Characteristics
- ✅ Clear, actionable conclusions
- ✅ Evidence-based recommendations
- ✅ Proper source attribution
- ✅ Reproducible methodology
- ✅ Date and context included

### Finding Red Flags
- ❌ No sources cited
- ❌ Vague conclusions
- ❌ Missing context
- ❌ No date information
- ❌ Unverified claims

## Maintenance

### Regular Review
- Weekly: Review new findings for accuracy
- Monthly: Update status of ongoing investigations
- Quarterly: Archive outdated findings

### Cross-Referencing
- Link findings to related plans
- Reference in context/ documentation
- Update when patterns change

## Examples of Valuable Findings

### Performance Research
`20250113_document-indexing-benchmarks.md`
- Tested different chunking strategies
- Measured indexing throughput
- Recommended optimal batch sizes

### Pattern Analysis
`20250114_async-error-handling-patterns.md`
- Analyzed codebase error handling
- Identified best practices
- Created reusable patterns

### Technology Evaluation
`20250115_embedding-model-comparison.md`
- Compared different embedding models
- Evaluated performance vs quality
- Recommended model selection

## Best Practices

1. **Date everything** - Always use YYYYMMDD prefix
2. **Be specific** - Clear, descriptive filenames
3. **Cite sources** - Include all references
4. **Stay objective** - Evidence-based conclusions
5. **Update regularly** - Keep findings current
6. **Link connections** - Reference related work

---

*This directory serves as the project's research knowledge base, capturing valuable insights and investigations that inform development decisions.*