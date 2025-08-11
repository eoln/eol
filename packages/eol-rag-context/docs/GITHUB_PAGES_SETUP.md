# GitHub Pages Setup Guide

## Enabling GitHub Pages for Documentation

### 1. Repository Settings
Navigate to your repository settings:
1. Go to `https://github.com/eoln/eol`
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar

### 2. Configure GitHub Pages

#### Source Settings
- **Source**: Deploy from GitHub Actions (this is already configured in our workflow)
- **Branch**: Not applicable (using Actions deployment)

#### Custom Domain (Optional)
- If you have a custom domain, you can configure it here
- Add a CNAME file to the docs if using a custom domain

### 3. Deployment URLs

Once deployed, your documentation will be available at:

#### Main Documentation Site
- **Production URL**: `https://eoln.github.io/eol/packages/eol-rag-context/`
- **Version Selection**: Users can switch between versions using the version selector

#### Versioned URLs (managed by mike)
- **Latest (default)**: `https://eoln.github.io/eol/packages/eol-rag-context/latest/`
- **Development**: `https://eoln.github.io/eol/packages/eol-rag-context/dev/`
- **Specific versions**: `https://eoln.github.io/eol/packages/eol-rag-context/<version>/`
  - Example: `https://eoln.github.io/eol/packages/eol-rag-context/1.0.0/`

### 4. Deployment Process

The documentation is automatically deployed when:

1. **Automatic Deployment** (via GitHub Actions):
   ```bash
   # Happens automatically when you push to main
   git push origin main
   ```

2. **Manual Deployment** (using mike locally):
   ```bash
   # Deploy a specific version
   ./scripts/manage_docs_version.sh deploy 1.0.0 latest
   
   # Deploy development version
   ./scripts/manage_docs_version.sh deploy dev
   
   # List all deployed versions
   ./scripts/manage_docs_version.sh list
   ```

### 5. GitHub Actions Workflow

The `.github/workflows/docs.yml` workflow handles:
- Building documentation on every push to main
- Running quality checks (docstring coverage, link validation)
- Deploying to GitHub Pages using mike for versioning
- Creating PR previews (as downloadable artifacts)

### 6. First-Time Setup

For the first deployment:

1. **Ensure GitHub Pages is enabled** in repository settings
2. **Run the initial deployment**:
   ```bash
   # Clone the repository
   git clone https://github.com/eoln/eol.git
   cd eol/packages/eol-rag-context
   
   # Install dependencies
   pip install -r requirements-dev.txt
   
   # Build and deploy initial version
   mike deploy --push --update-aliases 0.1.0 latest
   mike set-default --push latest
   ```

3. **Verify deployment**:
   - Check Actions tab for workflow status
   - Visit the GitHub Pages URL after deployment completes
   - The deployment typically takes 2-5 minutes

### 7. Monitoring Deployments

#### GitHub Actions Status
- View deployment status: `https://github.com/eoln/eol/actions/workflows/docs.yml`
- Each deployment shows:
  - Build status
  - Docstring coverage report
  - Link validation results
  - Deployment URL

#### GitHub Pages Health
- Check Pages settings for deployment status
- Look for the green checkmark indicating successful deployment
- View deployment history in the Actions tab

### 8. Troubleshooting

#### If Pages doesn't show up:
1. Ensure you have pushed to the `main` branch
2. Check that GitHub Actions completed successfully
3. Verify Pages is enabled in repository settings
4. Wait 5-10 minutes for DNS propagation on first deployment

#### If versions don't appear:
1. Check mike configuration in `mkdocs.yml`
2. Ensure `gh-pages` branch exists and has content
3. Verify the GitHub Actions workflow has write permissions

#### Common Issues:
- **404 Error**: Check the URL path matches the site_url in mkdocs.yml
- **Old content**: Clear browser cache or use incognito mode
- **Missing styles**: Ensure Material theme is properly installed
- **Broken links**: Run `mkdocs build --strict` locally to find issues

### 9. PR Previews

When you create a Pull Request:
1. The workflow builds the documentation
2. Creates a downloadable artifact with the built site
3. Posts a comment on the PR with a link to download the preview
4. Full preview deployment only happens after merging to main

### 10. Local Testing

Before pushing, test locally:
```bash
# Build and serve locally
mkdocs serve

# Test with mike (versioned)
mike serve

# Validate documentation
python scripts/validate_docs.py

# Run pre-commit hooks
pre-commit run --all-files
```

## Important Notes

- **Branch Protection**: Consider protecting the `gh-pages` branch (created by mike) to prevent accidental deletion
- **Secrets**: If using Google Analytics, set up the `GOOGLE_ANALYTICS_KEY` secret in repository settings
- **Permissions**: The GitHub Actions workflow needs write permissions for Pages deployment
- **Storage**: GitHub Pages has a 1GB size limit for the published site

## Quick Reference

| Component | URL/Location |
|-----------|-------------|
| Live Documentation | `https://eoln.github.io/eol/packages/eol-rag-context/` |
| GitHub Repository | `https://github.com/eoln/eol` |
| Actions Workflow | `.github/workflows/docs.yml` |
| Version Management | `scripts/manage_docs_version.sh` |
| MkDocs Config | `mkdocs.yml` |
| Documentation Source | `docs/` directory |
| Built Site (local) | `site/` directory |
| Deployed Site Branch | `gh-pages` branch |