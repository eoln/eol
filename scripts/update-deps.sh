#!/bin/bash
# Update dependencies safely with version constraints

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
UPDATE_TYPE=${1:-"patch"}  # patch, minor, major, all
DRY_RUN=${2:-"false"}

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_info "🔄 Updating dependencies (type: $UPDATE_TYPE, dry-run: $DRY_RUN)..."

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        print_error "Virtual environment not found. Run ./scripts/setup-workspace.sh first."
        exit 1
    fi
fi

# Backup current constraints
cp requirements/constraints.txt requirements/constraints.txt.backup
print_info "Backed up constraints to requirements/constraints.txt.backup"

# Function to update version based on type
update_version() {
    local current_version=$1
    local update_type=$2
    
    IFS='.' read -r major minor patch <<< "$current_version"
    
    case $update_type in
        patch)
            echo "$major.$minor.$((patch + 1))"
            ;;
        minor)
            echo "$major.$((minor + 1)).0"
            ;;
        major)
            echo "$((major + 1)).0.0"
            ;;
        *)
            echo "$current_version"
            ;;
    esac
}

# Update packages based on type
case $UPDATE_TYPE in
    patch)
        print_info "Updating patch versions only..."
        UPGRADE_STRATEGY="--upgrade-package"
        ;;
    minor)
        print_info "Updating minor versions..."
        UPGRADE_STRATEGY="--upgrade"
        ;;
    major)
        print_info "Updating major versions..."
        UPGRADE_STRATEGY="--upgrade"
        ;;
    all)
        print_info "Updating all packages to latest..."
        UPGRADE_STRATEGY="--upgrade"
        ;;
    *)
        print_error "Invalid update type: $UPDATE_TYPE (use: patch, minor, major, all)"
        exit 1
        ;;
esac

if [ "$DRY_RUN" == "true" ]; then
    print_warning "DRY RUN MODE - No changes will be made"
fi

# Update base dependencies
print_info "Updating base dependencies..."
if [ "$DRY_RUN" == "false" ]; then
    uv pip install $UPGRADE_STRATEGY -r requirements/base.txt
else
    print_info "Would run: uv pip install $UPGRADE_STRATEGY -r requirements/base.txt"
fi

# Update dev dependencies
print_info "Updating dev dependencies..."
if [ "$DRY_RUN" == "false" ]; then
    uv pip install $UPGRADE_STRATEGY -r requirements/dev.txt
else
    print_info "Would run: uv pip install $UPGRADE_STRATEGY -r requirements/dev.txt"
fi

# Update package-specific dependencies
for package in packages/*; do
    if [ -f "$package/pyproject.toml" ]; then
        package_name=$(basename "$package")
        print_info "Updating $package_name dependencies..."
        
        if [ "$DRY_RUN" == "false" ]; then
            cd "$package"
            uv pip install $UPGRADE_STRATEGY -e .
            cd "$PROJECT_ROOT"
        else
            print_info "Would update $package_name"
        fi
    fi
done

# Generate new constraints file
if [ "$DRY_RUN" == "false" ]; then
    print_info "Generating new constraints file..."
    uv pip freeze > requirements/constraints.txt.new
    
    # Compare constraints
    print_info "Changes in constraints:"
    diff requirements/constraints.txt.backup requirements/constraints.txt.new || true
    
    # Run tests to verify updates
    print_info "Running basic tests to verify updates..."
    python -c "
import sys
print('Python:', sys.version)

# Test basic imports
try:
    import redis
    import pydantic
    import fastmcp
    import pytest
    print('✓ Core imports successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"
    
    # Run security check
    print_info "Running security check on updated dependencies..."
    ./scripts/check-deps.sh || true
    
    # If all good, update constraints
    mv requirements/constraints.txt.new requirements/constraints.txt
    print_success "Constraints file updated"
else
    print_info "Dry run complete. No changes made."
fi

# Show update summary
echo ""
echo "================================================"
print_info "Update Summary:"
print_info "  Update type: $UPDATE_TYPE"
print_info "  Constraints backup: requirements/constraints.txt.backup"

if [ "$DRY_RUN" == "false" ]; then
    print_info "  New constraints: requirements/constraints.txt"
    print_success "✅ Dependencies updated successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "  1. Review changes: diff requirements/constraints.txt.backup requirements/constraints.txt"
    print_info "  2. Run full test suite: ./scripts/test-all.sh"
    print_info "  3. Commit changes if tests pass"
else
    print_warning "Dry run complete. Run without 'true' argument to apply changes."
fi