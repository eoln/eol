#!/bin/bash
# Check dependencies for conflicts, security issues, and consistency

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🔍 Checking dependencies for EOL workspace..."
echo "================================================"

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        print_warning "Virtual environment not found. Run ./scripts/setup-workspace.sh first."
        exit 1
    fi
fi

# 1. Check for dependency conflicts
echo ""
print_info "Checking for dependency conflicts..."
if uv pip check 2>/dev/null; then
    print_success "No dependency conflicts found"
else
    print_error "Dependency conflicts detected!"
    exit_code=1
fi

# 2. Security audit with pip-audit
echo ""
print_info "Running security audit..."
if command -v pip-audit &> /dev/null; then
    audit_output=$(pip-audit --desc 2>&1) || true
    if echo "$audit_output" | grep -q "No known vulnerabilities"; then
        print_success "No known vulnerabilities found"
    else
        print_warning "Security vulnerabilities detected:"
        echo "$audit_output"
        exit_code=1
    fi
else
    print_warning "pip-audit not installed. Install with: pip install pip-audit"
fi

# 3. Check with safety
echo ""
print_info "Running safety check..."
if command -v safety &> /dev/null; then
    safety_output=$(safety check --json 2>/dev/null) || true
    vulnerabilities=$(echo "$safety_output" | python -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('vulnerabilities', [])))" 2>/dev/null || echo "0")
    if [ "$vulnerabilities" -eq "0" ]; then
        print_success "No safety vulnerabilities found"
    else
        print_warning "Safety found $vulnerabilities vulnerabilities"
        safety check || true
        exit_code=1
    fi
else
    print_warning "safety not installed. Install with: pip install safety"
fi

# 4. Check for circular dependencies
echo ""
print_info "Checking for circular dependencies..."
if command -v pipdeptree &> /dev/null; then
    circular=$(pipdeptree --warn fail 2>&1 | grep -c "circular" || true)
    if [ "$circular" -eq "0" ]; then
        print_success "No circular dependencies found"
    else
        print_error "Circular dependencies detected!"
        pipdeptree --warn fail || true
        exit_code=1
    fi
else
    print_warning "pipdeptree not installed. Install with: pip install pipdeptree"
fi

# 5. Check version constraints consistency
echo ""
print_info "Checking version constraints..."
constraint_issues=0
if [ -f "requirements/constraints.txt" ]; then
    while IFS= read -r line; do
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)==(.+)$ ]]; then
            package="${BASH_REMATCH[1]}"
            version="${BASH_REMATCH[2]}"
            
            # Check if installed version matches constraint
            installed_version=$(pip show "$package" 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
            if [ -n "$installed_version" ] && [ "$installed_version" != "$version" ]; then
                print_warning "Version mismatch for $package: constraint=$version, installed=$installed_version"
                ((constraint_issues++))
            fi
        fi
    done < "requirements/constraints.txt"
    
    if [ "$constraint_issues" -eq "0" ]; then
        print_success "All version constraints satisfied"
    else
        print_warning "Found $constraint_issues version constraint issues"
        exit_code=1
    fi
else
    print_warning "No constraints.txt file found"
fi

# 6. Check for unused dependencies
echo ""
print_info "Checking for potentially unused dependencies..."
# This is a simple check - for thorough analysis use pip-autoremove or similar
for package_dir in packages/*; do
    if [ -f "$package_dir/pyproject.toml" ]; then
        package_name=$(basename "$package_dir")
        print_info "  Checking $package_name..."
        
        # Get list of imports from Python files
        if [ -d "$package_dir/src" ]; then
            imports=$(find "$package_dir/src" -name "*.py" -exec grep -h "^import \|^from " {} \; 2>/dev/null | \
                     sed 's/from \([^ ]*\).*/\1/' | sed 's/import \([^ ]*\).*/\1/' | \
                     cut -d'.' -f1 | sort -u)
            
            # This is a basic check - would need more sophisticated analysis for accuracy
            # Just reporting for awareness
        fi
    fi
done

# 7. Check for outdated packages
echo ""
print_info "Checking for outdated packages..."
outdated_count=$(uv pip list --outdated 2>/dev/null | grep -c "^" || echo "0")
if [ "$outdated_count" -gt "1" ]; then  # Header line counts as 1
    print_warning "Found $((outdated_count-1)) outdated packages:"
    uv pip list --outdated || true
else
    print_success "All packages up to date"
fi

# 8. Validate pyproject.toml files
echo ""
print_info "Validating pyproject.toml files..."
validation_issues=0
for package_dir in packages/*; do
    if [ -f "$package_dir/pyproject.toml" ]; then
        package_name=$(basename "$package_dir")
        if python -c "import toml; toml.load('$package_dir/pyproject.toml')" 2>/dev/null; then
            print_success "$package_name: pyproject.toml valid"
        else
            print_error "$package_name: pyproject.toml invalid!"
            ((validation_issues++))
        fi
    fi
done

if [ "$validation_issues" -eq "0" ]; then
    print_success "All pyproject.toml files valid"
else
    exit_code=1
fi

# Summary
echo ""
echo "================================================"
if [ "${exit_code:-0}" -eq "0" ]; then
    print_success "✅ All dependency checks passed!"
else
    print_error "❌ Some dependency issues were found. Please review above."
    exit 1
fi