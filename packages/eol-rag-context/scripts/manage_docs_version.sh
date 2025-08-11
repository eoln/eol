#!/bin/bash
# Documentation version management script using mike

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  deploy <version> [alias]  Deploy documentation for a specific version"
    echo "  list                       List all deployed versions"
    echo "  delete <version>           Delete a specific version"
    echo "  set-default <version>      Set the default version"
    echo "  alias <version> <alias>    Create an alias for a version"
    echo "  serve                      Serve documentation locally with versions"
    echo ""
    echo "Examples:"
    echo "  $0 deploy 1.0.0 latest    # Deploy v1.0.0 and alias it as 'latest'"
    echo "  $0 deploy dev              # Deploy development version"
    echo "  $0 set-default 1.0.0       # Set v1.0.0 as default"
    echo "  $0 serve                   # Serve docs locally"
}

# Check if mike is installed
check_mike() {
    if ! command -v mike &> /dev/null; then
        print_error "mike is not installed. Install it with: pip install mike"
        exit 1
    fi
}

# Deploy documentation version
deploy_version() {
    local version=$1
    local alias=$2
    
    if [ -z "$version" ]; then
        print_error "Version is required"
        show_usage
        exit 1
    fi
    
    print_info "Building documentation..."
    mkdocs build --clean
    
    if [ -n "$alias" ]; then
        print_info "Deploying version $version with alias $alias..."
        mike deploy --push --update-aliases "$version" "$alias"
    else
        print_info "Deploying version $version..."
        mike deploy --push "$version"
    fi
    
    print_info "Documentation deployed successfully!"
}

# List all versions
list_versions() {
    print_info "Deployed documentation versions:"
    mike list
}

# Delete a version
delete_version() {
    local version=$1
    
    if [ -z "$version" ]; then
        print_error "Version is required"
        show_usage
        exit 1
    fi
    
    print_warning "Deleting version $version..."
    mike delete --push "$version"
    print_info "Version $version deleted"
}

# Set default version
set_default() {
    local version=$1
    
    if [ -z "$version" ]; then
        print_error "Version is required"
        show_usage
        exit 1
    fi
    
    print_info "Setting $version as default..."
    mike set-default --push "$version"
    print_info "Default version set to $version"
}

# Create alias
create_alias() {
    local version=$1
    local alias=$2
    
    if [ -z "$version" ] || [ -z "$alias" ]; then
        print_error "Both version and alias are required"
        show_usage
        exit 1
    fi
    
    print_info "Creating alias $alias for version $version..."
    mike alias --push "$version" "$alias"
    print_info "Alias created successfully"
}

# Serve documentation locally
serve_docs() {
    print_info "Serving documentation locally..."
    print_info "Access at http://localhost:8000"
    mike serve
}

# Main script logic
check_mike

case "$1" in
    deploy)
        deploy_version "$2" "$3"
        ;;
    list)
        list_versions
        ;;
    delete)
        delete_version "$2"
        ;;
    set-default)
        set_default "$2"
        ;;
    alias)
        create_alias "$2" "$3"
        ;;
    serve)
        serve_docs
        ;;
    *)
        show_usage
        exit 1
        ;;
esac