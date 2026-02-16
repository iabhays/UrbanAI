#!/bin/bash

# SENTIENTCITY Setup Verification Script
# Ensures all configurations are correct before running Start.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   SENTIENTCITY Setup Verification                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check function
check_file() {
    local file="$1"
    local description="$2"
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description - FILE NOT FOUND: $file"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Check directory
check_dir() {
    local dir="$1"
    local description="$2"
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description - DIRECTORY NOT FOUND: $dir"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Check command
check_command() {
    local cmd="$1"
    local description="$2"
    
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $description - NOT INSTALLED: $cmd"
        WARNINGS=$((WARNINGS + 1))
        return 1
    fi
}

echo -e "${BLUE}[1] Checking Project Structure${NC}"
echo "────────────────────────────────────────────────────────────"
check_dir "$PROJECT_ROOT/dashboard/react_ui" "Dashboard Vite project exists"
check_dir "$PROJECT_ROOT/sentient_city" "Backend Python package exists"
check_dir "$PROJECT_ROOT/deployment/docker" "Docker configuration directory exists"
check_dir "$PROJECT_ROOT/configs" "Configuration directory exists"
echo ""

echo -e "${BLUE}[2] Checking Vite Configuration${NC}"
echo "────────────────────────────────────────────────────────────"
check_file "$PROJECT_ROOT/dashboard/react_ui/vite.config.js" "Vite config exists"
check_file "$PROJECT_ROOT/dashboard/react_ui/package.json" "Package.json exists"
check_file "$PROJECT_ROOT/dashboard/react_ui/src/main.jsx" "React entry point exists"
echo ""

echo -e "${BLUE}[3] Checking Docker Configuration${NC}"
echo "────────────────────────────────────────────────────────────"
check_file "$PROJECT_ROOT/deployment/docker/Dockerfile.dashboard" "Dashboard Dockerfile exists"
check_file "$PROJECT_ROOT/deployment/docker/nginx.conf" "Nginx configuration exists"
check_file "$PROJECT_ROOT/docker-compose.yml" "docker-compose.yml exists"
echo ""

echo -e "${BLUE}[4] Checking Startup Scripts${NC}"
echo "────────────────────────────────────────────────────────────"
check_file "$PROJECT_ROOT/Start.sh" "Start.sh exists"
if [ -x "$PROJECT_ROOT/Start.sh" ]; then
    echo -e "${GREEN}✓${NC} Start.sh is executable"
else
    echo -e "${YELLOW}⚠${NC} Start.sh is NOT executable - fixing..."
    chmod +x "$PROJECT_ROOT/Start.sh"
    echo -e "${GREEN}✓${NC} Start.sh made executable"
fi
echo ""

echo -e "${BLUE}[5] Checking Required Commands${NC}"
echo "────────────────────────────────────────────────────────────"
check_command "docker" "Docker CLI installed"
check_command "python3" "Python 3 installed"
check_command "npm" "Node.js npm installed"
echo ""

echo -e "${BLUE}[6] Verifying Dockerfile Content${NC}"
echo "────────────────────────────────────────────────────────────"

# Check if Dockerfile has correct Vite paths
if grep -q "dashboard/react_ui" "$PROJECT_ROOT/deployment/docker/Dockerfile.dashboard"; then
    echo -e "${GREEN}✓${NC} Dockerfile references correct dashboard path"
else
    echo -e "${RED}✗${NC} Dockerfile does NOT reference dashboard/react_ui - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "/app/dist" "$PROJECT_ROOT/deployment/docker/Dockerfile.dashboard"; then
    echo -e "${GREEN}✓${NC} Dockerfile uses dist/ output (Vite correct)"
else
    echo -e "${RED}✗${NC} Dockerfile does NOT use dist/ output - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "npm run build" "$PROJECT_ROOT/deployment/docker/Dockerfile.dashboard"; then
    echo -e "${GREEN}✓${NC} Dockerfile runs npm run build"
else
    echo -e "${RED}✗${NC} Dockerfile does NOT run npm run build - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo -e "${BLUE}[7] Verifying Nginx Configuration${NC}"
echo "────────────────────────────────────────────────────────────"

if grep -q "try_files \$uri /index.html" "$PROJECT_ROOT/deployment/docker/nginx.conf"; then
    echo -e "${GREEN}✓${NC} Nginx has SPA routing configured"
else
    echo -e "${RED}✗${NC} Nginx missing SPA routing - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "/assets/" "$PROJECT_ROOT/deployment/docker/nginx.conf"; then
    echo -e "${GREEN}✓${NC} Nginx has Vite asset caching configured"
else
    echo -e "${RED}✗${NC} Nginx missing Vite asset caching - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "proxy_pass http://backend" "$PROJECT_ROOT/deployment/docker/nginx.conf"; then
    echo -e "${GREEN}✓${NC} Nginx has API proxy configured"
else
    echo -e "${RED}✗${NC} Nginx missing API proxy - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo -e "${BLUE}[8] Verifying docker-compose.yml${NC}"
echo "────────────────────────────────────────────────────────────"

if grep -q "REACT_APP_API_URL" "$PROJECT_ROOT/docker-compose.yml"; then
    echo -e "${GREEN}✓${NC} docker-compose has correct env var (REACT_APP_API_URL)"
else
    echo -e "${RED}✗${NC} docker-compose has wrong env var - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi

if grep -q '"3000:80"' "$PROJECT_ROOT/docker-compose.yml" || grep -q "'3000:80'" "$PROJECT_ROOT/docker-compose.yml"; then
    echo -e "${GREEN}✓${NC} docker-compose has correct port mapping (3000:80)"
else
    echo -e "${RED}✗${NC} docker-compose has wrong port mapping - NEEDS FIX"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo -e "${BLUE}[9] Checking for Incorrect Files${NC}"
echo "────────────────────────────────────────────────────────────"

if [ -f "$PROJECT_ROOT/dashboard/next.config.js" ]; then
    echo -e "${RED}✗${NC} Found incorrect next.config.js (Vite doesn't need this)"
    echo "   Remove with: rm $PROJECT_ROOT/dashboard/next.config.js"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓${NC} No incorrect Next.js config files found"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      VERIFICATION SUMMARY                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready to run ./Start.sh${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./Start.sh"
    echo "  2. Wait for startup to complete"
    echo "  3. Open: http://localhost:3000 in your browser"
    echo ""
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found. Please fix before running Start.sh${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) - may affect functionality${NC}"
    fi
    echo ""
    exit 1
fi
