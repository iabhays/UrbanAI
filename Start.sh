#!/bin/bash

################################################################################
# SENTIENTCITY - Production Startup Script
#
# This script automates the complete startup of SENTIENTCITY with:
# - Docker validation and automatic startup
# - Service orchestration via docker-compose
# - Automatic health checks
################################################################################

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CHECK_INTERVAL=2
DOCKER_CHECK_TIMEOUT=60
DOCKER_STARTUP_WAIT=30

################################################################################
# Logging Functions
################################################################################

log_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Docker CLI Check
################################################################################

check_docker_cli() {
    log_section "Checking Docker CLI Installation"
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker CLI is not installed"
        echo "   Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    DOCKER_VERSION=$(docker --version)
    log_success "Docker CLI found"
    log_info "$DOCKER_VERSION"
}

################################################################################
# Docker Daemon Checks
################################################################################

is_docker_running() {
    docker ps > /dev/null 2>&1
    return $?
}

wait_for_docker() {
    local elapsed=0
    
    log_section "Waiting for Docker Daemon to Start"
    log_info "Timeout: ${DOCKER_CHECK_TIMEOUT}s (checking every ${DOCKER_CHECK_INTERVAL}s)"
    
    while ! is_docker_running; do
        if [ $elapsed -ge $DOCKER_CHECK_TIMEOUT ]; then
            log_error "Docker daemon failed to start within ${DOCKER_CHECK_TIMEOUT}s"
            exit 1
        fi
        
        echo -ne "\r   Waiting... ${elapsed}s"
        sleep $DOCKER_CHECK_INTERVAL
        elapsed=$((elapsed + DOCKER_CHECK_INTERVAL))
    done
    
    echo -ne "\r"
    log_success "Docker daemon is running"
}

start_docker_desktop() {
    log_section "Starting Docker Desktop"
    
    if ! is_docker_running; then
        log_info "Opening Docker Desktop application..."
        open -a Docker 2>/dev/null || {
            log_error "Failed to open Docker Desktop"
            log_info "Please manually start Docker Desktop and run this script again"
            exit 1
        }
        
        log_info "Waiting ${DOCKER_STARTUP_WAIT}s for Docker to initialize..."
        sleep $DOCKER_STARTUP_WAIT
        
        wait_for_docker
    else
        log_success "Docker daemon is already running"
    fi
}

check_docker_daemon() {
    log_section "Checking Docker Daemon"
    
    if is_docker_running; then
        log_success "Docker daemon is running"
        docker info --format='Server Version: {{.ServerVersion}}'
    else
        start_docker_desktop
    fi
}

################################################################################
# Docker Compose Operations
################################################################################

build_images() {
    log_section "Building Docker Images"
    log_info "Building frontend and backend services..."
    
    cd "$SCRIPT_DIR"
    
    if ! docker-compose build --no-cache; then
        log_error "Failed to build Docker images"
        log_info "Run 'docker-compose logs' to see detailed error messages"
        exit 1
    fi
    
    log_success "Docker images built successfully"
}

start_services() {
    log_section "Starting Services"
    log_info "Starting backend, dashboard, Kafka, Redis, and Zookeeper..."
    
    cd "$SCRIPT_DIR"
    
    if ! docker-compose up -d; then
        log_error "Failed to start services"
        log_info "Run 'docker-compose logs' to see detailed error messages"
        exit 1
    fi
    
    log_success "Services started successfully"
}

check_service_health() {
    log_section "Waiting for Services to Be Ready"
    
    local backend_ready=false
    local dashboard_ready=false
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        # Check backend health
        if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            log_success "Backend API is ready"
            backend_ready=true
        else
            echo -ne "\r   Waiting for backend... attempt $((attempt + 1))/$max_attempts"
        fi
        
        # Check dashboard health
        if curl -s http://localhost:3000/health > /dev/null 2>&1; then
            log_success "Dashboard is ready"
            dashboard_ready=true
        else
            echo -ne "\r   Waiting for dashboard... attempt $((attempt + 1))/$max_attempts"
        fi
        
        if [ "$backend_ready" = true ] && [ "$dashboard_ready" = true ]; then
            echo -ne "\r"
            break
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -ne "\r"
    
    if [ "$backend_ready" = false ]; then
        log_warning "Backend API health check failed (may still be starting)"
        log_info "Run: docker-compose logs backend"
    fi
    
    if [ "$dashboard_ready" = false ]; then
        log_warning "Dashboard health check failed (may still be starting)"
        log_info "Run: docker-compose logs dashboard"
    fi
}

show_service_status() {
    log_section "Service Status"
    
    docker-compose ps
    
    echo ""
    log_info "Running services:"
    docker-compose ps --services | while read service; do
        log_success "$service"
    done
}

################################################################################
# Main Execution
################################################################################

main() {
    log_section "SENTIENTCITY AI - Startup Script"
    log_info "Starting all services for production deployment"
    
    # Step 1: Validate Docker CLI
    check_docker_cli
    
    # Step 2: Ensure Docker daemon is running
    check_docker_daemon
    
    # Step 3: Build Docker images
    build_images
    
    # Step 4: Start services
    start_services
    
    # Step 5: Wait for services to be healthy
    check_service_health
    
    # Step 6: Show final status
    show_service_status
    
    # Final summary
    log_section "✨ SENTIENTCITY is Ready"
    echo ""
    echo -e "${GREEN}All services have been started successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Dashboard URL:${NC}"
    echo -e "  → ${BLUE}http://localhost:3000${NC}"
    echo ""
    echo -e "${YELLOW}Backend API:${NC}"
    echo -e "  → ${BLUE}http://localhost:8000${NC}"
    echo -e "  → Health: ${BLUE}http://localhost:8000/api/v1/health${NC}"
    echo ""
    echo -e "${YELLOW}Useful Commands:${NC}"
    echo "  View logs:        ${BLUE}docker-compose logs -f${NC}"
    echo "  Backend logs:     ${BLUE}docker-compose logs -f backend${NC}"
    echo "  Dashboard logs:   ${BLUE}docker-compose logs -f dashboard${NC}"
    echo "  Stop services:    ${BLUE}docker-compose down${NC}"
    echo "  Restart services: ${BLUE}docker-compose restart${NC}"
    echo ""
    echo -e "${GREEN}Happy monitoring! 🚀${NC}"
    echo ""
}

# Execute main function
main
exit 0
