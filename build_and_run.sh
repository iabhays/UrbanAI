#!/bin/bash

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Building SentientCity Docker Images"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ensure Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker daemon is not responding"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Build individual services
echo "📦 Building Dashboard..."
docker build -f deployment/docker/Dockerfile.dashboard -t sentientcity-dashboard . --progress=plain

echo ""
echo "✓ Dashboard build complete"
echo ""

echo "🚀 Starting services with docker-compose..."
docker-compose up -d

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Services started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Services:"
echo "  🌐 Dashboard:  http://localhost:3000"
echo "  🔌 Backend API: http://localhost:8000"
echo ""
echo "View logs: docker-compose logs -f"
echo "Stop:      docker-compose down"
