# SENTIENTCITY - Startup Script Setup Complete ✓

## 📋 Summary

A production-grade startup script has been created to automate the complete SENTIENTCITY deployment with a single command.

---

## 🚀 Quick Start

### Run the startup script:

```bash
cd /Users/abhijeetkumar/Desktop/SENTIENTCITY
./Start.sh
```

That's it! The script will:
1. ✓ Check Docker CLI installation
2. ✓ Verify Docker daemon is running (auto-start if needed)
3. ✓ Build all Docker images
4. ✓ Start all services (Zookeeper, Kafka, Redis, Backend, Dashboard)
5. ✓ Health check services
6. ✓ Display dashboard URL

---

## 📁 Files Created/Modified

### 1. **Start.sh** (NEW)
- **Location**: `/Users/abhijeetkumar/Desktop/SENTIENTCITY/Start.sh`
- **Size**: 8.1 KB
- **Purpose**: Production startup script with:
  - Docker daemon auto-detection and startup
  - Service orchestration via docker-compose
  - Health checks for all services
  - Clear, colored output logging
  - Error handling and timeout management

### 2. **deployment/docker/nginx.conf** (NEW)
- **Location**: `/Users/abhijeetkumar/Desktop/SENTIENTCITY/deployment/docker/nginx.conf`
- **Size**: 1.7 KB
- **Purpose**: Nginx configuration for dashboard
  - SPA routing (all requests → index.html)
  - API proxy to backend (http://backend:8000)
  - WebSocket support
  - Gzip compression
  - Health check endpoint

### 3. **deployment/docker/Dockerfile.dashboard** (UPDATED)
- **Previous**: Used Node.js `npm run preview`
- **New**: Multi-stage build with Nginx
- **Changes**:
  - Build stage: Node.js for building React app
  - Production stage: Nginx for serving static files
  - Health checks included
  - Reduced image size

### 4. **docker-compose.yml** (FIXED)
- **Removed**: Deprecated `version: '3.8'` field
- **Fixed**: Dashboard build context paths
- **Verified**: All service configurations

---

## 📊 Architecture

```
┌─────────────────────────────────────────┐
│  Start.sh (Script Entry Point)          │
└────────────┬────────────────────────────┘
             │
             ├─→ Check Docker CLI
             ├─→ Verify Docker daemon
             │   └─→ Auto-start if needed
             ├─→ docker-compose build
             ├─→ docker-compose up -d
             └─→ Health checks
                 └─→ Display dashboard URL
```

---

## 🌐 Service Endpoints

Once `./Start.sh` completes:

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:3000 | React UI (Nginx served) |
| **Backend API** | http://localhost:8000 | FastAPI service |
| **API Health** | http://localhost:8000/api/v1/health | Backend health check |
| **Kafka** | localhost:9092 | Event streaming |
| **Redis** | localhost:6379 | Caching |
| **Zookeeper** | localhost:2181 | Kafka coordination |

---

## 🛠️ Script Features

### ✓ Docker Management
- Detects Docker CLI installation
- Checks if Docker daemon is running
- Automatically opens Docker Desktop if needed
- 60-second timeout with user feedback
- Color-coded status messages

### ✓ Service Orchestration
- Builds images with cache invalidation
- Starts all services in correct order
- Waits for service readiness
- Health checks (30 attempts, 2-second intervals)

### ✓ Error Handling
- Exits with error code 1 on failure
- Detailed error messages
- Logging suggestions (docker-compose logs)
- No interactive prompts (fully automated)

### ✓ User Experience
- Clear section headers
- Color-coded output (✓✗⚠ℹ)
- Progress indicators
- Final summary with useful commands

---

## 📝 Useful Commands

After starting with `./Start.sh`:

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f dashboard
docker-compose logs -f kafka

# Check service status
docker-compose ps

# Stop all services
docker-compose down

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# Remove volumes (data cleanup)
docker-compose down -v
```

---

## 🔍 Production Readiness Checklist

- [x] Single entry point script
- [x] Automated Docker startup
- [x] Health checks for all services
- [x] Proper error handling
- [x] No manual steps required
- [x] Nginx production web server
- [x] Multi-stage Docker builds
- [x] Proper service dependencies
- [x] Clear logging and output
- [x] macOS compatible (bash/zsh)

---

## ⚠️ Important Notes

1. **First Run**: Script will download ~2-3GB of Docker images (one-time only)
2. **Docker Desktop**: Must be available on the system
3. **Ports**: Ensure ports 3000, 8000, 6379, 9092, 2181 are available
4. **Network**: Services communicate via Docker internal network
5. **Build Time**: First build takes 5-10 minutes depending on internet speed

---

## 🐛 Troubleshooting

### Docker daemon won't start
```bash
# Check if Docker Desktop is in Applications
ls /Applications/Docker.app

# Manually start Docker Desktop
open -a Docker
```

### Port already in use
```bash
# Find process using port
lsof -i :3000

# Kill process if needed
kill -9 <PID>
```

### Services failing to start
```bash
# Check detailed logs
docker-compose logs backend
docker-compose logs dashboard

# Rebuild with fresh images
docker-compose down
./Start.sh
```

### Out of disk space
```bash
# Clean up Docker resources
docker system prune -a
```

---

## 📚 Documentation References

- [DEPLOYMENT.md](../DEPLOYMENT.md) - Full deployment guide
- [README.md](../README.md) - Project overview
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture

---

## ✅ Verification

To verify everything is working:

```bash
# 1. Run startup script
./Start.sh

# 2. Open dashboard in browser
open http://localhost:3000

# 3. Check backend
curl http://localhost:8000/api/v1/health

# 4. View logs
docker-compose logs -f
```

---

**Status**: ✅ Ready for Production
**Last Updated**: February 7, 2026
**Compatibility**: macOS with Docker Desktop
