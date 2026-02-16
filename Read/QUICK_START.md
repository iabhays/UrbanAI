# 🚀 SENTIENTCITY Quick Start Guide

## Prerequisites Installed ✅
- Docker Desktop (verified)
- Python 3.10+ (verified)
- Node.js npm (verified)
- Project files in `/Users/abhijeetkumar/Desktop/SENTIENTCITY`

---

## Start SENTIENTCITY (1 Command)

```bash
cd /Users/abhijeetkumar/Desktop/SENTIENTCITY
./Start.sh
```

**What happens:**
1. Verifies Docker is installed and running
2. Builds Vite React app to static files
3. Starts all services (Backend, Dashboard, Kafka, Redis, Zoekeeper)
4. Runs health checks
5. Shows dashboard URL

**Expected output:**
```
✓ Docker is installed
✓ Docker daemon is running
✓ Building services...
✓ Waiting for services to start...
✓ Health check passed
✓ Dashboard ready: http://localhost:3000
```

---

## Access Dashboard

### In Browser
```
http://localhost:3000
```

### What You'll See
- React application running on Nginx
- Connects to backend at http://localhost:8000
- Real-time updates via WebSocket

---

## Service Endpoints

| Service | URL | Port |
|---------|-----|------|
| **Dashboard** | http://localhost:3000 | 3000 |
| **API** | http://localhost:8000 | 8000 |
| **Health Check** | http://localhost:8000/api/v1/health | 8000 |
| **Kafka** | localhost:9092 | 9092 |
| **Redis** | localhost:6379 | 6379 |

---

## View Logs

```bash
# All services
docker-compose logs -f

# Dashboard only
docker-compose logs -f dashboard

# Backend only
docker-compose logs -f backend
```

---

## Stop Services

```bash
docker-compose down
```

---

## Restart Services

```bash
./Start.sh
```

Or:
```bash
docker-compose restart
```

---

## Development vs Production

### Production (What ./Start.sh Uses)
✅ Docker-based
✅ Optimized Vite build
✅ Nginx serving static files
✅ Recommended for testing real setup

### Development (If Modifying Frontend)
```bash
cd dashboard/react_ui
npm install
npm run dev
```
Then open http://localhost:3000 (Vite dev server with HMR)

---

## Troubleshooting

### Dashboard shows blank page
```bash
# Check Nginx is serving files
curl http://localhost:3000/index.html

# Rebuild
docker-compose down
./Start.sh
```

### API calls failing
```bash
# Check backend
curl http://localhost:8000/api/v1/health

# View logs
docker-compose logs backend
```

### Docker won't start
```bash
# Ensure Docker Desktop is running
docker ps

# Start Docker Desktop manually if needed
```

### Port already in use
```bash
# Kill process using port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port in docker-compose.yml
# Change: ports: - "3000:80" to "3001:80"
```

---

## Architecture

```
Browser (localhost:3000)
    ↓
Nginx Container (port 80)
    ├─→ Static files from Vite build
    ├─→ /api/* → Backend (port 8000)
    └─→ /ws/* → WebSocket to Backend
        ↓
    FastAPI Backend
    ├─→ Kafka (event streaming)
    ├─→ Redis (caching)
    └─→ ML models (inference)
```

---

## Key Files

- **[Start.sh](./Start.sh)** - Main startup script
- **[docker-compose.yml](./docker-compose.yml)** - Service configuration
- **[deployment/docker/Dockerfile.dashboard](./deployment/docker/Dockerfile.dashboard)** - Dashboard Docker build
- **[deployment/docker/nginx.conf](./deployment/docker/nginx.conf)** - Nginx configuration
- **[VITE_PRODUCTION_SETUP.md](./VITE_PRODUCTION_SETUP.md)** - Detailed documentation

---

## Framework Details

**Frontend:** React 18 + Vite 5
- Source: `dashboard/react_ui/`
- Build output: `dist/` (optimized static files)
- Web server: Nginx Alpine
- Routing: SPA (client-side via React Router)

**Backend:** FastAPI
- Port: 8000
- Health: `/api/v1/health`
- API prefix: `/api/`

---

## Next Steps

1. **Run start:** `./Start.sh`
2. **Open browser:** http://localhost:3000
3. **Check logs:** `docker-compose logs -f`
4. **Develop:** Modify code and rebuild with `./Start.sh` or edit frontend files and use `npm run dev`

---

**Status:** ✅ Production Ready
**Last Updated:** February 7, 2025
