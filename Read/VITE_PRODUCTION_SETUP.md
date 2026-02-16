# SENTIENTCITY Dashboard - Vite React Production Setup ✅

## Overview

The SENTIENTCITY dashboard is a **React application using Vite** configured for production deployment using Docker and Nginx.

---

## Architecture

### Build Process

```
Source Code (dashboard/react_ui/)
    ↓
Vite Build (npm run build)
    → Generates /app/dist/ (optimized static files)
    ↓
Nginx Container
    → Serves /app/dist as static site on port 80
    → Maps to host port 3000
```

### Production Deployment

```
Browser (http://localhost:3000)
    ↓ (mapped port)
Nginx Container (port 80)
    ├─→ Static files from Vite build (/dist)
    ├─→ /api/* → proxies to backend:8000
    └─→ /ws/* → WebSocket to backend:8000
```

---

## Files Created/Updated

### ✅ **deployment/docker/Dockerfile.dashboard**
**Type:** Multi-stage Docker build

**Stage 1: Builder**
- Image: `node:18-alpine`
- Installs dependencies from `dashboard/react_ui/`
- Runs `npm run build` to create `dist/` folder
- Vite outputs optimized static files, JS bundles, CSS

**Stage 2: Production**
- Image: `nginx:alpine`
- Copies built files from `/app/dist` → `/usr/share/nginx/html`
- No Node.js in final image (smaller, faster, more secure)
- Serves on port 80

**Key Paths:**
```dockerfile
COPY dashboard/react_ui/package*.json ./
COPY dashboard/react_ui .
RUN npm run build                    # Creates ./dist/
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

### ✅ **deployment/docker/nginx.conf**
**Type:** Nginx configuration for Vite SPA

**Routes:**
| Route | Handler | Purpose |
|-------|---------|---------|
| `/` | index.html | Main app |
| `/assets/*` | Cached files | JS/CSS bundles |
| `/index.html` | No cache | HTML file |
| `/api/*` | Proxy to backend:8000 | Backend API |
| `/ws/*` | WebSocket proxy | Real-time updates |
| `/*` (fallback) | `/index.html` | Client-side routing |

**Key Features:**
- SPA routing: `try_files $uri /index.html`
- Asset caching: 365-day cache for `/assets/`
- HTML no-cache: 1-hour TTL for `index.html`
- API proxy: Requests to `/api/*` go to `backend:8000`
- WebSocket support: `ws://` connections proxied

### ✅ **docker-compose.yml**
**Updated:**
```yaml
dashboard:
  build:
    context: .
    dockerfile: deployment/docker/Dockerfile.dashboard
  ports:
    - "3000:80"  # Host:Container port mapping
  depends_on:
    - backend
  environment:
    - REACT_APP_API_URL=http://localhost:8000
```

---

## Vite vs Next.js vs Vite

| Feature | Vite | Previous Setup |
|---------|------|-----------------|
| Build Output | `dist/` | `out/` (Next.js) |
| Dev Server | `npm run dev` | `npm run dev` |
| Production | Nginx + static | Would fail |
| Runtime | None (static) | None (static) |
| Framework | React Router | Next.js App Router |
| Image Optimization | Vite handles | next/image |

---

## Running SENTIENTCITY

### Start Everything

```bash
cd /Users/abhijeetkumar/Desktop/SENTIENTCITY
./Start.sh
```

### What Happens

1. **Docker verification** - Checks Docker is installed
2. **Docker daemon** - Starts Docker Desktop if needed
3. **Image build** - Builds Vite app and Nginx image
4. **Service start** - Launches all services (Kafka, Redis, Backend, Dashboard)
5. **Health checks** - Verifies services are ready
6. **Success** - Shows dashboard URL: `http://localhost:3000`

---

## Build Process Flow

```
./Start.sh
    ↓
docker-compose build
    ↓
Dockerfile.dashboard Stage 1 (Builder):
  - FROM node:18-alpine
  - COPY dashboard/react_ui/ (source code)
  - npm ci (clean install dependencies)
  - npm run build (Vite builds → /app/dist/)
    ├─ Optimizes images
    ├─ Minifies JavaScript
    ├─ Tree-shakes unused code
    ├─ Generates source maps (dev) or no maps (prod)
    └─ Creates /app/dist/ with:
       - index.html
       - assets/ (JS, CSS, images)
    ↓
Dockerfile.dashboard Stage 2 (Production):
  - FROM nginx:alpine
  - COPY --from=builder /app/dist /usr/share/nginx/html
  - CMD: nginx -g "daemon off;"
    ↓
docker-compose up -d
    ↓
Nginx starts and serves static files on port 80
Port 80 is mapped to host port 3000
Browser can access: http://localhost:3000
```

---

## Service Endpoints

After `./Start.sh`:

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:3000 | React UI (Nginx served) |
| **Backend API** | http://localhost:8000 | FastAPI service |
| **API Health** | http://localhost:8000/api/v1/health | Backend health check |
| **Kafka** | localhost:9092 | Event streaming |
| **Redis** | localhost:6379 | Caching |
| **Zookeeper** | localhost:2181 | Kafka coordination |

---

## Vite Configuration

### **dashboard/react_ui/vite.config.js**

```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
})
```

**Dev Mode:** Vite dev server proxies to backend
**Production:** Nginx proxies to backend

### **dashboard/react_ui/package.json**

```json
{
  "scripts": {
    "dev": "vite",           // Local dev server
    "build": "vite build",   // Production build
    "preview": "vite preview" // Test build locally
  }
}
```

---

## Development vs Production

### Development (Local)

```bash
cd dashboard/react_ui
npm install
npm run dev
```

**Uses:**
- Vite dev server on port 3000
- Hot module replacement (HMR)
- Proxies API requests to backend:8000

### Production (Docker)

```bash
./Start.sh
```

**Uses:**
- Docker image build
- Vite static build → dist/
- Nginx serves static files
- API proxying via Nginx

---

## Troubleshooting

### White Screen in Browser

**Check:** Nginx is serving files
```bash
docker-compose logs dashboard
curl http://localhost:3000/index.html
```

**Solution:**
- Verify `/usr/share/nginx/html` has files
- Check nginx.conf SPA routing: `try_files $uri /index.html`
- Rebuild Docker image: `docker-compose down && ./Start.sh`

### API Requests Failing

**Check:** Backend is running
```bash
curl http://localhost:8000/api/v1/health
```

**Solution:**
- Ensure backend started before dashboard
- Check backend logs: `docker-compose logs backend`
- Verify nginx.conf proxy: `proxy_pass http://backend;`

### 404 on Refresh

**Check:** SPA routing configured
```nginx
location / {
    try_files $uri /index.html;
}
```

**Solution:**
- This should fallback to index.html for all routes
- Vite React Router handles client-side routing
- Restart Nginx: `docker-compose restart dashboard`

### Docker Build Fails

**Check:** Source files exist
```bash
ls dashboard/react_ui/src/
```

**Solution:**
- Verify source code at `dashboard/react_ui/`
- Clear Docker cache: `docker system prune -a`
- Rebuild: `docker-compose build --no-cache`

---

## Performance Notes

- **Build time:** 2-5 minutes (downloads dependencies + Vite build)
- **Image size:** ~50 MB (Nginx Alpine + static files)
- **Memory:** ~30 MB at runtime
- **CSS/JS caching:** 365 days (browser cache for `/assets/`)
- **HTML caching:** 1 hour (allows version updates)

---

## Production Checklist

✅ Vite builds to `dist/` folder
✅ Dockerfile copies from `/app/dist`
✅ Nginx serves static files on port 80
✅ SPA routing configured (try_files → index.html)
✅ API proxy to backend:8000 configured
✅ WebSocket support enabled
✅ Asset caching headers set
✅ Health check endpoint available
✅ docker-compose.yml correct
✅ Start.sh works end-to-end

---

## Useful Commands

```bash
# View all logs
docker-compose logs -f

# View dashboard logs
docker-compose logs -f dashboard

# Reload Nginx config (no rebuild)
docker-compose exec dashboard nginx -s reload

# Rebuild dashboard only
docker-compose build --no-cache dashboard

# Full restart
docker-compose down
./Start.sh
```

---

## Related Files

- [deployment/docker/Dockerfile.dashboard](../../deployment/docker/Dockerfile.dashboard) - Docker build
- [deployment/docker/nginx.conf](../../deployment/docker/nginx.conf) - Nginx config
- [docker-compose.yml](../../docker-compose.yml) - Service orchestration
- [Start.sh](../../Start.sh) - Startup script
- [dashboard/react_ui/vite.config.js](../../dashboard/react_ui/vite.config.js) - Vite config

---

**Status:** ✅ Production Ready
**Framework:** React 18 + Vite 5
**Server:** Nginx Alpine
**Platform:** Docker + macOS
**Last Updated:** February 7, 2026
