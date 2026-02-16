# SENTIENTCITY Dashboard - Next.js 14 Production Setup

## Overview

The SENTIENTCITY dashboard is a **Next.js 14 application** configured for production deployment using Docker and Nginx.

### Key Changes Made

1. ✅ **next.config.js** - Created with static export configuration
2. ✅ **Dockerfile.dashboard** - Updated for Next.js 14 static export + Nginx
3. ✅ **nginx.conf** - Configured for Next.js static files and API proxying
4. ✅ **docker-compose.yml** - Fixed port mapping (3000:80)
5. ✅ **Start.sh** - Compatible with all changes

---

## Architecture

### Build Process

```
Source Code (dashboard/)
    ↓
Next.js Build (next build)
    → Generates /app/out (static export)
    ↓
Nginx Container
    → Serves /app/out as static site on port 80
    → Maps to host port 3000
```

### Production Deployment

```
Browser (http://localhost:3000)
    ↓ (mapped port)
Nginx Container (port 80)
    ├─→ Static files from Next.js export
    ├─→ /api/* → proxies to backend:8000
    └─→ /ws/* → WebSocket to backend:8000
```

---

## Next.js Configuration

### `dashboard/next.config.js`

```javascript
output: 'export'  // Enable static HTML export
images.unoptimized: true  // No image optimization for static export
trailingSlash: true  // URLs with trailing slashes
```

**Why needed for Docker:**
- `output: 'export'` generates static files in `./out` directory
- This allows serving via Nginx without Node.js runtime
- Compatible with static hosting platforms

---

## Docker Setup

### Multi-Stage Build

**Stage 1: Builder**
- Image: `node:18-alpine`
- Action: Builds Next.js app with `npm run build`
- Output: Static files in `/app/out`

**Stage 2: Production**
- Image: `nginx:alpine`
- Action: Serves static files with Nginx
- No Node.js in final image (smaller, faster)

### Dockerfile Key Paths

```dockerfile
# Build Next.js
COPY dashboard/package*.json ./
RUN npm run build  # Generates /app/out

# Serve with Nginx
COPY --from=builder /app/out /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## Nginx Configuration

### Routes

| Route | Handler | Purpose |
|-------|---------|---------|
| `/` | Static HTML | Next.js pages |
| `/_next/static/*` | Cached assets | JS/CSS with 1-year cache |
| `/api/*` | Proxied to backend:8000 | Backend API |
| `/ws/*` | WebSocket proxy | Real-time updates |
| `/*` (fallback) | `/index.html` | Client-side routing |

### SPA Routing

```nginx
location / {
    try_files $uri $uri/ /index.html;
}
```

This enables client-side routing for Next.js pages.

---

## Docker Compose Configuration

### Dashboard Service

```yaml
dashboard:
  build:
    context: .
    dockerfile: deployment/docker/Dockerfile.dashboard
  ports:
    - "3000:80"  # Host:Container
  depends_on:
    - backend
  environment:
    - NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Key Details:**
- Context: Project root (needed for paths to work)
- Port: 3000:80 (Nginx port 80 maps to host port 3000)
- Backend dependency: Dashboard waits for backend

---

## Build & Runtime

### Build Process

```bash
./Start.sh
  ↓
docker-compose build
  ↓
docker build -f Dockerfile.dashboard
  ├─→ npm ci (clean install)
  ├─→ npm run build (Next.js compilation)
  ├─→ Generates /app/out
  └─→ Copy to Nginx
```

### Runtime

```bash
docker-compose up -d
  ↓
Nginx container starts
  ├─→ Serves /usr/share/nginx/html (static files)
  ├─→ Proxies /api/* to backend
  └─→ Listens on port 80
```

---

## Accessing the Dashboard

### After running `./Start.sh`

```
http://localhost:3000
```

**Endpoints:**
- Dashboard UI: `http://localhost:3000`
- API calls: `GET http://localhost:8000/api/*`
- WebSocket: `ws://localhost:3000/ws/*` (proxied to backend)

---

## Production Benefits

✅ **No Node.js in production**
- Smaller image size
- Lower memory footprint
- Better security (minimal attack surface)

✅ **Static file serving via Nginx**
- Optimized for serving static content
- Built-in compression (gzip)
- Cache control headers for browser caching

✅ **Efficient API proxying**
- Nginx handles API requests
- Browsers can make API calls to same origin
- No CORS issues

✅ **Client-side routing**
- Next.js pages work with SPA routing
- Fallback to index.html for all non-file requests
- Smooth navigation without page reloads

---

## Development vs. Production

### Development
```bash
cd dashboard
npm run dev  # Runs Next.js dev server with hot reload
```

### Production (Docker)
```bash
./Start.sh  # Builds static export and serves with Nginx
```

**Note:** The docker setup does NOT use `next dev` or Vite. It's pure static export + Nginx.

---

## File Structure

```
SENTIENTCITY/
├── dashboard/
│   ├── next.config.js (new)
│   ├── package.json (scripts: dev, build, start)
│   ├── src/
│   └── public/
├── deployment/docker/
│   ├── Dockerfile.dashboard (updated)
│   ├── nginx.conf (updated)
│   ├── Dockerfile.backend
│   └── ...
├── docker-compose.yml (updated)
├── Start.sh (works with all changes)
└── ...
```

---

## Verification Checklist

✅ `next.config.js` has `output: 'export'`
✅ `Dockerfile.dashboard` copies from `/app/out`
✅ `nginx.conf` has proper routing
✅ `docker-compose.yml` maps `3000:80`
✅ `Start.sh` is executable
✅ No `next export` command used
✅ No Node.js in final Nginx image
✅ Backend service running on 8000
✅ Dashboard accessible at localhost:3000

---

## Troubleshooting

### Dashboard shows 404

**Check:** Nginx routing configuration
```bash
docker-compose logs dashboard
```

**Solution:** Verify `nginx.conf` has `try_files $uri /index.html`

### API requests fail

**Check:** Backend is running
```bash
curl http://localhost:8000/api/v1/health
```

**Solution:** Ensure backend service started before dashboard

### Static files not loading

**Check:** Docker build completed successfully
```bash
docker-compose logs dashboard | grep "npm run build"
```

**Solution:** Check `Dockerfile.dashboard` copies from `/app/out`

---

## Performance Notes

- **Build time:** 2-5 minutes (downloads dependencies + builds)
- **Image size:** ~100 MB (Nginx Alpine + static files)
- **Memory:** ~50 MB at runtime
- **CSS/JS caching:** 365 days (browser cache)

---

## Related Documentation

- [DEPLOYMENT.md](../DEPLOYMENT.md) - Full deployment guide
- [STARTUP_SETUP.md](./STARTUP_SETUP.md) - Startup script guide
- [README.md](../README.md) - Project overview

---

**Status:** ✅ Production Ready
**Last Updated:** February 7, 2026
**Framework:** Next.js 14 with Static Export
**Server:** Nginx Alpine
**Platform:** Docker + macOS
