# 🚀 UrbanAI Deployment Comparison & Recommendations

**Updated: February 2026**
**Repository**: https://github.com/iabhays/UrbanAI

---

## 📊 Deployment Platform Comparison

| Feature | Vercel | Railway | Render |
|---------|--------|---------|--------|
| **Primary Use** | ✅ Frontend (Next.js) | ✅ Backend (Python) | ✅ Backend (Python) |
| **Free Tier** | Yes (Recommended) | Yes (500 credits/mo) | Yes (Very Limited) |
| **Deployment** | GitHub Auto-Deploy | GitHub Auto-Deploy | GitHub Auto-Deploy |
| **Cold Starts** | Fast | ~10s for Python | ~20s for Python |
| **Scaling** | Automatic | Automatic | Automatic |
| **Databases** | Not Included | PostgreSQL, Redis | PostgreSQL incl. |
| **Environment Variables** | Easy UI | Easy UI | Easy UI |
| **HTTPS** | Automatic | Automatic | Automatic |
| **Custom Domain** | Yes | Yes | Yes |
| **Best For** | React/Next.js Apps | Production APIs | Side Projects |

---

## 🏆 Our Recommendation

### **Frontend Deployment: Vercel** ✅ RECOMMENDED

**Why:**
- Specifically optimized for Next.js
- Fastest deployments and performance
- Generous free tier (100GB/month bandwidth)
- CDN integrated globally
- Automatic preview deployments for PRs
- Zero-config deployments

### **Backend Deployment: Railway** ✅ RECOMMENDED

**Why:**
- Modern Python support with environment isolation
- Better pricing ($5/month after free credits) vs Render
- Integrated PostgreSQL and Redis out-of-the-box
- 500 credits/month free (enough for side projects)
- Faster boot times than Render
- Better developer experience

---

## 📋 Pre-Deployment Checklist

### Backend Requirements
```
✅ Python 3.10+ available
✅ requirements.txt with all dependencies
✅ Procfile configured (we created one)
✅ .env.example with all required variables
✅ FastAPI app properly configured
✅ CORS configured for frontend domain
```

### Frontend Requirements
```
✅ Next.js 14+ configured
✅ dashboard/package.json exists
✅ Build script working locally
✅ Environment variables setup
✅ next.config.js created
✅ vercel.json created
```

### Both
```
✅ Git repository linked (GitHub)
✅ .env files NOT in git (use .env.example)
✅ Project builds locally without errors
✅ All dependencies pinned to versions
```

---

## 🎯 Step-by-Step Deployment Guide

### **Part 1: Deploy Frontend to Vercel**

#### 1.1 Create Vercel Account
```bash
1. Go to https://vercel.com
2. Click "Sign Up"
3. Choose "Continue with GitHub"
4. Authorize Vercel
```

#### 1.2 Import Project
```bash
1. Go to https://vercel.com/new
2. Click "Import Git Repository"
3. Paste: https://github.com/iabhays/UrbanAI
4. Click "Continue"
```

#### 1.3 Configure Project
```
Project Name: urbanai-dashboard

Root Directory: ./dashboard/

Framework: Next.js

Build Command: npm run build

Output Directory: .next

Install Command: npm install
```

#### 1.4 Environment Variables
Add these in Vercel Dashboard → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL = https://urbanai-api.railway.app
NEXT_PUBLIC_APP_NAME = UrbanAI
NEXT_PUBLIC_ENV = production
```

#### 1.5 Deploy
```bash
Click "Deploy" button
Wait 2-3 minutes
You'll get: https://urbanai-xxx.vercel.app
```

**Vercel Deployment Tips:**
- Every push to `master` auto-deploys
- Preview deployments for every PR
- Easy rollback to previous versions
- Check deployment logs in Vercel Dashboard

---

### **Part 2: Deploy Backend to Railway**

#### 2.1 Create Railway Account
```bash
1. Go to https://railway.app
2. Click "Start Free"
3. Choose "Login with GitHub"
4. Authorize Railway
```

#### 2.2 Create New Project
```bash
1. Go to https://railway.app/dashboard
2. Click "+ New Project"
3. Select "Deploy from GitHub repo"
4. Choose UrbanAI repository
5. Click "Deploy"
```

#### 2.3 Configure Build & Start Commands
Go to Settings:

```
Build Command: pip install -r requirements.txt

Start Command: uvicorn sentient_city.backend_api.main:app --host 0.0.0.0 --port $PORT --workers 4
```

#### 2.4 Add Environment Variables
Go to Variables tab, add:

```
ENVIRONMENT = production
API_HOST = 0.0.0.0
API_PORT = 8000
SECRET_KEY = your-secret-key-here (generate random)
ALLOWED_ORIGINS = https://urbanai-xxx.vercel.app,http://localhost:3000
DATABASE_URL = (Railway auto-generates if you add PostgreSQL plugin)
REDIS_URL = (Railway auto-generates if you add Redis plugin)
DASHBOARD_URL = https://urbanai-xxx.vercel.app
```

#### 2.5 Add Database & Cache (Optional but Recommended)
```bash
1. Go to your Railway project
2. Click "+ Add Service"
3. Choose "PostgreSQL" → Deploy
4. Choose "Redis" → Deploy
5. They auto-inject environment variables
```

#### 2.6 Deploy
```bash
1. Railway automatically deploys when config is complete
2. Wait 5-10 minutes
3. You'll see: https://urbanai-api-xxx.railway.app
4. Check health: https://urbanai-api-xxx.railway.app/health
```

**Railway Deployment Tips:**
- Every push to `master` auto-deploys
- Free tier: 500 credits/month ($5 value)
- Paid tier: $5/month for production use
- Can add PostgreSQL/Redis easily
- Monitor logs in Railway Dashboard

---

### **Part 3: Connect Frontend to Backend**

After backend is deployed:

#### 3.1 Update Vercel Environment Variables
```bash
1. Vercel Dashboard → Your Project → Settings
2. Environment Variables
3. Update NEXT_PUBLIC_API_URL:
   Value: https://urbanai-api-xxx.railway.app
4. Save and redeploy
```

#### 3.2 Verify Connection
```bash
1. Open https://urbanai-xxx.vercel.app
2. Open browser console (F12)
3. Check network tab
4. API calls should go to railway.app URL
```

---

## 🔄 Alternative: Render.com

If you prefer Render over Railway:

### Pros:
- Simple deployment
- PostgreSQL included in free tier
- Good documentation

### Cons:
- Slower cold starts (~20s)
- Lower free tier limits
- Less generous free credits
- Requires more manual setup

### Deploy to Render:
```bash
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Choose UrbanAI repo
5. Set commands same as Railway
6. Add environment variables
7. Deploy
```

---

## 📊 Cost Comparison (Monthly)

### **UrbanAI Typical Usage:**

| Platform | Free Tier | Cost | Notes |
|----------|-----------|------|-------|
| Vercel | ✅ Sufficient | $0-20 | Bandwidth included |
| Railway | ✅ Very Good | $0-10 | 500 credits free |
| Render | ⚠️ Limited | $7-15+ | Less generous |

**Recommended Setup Cost: $0-10/month** (if using Railway free tier)

---

## 🛠️ Post-Deployment Tasks

### 1. Monitor Deployments
```bash
# Check Vercel
https://vercel.com/dashboard

# Check Railway
https://railway.app/dashboard

# Check live status
curl https://urbanai-xxx.vercel.app
curl https://urbanai-api-xxx.railway.app/health
```

### 2. Setup Monitoring
```bash
# Railway provides built-in monitoring
# Log into Railway Dashboard
# View: Deployments → Logs tab
# View: Metrics tab for CPU/Memory

# Vercel provides:
# Analytics, Real Experience Score (RES)
# Check: Analytics tab
```

### 3. Setup Auto-Scaling
```bash
# Vercel: Automatic (nothing needed)

# Railway:
# Go to Project → Settings → Compute
# Enable auto-scaling (recommended)
```

### 4. Custom Domain (Optional)
```bash
# Vercel:
# Settings → Domains → Add Domain
# Update DNS records

# Railway:
# Project → Network → Add Custom Domain
# Update DNS records
```

---

## 🚨 Troubleshooting

### **Frontend not connecting to backend**
```bash
# Check CORS in backend
# Check NEXT_PUBLIC_API_URL in Vercel

# Solution:
# 1. Verify backend is running
# 2. Check ALLOWED_ORIGINS includes Vercel URL
# 3. Restart Vercel deployment
```

### **Backend deployment failing**
```bash
# Common issues:
# 1. Missing requirements.txt dependencies
# 2. Python version mismatch
# 3. Port not set to $PORT environment variable

# Solution:
# 1. Check Railway logs for errors
# 2. Test locally: python -m pip install -r requirements.txt
# 3. Verify Procfile has correct start command
```

### **Cold start delays**
```bash
# This is normal for free tier
# Railway: ~10s, Render: ~20s

# Solution:
# 1. Use paid tier for always-on instances
# 2. Add monitoring/health checks
# 3. Some services may auto-spin down
```

---

## 📚 Useful Links

**Documentation:**
- [Vercel Next.js Deployment](https://vercel.com/docs/frameworks/nextjs)
- [Railway Python Deployment](https://docs.railway.app/guides/deployments)
- [Render Python Deployment](https://render.com/docs)

**Our Repository:**
- GitHub: https://github.com/iabhays/UrbanAI
- Frontend: https://urbanai.vercel.app (when deployed)
- Backend API: https://urbanai-api-xxx.railway.app (when deployed)
- API Docs: https://urbanai-api-xxx.railway.app/docs (when deployed)

**Monitoring & Management:**
- Vercel Dashboard: https://vercel.com/dashboard
- Railway Dashboard: https://railway.app/dashboard

---

## ✅ Final Checklist Before Going Live

```
[ ] Git remote updated to UrbanAI repo
[ ] All deployment files committed and pushed (vercel.json, Procfile, next.config.js)
[ ] Frontend deployed to Vercel
[ ] Backend deployed to Railway
[ ] Environment variables configured
[ ] Frontend connects to backend (test API call)
[ ] Health check endpoint responds
[ ] Logs are being captured
[ ] Custom domain setup (optional)
[ ] Monitoring configured
[ ] Team has access to dashboards
```

---

## 🎓 Next Steps

1. **Deploy this week** to test the setup
2. **Monitor for 24 hours** to catch any issues
3. **Setup alerts** for downtime
4. **Document API endpoints** for your team
5. **Add SSL certificate** (usually automatic)
6. **Setup CI/CD pipeline** for testing before deploy

---

**Questions?** Check the deployment logs or visit our GitHub issues: https://github.com/iabhays/UrbanAI/issues
