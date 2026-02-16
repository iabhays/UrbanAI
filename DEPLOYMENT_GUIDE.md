# 🚀 SENTIENTCITY Deployment Guide

Complete guide to deploy SENTIENTCITY dashboard and backend to the cloud for free.

---

## 📋 Deployment Options

### Frontend (React Dashboard)
- **Vercel** (Recommended) - Free tier perfect for React apps, automatic deployments from GitHub

### Backend (FastAPI)
- **Railway.app** (Recommended) - Free tier, easy setup, automatic deployments
- **Render.com** - Alternative option, also free tier

---

## 🎨 Part 1: Deploy Frontend to Vercel

### Prerequisites
- GitHub fork/repository of SentientCity
- Vercel account (free at https://vercel.com)

### Step-by-Step Instructions

#### 1. Create Vercel Account
```
1. Go to https://vercel.com
2. Click "Sign Up"
3. Choose "Continue with GitHub"
4. Authorize Vercel to access your GitHub account
5. Complete signup
```

#### 2. Create New Project on Vercel
```
1. Go to https://vercel.com/new
2. Click "Import Git Repository"
3. Paste your SentientCity GitHub URL:
   https://github.com/iabhays/SentientCity
4. Click "Continue"
```

#### 3. Configure Project Settings
```
In the "Configure Project" section:

Project Name: sentientcity-dashboard
(or your preferred name)

Root Directory: ./dashboard/react_ui

Framework: React

Build Command: npm run build

Output Directory: dist

Install Command: npm install

Environment Variables:
  Key: REACT_APP_API_URL
  Value: https://your-backend-url.railway.app
  (We'll update this after backend deployment)
```

#### 4. Deploy
```
1. Click "Deploy"
2. Wait for deployment to complete (~2-3 minutes)
3. You'll get a Vercel URL: https://sentientcity-xxx.vercel.app
4. Note this URL - we'll add it to README
```

#### 5. Set Up Custom Domain (Optional)
```
In Vercel Dashboard:
1. Go to your project settings
2. Click "Domains"
3. Add custom domain (if you have one)
```

---

## 🔧 Part 2: Deploy Backend to Railway.app

### Prerequisites
- Vercel URL (from Part 1)
- Railway account (free at https://railway.app)

### Step-by-Step Instructions

#### 1. Create Railway Account
```
1. Go to https://railway.app
2. Click "Start Free"
3. Choose "Login with GitHub" or "Sign up"
4. Authorize Railway to access GitHub
5. Complete signup
```

#### 2. Create New Project
```
1. Go to https://railway.app/dashboard
2. Click "+ New Project"
3. Select "Deploy from GitHub repo"
4. Choose your SentientCity repository
5. Click "Deploy"
```

#### 3. Configure Environment Variables
```
In Railway Dashboard, click on your project:

1. Go to "Variables" tab
2. Add the following environment variables:

ENVIRONMENT=production
REDIS_URL=redis://default:password@localhost:6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
DATABASE_URL=postgresql://user:password@host/db
DASHBOARD_URL=https://sentientcity-xxx.vercel.app

(Replace sentientcity-xxx with your actual Vercel URL)
```

#### 4. Configure Deployment Settings
```
1. Go to "Settings" tab
2. Start Command: 
   uvicorn sentient_city.backend_api.main:app --host 0.0.0.0 --port $PORT
3. Build Command: 
   pip install -r requirements.txt
4. Python Version: 3.10
```

#### 5. Deploy
```
1. Railway automatically starts deployment
2. Wait 5-10 minutes for completion
3. You'll get a Railway URL: https://sentientcity-api-xxx.railway.app
4. Note this URL for the README
```

#### 6. Update Vercel with Backend URL
```
Go back to Vercel:
1. Project Settings → Environment Variables
2. Update REACT_APP_API_URL:
   Value: https://sentientcity-api-xxx.railway.app
   (Use your actual Railway URL)
3. Redeploy project (automatic)
```

---

## 🚂 Part 3 (Alternative): Deploy Backend to Render.com

### Prerequisites
- Vercel URL (from Part 1)
- Render account (free at https://render.com)

### Step-by-Step Instructions

#### 1. Create Render Account
```
1. Go to https://render.com
2. Click "Get Started"
3. Choose "Sign up with GitHub"
4. Authorize Render
5. Complete signup
```

#### 2. Create Web Service
```
1. Go to https://dashboard.render.com
2. Click "+ New +"
3. Select "Web Service"
4. Choose "Deploy an existing repository"
5. Select SentientCity repo
6. Click "Connect"
```

#### 3. Configure Service
```
Service Details:
- Name: sentientcity-api
- Environment: Python 3 (or 3.10 if available)
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn sentient_city.backend_api.main:app --host 0.0.0.0 --port $PORT
- Plan: Free

Environment Variables:
ENVIRONMENT=production
DASHBOARD_URL=https://sentientcity-xxx.vercel.app
```

#### 4. Deploy
```
1. Click "Create Web Service"
2. Wait 10-15 minutes for deployment
3. Get your Render URL: https://sentientcity-api.onrender.com
4. Note this URL for updates
```

---

## 📝 Update README with Live URLs

After deployment, update your README.md with:

```markdown
## 🌐 Live Demo

**Frontend Dashboard**: https://sentientcity-xxx.vercel.app  
**Backend API**: https://sentientcity-api-xxx.railway.app  
**API Documentation**: https://sentientcity-api-xxx.railway.app/docs  

Try it live:
- Dashboard: [Open Dashboard](https://sentientcity-xxx.vercel.app)
- API Docs: [View API](https://sentientcity-api-xxx.railway.app/docs)
- Health Check: [API Status](https://sentientcity-api-xxx.railway.app/health)
```

---

## 🔄 Automatic Deployments

### Vercel - Auto Deploys on GitHub Push
```
Every time you push to main branch:
1. GitHub webhook triggers Vercel
2. Vercel pulls latest code
3. Runs build command
4. Deploys automatically (2-3 mins)

No manual action needed!
```

### Railway/Render - Auto Deploys on GitHub Push
```
Similarly, backend auto-deploys:
1. GitHub webhook triggers Railway/Render
2. Latest code pulled
3. Dependencies installed
4. API restarts (5-10 mins)

Enable in settings → Auto Deploy: ON
```

---

## 🐛 Troubleshooting Deployments

### Frontend (Vercel) Not Working

**Issue: Build fails**
```
Solution:
1. Check build logs in Vercel dashboard
2. Ensure NODE_ENV is not set to production locally
3. Verify package.json has all dependencies
4. Run npm install locally to verify
```

**Issue: Blank page or 404 errors**
```
Solution:
1. Check console (F12) for errors
2. Verify REACT_APP_API_URL environment variable
3. Check if backend API is accessible from frontend
4. CORS might be blocking - contact backend admin
```

**Issue: Real-time updates not working**
```
Solution:
1. WebSocket (ws://) not upgrading to WSS (wss://)
2. Check browser console for WebSocket errors
3. Vercel may need special WebSocket configuration
4. Use polling as fallback in development
```

### Backend (Railway/Render) Not Working

**Issue: Service times out during startup**
```
Solution:
1. Check if dependencies install is taking too long
2. Reduce package.json size (remove dev dependencies)
3. Use wheel files for faster installation
4. Contact Railway/Render support
```

**Issue: Database connection failed**
```
Solution:
1. Verify DATABASE_URL environment variable
2. Check if database service is running
3. Test locally: psql $DATABASE_URL
4. May need to provision database separately
```

**Issue: API returns 502 Bad Gateway**
```
Solution:
1. Check Railway/Render logs for errors
2. Verify start command is correct
3. Ensure port is set to $PORT variable
4. Check if dependencies are installed
```

---

## 📊 Monitoring Deployments

### Vercel Analytics
```
1. Go to Vercel Dashboard
2. Select your project
3. Click "Analytics"
- View real-time traffic
- Response times
- Error rates
- Build history
```

### Railway Monitoring
```
1. Go to Railway project
2. View real-time logs
3. Monitor CPU/memory usage
4. View deployment history
```

---

## 🔐 Security Considerations

### Frontend (Vercel)
```
✅ Enable "Trusted IPs" for private projects
✅ Set up password protection if needed
✅ Use HTTPS (automatic with Vercel)
✅ Enable spam protection
```

### Backend (Railway/Render)
```
✅ Use environment variables for secrets
✅ Don't commit .env files
✅ Enable HTTPS (automatic)
✅ Set up rate limiting
✅ Use authentication tokens for API
```

---

## 💰 Pricing

### Vercel (Free Tier)
- ✅ 15GB bandwidth per month
- ✅ Unlimited projects
- ✅ Automatic deployments
- ✅ SSL/HTTPS included
- ✅ CDN included
- **Cost**: Free

### Railway (Free Tier)
- ✅ $5/month free credits
- ✅ Databases included
- ✅ Auto-scaling
- ✅ SSL/HTTPS included
- ✅ Generous free tier
- **Cost**: Free with $5 credits

### Render (Free Tier)
- ✅ Free tier available
- ✅ Auto-sleep after 15 min inactivity
- ✅ 100 GB bandwidth
- ✅ SSL/HTTPS included
- **Cost**: Free (with limitations)

---

## 📞 Support

Having issues? Check:
1. Framework-specific docs:
   - Vercel: https://vercel.com/docs
   - Railway: https://railway.app/docs
   - Render: https://render.com/docs

2. GitHub Issues: https://github.com/iabhays/SentientCity/issues

3. Contact: abhays2103@gmail.com

---

## ✅ Next Steps

1. ✅ Deploy frontend to Vercel
2. ✅ Deploy backend to Railway/Render
3. ✅ Update README with live URLs
4. ✅ Test deployment
5. ✅ Enable auto-deployments
6. ✅ Monitor performance
7. ✅ Share live demo with team

---

**Last Updated**: February 16, 2026  
**Deployment Status**: Ready for Production
