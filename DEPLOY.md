# 🚀 UrbanAI Deployment Guide

This guide will help you deploy UrbanAI for free using:
- **Vercel** for the frontend (React dashboard)
- **Render** or **Railway** for the backend (FastAPI)

No Docker required! ✨

---

## 📋 Prerequisites

- GitHub account
- Vercel account (free tier)
- Render or Railway account (free tier)

---

## 🎯 Quick Start - Local Development

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd UrbanAI
```

### 2. Run Locally

Simply run:

```bash
./run.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Start the backend API on `http://localhost:8000`
- Start the frontend on `http://localhost:3000`

**That's it!** No Docker, no complex setup.

---

## 🌐 Deploy Frontend to Vercel (FREE)

### Option 1: Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy frontend
cd dashboard/react_ui
vercel

# Follow the prompts:
# - Set up and deploy: Y
# - Scope: Your account
# - Link to existing project: N
# - Project name: urbanai
# - Directory: ./
# - Override settings: N
```

### Option 2: Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Set **Root Directory** to: `dashboard/react_ui`
5. Framework Preset: **Vite**
6. Click "Deploy"

### Configure Environment Variables on Vercel

In your Vercel project settings, add:

```
VITE_API_URL=<your-backend-url>
```

(You'll get the backend URL after deploying the backend)

---

## 🔌 Deploy Backend to Render (FREE)

### Step 1: Create Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"

### Step 2: Connect Repository

1. Select your UrbanAI repository
2. Give it a name: `urbanai-backend`

### Step 3: Configure Service

```
Name: urbanai-backend
Region: Oregon (US West) or closest to you
Branch: main
Root Directory: backend
Runtime: Python 3
Build Command: pip install -r ../requirements.txt
Start Command: python main.py
```

### Step 4: Set Environment Variables

```
PYTHON_VERSION=3.11
PORT=8000
```

### Step 5: Deploy

Click "Create Web Service" and wait 3-5 minutes.

Your backend URL will be: `https://urbanai-backend.onrender.com`

### Important: Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- **Solution**: Use a service like [UptimeRobot](https://uptimerobot.com) to ping your backend every 10 minutes

---

## 🚂 Alternative: Deploy Backend to Railway (FREE)

### Step 1: Create Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"

### Step 2: Configure

1. Select your repository
2. Click "Add variables"
3. Add:
   ```
   PORT=8000
   ```

### Step 3: Set Start Command

In the Settings:
- **Root Directory**: `backend`
- **Start Command**: `python main.py`
- **Build Command**: `pip install -r ../requirements.txt`

### Step 4: Deploy

Railway will automatically deploy. Your URL will be:
`https://urbanai-backend.up.railway.app`

---

## 🔗 Connect Frontend to Backend

After deploying both:

1. Copy your backend URL (from Render or Railway)
2. Go to your Vercel project → Settings → Environment Variables
3. Add or update:
   ```
   VITE_API_URL=https://your-backend-url.onrender.com
   ```
4. Redeploy frontend (Vercel does this automatically)

---

## ✅ Verify Deployment

### Test Backend

```bash
curl https://your-backend-url.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-03-01T10:30:00",
  "service": "urbanai-backend"
}
```

### Test Frontend

1. Open `https://your-app.vercel.app`
2. Should see the UrbanAI dashboard
3. Check browser console for any connection errors

---

## 🎨 Custom Domain (Optional)

### Vercel (Frontend)

1. Go to your project → Settings → Domains
2. Add your domain
3. Update DNS records as instructed

### Render (Backend)

1. Go to your service → Settings
2. Click "Add Custom Domain"
3. Follow DNS setup instructions

---

## 📊 Monitoring

### Backend Logs (Render)

```bash
# View logs in dashboard
render logs -f urbanai-backend
```

### Backend Logs (Railway)

View logs directly in the Railway dashboard.

### Vercel Logs

View deployment and runtime logs in the Vercel dashboard.

---

## 🐛 Troubleshooting

### Backend not responding

1. Check logs in Render/Railway dashboard
2. Ensure environment variables are set correctly
3. Verify Python version is 3.10+

### Frontend can't connect to backend

1. Check CORS settings in `backend/main.py`
2. Verify `VITE_API_URL` is set correctly in Vercel
3. Check browser console for errors

### Frontend build fails

1. Ensure Node.js version is 18+
2. Delete `node_modules` and `package-lock.json`
3. Run `npm install` again

---

## 💰 Cost Breakdown

| Service | Free Tier | Limits |
|---------|-----------|--------|
| **Vercel** | ✅ Free | 100 GB bandwidth/month |
| **Render** | ✅ Free | 750 hours/month, auto-sleep |
| **Railway** | ✅ $5 credit/month | ~500 hours/month |

**Total Cost: $0/month** for low-traffic applications!

---

## 🔄 Continuous Deployment

Both Vercel and Render/Railway support automatic deployments:

1. Push to your `main` branch
2. Services automatically rebuild and deploy
3. Zero downtime deployments

---

## 📦 What's Different from Original Setup?

### Removed:
- ❌ Docker and Docker Compose
- ❌ Kafka (replaced with in-memory storage)
- ❌ Redis (replaced with in-memory storage)
- ❌ PostgreSQL (using in-memory for demo)
- ❌ Research and experimental code
- ❌ Heavy ML dependencies (for easier deployment)

### Kept:
- ✅ FastAPI backend with REST API
- ✅ WebSocket support for real-time updates
- ✅ React frontend dashboard
- ✅ Core crowd analysis algorithms
- ✅ Image upload and analysis

---

## 🚀 Next Steps

1. **Add Authentication**: Implement proper JWT authentication
2. **Add Database**: Use PostgreSQL on Render (free tier) or Supabase
3. **Add Caching**: Use Upstash Redis (free tier)
4. **Scale Up**: Upgrade to paid tiers as needed

---

## 📞 Support

If you encounter issues:

1. Check the logs in your deployment dashboard
2. Open an issue on GitHub
3. Review the main README.md for additional context

---

**Happy Deploying! 🎉**
