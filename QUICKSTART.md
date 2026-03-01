# ⚡ UrbanAI Quick Start

## 🎯 Get Running in 5 Minutes

### Step 1: Run Locally

```bash
./run.sh
```

**That's it!** 🎉

The app will automatically:
- ✅ Create virtual environment
- ✅ Install dependencies  
- ✅ Start backend (http://localhost:8000)
- ✅ Start frontend (http://localhost:3000)

---

## 🌐 Deploy FREE (10 Minutes)

### Frontend → Vercel

```bash
cd dashboard/react_ui
npm install -g vercel
vercel
```

### Backend → Render

1. Go to [render.com](https://render.com)
2. New Web Service → Connect GitHub
3. Root Directory: `backend`
4. Build: `pip install -r ../requirements.txt`
5. Start: `python main.py`
6. Deploy! ✨

**See [DEPLOY.md](./DEPLOY.md) for details**

---

## 📚 Documentation

- **Full Guide**: [README_SIMPLE.md](./README_SIMPLE.md)
- **Deployment**: [DEPLOY.md](./DEPLOY.md)
- **Changes**: [CHANGES.md](./CHANGES.md)
- **API Docs**: http://localhost:8000/docs (when running)

---

## 🐛 Issues?

### Backend won't start
```bash
source venv/bin/activate
pip install -r requirements.txt
cd backend && python main.py
```

### Frontend won't start
```bash
cd dashboard/react_ui
rm -rf node_modules
npm install
npm run dev
```

---

## ✨ What's Different?

| Before | After |
|--------|-------|
| Docker required | ❌ No Docker |
| Kafka/Redis needed | ✅ In-memory |
| 10+ setup steps | ✅ 1 command |
| $50-100/month | ✅ FREE |

---

**Need help?** Read [CHANGES.md](./CHANGES.md) to see what was simplified.
