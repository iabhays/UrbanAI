# UrbanAI Video Analysis - Demo vs Production

## 🎯 Architecture Overview

The system has **two completely independent analysis flows**:

### 1️⃣ **DEMO VIDEO ANALYSIS** (Dashboard Feature)
- **Location:** `datasets/raw/` directory
- **Access:** Dashboard UI → "Video Analysis" tab
- **Purpose:** Testing and demonstration
- **Impact:** **ZERO impact** on production cameras
- **Max Videos:** Unlimited (file system dependent)
- **Status:** Isolated, mock results

### 2️⃣ **PRODUCTION CAMERA SOURCES** (Real-time Pipeline)
- **Location:** Configured in `configs/config.yaml`
- **Access:** Command line: `python scripts/run_pipeline.py --camera <RTSP_URL>`
- **Purpose:** Real-time monitoring with live cameras/RTSP streams
- **Impact:** **ZERO impact** from demo videos
- **Sources:** RTSP, IP cameras, local webcam, video files
- **Status:** Real inference, real alerts to Kafka/Redis

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UrbanAI SYSTEM                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────┐      ┌──────────────────────────┐
│   DEMO VIDEO ANALYSIS    │      │  PRODUCTION CAMERAS      │
│     (Dashboard Only)     │      │   (Real-time Pipeline)   │
├──────────────────────────┤      ├──────────────────────────┤
│ • Local video files      │      │ • RTSP streams           │
│ • datasets/raw/          │      │ • IP cameras             │
│ • 6 demo videos          │      │ • Network streams        │
│ • Mock results           │      │ • Webcam (index 0)       │
│ • No real inference      │      │ • Real YOLOv26 models    │
│ • For testing only       │      │ • Real-time alerts       │
│ • Isolated state         │      │ • Kafka/Redis output     │
└──────┬───────────────────┘      └──────┬───────────────────┘
       │                                   │
       ▼                                   ▼
   ┌────────────┐                    ┌────────────┐
   │ API Server │                    │  Pipeline  │
   │ /api/v1/   │                    │   Daemon   │
   │  videos/*  │                    │            │
   └────────────┘                    └────────────┘
       │                                   │
       ▼                                   ▼
   ┌────────────┐                    ┌────────────┐
   │ Dashboard  │                    │   Kafka    │
   │ React UI   │                    │   Redis    │
   └────────────┘                    └────────────┘

🔒 COMPLETELY ISOLATED - No interference!
```

---

## 🚀 Quick Start Guide

### Run Demo Videos (Dashboard)

```bash
# 1. Start all services
./Start.sh

# 2. Open dashboard
http://localhost:3000

# 3. Go to "Video Analysis" tab
# 4. Click "Analyze" on any video
# 5. Watch demo results populate
```

**This does NOT affect production cameras.**

### Run Production Cameras

```bash
# 1. Ensure services are running (from ./Start.sh)
# Already includes Kafka, Redis, Backend

# 2. In new terminal, run pipeline with RTSP camera
python scripts/run_pipeline.py \
  --camera "rtsp://192.168.1.100:554/stream" \
  --camera-id "main_entrance"

# 3. For multiple cameras (parallel)
python scripts/run_pipeline.py --camera "rtsp://camera1" --camera-id cam1 &
python scripts/run_pipeline.py --camera "rtsp://camera2" --camera-id cam2 &
python scripts/run_pipeline.py --camera "rtsp://camera3" --camera-id cam3 &
```

**This does NOT affect demo videos.**

### Both Running Simultaneously ✅

```bash
# Terminal 1: Start services
./Start.sh

# Terminal 2: Start demo via dashboard
# (Visit http://localhost:3000 → Video Analysis)

# Terminal 3: Start production camera
python scripts/run_pipeline.py --camera "rtsp://your_camera_url"

# Terminal 4: Start another camera
python scripts/run_pipeline.py --camera "rtsp://another_camera"

# All run independently, zero interference!
```

---

## 📝 Configuration

### Demo Videos (No Configuration Needed)
- Automatic: Scans `datasets/raw/` directory
- No config changes required
- Plug and play

### Production Cameras

**Option 1: Command Line (Recommended)**
```bash
python scripts/run_pipeline.py --camera "rtsp://192.168.1.100:554/stream"
python scripts/run_pipeline.py --camera "datasets/raw/video1.mp4"  # File
python scripts/run_pipeline.py --camera 0  # Webcam
```

**Option 2: Configuration File (Optional)**
Edit `configs/config.yaml`:
```yaml
cameras:
  - id: "camera_001"
    source: "rtsp://192.168.1.100:554/stream1"
    location: "Main Entrance"
    enabled: true
  
  - id: "camera_002"
    source: "rtsp://192.168.1.101:554/stream2"
    location: "Side Entrance"
    enabled: true
```

Then run:
```bash
python scripts/run_pipeline.py  # Uses config.yaml
```

---

## 🎬 Demo Videos Location

Add your test videos here:
```
UrbanAI/
└── datasets/
    └── raw/
        ├── video1.mp4
        ├── video2.mp4
        ├── video3.mp4
        ├── video4.mp4
        ├── video5.mp4
        ├── video6.mp4
        ├── pedestrian_test.avi
        └── crowd_test.mov
```

**All videos in this folder automatically appear in dashboard demo!**

---

## 🔧 API Endpoints

### Demo Videos API (Isolated)
```
GET  /api/v1/videos              # List demo videos
POST /api/v1/videos/{id}/analyze # Start demo analysis
POST /api/v1/videos/{id}/stop    # Stop demo analysis
GET  /api/v1/videos/{id}/status  # Get analysis status
GET  /api/v1/videos/{id}/results # Get mock results
```

**Type:** "demo" (clearly marked)
**Impact:** Zero on production

### Production Cameras
```
Via: python scripts/run_pipeline.py --camera <source>

Events to Kafka:
- Topic: detections
- Topic: tracks
- Topic: alerts

Cache to Redis:
- Key: frame:{camera_id}:{frame_count}
```

---

## 📊 Data Flow Comparison

### DEMO FLOW
```
Dashboard Click
    ↓
API: /videos/{id}/analyze
    ↓
Backend: _demo_analyzing_videos dictionary
    ↓
Simulate 10 second processing
    ↓
Generate mock results (random)
    ↓
Dashboard displays results
    ↓
❌ NO real inference
❌ NO Kafka events
❌ NO alerts to production system
```

### PRODUCTION FLOW
```
python scripts/run_pipeline.py --camera <RTSP>
    ↓
VideoProcessor opens video stream
    ↓
EdgeDetector runs YOLOv26 inference ✅
    ↓
Tracker (OC-SORT) processes detections ✅
    ↓
Pose extraction, behavior analysis ✅
    ↓
Risk assessment, alerts generated ✅
    ↓
Kafka pub: detections, tracks, alerts ✅
Redis cache: frame results ✅
    ↓
Production system reacts to real alerts
```

---

## ✅ Checklist

### For Demo Testing
- [ ] Videos in `datasets/raw/`
- [ ] Dashboard running at http://localhost:3000
- [ ] Visit "Video Analysis" tab
- [ ] Click "Analyze" on a video
- [ ] See mock results

### For Production Deployment
- [ ] RTSP cameras ready (IP, routing correct)
- [ ] `./Start.sh` running (Kafka, Redis, Backend)
- [ ] Run: `python scripts/run_pipeline.py --camera <RTSP_URL>`
- [ ] Check Kafka for real events
- [ ] Verify alerts trigger correctly

### Running Both Simultaneously ✅
- [ ] Demo videos analyzed in UI
- [ ] Production cameras streaming
- [ ] No conflicts
- [ ] No performance issues
- [ ] Independent state management

---

## 🛡️ Isolation Guarantees

| Aspect | Demo | Production | Isolation |
|--------|------|-----------|-----------|
| **Memory** | Separate dict | Separate process | ✅ Yes |
| **State** | `_demo_*` vars | Pipeline object | ✅ Yes |
| **inference** | Mock (random) | Real YOLOv26 | ✅ Yes |
| **Output** | Mock results | Kafka/Redis | ✅ Yes |
| **Timing** | Fixed 10s | Variable | ✅ Yes |
| **Config** | Auto scan dir | config.yaml | ✅ Yes |
| **Scaling** | Limited | Unlimited | ✅ Yes |

---

## 🎯 Use Cases

### DEMO (What to do)
✅ Test UI with 6 sample videos
✅ Verify dashboard works
✅ Show to stakeholders
✅ Understand AI capabilities
✅ Develop UI features

### PRODUCTION (What NOT to do here)
❌ Don't replace real cameras with demos
❌ Don't rely on mock results for alerts
❌ Don't use demo for actual monitoring
❌ Don't expose demo to real environments

---

## 📞 Support

### Issues with Demo
- Check `datasets/raw/` has videos
- Restart dashboard
- Check browser console

### Issues with Production
- Verify RTSP URL works: `ffprobe rtsp://...`
- Check Kafka is running: `docker-compose ps`
- Verify logs: `docker-compose logs backend`
- Run: `python scripts/run_pipeline.py --camera 0` for webcam test

---

## Summary

**This architecture ensures:**
1. ✅ Demo videos for testing/demo purposes
2. ✅ Production cameras for real monitoring
3. ✅ Complete isolation (zero interference)
4. ✅ Both can run simultaneously
5. ✅ Easy to test, easy to deploy

**Demo is NOT production. Production is NOT affected by demo.**

---

**Status:** ✅ Ready for deployment
**Last Updated:** February 7, 2026
