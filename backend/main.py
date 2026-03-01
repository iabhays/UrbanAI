"""
UrbanAI - Simplified Backend API
Standalone FastAPI backend for easy deployment on Render, Railway, or similar platforms
"""

import os
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="UrbanAI API",
    version="1.0.0",
    description="Smart City Intelligence Platform - Simplified Backend",
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Models ==============

class SystemStatus(BaseModel):
    status: str
    uptime: float
    active_cameras: int
    total_detections: int
    timestamp: datetime

class DetectionResult(BaseModel):
    camera_id: str
    timestamp: datetime
    person_count: int
    crowd_density: float
    risk_level: str
    confidence: float

class AlertData(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    camera_id: str
    timestamp: datetime
    message: str
    confidence: float

# ============== In-Memory Storage ==============

class InMemoryStore:
    """Simple in-memory storage for demo purposes"""
    def __init__(self):
        self.cameras = {}
        self.detections = []
        self.alerts = []
        self.start_time = datetime.utcnow()
        
    def add_detection(self, detection: DetectionResult):
        self.detections.append(detection)
        # Keep only last 100 detections
        if len(self.detections) > 100:
            self.detections = self.detections[-100:]
    
    def add_alert(self, alert: AlertData):
        self.alerts.append(alert)
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def get_uptime(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()

store = InMemoryStore()

# ============== WebSocket Manager ==============

class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")

ws_manager = ConnectionManager()

# ============== Routes ==============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "UrbanAI Backend API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "urbanai-backend"
    }

@app.get("/api/v1/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    return SystemStatus(
        status="operational",
        uptime=store.get_uptime(),
        active_cameras=len(store.cameras),
        total_detections=len(store.detections),
        timestamp=datetime.utcnow()
    )

@app.get("/api/v1/cameras")
async def list_cameras():
    """List all active cameras"""
    return {
        "cameras": [
            {
                "camera_id": "cam_001",
                "name": "Main Entrance",
                "status": "active",
                "fps": 30.0,
                "location": "Building A"
            },
            {
                "camera_id": "cam_002",
                "name": "Parking Lot",
                "status": "active",
                "fps": 25.0,
                "location": "Zone B"
            }
        ]
    }

@app.get("/api/v1/detections")
async def get_detections(limit: int = 50):
    """Get recent detections"""
    recent = store.detections[-limit:]
    return {
        "total": len(recent),
        "detections": [d.dict() for d in recent]
    }

@app.get("/api/v1/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    recent = store.alerts[-limit:]
    return {
        "total": len(recent),
        "alerts": [a.dict() for a in recent]
    }

@app.post("/api/v1/detections")
async def create_detection(detection: DetectionResult):
    """Create a new detection record"""
    store.add_detection(detection)
    
    # Broadcast to WebSocket clients
    await ws_manager.broadcast({
        "type": "detection",
        "data": detection.dict()
    })
    
    return {"status": "success", "detection_id": len(store.detections)}

@app.post("/api/v1/alerts")
async def create_alert(alert: AlertData):
    """Create a new alert"""
    store.add_alert(alert)
    
    # Broadcast to WebSocket clients
    await ws_manager.broadcast({
        "type": "alert",
        "data": alert.dict()
    })
    
    return {"status": "success", "alert_id": alert.alert_id}

@app.post("/api/v1/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image for crowd detection
    This is a simplified version that returns mock data
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Mock analysis results
        # In production, this would run actual YOLO detection
        mock_result = DetectionResult(
            camera_id="uploaded",
            timestamp=datetime.utcnow(),
            person_count=np.random.randint(5, 25),
            crowd_density=np.random.uniform(0.5, 3.5),
            risk_level="low" if np.random.random() > 0.3 else "medium",
            confidence=np.random.uniform(0.75, 0.95)
        )
        
        store.add_detection(mock_result)
        
        return {
            "status": "success",
            "result": mock_result.dict()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== WebSocket Routes ==============

@app.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket):
    """
    WebSocket endpoint for live updates
    Clients can connect to receive real-time detection and alert updates
    """
    await ws_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to UrbanAI live feed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back for ping/pong
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)

# ============== Background Tasks ==============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("UrbanAI Backend starting up...")
    logger.info("Backend API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("UrbanAI Backend shutting down...")

# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
