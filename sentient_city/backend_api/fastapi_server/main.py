"""
FastAPI backend main application.

Provides REST API and WebSocket endpoints for the dashboard.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import asyncio
import json
from loguru import logger

from ...utils.config import get_config
from ...utils.logger import setup_logger
from .routes import alerts, cameras, analytics, health, videos
from .websocket_manager import WebSocketManager
from .middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    AuthenticationMiddleware
)

# Setup logging
config = get_config()
system_config = config.get_section("system")
setup_logger(
    log_level=system_config.get("log_level", "INFO"),
    log_file=system_config.get("log_file")
)

# Initialize FastAPI app
app = FastAPI(
    title="SENTIENTCITY AI API",
    description="Multi-Agent Smart City Intelligence Platform API",
    version="1.0.0"
)

# Security middleware (order matters!)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=60, requests_per_hour=1000)

# Authentication middleware
app.add_middleware(AuthenticationMiddleware)

# CORS middleware
backend_config = config.get_section("backend")
cors_config = backend_config.get("cors", {})
if cors_config.get("enabled", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

# WebSocket manager
websocket_manager = WebSocketManager()

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(cameras.router, prefix="/api/v1", tags=["Cameras"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(videos.router, prefix="/api/v1", tags=["Videos"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process message
            await websocket_manager.send_personal_message(
                f"Echo: {data}", websocket
            )
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("SENTIENTCITY AI API starting up...")
    # Initialize components here if needed


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("SENTIENTCITY AI API shutting down...")
    await websocket_manager.broadcast(json.dumps({"type": "shutdown"}))


if __name__ == "__main__":
    import uvicorn
    
    backend_config = get_config().get_section("backend")
    uvicorn.run(
        "sentient_city.backend_api.fastapi_server.main:app",
        host=backend_config.get("host", "0.0.0.0"),
        port=backend_config.get("port", 8000),
        reload=backend_config.get("reload", False),
        workers=backend_config.get("workers", 1)
    )
