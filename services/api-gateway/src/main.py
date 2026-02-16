"""
SENTIENTCITY AI - Backend API Gateway
Unified API for all client applications
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import jwt

from sentientcity.core.logging import get_logger, setup_logging
from sentientcity.core.settings import get_settings
from sentientcity.schemas.events import AlertEvent, Severity

logger = get_logger(__name__)
settings = get_settings()
security = HTTPBearer()


# ============== Models ==============

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    user_id: str
    username: str
    roles: list[str]


class CameraStatus(BaseModel):
    camera_id: str
    name: str
    status: str
    last_frame_time: datetime | None
    fps: float
    active_tracks: int


class AlertSummary(BaseModel):
    alert_id: UUID
    alert_type: str
    severity: Severity
    camera_id: str
    timestamp: datetime
    description: str
    acknowledged: bool


class SystemHealth(BaseModel):
    status: str
    services: dict[str, str]
    uptime_seconds: float
    active_cameras: int
    alerts_last_hour: int


# ============== Auth ==============

class AuthManager:
    """JWT authentication manager."""

    def __init__(self) -> None:
        self.secret = settings.secret_key.get_secret_value()
        self.algorithm = settings.jwt_algorithm
        self.expiry_minutes = settings.jwt_expiry_minutes

    def create_token(self, user_id: str, roles: list[str]) -> str:
        """Create JWT token."""
        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": datetime.utcnow() + timedelta(minutes=self.expiry_minutes),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")


auth_manager = AuthManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    """Dependency to get current authenticated user."""
    payload = auth_manager.verify_token(credentials.credentials)
    return payload


def require_role(role: str):
    """Dependency factory for role-based access control."""
    async def check_role(user: dict = Depends(get_current_user)) -> dict:
        if role not in user.get("roles", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return check_role


# ============== WebSocket Manager ==============

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)
        logger.info("websocket_connected", channel=channel)

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        if channel in self.active_connections:
            self.active_connections[channel].remove(websocket)
            logger.info("websocket_disconnected", channel=channel)

    async def broadcast(self, channel: str, message: dict[str, Any]) -> None:
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


ws_manager = ConnectionManager()


# ============== Service Client ==============

class ServiceClient:
    """HTTP client for internal service communication."""

    def __init__(self) -> None:
        self.services = {
            "edge-inference": "http://edge-inference:8001",
            "tracking": "http://tracking:8002",
            "pose": "http://pose:8003",
            "intelligence-engine": "http://intelligence-engine:8005",
            "memory": "http://memory:8006",
            "explainability": "http://explainability:8007",
        }
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    async def call(
        self,
        service: str,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call an internal service."""
        if service not in self.services:
            raise HTTPException(status_code=404, detail=f"Service {service} not found")
        
        url = f"{self.services[service]}{path}"
        
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("service_call_failed", service=service, error=str(e))
            raise HTTPException(status_code=502, detail=f"Service {service} unavailable")


service_client = ServiceClient()


# ============== App Setup ==============

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    logger.info("api_gateway_starting")
    await service_client.start()
    yield
    await service_client.stop()
    logger.info("api_gateway_stopped")


app = FastAPI(
    title="SENTIENTCITY API Gateway",
    version="0.1.0",
    description="Unified API for SENTIENTCITY Smart City Intelligence System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Routes ==============

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "api-gateway"}


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate user and return JWT token."""
    # Placeholder - integrate with actual auth system
    if request.username == "admin" and request.password == "admin":
        token = auth_manager.create_token(
            user_id="admin",
            roles=["admin", "operator", "viewer"],
        )
        return TokenResponse(
            access_token=token,
            expires_in=settings.jwt_expiry_minutes * 60,
        )
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/api/v1/auth/me", response_model=UserInfo)
async def get_me(user: dict = Depends(get_current_user)) -> UserInfo:
    """Get current user info."""
    return UserInfo(
        user_id=user["sub"],
        username=user["sub"],
        roles=user.get("roles", []),
    )


@app.get("/api/v1/cameras", response_model=list[CameraStatus])
async def list_cameras(user: dict = Depends(get_current_user)) -> list[CameraStatus]:
    """List all cameras and their status."""
    # Placeholder - would query edge-inference and tracking services
    return [
        CameraStatus(
            camera_id="cam_001",
            name="Main Entrance",
            status="active",
            last_frame_time=datetime.utcnow(),
            fps=30.0,
            active_tracks=5,
        ),
        CameraStatus(
            camera_id="cam_002",
            name="Parking Lot A",
            status="active",
            last_frame_time=datetime.utcnow(),
            fps=25.0,
            active_tracks=12,
        ),
    ]


@app.get("/api/v1/alerts", response_model=list[AlertSummary])
async def list_alerts(
    severity: Severity | None = None,
    camera_id: str | None = None,
    limit: int = Query(default=50, le=200),
    user: dict = Depends(get_current_user),
) -> list[AlertSummary]:
    """List recent alerts with optional filters."""
    # Placeholder - would query memory service
    return []


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: UUID,
    user: dict = Depends(require_role("operator")),
) -> dict[str, str]:
    """Acknowledge an alert."""
    return {"status": "acknowledged", "alert_id": str(alert_id)}


@app.get("/api/v1/system/health", response_model=SystemHealth)
async def system_health(user: dict = Depends(get_current_user)) -> SystemHealth:
    """Get overall system health."""
    # Would check all services
    return SystemHealth(
        status="healthy",
        services={
            "edge-inference": "healthy",
            "tracking": "healthy",
            "intelligence-engine": "healthy",
        },
        uptime_seconds=3600.0,
        active_cameras=10,
        alerts_last_hour=3,
    )


@app.get("/api/v1/analytics/crowd")
async def crowd_analytics(
    camera_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Get crowd analytics data."""
    return {
        "total_count": 150,
        "density_map": [],
        "flow_vectors": [],
        "hotspots": [],
    }


@app.get("/api/v1/tracks/{camera_id}")
async def get_tracks(
    camera_id: str,
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Get active tracks for a camera."""
    result = await service_client.call(
        "tracking", "GET", f"/api/v1/tracks/{camera_id}"
    )
    return result


# ============== WebSocket Routes ==============

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time alert updates."""
    await ws_manager.connect(websocket, "alerts")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "alerts")


@app.websocket("/ws/camera/{camera_id}")
async def websocket_camera(websocket: WebSocket, camera_id: str) -> None:
    """WebSocket endpoint for camera-specific updates."""
    channel = f"camera:{camera_id}"
    await ws_manager.connect(websocket, channel)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
