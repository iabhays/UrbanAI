"""Alert endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# In-memory storage (in production, use database)
alerts_storage: List[dict] = []


class AlertResponse(BaseModel):
    """Alert response model."""
    alert_id: str
    timestamp: str
    severity: str
    risk_level: str
    risk_score: float
    camera_id: Optional[str]
    location: Optional[str]
    explanation: str
    acknowledged: bool


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None
):
    """Get alerts."""
    filtered = alerts_storage
    
    if severity:
        filtered = [a for a in filtered if a.get("severity") == severity]
    
    if acknowledged is not None:
        filtered = [a for a in filtered if a.get("acknowledged") == acknowledged]
    
    return filtered[-limit:]


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    """Get specific alert."""
    for alert in alerts_storage:
        if alert.get("alert_id") == alert_id:
            return alert
    
    raise HTTPException(status_code=404, detail="Alert not found")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge alert."""
    for alert in alerts_storage:
        if alert.get("alert_id") == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = datetime.now().isoformat()
            return {"status": "acknowledged"}
    
    raise HTTPException(status_code=404, detail="Alert not found")
