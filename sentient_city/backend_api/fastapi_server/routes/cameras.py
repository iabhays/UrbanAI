"""Camera endpoints."""

from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

router = APIRouter()

from ...utils.config import get_config


class CameraResponse(BaseModel):
    """Camera response model."""
    id: str
    source: str
    location: str
    coordinates: List[float]
    enabled: bool


@router.get("/cameras", response_model=List[CameraResponse])
async def get_cameras():
    """Get list of cameras."""
    config = get_config()
    cameras_config = config.get_section("cameras")
    
    cameras = []
    for camera in cameras_config:
        cameras.append(CameraResponse(
            id=camera.get("id"),
            source=camera.get("source"),
            location=camera.get("location"),
            coordinates=camera.get("coordinates", [0.0, 0.0]),
            enabled=camera.get("enabled", True)
        ))
    
    return cameras
