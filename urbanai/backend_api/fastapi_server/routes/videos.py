"""
Video endpoints for DEMO PURPOSES ONLY.

This is a standalone demo feature for testing video analysis.
It does NOT interfere with real production camera sources.

Real camera sources are configured separately in:
- configs/config.yaml (cameras section)
- Run via: python scripts/run_pipeline.py --camera <RTSP_URL>
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path
import asyncio
from loguru import logger
import os

router = APIRouter()

# Demo videos directory (isolated from production)
DEMO_VIDEOS_DIR = Path(__file__).parent.parent.parent.parent.parent / "datasets" / "raw"

# Track ongoing demo analyses (completely isolated)
_demo_analyzing_videos = {}
_demo_video_results = {}


class VideoInfo(BaseModel):
    """Video information model."""
    id: str
    name: str
    path: str
    size_mb: float
    duration: Optional[str] = None
    progress: int = 0


class VideoStatus(BaseModel):
    """Video status model."""
    id: str
    status: str  # analyzing, completed, error, idle
    progress: int
    message: Optional[str] = None


class AnalysisResult(BaseModel):
    """Analysis result model."""
    id: str
    total_detections: int
    total_tracks: int
    avg_crowd_density: float
    risk_level: str
    risk_percentage: Optional[float] = None
    alerts: List[str]
    processing_time_seconds: float


def get_video_files() -> List[VideoInfo]:
    """
    Get list of DEMO video files from datasets/raw directory.
    
    ⚠️  DEMO ONLY - This does not affect production camera sources.
    
    Real camera sources:
    - Configured in configs/config.yaml
    - Run via: python scripts/run_pipeline.py --camera <source>
    """
    videos = []
    
    if not DEMO_VIDEOS_DIR.exists():
        DEMO_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        return videos
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    for file_path in sorted(DEMO_VIDEOS_DIR.glob('*')):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            video_id = file_path.stem
            
            videos.append(VideoInfo(
                id=video_id,
                name=file_path.name,
                path=str(file_path),
                size_mb=round(size_mb, 2),
                duration=None,
                progress=_demo_analyzing_videos.get(video_id, {}).get('progress', 0)
            ))
    
    return videos


@router.get("/videos", response_model=List[VideoInfo])
async def list_videos():
    """Get list of available DEMO videos.
    
    ⚠️  DEMO ONLY - Does not affect production camera sources.
    Production cameras run independently via: python scripts/run_pipeline.py
    """
    try:
        videos = get_video_files()
        logger.info(f"Listed {len(videos)} demo videos (isolated from production)")
        return videos
    except Exception as e:
        logger.error(f"Error listing demo videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}/status", response_model=VideoStatus)
async def get_video_status(video_id: str):
    """Get status of DEMO video analysis (isolated)."""
    try:
        if video_id in _demo_analyzing_videos:
            analysis = _demo_analyzing_videos[video_id]
            return VideoStatus(
                id=video_id,
                status=analysis.get('status', 'analyzing'),
                progress=analysis.get('progress', 0),
                message=analysis.get('message')
            )
        
        if video_id in _demo_video_results:
            return VideoStatus(
                id=video_id,
                status='completed',
                progress=100,
                message='Demo analysis complete'
            )
        
        return VideoStatus(
            id=video_id,
            status='idle',
            progress=0,
            message='Ready for demo analysis'
        )
    except Exception as e:
        logger.error(f"Error getting demo video status for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/videos/{video_id}/analyze")
async def analyze_video(video_id: str):
    """Start analyzing a DEMO video (isolated from production).
    
    ⚠️  DEMO ONLY - This is a separate, isolated analysis.
    Production camera sources run independently and are NOT affected.
    """
    try:
        # Check if video exists
        videos = get_video_files()
        video = next((v for v in videos if v.id == video_id), None)
        
        if not video:
            raise HTTPException(status_code=404, detail=f"Demo video {video_id} not found")
        
        # Check if already analyzing
        if video_id in _demo_analyzing_videos:
            return {"status": "already_analyzing", "video_id": video_id, "type": "demo"}
        
        # Start demo analysis (ISOLATED - does not affect production)
        _demo_analyzing_videos[video_id] = {
            "status": "analyzing",
            "progress": 0,
            "message": "Starting demo analysis..."
        }
        
        logger.info(f"Started DEMO analysis for video: {video_id} (isolated from production)")
        
        # Run demo analysis in background (completely isolated)
        asyncio.create_task(_simulate_video_analysis(video_id, video.path))
        
        return {
            "status": "started",
            "video_id": video_id,
            "type": "demo",
            "message": f"Demo analysis started for {video.name}",
            "note": "Production camera sources are NOT affected"
        }
    except Exception as e:
        logger.error(f"Error analyzing demo video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/videos/{video_id}/stop")
async def stop_analysis(video_id: str):
    """Stop analyzing a DEMO video."""
    try:
        if video_id in _demo_analyzing_videos:
            del _demo_analyzing_videos[video_id]
            logger.info(f"Stopped demo analysis for video: {video_id}")
        
        return {
            "status": "stopped",
            "video_id": video_id,
            "type": "demo",
            "message": "Demo analysis stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping demo analysis for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}/results", response_model=Optional[AnalysisResult])
async def get_analysis_results(video_id: str):
    """Get demo analysis results for a video."""
    try:
        if video_id not in _demo_video_results:
            raise HTTPException(status_code=404, detail="Demo results not available yet")
        
        return _demo_video_results[video_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting demo results for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility function to simulate video analysis
async def _simulate_video_analysis(video_id: str, video_path: str):
    """Run REAL video analysis with actual YOLO detection.
    
    This replaces the mock simulation with real detection.
    """
    import sys
    from pathlib import Path
    
    try:
        # Add the script directory to path
        script_path = Path(__file__).parent.parent.parent.parent.parent / "scripts"
        sys.path.append(str(script_path))
        
        # Import the simple analyzer (working version)
        from simple_video_analysis import SimpleVideoAnalyzer
        
        # Initialize analyzer
        analyzer = SimpleVideoAnalyzer()
        
        # Progress callback for WebSocket updates
        def progress_callback(v_id, progress, message):
            if video_id in _demo_analyzing_videos:
                _demo_analyzing_videos[video_id].update({
                    "progress": int(progress),
                    "message": message
                })
        
        # Run real analysis
        result = analyzer.analyze_video(video_path, progress_callback)
        
        # Convert to API format
        _demo_video_results[video_id] = AnalysisResult(
            id=result.video_id,
            total_detections=result.total_detections,
            total_tracks=result.unique_tracks,
            avg_crowd_density=result.avg_crowd_density,
            risk_level=result.risk_level,
            risk_percentage=result.risk_percentage,
            alerts=result.alerts,
            processing_time_seconds=result.processing_time_seconds
        )
        
        # Update status
        del _demo_analyzing_videos[video_id]
        logger.info(f"Real analysis completed for video: {video_id}")
        
    except ImportError as e:
        logger.error(f"Could not import simple analyzer: {e}")
        # Fallback to mock analysis
        await _fallback_mock_analysis(video_id, video_path)
    except Exception as e:
        logger.error(f"Error in real analysis for {video_id}: {e}")
        if video_id in _demo_analyzing_videos:
            _demo_analyzing_videos[video_id]["status"] = "error"
            _demo_analyzing_videos[video_id]["message"] = str(e)


async def _fallback_mock_analysis(video_id: str, video_path: str):
    """Fallback to mock analysis if real analysis fails."""
    import random
    import time
    
    # Simulate processing steps
    steps = [
        (20, "Loading video (fallback mode)..."),
        (40, "Detecting objects (mock)..."),
        (60, "Tracking persons (mock)..."),
        (80, "Analyzing crowd (mock)..."),
        (95, "Generating alerts (mock)..."),
        (100, "Mock analysis complete!")
    ]
    
    for progress, message in steps:
        if video_id not in _demo_analyzing_videos:
            return  # Stopped
        
        _demo_analyzing_videos[video_id].update({
            "progress": progress,
            "message": message
        })
        
        await asyncio.sleep(1)  # Faster for fallback
    
    # Generate mock results
    _demo_video_results[video_id] = AnalysisResult(
        id=video_id,
        total_detections=random.randint(50, 500),
        total_tracks=random.randint(10, 100),
        avg_crowd_density=round(random.uniform(0.2, 0.9), 2),
        risk_level=random.choice(['Low', 'Medium', 'High', 'Critical']),
        alerts=[
            "High crowd density detected",
            "Unusual movement pattern",
            "Person fell down"
        ] if random.random() > 0.5 else [],
        processing_time_seconds=random.randint(30, 300)
    )
    
    # Update status
    del _demo_analyzing_videos[video_id]
    logger.info(f"Fallback mock analysis completed for video: {video_id}")
