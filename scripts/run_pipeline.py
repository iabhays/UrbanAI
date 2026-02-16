#!/usr/bin/env python3
"""
Main pipeline runner script.

Runs the SENTIENTCITY AI processing pipeline.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentient_city.pipeline import SentientCityPipeline
from sentient_city.utils.logger import setup_logger
from sentient_city.utils.config import get_config
from loguru import logger


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SENTIENTCITY AI Pipeline Runner")
    parser.add_argument(
        "--camera",
        type=str,
        help="Camera source (RTSP URL, file path, or camera index)"
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        help="Camera ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level=args.log_level)
    
    # Load configuration
    config = get_config(args.config) if args.config else get_config()
    
    # Get camera source
    if args.camera:
        camera_source = args.camera
        camera_id = args.camera_id or f"camera_{hash(camera_source)}"
    else:
        # Use first enabled camera from config
        cameras_config = config.get_section("cameras")
        enabled_cameras = [c for c in cameras_config if c.get("enabled", True)]
        
        if not enabled_cameras:
            logger.error("No enabled cameras found in configuration")
            return
        
        camera = enabled_cameras[0]
        camera_source = camera["source"]
        camera_id = camera["id"]
    
    logger.info(f"Starting pipeline for camera: {camera_id} ({camera_source})")
    
    # Create and run pipeline
    try:
        pipeline = SentientCityPipeline(
            camera_source=camera_source,
            camera_id=camera_id
        )
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
