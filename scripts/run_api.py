#!/usr/bin/env python3
"""
API server runner script.

Runs the FastAPI backend server.
"""

import argparse
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentient_city.utils.logger import setup_logger
from sentient_city.utils.config import get_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SENTIENTCITY AI API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level=args.log_level.upper())
    
    # Load configuration
    config = get_config()
    backend_config = config.get_section("backend")
    
    # Override with command line args
    host = args.host or backend_config.get("host", "0.0.0.0")
    port = args.port or backend_config.get("port", 8000)
    workers = args.workers if not args.reload else 1
    reload = args.reload or backend_config.get("reload", False)
    
    # Run server
    uvicorn.run(
        "sentient_city.backend_api.fastapi_server.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
