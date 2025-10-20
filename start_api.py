#!/usr/bin/env python3
"""
Simple script to start the FPL Predictor API server.
This ensures all dependencies are properly loaded.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ.setdefault('PYTHONPATH', str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FPL Predictor API...")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            reload_dirs=[str(project_root / "api"), str(project_root / "src")]
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
