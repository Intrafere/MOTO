"""
Middleware for CORS and error handling.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

# Default allowed origins for local development
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def setup_middleware(app: FastAPI) -> None:
    """Setup middleware for the FastAPI app."""
    
    # Allow custom origins via environment variable (comma-separated)
    # Example: CORS_ORIGINS=http://localhost:3000,http://example.com
    custom_origins = os.environ.get("CORS_ORIGINS", "")
    if custom_origins:
        origins = [o.strip() for o in custom_origins.split(",") if o.strip()]
        logger.info(f"Using custom CORS origins: {origins}")
    else:
        origins = DEFAULT_ORIGINS
        logger.info(f"Using default CORS origins: {origins}")
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info("Middleware configured")
