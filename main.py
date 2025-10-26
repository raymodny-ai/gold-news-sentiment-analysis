"""
"""
Main FastAPI application for the gold news sentiment analysis system.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from app.core.config import settings
from app.models.database import create_tables
from app.api.routes import router


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up the application...")

    # Create database tables
    try:
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

    yield

    # Shutdown
    logger.info("Shutting down the application...")


# Create FastAPI app
app = FastAPI(
    title="Gold News Sentiment Analysis API",
    description="API for analyzing news sentiment and predicting gold prices",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add trusted host middleware (for production)
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure allowed hosts in production
    )


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()

    # Log request
    logger.info(f"{request.method} {request.url.path} - Start")

    try:
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"{request.method} {request.url.path} - "
            f"Error: {str(e)} - Time: {process_time:.3f}s"
        )
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )


# Include API routes
app.include_router(
    router,
    prefix=f"/api/{settings.api_version}",
    tags=["api"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Gold News Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "api_info": "/api/info",
        "health": "/api/v1/health"
    }


# API info endpoint (moved to /api/info to avoid conflicts)
@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Gold News Sentiment Analysis API",
        "version": "1.0.0",
        "description": "API for analyzing news sentiment and predicting gold prices",
        "endpoints": {
            "news": "/api/v1/news",
            "sentiment": "/api/v1/sentiment",
            "predictions": "/api/v1/predictions",
            "gold_prices": "/api/v1/gold-prices",
            "analytics": "/api/v1/analytics",
            "health": "/api/v1/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )
