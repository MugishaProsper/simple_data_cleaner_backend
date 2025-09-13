import os
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
import pandas as pd
import logging

# Import our modules
from config import settings
from database import get_db, init_db, cleanup_expired_files, DataFile
from security import (
    RateLimitMiddleware, SecurityHeadersMiddleware, RequestLoggingMiddleware,
    generate_request_id
)
from file_manager import file_manager, plot_manager
from logging_config import setup_logging, get_logger, ErrorResponse, ErrorCodes
from services.data_cleaner import DataCleaner
from services.data_visualizer import DataVisualizer
from services.data_transformer import DataTransformer
from models.schemas import (
    CleaningOptions, VisualizationRequest, TransformationRequest
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create necessary directories
os.makedirs("static/plots", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Public Data Cleaner API...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Data Cleaner API...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# Create FastAPI app
app = FastAPI(
    title="Public Data Cleaning and Analysis API",
    version=settings.app_version,
    description="Public data cleaning and analysis API - no authentication required",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Background tasks
async def periodic_cleanup():
    """Periodic cleanup of expired files."""
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_hours * 3600)
            await file_manager.cleanup_expired_files()
            plot_manager.cleanup_old_plots(settings.max_file_age_hours)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment,
        "public": True
    }


# File upload endpoint
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload a CSV file and return initial data summary."""
    try:
        # Read file content
        content = await file.read()
        
        # Save file and get metadata (no user_id needed for public app)
        data_file = await file_manager.save_uploaded_file_public(
            file_content=content,
            original_filename=file.filename,
            db_session=db
        )
        
        # Load dataframe for summary
        df = await file_manager.get_file_dataframe(data_file)
        
        # Generate summary
        summary = {
            "file_id": data_file.file_id,
            "filename": data_file.original_filename,
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head().to_dict(orient="records")
        }
        
        logger.info(f"File uploaded successfully: {data_file.file_id}")
        return summary
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


# Data cleaning endpoint
@app.post("/clean")
async def clean_data(
    request: CleaningOptions,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Clean the uploaded data based on provided options."""
    try:
        # Get data file (no user restriction for public app)
        data_file = db.query(DataFile).filter(
            DataFile.file_id == request.file_id
        ).first()
        
        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Load dataframe
        df = await file_manager.get_file_dataframe(data_file)
        
        # Initialize cleaner
        cleaner = DataCleaner(df)
        
        # Apply cleaning
        cleaned_df = cleaner.clean_data(request)
        
        # Save processed data
        output_path = await file_manager.save_processed_dataframe(
            cleaned_df, data_file, "cleaned"
        )
        
        # Update database
        data_file.is_processed = True
        data_file.processing_status = "completed"
        db.commit()
        
        # Get cleaning summary
        cleaning_summary = cleaner.get_cleaning_summary()
        
        # Generate response
        summary = {
            "file_id": request.file_id,
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "columns": cleaned_df.columns.tolist(),
            "missing_values_before": df.isnull().sum().to_dict(),
            "missing_values_after": cleaned_df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            "preview": cleaned_df.head().to_dict(orient="records"),
            "describe": cleaned_df.describe().to_dict(),
            "cleaning_summary": cleaning_summary
        }
        
        logger.info(f"Data cleaned successfully: {request.file_id}")
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data cleaning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data cleaning failed: {str(e)}"
        )


# Visualization endpoint
@app.post("/visualize")
async def visualize_data(
    request: VisualizationRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Generate visualization based on user request."""
    try:
        # Get data file (no user restriction for public app)
        data_file = db.query(DataFile).filter(
            DataFile.file_id == request.file_id
        ).first()
        
        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Load dataframe
        df = await file_manager.get_file_dataframe(data_file)
        
        # Validate dataframe and columns
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot visualize empty dataframe"
            )
        
        if request.x_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{request.x_column}' not found in dataframe"
            )
        
        if request.y_column and request.y_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{request.y_column}' not found in dataframe"
            )
        
        # Initialize visualizer
        visualizer = DataVisualizer(df)
        
        # Create visualization
        plot_filename = visualizer.create_plot(
            plot_type=request.plot_type,
            x_column=request.x_column,
            y_column=request.y_column,
            title=request.title,
            file_id=request.file_id
        )
        
        logger.info(f"Visualization created: {plot_filename}")
        
        return {
            "plot_url": plot_filename,
            "columns": df.columns.tolist()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visualization failed: {str(e)}"
        )


# Data transformation endpoint
@app.post("/transform")
async def transform_data(
    request: TransformationRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Apply data transformation."""
    try:
        # Get data file (no user restriction for public app)
        data_file = db.query(DataFile).filter(
            DataFile.file_id == request.file_id
        ).first()
        
        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Load dataframe
        df = await file_manager.get_file_dataframe(data_file)
        
        # Initialize transformer
        transformer = DataTransformer(df)
        
        # Apply transformation
        transformed_df = transformer.apply_transformation(
            columns=request.columns,
            transformation_type=request.transformation_type,
            params=request.params
        )
        
        # Save processed data
        output_path = await file_manager.save_processed_dataframe(
            transformed_df, data_file, "transformed"
        )
        
        logger.info(f"Data transformed successfully: {request.file_id}")
        
        return {
            "columns": transformed_df.columns.tolist(),
            "preview": transformed_df.head().to_dict(orient="records"),
            "transformation_summary": transformer.get_transformation_summary()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data transformation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data transformation failed: {str(e)}"
        )


# Download endpoint
@app.get("/download/{file_id}")
async def download_data(
    file_id: str,
    db: Session = Depends(get_db)
):
    """Download the processed data as CSV."""
    try:
        # Get data file (no user restriction for public app)
        data_file = db.query(DataFile).filter(
            DataFile.file_id == file_id
        ).first()
        
        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Load dataframe
        df = await file_manager.get_file_dataframe(data_file)
        
        # Save as CSV
        output_path = await file_manager.save_processed_dataframe(
            df, data_file, "download"
        )
        
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename=f"processed_{data_file.original_filename}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )


# Get file info endpoint (public)
@app.get("/files/{file_id}")
async def get_file_info(
    file_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get information about a specific file."""
    try:
        data_file = db.query(DataFile).filter(
            DataFile.file_id == file_id
        ).first()
        
        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return {
            "file_id": data_file.file_id,
            "filename": data_file.original_filename,
            "size": data_file.file_size,
            "created_at": data_file.created_at.isoformat(),
            "is_processed": data_file.is_processed,
            "processing_status": data_file.processing_status,
            "columns": json.loads(data_file.columns) if data_file.columns else [],
            "shape": json.loads(data_file.shape) if data_file.shape else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file information"
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error_code=ErrorCodes.INTERNAL_SERVER_ERROR,
        message="An internal server error occurred",
        request_id=generate_request_id()
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.to_dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_public:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if settings.environment == "production" else 1,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
