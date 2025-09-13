import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
import pandas as pd
import logging

# Import our modules
from config import settings
from database import get_db, init_db, cleanup_expired_files, DataFile, User
from auth import (
    auth_manager, get_current_user, get_current_active_user, 
    authenticate_user, create_user, User
)
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
    CleaningOptions, VisualizationRequest, TransformationRequest,
    UserCreate, UserLogin, Token, TokenRefresh
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
    logger.info("Starting Data Cleaner API...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Create default admin user if it doesn't exist
    await create_default_admin()
    
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
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready data cleaning and analysis API",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
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


async def create_default_admin():
    """Create default admin user if it doesn't exist."""
    db = next(get_db())
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            create_user(
                db=db,
                username="admin",
                email="admin@datacleaner.com",
                password="admin123",  # Change this in production!
                is_admin=True
            )
            logger.info("Default admin user created")
    except Exception as e:
        logger.error(f"Error creating default admin: {e}")
    finally:
        db.close()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment
    }


# Authentication endpoints
@app.post("/auth/register", response_model=Dict[str, str])
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        # Create user
        user = create_user(
            db=db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )
        
        logger.info(f"New user registered: {user.username}")
        
        return {"message": "User registered successfully", "user_id": str(user.id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return access token."""
    try:
        user = authenticate_user(db, user_credentials.username, user_credentials.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token = auth_manager.create_access_token(data={"sub": str(user.id)})
        refresh_token = auth_manager.create_refresh_token(data={"sub": str(user.id)})
        
        # Create session
        token_id = generate_request_id()
        auth_manager.create_user_session(db, user.id, token_id)
        
        logger.info(f"User logged in: {user.username}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.post("/auth/refresh", response_model=Token)
async def refresh_token(refresh_data: TokenRefresh, db: Session = Depends(get_db)):
    """Refresh access token."""
    try:
        payload = auth_manager.verify_token(refresh_data.refresh_token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        access_token = auth_manager.create_access_token(data={"sub": str(user.id)})
        new_refresh_token = auth_manager.create_refresh_token(data={"sub": str(user.id)})
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


# File upload endpoint
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload a CSV file and return initial data summary."""
    try:
        # Read file content
        content = await file.read()
        
        # Save file and get metadata
        data_file = await file_manager.save_uploaded_file(
            file_content=content,
            original_filename=file.filename,
            user_id=current_user.id,
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Clean the uploaded data based on provided options."""
    try:
        # Get data file
        data_file = db.query(DataFile).filter(
            DataFile.file_id == request.file_id,
            DataFile.owner_id == current_user.id
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Generate visualization based on user request."""
    try:
        # Get data file
        data_file = db.query(DataFile).filter(
            DataFile.file_id == request.file_id,
            DataFile.owner_id == current_user.id
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Apply data transformation."""
    try:
        # Get data file
        data_file = db.query(DataFile).filter(
            DataFile.file_id == request.file_id,
            DataFile.owner_id == current_user.id
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Download the processed data as CSV."""
    try:
        # Get data file
        data_file = db.query(DataFile).filter(
            DataFile.file_id == file_id,
            DataFile.owner_id == current_user.id
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


# User files endpoint
@app.get("/files")
async def get_user_files(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get list of user's uploaded files."""
    try:
        files = db.query(DataFile).filter(
            DataFile.owner_id == current_user.id
        ).order_by(DataFile.created_at.desc()).all()
        
        return [
            {
                "file_id": file.file_id,
                "filename": file.original_filename,
                "size": file.file_size,
                "created_at": file.created_at.isoformat(),
                "is_processed": file.is_processed,
                "processing_status": file.processing_status
            }
            for file in files
        ]
        
    except Exception as e:
        logger.error(f"Error getting user files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve files"
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
        "main_production:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if settings.environment == "production" else 1,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
