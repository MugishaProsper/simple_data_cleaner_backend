import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Generator
import uuid

from config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
    pool_recycle=300
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    files = relationship("DataFile", back_populates="owner")
    sessions = relationship("UserSession", back_populates="user")


class UserSession(Base):
    """User session model for JWT token management."""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_id = Column(String(255), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class DataFile(Base):
    """Data file model for uploaded files."""
    __tablename__ = "data_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(255), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Optional for public files
    
    # File metadata
    columns = Column(Text)  # JSON string of column names
    shape = Column(Text)    # JSON string of (rows, cols)
    data_types = Column(Text)  # JSON string of column data types
    missing_values = Column(Text)  # JSON string of missing value counts
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="files")
    operations = relationship("DataOperation", back_populates="data_file")


class DataOperation(Base):
    """Data operation model for tracking cleaning/transformation operations."""
    __tablename__ = "data_operations"
    
    id = Column(Integer, primary_key=True, index=True)
    data_file_id = Column(Integer, ForeignKey("data_files.id"), nullable=False)
    operation_type = Column(String(50), nullable=False)  # clean, transform, visualize
    operation_params = Column(Text)  # JSON string of operation parameters
    operation_result = Column(Text)  # JSON string of operation results
    
    # Performance metrics
    execution_time_ms = Column(Integer)
    memory_usage_mb = Column(Float)
    
    # Status
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    data_file = relationship("DataFile", back_populates="operations")


class PlotFile(Base):
    """Plot file model for generated visualizations."""
    __tablename__ = "plot_files"
    
    id = Column(Integer, primary_key=True, index=True)
    plot_id = Column(String(255), unique=True, index=True, nullable=False)
    data_file_id = Column(Integer, ForeignKey("data_files.id"), nullable=False)
    plot_type = Column(String(50), nullable=False)
    plot_path = Column(String(500), nullable=False)
    plot_params = Column(Text)  # JSON string of plot parameters
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)


# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database initialization
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def cleanup_expired_files():
    """Clean up expired files and sessions."""
    db = SessionLocal()
    try:
        # Clean up expired data files
        expired_files = db.query(DataFile).filter(
            DataFile.expires_at < datetime.utcnow()
        ).all()
        
        for file in expired_files:
            # Delete physical file
            if os.path.exists(file.file_path):
                os.remove(file.file_path)
            
            # Delete associated plot files
            plot_files = db.query(PlotFile).filter(
                PlotFile.data_file_id == file.id
            ).all()
            
            for plot in plot_files:
                if os.path.exists(plot.plot_path):
                    os.remove(plot.plot_path)
                db.delete(plot)
            
            db.delete(file)
        
        # Clean up expired sessions
        expired_sessions = db.query(UserSession).filter(
            UserSession.expires_at < datetime.utcnow()
        ).all()
        
        for session in expired_sessions:
            db.delete(session)
        
        db.commit()
        print(f"Cleaned up {len(expired_files)} expired files and {len(expired_sessions)} expired sessions")
        
    except Exception as e:
        db.rollback()
        print(f"Error during cleanup: {e}")
    finally:
        db.close()


def generate_file_id() -> str:
    """Generate unique file ID."""
    return str(uuid.uuid4())


def get_file_expiry_time() -> datetime:
    """Get file expiry time based on settings."""
    return datetime.utcnow() + timedelta(hours=settings.max_file_age_hours)
