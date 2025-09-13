import os
import shutil
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

from config import settings
from database import DataFile, get_db, generate_file_id, get_file_expiry_time
from security import FileUploadSecurity, InputValidation
from logging_config import FileProcessingError, ValidationError, get_logger

logger = get_logger(__name__)


class FileManager:
    """Enhanced file management with async support and validation."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.max_file_size = settings.max_file_size
        self.allowed_types = settings.allowed_file_types
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(
        self,
        file_content: bytes,
        original_filename: str,
        user_id: int,
        db_session
    ) -> DataFile:
        """Save uploaded file with validation and metadata extraction."""
        
        # Validate file
        if not FileUploadSecurity.validate_file_size(len(file_content)):
            raise FileProcessingError(f"File too large. Maximum size: {self.max_file_size} bytes")
        
        if not FileUploadSecurity.validate_file_type(original_filename):
            raise FileProcessingError(f"Invalid file type. Allowed types: {', '.join(self.allowed_types)}")
        
        # Generate secure filename and file ID
        file_id = generate_file_id()
        secure_filename = FileUploadSecurity.generate_secure_filename(original_filename)
        file_path = self.upload_dir / secure_filename
        
        # Save file asynchronously
        await self._save_file_async(file_content, file_path)
        
        # Extract metadata
        metadata = await self._extract_file_metadata(file_path, original_filename)
        
        # Create database record
        data_file = DataFile(
            file_id=file_id,
            filename=secure_filename,
            original_filename=original_filename,
            file_path=str(file_path),
            file_size=len(file_content),
            mime_type=self._get_mime_type(original_filename),
            owner_id=user_id,
            columns=json.dumps(metadata.get('columns')),
            shape=json.dumps(metadata.get('shape')),
            data_types=json.dumps(metadata.get('data_types')),
            missing_values=json.dumps(metadata.get('missing_values')),
            expires_at=get_file_expiry_time()
        )
        
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)
        
        logger.info(f"File saved successfully: {file_id}")
        return data_file
    
    async def save_uploaded_file_public(
        self,
        file_content: bytes,
        original_filename: str,
        db_session
    ) -> DataFile:
        """Save uploaded file for public access (no user authentication)."""
        
        # Validate file
        if not FileUploadSecurity.validate_file_size(len(file_content)):
            raise FileProcessingError(f"File too large. Maximum size: {self.max_file_size} bytes")
        
        if not FileUploadSecurity.validate_file_type(original_filename):
            raise FileProcessingError(f"Invalid file type. Allowed types: {', '.join(self.allowed_types)}")
        
        # Generate secure filename and file ID
        file_id = generate_file_id()
        secure_filename = FileUploadSecurity.generate_secure_filename(original_filename)
        file_path = self.upload_dir / secure_filename
        
        # Save file asynchronously
        await self._save_file_async(file_content, file_path)
        
        # Extract metadata
        metadata = await self._extract_file_metadata(file_path, original_filename)
        
        # Create database record (no user_id for public access)
        data_file = DataFile(
            file_id=file_id,
            filename=secure_filename,
            original_filename=original_filename,
            file_path=str(file_path),
            file_size=len(file_content),
            mime_type=self._get_mime_type(original_filename),
            owner_id=None,  # No user required for public files
            columns=json.dumps(metadata.get('columns')),
            shape=json.dumps(metadata.get('shape')),
            data_types=json.dumps(metadata.get('data_types')),
            missing_values=json.dumps(metadata.get('missing_values')),
            expires_at=get_file_expiry_time()
        )
        
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)
        
        logger.info(f"Public file saved successfully: {file_id}")
        return data_file
    
    async def _save_file_async(self, content: bytes, file_path: Path) -> None:
        """Save file content asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._save_file_sync,
            content,
            file_path
        )
    
    def _save_file_sync(self, content: bytes, file_path: Path) -> None:
        """Save file content synchronously."""
        with open(file_path, 'wb') as f:
            f.write(content)
    
    async def _extract_file_metadata(self, file_path: Path, original_filename: str) -> Dict[str, Any]:
        """Extract metadata from uploaded file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_metadata_sync,
            file_path,
            original_filename
        )
    
    def _extract_metadata_sync(self, file_path: Path, original_filename: str) -> Dict[str, Any]:
        """Extract metadata synchronously."""
        try:
            # Read file based on extension
            file_extension = original_filename.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise FileProcessingError(f"Unsupported file type: {file_extension}")
            
            # Extract metadata
            metadata = {
                'columns': df.columns.tolist(),
                'shape': [df.shape[0], df.shape[1]],
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': df.isnull().sum().to_dict()
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            raise FileProcessingError(f"Failed to extract file metadata: {str(e)}")
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type based on file extension."""
        extension = filename.lower().split('.')[-1]
        mime_types = {
            'csv': 'text/csv',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'xls': 'application/vnd.ms-excel'
        }
        return mime_types.get(extension, 'application/octet-stream')
    
    async def get_file_dataframe(self, data_file: DataFile) -> pd.DataFrame:
        """Load file as pandas DataFrame."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._load_dataframe_sync,
            data_file
        )
    
    def _load_dataframe_sync(self, data_file: DataFile) -> pd.DataFrame:
        """Load DataFrame synchronously."""
        file_extension = data_file.original_filename.lower().split('.')[-1]
        
        if file_extension == 'csv':
            return pd.read_csv(data_file.file_path)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(data_file.file_path)
        else:
            raise FileProcessingError(f"Unsupported file type: {file_extension}")
    
    async def save_processed_dataframe(
        self,
        df: pd.DataFrame,
        data_file: DataFile,
        operation_type: str
    ) -> str:
        """Save processed DataFrame to file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._save_dataframe_sync,
            df,
            data_file,
            operation_type
        )
    
    def _save_dataframe_sync(self, df: pd.DataFrame, data_file: DataFile, operation_type: str) -> str:
        """Save DataFrame synchronously."""
        # Generate output filename
        output_filename = f"{operation_type}_{data_file.file_id}.csv"
        output_path = self.upload_dir / output_filename
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Processed data saved: {output_path}")
        return str(output_path)
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from filesystem."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    async def cleanup_expired_files(self) -> Dict[str, int]:
        """Clean up expired files."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._cleanup_expired_files_sync
        )
    
    def _cleanup_expired_files_sync(self) -> Dict[str, int]:
        """Clean up expired files synchronously."""
        from database import SessionLocal
        
        db = SessionLocal()
        deleted_files = 0
        deleted_plots = 0
        
        try:
            # Get expired files
            expired_files = db.query(DataFile).filter(
                DataFile.expires_at < datetime.utcnow()
            ).all()
            
            for file in expired_files:
                # Delete physical file
                if self.delete_file(file.file_path):
                    deleted_files += 1
                
                # Delete associated plot files
                from database import PlotFile
                plot_files = db.query(PlotFile).filter(
                    PlotFile.data_file_id == file.id
                ).all()
                
                for plot in plot_files:
                    if self.delete_file(plot.plot_path):
                        deleted_plots += 1
                    db.delete(plot)
                
                db.delete(file)
            
            db.commit()
            logger.info(f"Cleanup completed: {deleted_files} files, {deleted_plots} plots deleted")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error during cleanup: {e}")
        finally:
            db.close()
        
        return {
            "deleted_files": deleted_files,
            "deleted_plots": deleted_plots
        }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        if not os.path.exists(file_path):
            return {}
        
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "exists": True
        }
    
    def validate_file_integrity(self, file_path: str, expected_hash: str = None) -> bool:
        """Validate file integrity."""
        if not os.path.exists(file_path):
            return False
        
        if expected_hash:
            import hashlib
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash == expected_hash
        
        return True


class PlotManager:
    """Manage plot file generation and storage."""
    
    def __init__(self):
        self.plots_dir = Path("static/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def save_plot(self, plot_data: bytes, plot_id: str, plot_type: str) -> str:
        """Save plot data to file."""
        filename = f"{plot_type}_{plot_id}.png"
        file_path = self.plots_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(plot_data)
        
        return str(file_path)
    
    def delete_plot(self, plot_path: str) -> bool:
        """Delete plot file."""
        try:
            if os.path.exists(plot_path):
                os.remove(plot_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting plot {plot_path}: {e}")
            return False
    
    def cleanup_old_plots(self, max_age_hours: int = 24) -> int:
        """Clean up old plot files."""
        deleted_count = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for plot_file in self.plots_dir.glob("*.png"):
            if datetime.fromtimestamp(plot_file.stat().st_mtime) < cutoff_time:
                if self.delete_plot(str(plot_file)):
                    deleted_count += 1
        
        return deleted_count


# Global instances
file_manager = FileManager()
plot_manager = PlotManager()
