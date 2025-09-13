import time
import hashlib
import secrets
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from config import settings
from logging_config import ErrorResponse, ErrorCodes

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            error_response = ErrorResponse(
                error_code=ErrorCodes.RATE_LIMIT_EXCEEDED,
                message="Rate limit exceeded. Please try again later.",
                details={"retry_after": 60}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response.to_dict()
            )
        
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded IP first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.requests[client_ip] = []
        
        # Check if limit exceeded
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_ip].append(current_time)
        return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("User-Agent", ""),
            }
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "client_ip": request.client.host if request.client else "unknown",
                }
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": process_time,
                    "client_ip": request.client.host if request.client else "unknown",
                },
                exc_info=True
            )
            
            raise


class FileUploadSecurity:
    """File upload security utilities."""
    
    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """Validate file type based on extension."""
        if not filename:
            return False
        
        file_extension = filename.lower().split('.')[-1]
        return file_extension in settings.allowed_file_types
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate file size."""
        return file_size <= settings.max_file_size
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        import os
        import re
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    @staticmethod
    def generate_secure_filename(original_filename: str) -> str:
        """Generate a secure filename with random prefix."""
        import uuid
        
        # Get file extension
        file_extension = original_filename.lower().split('.')[-1]
        
        # Generate secure filename
        secure_name = f"{uuid.uuid4().hex}.{file_extension}"
        
        return secure_name


class InputValidation:
    """Input validation utilities."""
    
    @staticmethod
    def validate_csv_data(data: str) -> Tuple[bool, Optional[str]]:
        """Validate CSV data format."""
        try:
            import pandas as pd
            from io import StringIO
            
            # Try to read the CSV
            df = pd.read_csv(StringIO(data))
            
            # Check if dataframe is empty
            if df.empty:
                return False, "CSV file is empty"
            
            # Check for reasonable number of columns
            if len(df.columns) > 1000:
                return False, "Too many columns (max 1000)"
            
            # Check for reasonable number of rows
            if len(df) > 1000000:  # 1 million rows
                return False, "Too many rows (max 1,000,000)"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid CSV format: {str(e)}"
    
    @staticmethod
    def validate_plot_parameters(plot_type: str, x_column: str, y_column: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate plot parameters."""
        valid_plot_types = ["line", "bar", "scatter", "histogram", "heatmap"]
        
        if plot_type not in valid_plot_types:
            return False, f"Invalid plot type. Must be one of: {', '.join(valid_plot_types)}"
        
        if not x_column or not x_column.strip():
            return False, "X column is required"
        
        # For scatter plots, y_column is required
        if plot_type == "scatter" and (not y_column or not y_column.strip()):
            return False, "Y column is required for scatter plots"
        
        return True, None
    
    @staticmethod
    def validate_transformation_parameters(transformation_type: str, columns: list) -> Tuple[bool, Optional[str]]:
        """Validate transformation parameters."""
        valid_transformations = ["normalize", "standardize", "log", "sqrt", "scale"]
        
        if transformation_type not in valid_transformations:
            return False, f"Invalid transformation type. Must be one of: {', '.join(valid_transformations)}"
        
        if not columns or len(columns) == 0:
            return False, "At least one column must be specified"
        
        return True, None


class CSRFProtection:
    """CSRF protection utilities."""
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate a CSRF token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_csrf_token(token: str, session_token: str) -> bool:
        """Validate CSRF token."""
        return token == session_token and len(token) > 0


# Utility functions
def hash_file_content(content: bytes) -> str:
    """Generate hash of file content for integrity checking."""
    return hashlib.sha256(content).hexdigest()


def generate_request_id() -> str:
    """Generate unique request ID."""
    return secrets.token_urlsafe(16)


def is_safe_path(basedir: str, path: str) -> bool:
    """Check if path is safe (prevents directory traversal)."""
    import os
    
    # Resolve the real path
    real_path = os.path.realpath(path)
    real_basedir = os.path.realpath(basedir)
    
    # Check if the resolved path starts with the base directory
    return real_path.startswith(real_basedir)
