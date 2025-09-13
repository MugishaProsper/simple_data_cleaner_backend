from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from enum import Enum

class PlotType(str, Enum):
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"

class TransformationType(str, Enum):
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    LOG = "log"
    SQRT = "sqrt"
    SCALE = "scale"

class CleaningOptions(BaseModel):
    fill_missing: bool = True
    drop_duplicates: bool = True
    standardize_columns: bool = True
    fix_datatypes: bool = True
    handle_outliers: bool = True
    strip_whitespace: bool = True
    fix_dates: bool = True
    remove_constant_columns: bool = True
    custom_na_values: Optional[List[str]] = None
    date_columns: Optional[List[str]] = None
    outlier_threshold: Optional[float] = 3.0

class VisualizationRequest(BaseModel):
    file_id: str
    plot_type: PlotType
    x_column: str
    y_column: Optional[str] = None
    title: Optional[str] = None

class TransformationRequest(BaseModel):
    file_id: str
    columns: List[str]
    transformation_type: TransformationType
    params: Optional[Dict[str, Any]] = None

class ColumnSummary(BaseModel):
    name: str
    dtype: str
    missing_count: int
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    sample_values: List[Any]

class DataSummary(BaseModel):
    total_rows: int
    total_columns: int
    missing_cells: int
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    column_summaries: List[ColumnSummary]


# Authentication schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class TokenRefresh(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: str


# File management schemas
class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    columns: List[str]
    shape: List[int]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    preview: List[Dict[str, Any]]


class FileListResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    created_at: str
    is_processed: bool
    processing_status: str