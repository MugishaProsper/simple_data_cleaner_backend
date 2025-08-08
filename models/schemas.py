from pydantic import BaseModel
from typing import List, Optional, Dict
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
    SQUARE_ROOT = "square_root"
    MIN_MAX_SCALE = "min_max_scale"

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
    outlier_threshold: Optional[float] = 3.0  # For z-score method

class VisualizationRequest(BaseModel):
    plot_type: PlotType
    x_column: str
    y_column: Optional[str] = None
    title: Optional[str] = None
    additional_params: Optional[Dict] = None

class TransformationRequest(BaseModel):
    columns: List[str]
    transformation_type: TransformationType
    params: Optional[Dict] = None

class ColumnSummary(BaseModel):
    name: str
    dtype: str
    missing_count: int
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    sample_values: List[str]

class DataSummary(BaseModel):
    total_rows: int
    total_columns: int
    memory_usage: str
    columns: List[ColumnSummary]
    missing_data_summary: Dict[str, int]
    duplicate_rows: int