import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, Normalizer, Binarizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    A class for performing various data transformations on pandas DataFrames.
    Handles both numerical and categorical data with support for custom transformations.
    """
    
    def __init__(self):
        """Initialize the DataTransformer with default settings."""
        self.df = None
        self.transformation_summary = {
            'transformations': [],
            'warnings': [],
            'errors': []
        }
        self._fitted = False
        self._transformers = {}
        self._preprocessor = None
        
    def _log_transformation(self, operation: str, columns: List[str], **kwargs) -> None:
        """Helper method to log transformation details."""
        log_entry = {
            'operation': operation,
            'columns': columns,
            'timestamp': pd.Timestamp.now().isoformat(),
            'parameters': kwargs
        }
        self.transformation_summary['transformations'].append(log_entry)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set the dataframe to be transformed."""
        self.df = df.copy()

    def normalize(self, columns: List[str]) -> None:
        """Apply min-max normalization to scale data between 0 and 1."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.transformation_summary["transformations_applied"].append("normalize")
        self.transformation_summary["columns_transformed"].extend(columns)

    def standardize(self, columns: List[str]) -> None:
        """Apply standardization (z-score normalization)."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.transformation_summary["transformations_applied"].append("standardize")
        self.transformation_summary["columns_transformed"].extend(columns)

    def log_transform(self, columns: List[str]) -> None:
        """Apply logarithmic transformation."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        for col in columns:
            # Handle negative values by shifting
            min_val = self.df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                self.df[col] = np.log(self.df[col] + shift)
            else:
                self.df[col] = np.log(self.df[col])
        
        self.transformation_summary["transformations_applied"].append("log_transform")
        self.transformation_summary["columns_transformed"].extend(columns)

    def square_root_transform(self, columns: List[str]) -> None:
        """Apply square root transformation."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        for col in columns:
            # Handle negative values by shifting
            min_val = self.df[col].min()
            if min_val < 0:
                shift = abs(min_val)
                self.df[col] = np.sqrt(self.df[col] + shift)
            else:
                self.df[col] = np.sqrt(self.df[col])
        
        self.transformation_summary["transformations_applied"].append("square_root")
        self.transformation_summary["columns_transformed"].extend(columns)

    def min_max_scale(self, columns: List[str], feature_range: tuple = (0, 1)) -> None:
        """Scale features to a specific range."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        scaler = MinMaxScaler(feature_range=feature_range)
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.transformation_summary["transformations_applied"].append("min_max_scale")
        self.transformation_summary["columns_transformed"].extend(columns)

    def get_transformation_summary(self) -> Dict:
        """Get summary of all transformations performed."""
        return self.transformation_summary