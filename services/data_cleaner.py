import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.original_df = None
        self.df = None
        self.cleaning_summary = {
            'original_shape': None,
            'cleaned_shape': None,
            'columns_processed': [],
            'steps_applied': [],
            'warnings': [],
            'errors': [],
            'missing_values_filled': 0,
            'rows_removed': 0
        }
        self.column_metadata = {}
        
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set the dataframe to be cleaned."""
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_summary['original_shape'] = df.shape
        
    def clean_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """
        Main method to clean the dataframe based on provided options.
        
        Args:
            df: Input pandas DataFrame
            options: Dictionary of cleaning options
            
        Returns:
            Cleaned pandas DataFrame
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.cleaning_summary = {
            'original_shape': self.df.shape,
            'columns_processed': [],
            'steps_applied': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Standardize column names first
            if options.get('standardize_columns', True):
                self._standardize_column_names()
            
            # Handle missing values
            if options.get('fill_na', True) or options.get('drop_na', False):
                self._handle_missing_values(
                    fill_na=options.get('fill_na', True),
                    drop_na=options.get('drop_na', False),
                    fill_strategy=options.get('fill_strategy', 'mean'),
                    na_threshold=options.get('na_threshold', 0.5)
                )
            
            # Handle duplicates
            if options.get('drop_duplicates', True):
                self._remove_duplicates()
            
            # Handle data types
            if options.get('infer_types', True):
                self._infer_and_convert_dtypes()
            
            # Handle outliers
            if options.get('handle_outliers', True):
                self._handle_outliers(
                    method=options.get('outlier_method', 'zscore'),
                    z_threshold=options.get('z_threshold', 3.0),
                    iqr_multiplier=options.get('iqr_multiplier', 1.5)
                )
            
            # Clean string data
            if options.get('strip_strings', True):
                self._clean_string_columns(
                    strip_whitespace=options.get('strip_strings', True),
                    lowercase=options.get('lowercase_columns', True)
                )
            
            # Parse dates
            if options.get('parse_dates', True):
                self._parse_dates()
            
            # Remove constant columns
            if options.get('remove_constant_columns', True):
                self._remove_constant_columns()
            
            # Update cleaning summary
            self.cleaning_summary['cleaned_shape'] = self.df.shape
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            self.cleaning_summary['errors'].append(str(e))
            raise

    def _standardize_column_names(self) -> None:
        """Standardize column names by removing special chars and converting to snake_case."""
        new_columns = {}
        for col in self.df.columns:
            # Convert to string in case column names are not strings
            col_str = str(col)
            
            # Replace special characters and spaces with underscore
            new_col = re.sub(r'[^\w\s]', ' ', col_str)
            
            # Convert to snake_case
            new_col = new_col.strip().lower().replace(' ', '_')
            new_col = re.sub('_+', '_', new_col)  # Replace multiple underscores with one
            
            # Ensure column name is valid Python identifier
            if not new_col or new_col[0].isdigit():
                new_col = f'col_{new_col}'
                
            new_columns[col] = new_col
        
        if new_columns:
            self.df.rename(columns=new_columns, inplace=True)
            self.cleaning_summary['steps_applied'].append('standardized_column_names')
            self.cleaning_summary['columns_renamed'] = new_columns
    
    def _handle_missing_values(self, fill_na: bool = True, drop_na: bool = False, 
                             fill_strategy: str = 'mean', na_threshold: float = 0.5) -> None:
        """Handle missing values based on the specified strategy."""
        # Remove columns with too many missing values
        na_ratio = self.df.isnull().mean()
        cols_to_drop = na_ratio[na_ratio > na_threshold].index.tolist()
        
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.cleaning_summary['steps_applied'].append(f'dropped_high_na_columns: {cols_to_drop}')
            self.cleaning_summary['columns_dropped'] = cols_to_drop
        
        if drop_na:
            initial_rows = len(self.df)
            self.df.dropna(inplace=True)
            rows_removed = initial_rows - len(self.df)
            if rows_removed > 0:
                self.cleaning_summary['steps_applied'].append(f'dropped_{rows_removed}_rows_with_na')
        
        if fill_na and fill_strategy:
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        if fill_strategy == 'mean':
                            fill_value = self.df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = self.df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                        elif fill_strategy == 'ffill':
                            self.df[col].fillna(method='ffill', inplace=True)
                            continue
                        elif fill_strategy == 'bfill':
                            self.df[col].fillna(method='bfill', inplace=True)
                            continue
                        else:  # default to 0 or specified value
                            try:
                                fill_value = float(fill_strategy)
                            except (ValueError, TypeError):
                                fill_value = 0
                    else:  # For non-numeric columns
                        if fill_strategy == 'mode':
                            fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else ''
                        else:
                            fill_value = ''  # Default empty string for non-numeric
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    self.cleaning_summary['steps_applied'].append(f'filled_na_in_{col}_with_{fill_strategy}')
    
    def _remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataframe."""
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(self.df)
        if duplicates_removed > 0:
            self.cleaning_summary['steps_applied'].append(f'removed_{duplicates_removed}_duplicate_rows')
    
    def _infer_and_convert_dtypes(self) -> None:
        """Infer and convert column data types where possible."""
        for col in self.df.select_dtypes(include=['object']).columns:
            # Skip if all values are null
            if self.df[col].isnull().all():
                continue
                
            # Try to convert to numeric first
            try:
                # Clean string representation of numbers
                cleaned_series = self.df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
                converted = pd.to_numeric(cleaned_series, errors='raise')
                self.df[col] = converted
                self.cleaning_summary['steps_applied'].append(f'converted_{col}_to_numeric')
                continue
            except (ValueError, TypeError):
                pass
                
            # Try to convert to datetime
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='raise')
                self.cleaning_summary['steps_applied'].append(f'converted_{col}_to_datetime')
            except (ValueError, TypeError):
                pass
    
    def _handle_outliers(self, method: str = 'zscore', z_threshold: float = 3.0, 
                        iqr_multiplier: float = 1.5) -> None:
        """Handle outliers in numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_mask = z_scores > z_threshold
                if outlier_mask.any():
                    # Cap outliers at threshold * std from mean
                    col_mean = self.df[col].mean()
                    col_std = self.df[col].std()
                    upper_bound = col_mean + (z_threshold * col_std)
                    lower_bound = col_mean - (z_threshold * col_std)
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                    self.cleaning_summary['steps_applied'].append(f'capped_outliers_in_{col}_using_zscore')
            
            elif method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (iqr_multiplier * IQR)
                upper_bound = Q3 + (iqr_multiplier * IQR)
                
                # Cap values at the bounds
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                self.cleaning_summary['steps_applied'].append(f'capped_outliers_in_{col}_using_iqr')
    
    def _clean_string_columns(self, strip_whitespace: bool = True, lowercase: bool = True) -> None:
        """Clean string columns by stripping whitespace and converting to lowercase."""
        for col in self.df.select_dtypes(include=['object']).columns:
            if strip_whitespace:
                self.df[col] = self.df[col].astype(str).str.strip()
            if lowercase:
                self.df[col] = self.df[col].str.lower()
        
        if strip_whitespace or lowercase:
            self.cleaning_summary['steps_applied'].append('cleaned_string_columns')
    
    def _parse_dates(self) -> None:
        """Attempt to parse date columns."""
        for col in self.df.select_dtypes(include=['object']).columns:
            # Skip if the column name suggests it's not a date
            if 'date' not in col.lower() and 'time' not in col.lower():
                continue
                
            try:
                # Try to infer datetime format
                self.df[col] = pd.to_datetime(self.df[col], infer_datetime_format=True, errors='ignore')
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self.cleaning_summary['steps_applied'].append(f'parsed_{col}_as_datetime')
            except (ValueError, TypeError):
                continue
    
    def _remove_constant_columns(self) -> None:
        """Remove columns that have a single unique value (constants)."""
        constant_columns = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_columns:
            self.df.drop(columns=constant_columns, inplace=True)
            self.cleaning_summary['steps_applied'].append(f'removed_constant_columns: {constant_columns}')
    
    def get_summary(self) -> Dict:
        """Return a summary of the cleaning operations performed."""
        if self.original_df is None or self.df is None:
            return {}
            
        return {
            'original_shape': self.cleaning_summary.get('original_shape'),
            'cleaned_shape': self.cleaning_summary.get('cleaned_shape'),
            'steps_applied': self.cleaning_summary.get('steps_applied', []),
            'warnings': self.cleaning_summary.get('warnings', []),
            'errors': self.cleaning_summary.get('errors', [])
        }

    def standardize_column_names(self):
        """Standardize column names by removing special chars and converting to snake_case."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        new_columns = {}
        for col in self.df.columns:
            # Convert to string in case column names are not strings
            col_str = str(col)
            
            # Replace special characters and spaces with underscore
            new_col = re.sub(r'[^\w\s]', ' ', col_str)
            
            # Convert to snake_case
            new_col = new_col.strip().lower().replace(' ', '_')
            new_col = re.sub('_+', '_', new_col)  # Replace multiple underscores with one
            
            # Ensure column name is valid Python identifier
            if not new_col or new_col[0].isdigit():
                new_col = f'col_{new_col}'
                
            new_columns[col] = new_col
        
        if new_columns:
            self.df.rename(columns=new_columns, inplace=True)
            self.cleaning_summary['steps_applied'].append('standardized_column_names')
            self.cleaning_summary['columns_modified'] = new_columns

    def detect_and_convert_datatypes(self):
        """Detect and convert data types automatically."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        for col in self.df.columns:
            # Try to convert to numeric
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                if not self.df[col].isna().all():
                    self.cleaning_summary["steps_applied"].append(f"converted_{col}_to_numeric")
            except:
                pass
            
            # Try to convert to datetime
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                self.cleaning_summary["steps_applied"].append(f"converted_{col}_to_datetime")
            except:
                pass

    def handle_missing_values(self, custom_na_values: Optional[List[str]] = None):
        """Handle missing values using appropriate methods for each column type."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        if custom_na_values:
            self.df.replace(custom_na_values, np.nan, inplace=True)

        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue

            if missing_count == len(self.df):
                # Drop columns with all missing values
                self.df.drop(columns=[col], inplace=True)
                self.cleaning_summary["steps_applied"].append(f"dropped_empty_column_{col}")
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Fill numeric columns with median
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif pd.api.types.is_datetime64_dtype(self.df[col]):
                # Forward fill datetime values
                self.df[col].fillna(method='ffill', inplace=True)
            else:
                # Fill categorical/string columns with mode
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            self.cleaning_summary["missing_values_filled"] += missing_count
            self.cleaning_summary["steps_applied"].append(f"filled_missing_values_{col}")

    def handle_outliers(self, threshold: float = 3.0):
        """Handle outliers in numeric columns using Z-score method."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers_mask = z_scores > threshold
            if outliers_mask.any():
                # Replace outliers with column median
                median_value = self.df[col].median()
                self.df.loc[outliers_mask, col] = median_value
                self.cleaning_summary["steps_applied"].append(f"handled_outliers_{col}")

    def remove_constant_columns(self):
        """Remove columns with constant values."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        constant_columns = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_columns:
            self.df.drop(columns=constant_columns, inplace=True)
            self.cleaning_summary["steps_applied"].append("removed_constant_columns")

    def strip_whitespace(self):
        """Strip whitespace from string columns."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].str.strip()
        self.cleaning_summary["steps_applied"].append("stripped_whitespace")

    def drop_duplicates(self):
        """Remove duplicate rows."""
        if self.df is None:
            raise ValueError("DataFrame not set. Call set_dataframe() first.")
        
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        rows_removed = initial_rows - len(self.df)
        if rows_removed > 0:
            self.cleaning_summary["rows_removed"] += rows_removed
            self.cleaning_summary["steps_applied"].append("removed_duplicates")

    def get_cleaning_summary(self) -> Dict:
        """Get summary of all cleaning operations performed."""
        return self.cleaning_summary