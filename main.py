from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime
import uuid
from pydantic import BaseModel

from services.data_cleaner import DataCleaner
from services.data_visualizer import DataVisualizer
from services.data_transformer import DataTransformer
from models.schemas import (
    CleaningOptions,
    VisualizationRequest,
    TransformationRequest
)

# Create necessary directories
os.makedirs("static/plots", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app = FastAPI(title="Data Cleaning and Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for demo purposes (use a database in production)
data_store: Dict[str, Dict[str, Any]] = {}

class CleanDataRequest(BaseModel):
    file_id: str
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a CSV file and return initial data summary"""
    try:
        # Generate unique ID for this session
        file_id = str(uuid.uuid4())
        file_path = f"uploads/{file_id}_{file.filename}"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the file with pandas
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
        
        # Generate initial summary
        summary = {
            "file_id": file_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head().to_dict(orient="records")
        }
        
        # Store the dataframe in memory (in production, use a proper database)
        data_store[file_id] = {
            "file_path": file_path,
            "df": df,
            "original_summary": summary
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

@app.post("/clean")
async def clean_data(request: CleanDataRequest) -> Dict[str, Any]:
    """Clean the uploaded data based on provided options"""
    if request.file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        data = data_store[request.file_id]
        df = data["df"].copy()
        
        # Initialize DataCleaner with the DataFrame
        cleaner = DataCleaner(df)
        
        # Create cleaning options
        options = CleaningOptions(
            fill_missing=request.fill_missing,
            drop_duplicates=request.drop_duplicates,
            standardize_columns=request.standardize_columns,
            fix_datatypes=request.fix_datatypes,
            handle_outliers=request.handle_outliers,
            strip_whitespace=request.strip_whitespace,
            fix_dates=request.fix_dates,
            remove_constant_columns=request.remove_constant_columns,
            custom_na_values=request.custom_na_values,
            date_columns=request.date_columns,
            outlier_threshold=request.outlier_threshold
        )
        
        # Apply cleaning operations
        cleaned_df = cleaner.clean_data(options)
        
        # Update stored data
        data["df"] = cleaned_df
        data["cleaned"] = True
        
        # Get cleaning summary
        cleaning_summary = cleaner.get_cleaning_summary()
        
        # Generate response summary
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
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def visualize_data(
    request: VisualizationRequest
) -> Dict[str, Any]:
    """Generate visualization based on user request"""
    if request.file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        data = data_store[request.file_id]
        df = data["df"]
        
        # Validate dataframe and columns
        if df.empty:
            raise HTTPException(status_code=400, detail="Cannot visualize empty dataframe")
        
        if request.x_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{request.x_column}' not found in dataframe")
        
        if request.y_column and request.y_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{request.y_column}' not found in dataframe")
        
        # Initialize visualizer with the DataFrame
        visualizer = DataVisualizer(df)
        
        # Create visualization
        plot_filename = visualizer.create_plot(
            plot_type=request.plot_type,
            x_column=request.x_column,
            y_column=request.y_column,
            title=request.title,
            file_id=request.file_id
        )
        
        return {
            "plot_url": plot_filename,  # Remove the leading slash
            "columns": df.columns.tolist()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transform")
async def transform_data(
    request: TransformationRequest
) -> Dict[str, Any]:
    """Apply data transformation"""
    if request.file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        data = data_store[request.file_id]
        df = data["df"].copy()
        
        # Initialize transformer with the DataFrame
        transformer = DataTransformer(df)
        
        # Apply transformation
        transformed_df = transformer.apply_transformation(
            columns=request.columns,
            transformation_type=request.transformation_type,
            params=request.params
        )
        
        # Update stored data
        data["df"] = transformed_df
        
        return {
            "columns": transformed_df.columns.tolist(),
            "preview": transformed_df.head().to_dict(orient="records"),
            "transformation_summary": transformer.get_transformation_summary()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_data(file_id: str):
    """Download the processed data as CSV"""
    if file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not ftound")
    
    try:
        data = data_store[file_id]
        output_path = f"uploads/processed_{file_id}.csv"
        data["df"].to_csv(output_path, index=False)
        
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename=f"processed_data.csv"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))