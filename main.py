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

# Initialize services
data_cleaner = DataCleaner()
data_visualizer = DataVisualizer()
data_transformer = DataTransformer()

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
        
        # Convert request to options dictionary
        options = {
            'fill_na': request.fill_missing,
            'drop_na': False,  # We're not dropping rows with NA
            'fill_strategy': 'mean',
            'na_threshold': 0.5,
            'drop_duplicates': request.drop_duplicates,
            'standardize_columns': request.standardize_columns,
            'infer_types': request.fix_datatypes,
            'handle_outliers': request.handle_outliers,
            'outlier_method': 'zscore',
            'z_threshold': request.outlier_threshold,
            'iqr_multiplier': 1.5,
            'strip_strings': request.strip_whitespace,
            'lowercase_columns': False,  # Not implemented in frontend
            'parse_dates': request.fix_dates,
            'remove_constant_columns': request.remove_constant_columns
        }
        
        # Set the dataframe in the cleaner
        data_cleaner.df = df
        data_cleaner.original_df = df.copy()
        
        # Apply cleaning operations
        cleaned_df = data_cleaner.clean_data(df, options)
        
        # Update stored data
        data["df"] = cleaned_df
        data["cleaned"] = True
        
        # Generate summary of cleaning results
        summary = {
            "file_id": request.file_id,
            "original_shape": data["original_summary"]["shape"],
            "cleaned_shape": cleaned_df.shape,
            "columns": cleaned_df.columns.tolist(),
            "missing_values_before": data["original_summary"]["missing_values"],
            "missing_values_after": cleaned_df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            "preview": cleaned_df.head().to_dict(orient="records"),
            "describe": cleaned_df.describe().to_dict()
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def visualize_data(
    file_id: str,
    plot_type: str,
    x_axis: str,
    y_axis: Optional[str] = None,
    hue: Optional[str] = None
) -> Dict[str, Any]:
    """Generate visualization based on user request"""
    if file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        data = data_store[file_id]
        df = data["df"]
        
        # Generate visualization
        plot_path = data_visualizer.create_visualization(
            df=df,
            plot_type=plot_type,
            x=x_axis,
            y=y_axis or "",
            hue=hue or "",
            file_id=file_id
        )
        
        # Return URL to the generated plot
        plot_url = f"/static/plots/{os.path.basename(plot_path)}"
        
        return {"plot_url": plot_url}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transform")
async def transform_data(
    file_id: str,
    operation: str,
    columns: List[str]
) -> Dict[str, Any]:
    """Apply data transformation"""
    if file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        data = data_store[file_id]
        df = data["df"].copy()
        
        # Create a new transformer instance for this operation
        transformer = DataTransformer()
        transformer.df = df
        
        # Apply transformation
        if operation == "normalize":
            transformer.normalize(columns)
        elif operation == "standardize":
            transformer.standardize(columns)
        elif operation == "log_transform":
            transformer.log_transform(columns)
        else:
            raise HTTPException(status_code=400, detail="Unsupported operation")
        
        # Update stored data
        data["df"] = transformer.df
        
        return {
            "operation": operation,
            "columns": columns,
            "preview": transformer.df.head().to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_data(file_id: str):
    """Download the processed data as CSV"""
    if file_id not in data_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        data = data_store[file_id]
        output_path = f"uploads/processed_{file_id}.csv"
        data["df"].to_csv(output_path, index=False)
        
        return FileResponse(
            output_path,
            filename=f"cleaned_data_{file_id}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
