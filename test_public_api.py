#!/usr/bin/env python3
"""
Simple test script for the public Data Cleaner API
"""

import requests
import json
import io
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Public: {data.get('public', False)}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_file_upload():
    """Test file upload functionality."""
    print("\nTesting file upload...")
    
    # Create sample CSV data
    csv_data = """name,age,city,salary
John,25,New York,50000
Jane,30,Los Angeles,60000
Bob,35,Chicago,55000
Alice,28,Houston,52000
Charlie,32,Phoenix,58000"""
    
    # Prepare file for upload
    files = {
        'file': ('test_data.csv', io.StringIO(csv_data), 'text/csv')
    }
    
    try:
        response = requests.post(f"{BASE_URL}/upload", files=files)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ File upload successful")
            print(f"   File ID: {data['file_id']}")
            print(f"   Columns: {data['columns']}")
            print(f"   Shape: {data['shape']}")
            return data['file_id']
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def test_data_cleaning(file_id):
    """Test data cleaning functionality."""
    if not file_id:
        print("\n❌ Skipping data cleaning test - no file ID")
        return False
    
    print(f"\nTesting data cleaning for file: {file_id}")
    
    cleaning_options = {
        "file_id": file_id,
        "fill_missing": True,
        "drop_duplicates": True,
        "standardize_columns": True,
        "fix_datatypes": True,
        "handle_outliers": True,
        "strip_whitespace": True,
        "fix_dates": True,
        "remove_constant_columns": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/clean",
            json=cleaning_options,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data cleaning successful")
            print(f"   Original shape: {data['original_shape']}")
            print(f"   Cleaned shape: {data['cleaned_shape']}")
            print(f"   Cleaning steps: {len(data['cleaning_summary']['steps_applied'])}")
            return True
        else:
            print(f"❌ Data cleaning failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Data cleaning error: {e}")
        return False

def test_visualization(file_id):
    """Test visualization functionality."""
    if not file_id:
        print("\n❌ Skipping visualization test - no file ID")
        return False
    
    print(f"\nTesting visualization for file: {file_id}")
    
    viz_options = {
        "file_id": file_id,
        "plot_type": "bar",
        "x_column": "city",
        "y_column": "salary",
        "title": "Salary by City"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/visualize",
            json=viz_options,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Visualization successful")
            print(f"   Plot URL: {data['plot_url']}")
            return True
        else:
            print(f"❌ Visualization failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_file_info(file_id):
    """Test file info endpoint."""
    if not file_id:
        print("\n❌ Skipping file info test - no file ID")
        return False
    
    print(f"\nTesting file info for file: {file_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/files/{file_id}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ File info retrieved")
            print(f"   Filename: {data['filename']}")
            print(f"   Size: {data['size']} bytes")
            print(f"   Processed: {data['is_processed']}")
            return True
        else:
            print(f"❌ File info failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ File info error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Public Data Cleaner API")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("\n❌ API is not running or not accessible")
        return
    
    # Test file upload
    file_id = test_file_upload()
    
    # Test other functionality
    test_data_cleaning(file_id)
    test_visualization(file_id)
    test_file_info(file_id)
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print(f"📊 API Documentation: {BASE_URL}/docs")
    print(f"🔍 Health Check: {BASE_URL}/health")

if __name__ == "__main__":
    main()
