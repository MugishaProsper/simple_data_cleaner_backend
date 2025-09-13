import pytest
import io
from fastapi.testclient import TestClient


def test_upload_file(client, auth_headers, sample_csv):
    """Test file upload."""
    csv_file = io.BytesIO(sample_csv.encode())
    
    response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "test.csv"
    assert "columns" in data
    assert "shape" in data
    assert len(data["columns"]) == 4  # name, age, city, salary


def test_upload_invalid_file_type(client, auth_headers):
    """Test upload with invalid file type."""
    text_file = io.BytesIO(b"not a csv file")
    
    response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 500
    assert "Invalid file type" in response.json()["detail"]


def test_upload_file_too_large(client, auth_headers):
    """Test upload with file too large."""
    # Create a large file (simulate)
    large_content = "x" * (100 * 1024 * 1024 + 1)  # 100MB + 1 byte
    large_file = io.BytesIO(large_content.encode())
    
    response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("large.csv", large_file, "text/csv")}
    )
    
    assert response.status_code == 500
    assert "File too large" in response.json()["detail"]


def test_get_user_files(client, auth_headers):
    """Test getting user files."""
    response = client.get("/files", headers=auth_headers)
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_clean_data(client, auth_headers, sample_csv):
    """Test data cleaning."""
    # First upload a file
    csv_file = io.BytesIO(sample_csv.encode())
    upload_response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    file_id = upload_response.json()["file_id"]
    
    # Clean the data
    response = client.post(
        "/clean",
        headers=auth_headers,
        json={
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
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "cleaning_summary" in data
    assert "original_shape" in data
    assert "cleaned_shape" in data


def test_visualize_data(client, auth_headers, sample_csv):
    """Test data visualization."""
    # First upload a file
    csv_file = io.BytesIO(sample_csv.encode())
    upload_response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    file_id = upload_response.json()["file_id"]
    
    # Create visualization
    response = client.post(
        "/visualize",
        headers=auth_headers,
        json={
            "file_id": file_id,
            "plot_type": "bar",
            "x_column": "city",
            "y_column": "salary",
            "title": "Salary by City"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "plot_url" in data
    assert "columns" in data


def test_transform_data(client, auth_headers, sample_csv):
    """Test data transformation."""
    # First upload a file
    csv_file = io.BytesIO(sample_csv.encode())
    upload_response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    file_id = upload_response.json()["file_id"]
    
    # Transform the data
    response = client.post(
        "/transform",
        headers=auth_headers,
        json={
            "file_id": file_id,
            "columns": ["salary"],
            "transformation_type": "normalize"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "transformation_summary" in data
    assert "columns" in data


def test_download_data(client, auth_headers, sample_csv):
    """Test data download."""
    # First upload a file
    csv_file = io.BytesIO(sample_csv.encode())
    upload_response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    file_id = upload_response.json()["file_id"]
    
    # Download the data
    response = client.get(f"/download/{file_id}", headers=auth_headers)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"


def test_access_other_user_file(client, auth_headers, sample_csv):
    """Test accessing another user's file."""
    # Upload file with first user
    csv_file = io.BytesIO(sample_csv.encode())
    upload_response = client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    file_id = upload_response.json()["file_id"]
    
    # Create second user and try to access first user's file
    client.post("/auth/register", json={
        "username": "user2",
        "email": "user2@example.com",
        "password": "password123"
    })
    
    login_response = client.post("/auth/login", json={
        "username": "user2",
        "password": "password123"
    })
    
    user2_headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}
    
    # Try to download first user's file
    response = client.get(f"/download/{file_id}", headers=user2_headers)
    
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]
