import pytest
from fastapi.testclient import TestClient


def test_register_user(client):
    """Test user registration."""
    response = client.post("/auth/register", json={
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "newpassword123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "User registered successfully"


def test_register_duplicate_user(client, test_user):
    """Test registration with duplicate username."""
    response = client.post("/auth/register", json={
        "username": "testuser",
        "email": "different@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]


def test_login_success(client, test_user):
    """Test successful login."""
    response = client.post("/auth/login", json={
        "username": "testuser",
        "password": "testpassword123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials(client):
    """Test login with invalid credentials."""
    response = client.post("/auth/login", json={
        "username": "nonexistent",
        "password": "wrongpassword"
    })
    
    assert response.status_code == 401
    assert "Invalid credentials" in response.json()["detail"]


def test_refresh_token(client, test_user):
    """Test token refresh."""
    # First login
    login_response = client.post("/auth/login", json={
        "username": "testuser",
        "password": "testpassword123"
    })
    
    refresh_token = login_response.json()["refresh_token"]
    
    # Refresh token
    response = client.post("/auth/refresh", json={
        "refresh_token": refresh_token
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_protected_endpoint_without_auth(client):
    """Test accessing protected endpoint without authentication."""
    response = client.get("/files")
    
    assert response.status_code == 401


def test_protected_endpoint_with_auth(client, auth_headers):
    """Test accessing protected endpoint with authentication."""
    response = client.get("/files", headers=auth_headers)
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)
