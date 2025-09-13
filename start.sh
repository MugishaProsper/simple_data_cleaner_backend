#!/bin/bash

# Production startup script for Data Cleaner API

set -e

echo "Starting Data Cleaner API..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Using default configuration."
    echo "Please copy env.example to .env and configure your settings."
fi

# Create necessary directories
mkdir -p logs uploads static/plots

# Set proper permissions
chmod 755 logs uploads static/plots

# Initialize database if needed
echo "Initializing database..."
python -c "from database import init_db; init_db()"

# Run database migrations if using Alembic
if [ -f "alembic.ini" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Start the application
echo "Starting application server..."

if [ "$ENVIRONMENT" = "production" ]; then
    # Production mode with multiple workers
    exec uvicorn main_production:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --access-log \
        --log-level info
else
    # Development mode
    exec uvicorn main_production:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level debug
fi
