@echo off
REM Public Data Cleaner API startup script (Windows)

echo Starting Public Data Cleaner API...

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found. Using default configuration.
    echo Please copy env.example to .env and configure your settings.
)

REM Create necessary directories
if not exist logs mkdir logs
if not exist uploads mkdir uploads
if not exist static\plots mkdir static\plots

REM Initialize database if needed
echo Initializing database...
python -c "from database import init_db; init_db()"

REM Start the application
echo Starting public application server...

if "%ENVIRONMENT%"=="production" (
    REM Production mode with multiple workers
    uvicorn main_public:app --host 0.0.0.0 --port 8000 --workers 4 --access-log --log-level info
) else (
    REM Development mode
    uvicorn main_public:app --host 0.0.0.0 --port 8000 --reload --log-level debug
)
