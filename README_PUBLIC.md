# Public Data Cleaner API

A **public, no-authentication-required** FastAPI application for data cleaning, visualization, and transformation. Perfect for public demos, educational purposes, or simple data processing needs.

## 🚀 Quick Start

### Option 1: Direct Python (Recommended for testing)
```bash
# Install dependencies
pip install -r requirements.txt

# Start the public API
python main_public.py
```

### Option 2: Docker (Recommended for production)
```bash
# Start with Docker Compose
docker-compose -f docker-compose-public.yml up -d
```

### Option 3: Windows Batch
```bash
# Start with batch script
start_public.bat
```

## 📚 API Endpoints

All endpoints are **publicly accessible** - no authentication required!

### Core Endpoints
- `POST /upload` - Upload CSV/Excel files
- `POST /clean` - Clean uploaded data
- `POST /visualize` - Generate visualizations
- `POST /transform` - Apply data transformations
- `GET /download/{file_id}` - Download processed data
- `GET /files/{file_id}` - Get file information

### System Endpoints
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

## 🧪 Test the API

### Quick Test
```bash
# Run the test script
python test_public_api.py
```

### Manual Testing with curl

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Upload File
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_data.csv"
```

#### 3. Clean Data
```bash
curl -X POST "http://localhost:8000/clean" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "YOUR_FILE_ID",
    "fill_missing": true,
    "drop_duplicates": true,
    "standardize_columns": true
  }'
```

#### 4. Create Visualization
```bash
curl -X POST "http://localhost:8000/visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "YOUR_FILE_ID",
    "plot_type": "bar",
    "x_column": "category",
    "y_column": "value",
    "title": "My Chart"
  }'
```

## 📊 Supported Features

### Data Cleaning
- ✅ Missing value handling
- ✅ Duplicate removal
- ✅ Outlier detection and treatment
- ✅ Data type conversion
- ✅ Column standardization
- ✅ String cleaning
- ✅ Date parsing
- ✅ Constant column removal

### Visualizations
- ✅ Line plots
- ✅ Bar charts
- ✅ Scatter plots
- ✅ Histograms
- ✅ Correlation heatmaps

### Data Transformations
- ✅ Normalization (Min-Max)
- ✅ Standardization (Z-score)
- ✅ Logarithmic transformation
- ✅ Square root transformation
- ✅ Custom scaling

### File Support
- ✅ CSV files
- ✅ Excel files (.xlsx, .xls)
- ✅ File size validation (100MB max)
- ✅ Automatic cleanup

## 🔧 Configuration

### Environment Variables
```bash
# Copy the example file
cp env.example .env

# Key settings for public use
ENVIRONMENT=development
DEBUG=true
CORS_ORIGINS=http://localhost:5173,http://localhost:8080
MAX_FILE_SIZE=104857600  # 100MB
RATE_LIMIT_REQUESTS=100  # requests per hour
```

### Database
- **Development**: SQLite (automatic)
- **Production**: PostgreSQL (configure in .env)

## 🐳 Docker Deployment

### Quick Start
```bash
# Start all services
docker-compose -f docker-compose-public.yml up -d

# Check status
docker-compose -f docker-compose-public.yml ps

# View logs
docker-compose -f docker-compose-public.yml logs -f api
```

### Services Included
- **API**: FastAPI application (port 8000)
- **PostgreSQL**: Database (port 5432)
- **Redis**: Caching (port 6379)
- **Nginx**: Reverse proxy (port 80)

## 🔒 Security Features

Even though it's public, the API includes:
- ✅ Rate limiting (100 requests/hour per IP)
- ✅ File type validation
- ✅ File size limits
- ✅ Input sanitization
- ✅ CORS protection
- ✅ Security headers
- ✅ Request logging

## 📈 Performance

- **Async file processing** for better performance
- **Automatic cleanup** of old files
- **Connection pooling** for database
- **Static file serving** for generated plots
- **Background tasks** for maintenance

## 🎯 Use Cases

Perfect for:
- **Public demos** and showcases
- **Educational purposes** and tutorials
- **Quick data processing** without setup
- **API testing** and development
- **Prototyping** data workflows
- **Public data analysis** tools

## 🚨 Important Notes

### Public Access
- ⚠️ **No authentication** - anyone can access
- ⚠️ **No user isolation** - files are shared
- ⚠️ **Rate limited** to prevent abuse
- ⚠️ **Files auto-expire** after 72 hours

### Security Considerations
- Use behind a firewall for sensitive data
- Consider adding authentication for production
- Monitor usage and implement additional rate limiting
- Regular cleanup of uploaded files

## 🆘 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

#### 2. Database Errors
```bash
# Initialize database
python -c "from database import init_db; init_db()"
```

#### 3. Port Already in Use
```bash
# Change port in .env file
PORT=8001
```

#### 4. File Upload Fails
- Check file size (max 100MB)
- Ensure file is CSV or Excel format
- Verify file is not corrupted

### Getting Help
- Check API documentation at `/docs`
- Review logs in the `logs/` directory
- Run the test script to verify functionality

## 🎉 Ready to Use!

Your public data cleaner API is ready! Visit `http://localhost:8000/docs` to explore the interactive API documentation and start processing data immediately.

**No signup, no authentication, just upload and process!** 🚀
