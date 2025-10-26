#!/bin/bash

# Setup script for gold news sentiment analysis system

echo "🚀 Setting up Gold News Sentiment Analysis System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models data

# Copy environment file
if [ ! -f .env ]; then
    echo "📄 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
else
    echo "✅ .env file already exists"
fi

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."

# Check PostgreSQL
if docker-compose ps postgres | grep -q "Up"; then
    echo "✅ PostgreSQL is running"
else
    echo "❌ PostgreSQL failed to start"
    exit 1
fi

# Check Redis
if docker-compose ps redis | grep -q "Up"; then
    echo "✅ Redis is running"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# Check API
if docker-compose ps api | grep -q "Up"; then
    echo "✅ API service is running"
else
    echo "❌ API service failed to start"
    exit 1
fi

# Check Dashboard
if docker-compose ps dashboard | grep -q "Up"; then
    echo "✅ Dashboard service is running"
else
    echo "❌ Dashboard service failed to start"
    exit 1
fi

# Run database migrations
echo "🗄️  Running database migrations..."
docker-compose exec api alembic upgrade head

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Available services:"
echo "   🌐 API Documentation: http://localhost:8000/docs"
echo "   📊 Dashboard: http://localhost:8501"
echo "   🔧 Celery Monitor: http://localhost:5555"
echo "   🏥 Health Check: http://localhost:8000/api/v1/health"
echo ""
echo "📚 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Collect gold price data: curl -X POST 'http://localhost:8000/api/v1/gold-prices/collect'"
echo "   3. Collect news: curl -X POST 'http://localhost:8000/api/v1/news/collect'"
echo "   4. Update sentiment: curl -X POST 'http://localhost:8000/api/v1/sentiment/update'"
echo "   5. Train models: curl -X POST 'http://localhost:8000/api/v1/predictions/train'"
echo ""
echo "💡 To stop services: docker-compose down"
echo "💡 To view logs: docker-compose logs -f"
echo "💡 To restart: docker-compose restart"
