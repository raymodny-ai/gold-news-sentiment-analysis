-- Initial database setup for gold news sentiment analysis system

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS gold_news_db;

-- Connect to the database
\c gold_news_db;

-- Create user if it doesn't exist
CREATE USER IF NOT EXISTS gold_user WITH PASSWORD 'gold_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE gold_news_db TO gold_user;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set default schema
SET search_path TO public;

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_published_at ON news(published_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_category ON news(category);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_source ON news(source);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_news_id ON sentiment_analysis(news_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_analyzer ON sentiment_analysis(analyzer_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_weighted_sentiment_unique ON weighted_sentiment(date, category, time_horizon);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_date ON price_predictions(prediction_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_target ON price_predictions(target_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gold_prices_date ON gold_prices(date);

-- Create partitions for large tables (if needed in the future)
-- This is a basic setup, you can add partitioning as data grows

-- Set up logging
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    source VARCHAR(50),
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);

-- Grant permissions on logs table
GRANT ALL PRIVILEGES ON TABLE system_logs TO gold_user;
