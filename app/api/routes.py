"""
API routes for the gold news sentiment analysis system.
"""
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.schemas import (
    SentimentListResponse, PredictionResponse,
    ErrorResponse, NewsCategory, AnalyzerType, TimeHorizon, ModelType,
    NewsCreate, SentimentAnalysisCreate, WeightedSentimentCreate,
    PricePredictionCreate, GoldPriceCreate
)
from app.services.news_collector import NewsCollectorService
from app.services.sentiment_analyzer import SentimentAnalysisService
from app.services.data_manager import DataManager
from app.services.prediction_model import PredictionService
from app.core.config import settings


# Initialize services
news_collector = NewsCollectorService()
sentiment_analyzer = SentimentAnalysisService()
data_manager = DataManager()
prediction_service = PredictionService()

# Create router
router = APIRouter()


# News API endpoints
@router.get("/news")
async def get_news(
    category: Optional[str] = Query(None, description="News category filter"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of results per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """Get news articles with optional filtering and pagination."""
    try:
        news_items, total = data_manager.get_news(
            category=category,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        # Convert to simple response format
        news_data = []
        for item in news_items:
            news_dict = {
                'id': item.id,
                'title': item.title,
                'content': item.content,
                'category': item.category,
                'published_at': item.published_at.isoformat() if item.published_at else None,
                'source': item.source,
                'url': item.url,
                'language': item.language,
                'created_at': item.created_at.isoformat() if item.created_at else None,
                'updated_at': item.updated_at.isoformat() if item.updated_at else None
            }
            news_data.append(news_dict)

        return {
            "data": news_data,
            "total": total,
            "page": (offset // limit) + 1,
            "per_page": limit
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving news: {str(e)}")


@router.post("/news/collect")
async def collect_news(
    background_tasks: BackgroundTasks,
    query: str = Query("gold", description="Search query for news collection"),
    max_results: int = Query(100, ge=1, le=1000, description="Maximum results per source")
):
    """Trigger news collection from all sources."""
    try:
        # Add background task for news collection
        background_tasks.add_task(
            collect_and_process_news,
            query=query,
            max_results=max_results
        )

        return {
            "message": "News collection started in background",
            "query": query,
            "max_results": max_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting news collection: {str(e)}")


async def collect_and_process_news(query: str, max_results: int):
    """Background task to collect and process news."""
    try:
        # Collect news from all sources
        articles = await news_collector.collect_all_news(
            query=query,
            max_results_per_source=max_results
        )

        # Store news articles
        stored_count = 0
        for article in articles:
            news_data = news_collector.convert_to_news_create(article)
            stored_news = data_manager.store_news(news_data)
            if stored_news:
                stored_count += 1

                # Analyze sentiment for the news
                text_to_analyze = f"{article.title} {article.content or ''}"
                sentiment_results = await sentiment_analyzer.analyze_text(text_to_analyze)

                # Store sentiment analysis results
                for analyzer_type, result in sentiment_results.items():
                    sentiment_data = SentimentAnalysisCreate(
                        news_id=stored_news.id,
                        analyzer_type=analyzer_type,
                        bullish_score=result.bullish_score,
                        bearish_score=result.bearish_score,
                        attention_score=result.attention_score,
                        confidence=result.confidence
                    )
                    data_manager.store_sentiment_analysis(sentiment_data)

        # Update weighted sentiments
        data_manager.update_weighted_sentiments()

    except Exception as e:
        print(f"Error in background news collection: {e}")


# Sentiment API endpoints
@router.get("/sentiment", response_model=SentimentListResponse)
async def get_sentiment_data(
    category: Optional[NewsCategory] = Query(None, description="News category filter"),
    time_horizon: Optional[TimeHorizon] = Query(None, description="Time horizon filter"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results")
):
    """Get sentiment analysis data."""
    try:
        sentiment_data = data_manager.get_weighted_sentiment(
            category=category,
            time_horizon=time_horizon,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        # Convert to response format
        data = []
        for item in sentiment_data:
            data.append({
                "date": item.date.isoformat(),
                "category": item.category,
                "time_horizon": item.time_horizon,
                "weighted_score": float(item.weighted_score),
                "created_at": item.created_at.isoformat()
            })

        return SentimentListResponse(data=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sentiment data: {str(e)}")


@router.post("/sentiment/analyze")
async def analyze_sentiment(
    background_tasks: BackgroundTasks,
    news_id: int = Query(..., description="News ID to analyze")
):
    """Analyze sentiment for a specific news article."""
    try:
        # Get news article
        news = data_manager.get_news_by_id(news_id)
        if not news:
            raise HTTPException(status_code=404, detail="News article not found")

        # Add background task for sentiment analysis
        background_tasks.add_task(
            analyze_news_sentiment,
            news_id=news_id,
            title=news.title,
            content=news.content
        )

        return {
            "message": "Sentiment analysis started in background",
            "news_id": news_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting sentiment analysis: {str(e)}")


async def analyze_news_sentiment(news_id: int, title: str, content: Optional[str]):
    """Background task to analyze news sentiment."""
    try:
        text_to_analyze = f"{title} {content or ''}"
        sentiment_results = await sentiment_analyzer.analyze_text(text_to_analyze)

        # Store sentiment analysis results
        for analyzer_type, result in sentiment_results.items():
            sentiment_data = SentimentAnalysisCreate(
                news_id=news_id,
                analyzer_type=analyzer_type,
                bullish_score=result.bullish_score,
                bearish_score=result.bearish_score,
                attention_score=result.attention_score,
                confidence=result.confidence
            )
            data_manager.store_sentiment_analysis(sentiment_data)

    except Exception as e:
        print(f"Error in background sentiment analysis: {e}")


@router.post("/sentiment/update")
async def update_weighted_sentiment(
    background_tasks: BackgroundTasks,
    start_date: Optional[date] = Query(None, description="Start date for update"),
    end_date: Optional[date] = Query(None, description="End date for update")
):
    """Update weighted sentiment indices."""
    try:
        background_tasks.add_task(
            update_sentiment_indices,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "message": "Weighted sentiment update started in background",
            "start_date": start_date,
            "end_date": end_date
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting sentiment update: {str(e)}")


async def update_sentiment_indices(start_date: Optional[date], end_date: Optional[date]):
    """Background task to update sentiment indices."""
    try:
        updated_count = data_manager.update_weighted_sentiments(
            start_date=start_date,
            end_date=end_date
        )
        print(f"Updated {updated_count} weighted sentiment entries")

    except Exception as e:
        print(f"Error updating sentiment indices: {e}")


# Prediction API endpoints
@router.get("/predictions", response_model=PredictionResponse)
async def get_predictions(
    target_date: Optional[date] = Query(None, description="Target prediction date"),
    model_type: Optional[ModelType] = Query(None, description="Model type filter")
):
    """Get price predictions."""
    try:
        if target_date is None:
            target_date = date.today() + timedelta(days=1)

        predictions = data_manager.get_price_predictions(
            target_date=target_date,
            model_type=model_type
        )

        # Convert to response format
        data = {}
        for pred in predictions:
            data[f"{pred.model_type}_prediction"] = {
                "prediction_date": pred.prediction_date.isoformat(),
                "target_date": pred.target_date.isoformat(),
                "predicted_price": float(pred.predicted_price),
                "confidence_interval_lower": float(pred.confidence_interval_lower) if pred.confidence_interval_lower else None,
                "confidence_interval_upper": float(pred.confidence_interval_upper) if pred.confidence_interval_upper else None,
                "feature_importance": pred.feature_importance,
                "created_at": pred.created_at.isoformat()
            }

        return PredictionResponse(data=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving predictions: {str(e)}")


@router.post("/predictions/train")
async def train_models(
    background_tasks: BackgroundTasks,
    start_date: Optional[date] = Query(None, description="Training data start date"),
    end_date: Optional[date] = Query(None, description="Training data end date"),
    model_types: List[ModelType] = Query(default=None, description="Model types to train")
):
    """Train prediction models."""
    try:
        # Convert model_types to list if None
        if model_types is None:
            from app.models.schemas import ModelType
            model_types = list(ModelType)

        background_tasks.add_task(
            train_prediction_models,
            start_date=start_date,
            end_date=end_date,
            model_types=model_types
        )

        return {
            "message": "Model training started in background",
            "model_types": [mt.value for mt in model_types],
            "start_date": start_date,
            "end_date": end_date
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting model training: {str(e)}")


async def train_prediction_models(start_date: Optional[date], end_date: Optional[date], model_types: List[ModelType]):
    """Background task to train prediction models."""
    try:
        results = await prediction_service.train_models(
            start_date=start_date,
            end_date=end_date,
            models=model_types
        )

        print(f"Model training completed: {results}")

    except Exception as e:
        print(f"Error training models: {e}")


@router.post("/predictions/predict")
async def predict_prices(
    background_tasks: BackgroundTasks,
    target_date: Optional[date] = Query(None, description="Target prediction date"),
        model_types: List[ModelType] = Query(default=None, description="Model types for prediction")
):
    """Make price predictions."""
    try:
        if target_date is None:
            target_date = date.today() + timedelta(days=1)

        # Convert model_types to list if None
        if model_types is None:
            from app.models.schemas import ModelType
            model_types = list(ModelType)

        background_tasks.add_task(
            make_price_predictions,
            target_date=target_date,
            model_types=model_types
        )

        return {
            "message": "Price prediction started in background",
            "target_date": target_date.isoformat(),
            "model_types": [mt.value for mt in model_types]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting price prediction: {str(e)}")


async def make_price_predictions(target_date: date, model_types: List[ModelType]):
    """Background task to make price predictions."""
    try:
        predictions = await prediction_service.predict_price(
            target_date=target_date,
            model_types=model_types
        )

        print(f"Price predictions completed: {len(predictions)} models")

    except Exception as e:
        print(f"Error making price predictions: {e}")


# Gold Price API endpoints
@router.get("/gold-prices")
async def get_gold_prices(
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results")
):
    """Get gold price historical data."""
    try:
        prices = data_manager.get_gold_prices(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        data = []
        for price in prices:
            data.append({
                "date": price.date.isoformat(),
                "open_price": float(price.open_price) if price.open_price else None,
                "high_price": float(price.high_price) if price.high_price else None,
                "low_price": float(price.low_price) if price.low_price else None,
                "close_price": float(price.close_price) if price.close_price else None,
                "volume": price.volume
            })

        return {"data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving gold prices: {str(e)}")


@router.post("/gold-prices/collect")
async def collect_gold_prices(
    background_tasks: BackgroundTasks,
    symbol: str = Query("GC=F", description="Gold symbol"),
    start_date: Optional[date] = Query(None, description="Start date for collection")
):
    """Collect gold price data."""
    try:
        if start_date is None:
            start_date = date.today() - timedelta(days=365)  # 1 year of data

        background_tasks.add_task(
            collect_price_data,
            symbol=symbol,
            start_date=start_date
        )

        return {
            "message": "Gold price collection started in background",
            "symbol": symbol,
            "start_date": start_date.isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting price collection: {str(e)}")


async def collect_price_data(symbol: str, start_date: date):
    """Background task to collect gold price data."""
    try:
        import yfinance as yf
        import pandas as pd

        # Get data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=date.today())

        stored_count = 0
        for index, row in df.iterrows():
            price_data = GoldPriceCreate(
                date=index.date(),
                open_price=float(row['Open']) if not pd.isna(row['Open']) else None,
                high_price=float(row['High']) if not pd.isna(row['High']) else None,
                low_price=float(row['Low']) if not pd.isna(row['Low']) else None,
                close_price=float(row['Close']) if not pd.isna(row['Close']) else None,
                volume=int(row['Volume']) if not pd.isna(row['Volume']) else None
            )

            stored_price = data_manager.store_gold_price(price_data)
            if stored_price:
                stored_count += 1

        print(f"Collected {stored_count} gold price records")

    except Exception as e:
        print(f"Error collecting gold price data: {e}")


# Analytics and Summary endpoints
@router.get("/analytics/sentiment-summary")
async def get_sentiment_summary(
    start_date: Optional[date] = Query(None, description="Start date for summary"),
    end_date: Optional[date] = Query(None, description="End date for summary")
):
    """Get sentiment analysis summary."""
    try:
        summary = data_manager.get_sentiment_summary(start_date, end_date)
        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sentiment summary: {str(e)}")


@router.get("/analytics/news-sources")
async def get_news_sources_summary():
    """Get news sources summary."""
    try:
        summary = data_manager.get_news_sources_summary()
        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving news sources summary: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_data(
    background_tasks: BackgroundTasks,
    days_to_keep: int = Query(180, ge=30, le=3650, description="Days of data to keep")
):
    """Clean up old data."""
    try:
        background_tasks.add_task(
            perform_cleanup,
            days_to_keep=days_to_keep
        )

        return {
            "message": "Data cleanup started in background",
            "days_to_keep": days_to_keep
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting cleanup: {str(e)}")


async def perform_cleanup(days_to_keep: int):
    """Background task to clean up old data."""
    try:
        result = data_manager.cleanup_old_data(days_to_keep)
        print(f"Cleanup completed: {result}")

    except Exception as e:
        print(f"Error during cleanup: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
