"""
Celery tasks for news collection and processing.
"""
import logging
import asyncio
from celery import shared_task
from datetime import datetime, timedelta

from app.services.news_collector import NewsCollectorService
from app.services.sentiment_analyzer import SentimentAnalysisService
from app.services.data_manager import DataManager
from app.core.config import settings


logger = logging.getLogger(__name__)


def run_async_task(coro):
    """Helper function to run async tasks in sync Celery tasks."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


@shared_task(name='app.tasks.news_tasks.collect_news_periodically')
def collect_news_periodically(query: str = "gold", max_results: int = 100):
    """Periodic task to collect news from all sources."""
    logger.info(f"Starting periodic news collection: query={query}, max_results={max_results}")

    try:
        # Run async task in sync context
        result = run_async_task(_collect_news_async(query, max_results))

        return result

    except Exception as e:
        logger.error(f"Error in periodic news collection: {e}")
        raise


async def _collect_news_async(query: str, max_results: int):
    """Async implementation of news collection."""
    # Initialize services
    news_collector = NewsCollectorService()

    # Collect news from all sources
    articles = await news_collector.collect_all_news(
        query=query,
        max_results_per_source=max_results
    )

    logger.info(f"Collected {len(articles)} articles from all sources")

    # Store news articles
    data_manager = DataManager()
    stored_count = 0

    for article in articles:
        news_data = news_collector.convert_to_news_create(article)
        stored_news = data_manager.store_news(news_data)

        if stored_news:
            stored_count += 1

            # Analyze sentiment for the news (async)
            analyze_sentiment_for_news.delay(
                news_id=stored_news.id,
                title=article.title,
                content=article.content
            )

    logger.info(f"Stored {stored_count} news articles")

    # Update weighted sentiments (async)
    update_sentiment_indices.delay()

    return {
        "status": "success",
        "articles_collected": len(articles),
        "articles_stored": stored_count
    }


@shared_task(name='app.tasks.news_tasks.collect_news_once')
def collect_news_once(query: str = "gold", max_results: int = 100):
    """One-time task to collect news from all sources."""
    return collect_news_periodically(query, max_results)


@shared_task(name='app.tasks.news_tasks.analyze_sentiment_for_news')
def analyze_sentiment_for_news(news_id: int, title: str, content: str):
    """Analyze sentiment for a specific news article."""
    logger.info(f"Analyzing sentiment for news {news_id}")

    try:
        # Run async task in sync context
        result = run_async_task(_analyze_sentiment_async(news_id, title, content))

        return result

    except Exception as e:
        logger.error(f"Error analyzing sentiment for news {news_id}: {e}")
        raise


async def _analyze_sentiment_async(news_id: int, title: str, content: str):
    """Async implementation of sentiment analysis."""
    # Initialize services
    sentiment_analyzer = SentimentAnalysisService()
    data_manager = DataManager()

    # Analyze sentiment
    text_to_analyze = f"{title} {content or ''}"
    sentiment_results = await sentiment_analyzer.analyze_text(text_to_analyze)

    # Store sentiment analysis results
    stored_analyses = 0
    for analyzer_type, result in sentiment_results.items():
        from app.models.schemas import SentimentAnalysisCreate

        sentiment_data = SentimentAnalysisCreate(
            news_id=news_id,
            analyzer_type=analyzer_type,
            bullish_score=result.bullish_score,
            bearish_score=result.bearish_score,
            attention_score=result.attention_score,
            confidence=result.confidence
        )

        stored_analysis = data_manager.store_sentiment_analysis(sentiment_data)
        if stored_analysis:
            stored_analyses += 1

    logger.info(f"Stored {stored_analyses} sentiment analyses for news {news_id}")

    return {
        "status": "success",
        "news_id": news_id,
        "analyses_stored": stored_analyses
    }
