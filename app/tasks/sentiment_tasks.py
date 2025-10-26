"""
Celery tasks for sentiment analysis and processing.
"""
import logging
import asyncio
from celery import shared_task
from datetime import datetime, timedelta

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


@shared_task(name='app.tasks.sentiment_tasks.update_sentiment_indices')
def update_sentiment_indices(start_date: str = None, end_date: str = None):
    """Update weighted sentiment indices."""
    logger.info(f"Updating sentiment indices: start_date={start_date}, end_date={end_date}")

    try:
        data_manager = DataManager()

        # Convert string dates to date objects
        start = datetime.fromisoformat(start_date).date() if start_date else None
        end = datetime.fromisoformat(end_date).date() if end_date else None

        # Update weighted sentiments
        updated_count = data_manager.update_weighted_sentiments(start_date=start, end_date=end)

        logger.info(f"Updated {updated_count} sentiment indices")

        return {
            "status": "success",
            "indices_updated": updated_count,
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None
        }

    except Exception as e:
        logger.error(f"Error updating sentiment indices: {e}")
        raise


@shared_task(name='app.tasks.sentiment_tasks.update_sentiment_indices_daily')
def update_sentiment_indices_daily():
    """Daily task to update sentiment indices."""
    # Update for the last 7 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    return update_sentiment_indices(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )


@shared_task(name='app.tasks.sentiment_tasks.analyze_news_batch')
def analyze_news_batch(news_ids: list):
    """Analyze sentiment for a batch of news articles."""
    logger.info(f"Analyzing sentiment for {len(news_ids)} news articles")

    try:
        sentiment_analyzer = SentimentAnalysisService()
        data_manager = DataManager()

        results = []
        for news_id in news_ids:
            # Get news article
            news = data_manager.get_news_by_id(news_id)
            if not news:
                logger.warning(f"News article {news_id} not found")
                continue

            # Analyze sentiment
            result = run_async_task(
                _analyze_single_news_async(news_id, news.title, news.content)
            )
            results.append(result)

        logger.info(f"Completed sentiment analysis for {len(results)} articles")

        return {
            "status": "success",
            "articles_analyzed": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        raise


async def _analyze_single_news_async(news_id: int, title: str, content: str):
    """Analyze sentiment for a single news article."""
    sentiment_analyzer = SentimentAnalysisService()
    data_manager = DataManager()

    text_to_analyze = f"{title} {content or ''}"
    sentiment_results = await sentiment_analyzer.analyze_text(text_to_analyze)

    # Store results
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

    return {
        "news_id": news_id,
        "analyses_stored": stored_analyses
    }
