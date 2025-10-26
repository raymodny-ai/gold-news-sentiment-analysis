"""
Data management service for database operations.
"""
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, text
from sqlalchemy.exc import IntegrityError
import pandas as pd

from app.models.database import SessionLocal
from app.models.models import News, SentimentAnalysis, WeightedSentiment, PricePrediction, GoldPrice
from app.models.schemas import (
    NewsCreate, SentimentAnalysisCreate, WeightedSentimentCreate,
    PricePredictionCreate, GoldPriceCreate, NewsCategory, AnalyzerType,
    TimeHorizon, ModelType
)
from app.core.config import settings


class DataManager:
    """Data management service for database operations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # News Management
    def store_news(self, news_data: NewsCreate) -> Optional[News]:
        """Store news article in database."""
        try:
            db = SessionLocal()
            db_news = News(**news_data.dict())
            db.add(db_news)
            db.commit()
            db.refresh(db_news)
            self.logger.info(f"Stored news: {db_news.id} - {db_news.title[:50]}...")
            return db_news
        except IntegrityError as e:
            db.rollback()
            self.logger.warning(f"Duplicate news entry: {e}")
            return None
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing news: {e}")
            return None
        finally:
            db.close()

    def get_news(
        self,
        category: Optional[NewsCategory] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[News], int]:
        """Get news articles with optional filtering."""
        try:
            db = SessionLocal()

            query = db.query(News)

            if category:
                query = query.filter(News.category == category)
            if start_date:
                query = query.filter(News.published_at >= start_date)
            if end_date:
                query = query.filter(News.published_at <= end_date)

            total = query.count()
            news_items = query.order_by(desc(News.published_at)).offset(offset).limit(limit).all()

            return news_items, total

        except Exception as e:
            self.logger.error(f"Error getting news: {e}")
            return [], 0
        finally:
            db.close()

    def get_news_by_id(self, news_id: int) -> Optional[News]:
        """Get news article by ID."""
        try:
            db = SessionLocal()
            return db.query(News).filter(News.id == news_id).first()
        except Exception as e:
            self.logger.error(f"Error getting news by ID: {e}")
            return None
        finally:
            db.close()

    # Sentiment Analysis Management
    def store_sentiment_analysis(self, sentiment_data: SentimentAnalysisCreate) -> Optional[SentimentAnalysis]:
        """Store sentiment analysis results."""
        try:
            db = SessionLocal()
            db_sentiment = SentimentAnalysis(**sentiment_data.dict())
            db.add(db_sentiment)
            db.commit()
            db.refresh(db_sentiment)
            self.logger.info(f"Stored sentiment analysis: {db_sentiment.id} for news {db_sentiment.news_id}")
            return db_sentiment
        except IntegrityError as e:
            db.rollback()
            self.logger.warning(f"Duplicate sentiment analysis: {e}")
            return None
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing sentiment analysis: {e}")
            return None
        finally:
            db.close()

    def get_sentiment_analysis(
        self,
        news_id: Optional[int] = None,
        analyzer_type: Optional[AnalyzerType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SentimentAnalysis]:
        """Get sentiment analysis results."""
        try:
            db = SessionLocal()
            query = db.query(SentimentAnalysis)

            if news_id:
                query = query.filter(SentimentAnalysis.news_id == news_id)
            if analyzer_type:
                query = query.filter(SentimentAnalysis.analyzer_type == analyzer_type)

            return query.order_by(desc(SentimentAnalysis.created_at)).offset(offset).limit(limit).all()

        except Exception as e:
            self.logger.error(f"Error getting sentiment analysis: {e}")
            return []
        finally:
            db.close()

    # Weighted Sentiment Management
    def store_weighted_sentiment(self, sentiment_data: WeightedSentimentCreate) -> Optional[WeightedSentiment]:
        """Store weighted sentiment index."""
        try:
            db = SessionLocal()
            db_sentiment = WeightedSentiment(**sentiment_data.dict())
            db.add(db_sentiment)
            db.commit()
            db.refresh(db_sentiment)
            self.logger.info(f"Stored weighted sentiment: {db_sentiment.date} - {db_sentiment.category} - {db_sentiment.weighted_score}")
            return db_sentiment
        except IntegrityError as e:
            db.rollback()
            self.logger.warning(f"Duplicate weighted sentiment entry: {e}")
            return None
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing weighted sentiment: {e}")
            return None
        finally:
            db.close()

    def get_weighted_sentiment(
        self,
        category: Optional[NewsCategory] = None,
        time_horizon: Optional[TimeHorizon] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[WeightedSentiment]:
        """Get weighted sentiment data."""
        try:
            db = SessionLocal()
            query = db.query(WeightedSentiment)

            if category:
                query = query.filter(WeightedSentiment.category == category)
            if time_horizon:
                query = query.filter(WeightedSentiment.time_horizon == time_horizon)
            if start_date:
                query = query.filter(WeightedSentiment.date >= start_date)
            if end_date:
                query = query.filter(WeightedSentiment.date <= end_date)

            return query.order_by(desc(WeightedSentiment.date)).offset(offset).limit(limit).all()

        except Exception as e:
            self.logger.error(f"Error getting weighted sentiment: {e}")
            return []
        finally:
            db.close()

    def calculate_weighted_sentiment(
        self,
        target_date: date,
        category: NewsCategory,
        time_horizon: TimeHorizon
    ) -> Optional[float]:
        """Calculate weighted sentiment for a specific date and category."""
        try:
            db = SessionLocal()

            # Calculate date range based on time horizon
            if time_horizon == TimeHorizon.SHORT:
                days = settings.short_term_days
            elif time_horizon == TimeHorizon.MEDIUM:
                days = settings.medium_term_days
            else:  # LONG
                days = settings.long_term_days

            start_date = target_date - timedelta(days=days)
            end_date = target_date

            # Get all sentiment analyses for the period
            query = db.query(SentimentAnalysis).join(News).filter(
                and_(
                    News.published_at >= start_date,
                    News.published_at <= end_date,
                    News.category == category
                )
            )

            analyses = query.all()
            if not analyses:
                return None

            # Calculate time-decayed weighted sentiment
            total_weight = 0.0
            weighted_sum = 0.0

            for analysis in analyses:
                # Get news article for time decay calculation
                news = db.query(News).filter(News.id == analysis.news_id).first()
                if not news:
                    continue

                # Apply time decay based on news age
                age_days = (end_date - news.published_at.date()).days
                time_weight = settings.decay_factor ** age_days

                # Use combined sentiment score (bullish - bearish)
                sentiment_score = analysis.bullish_score - analysis.bearish_score if analysis.bullish_score and analysis.bearish_score else 0

                # Apply weights
                weighted_sum += sentiment_score * time_weight * analysis.confidence
                total_weight += time_weight * analysis.confidence

            if total_weight == 0:
                return 0.0

            weighted_sentiment = weighted_sum / total_weight

            # Normalize to -1 to 1 range
            weighted_sentiment = max(-1.0, min(1.0, weighted_sentiment))

            return weighted_sentiment

        except Exception as e:
            self.logger.error(f"Error calculating weighted sentiment: {e}")
            return None
        finally:
            db.close()

    def update_weighted_sentiments(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> int:
        """Update weighted sentiment indices for all categories and time horizons."""
        try:
            db = SessionLocal()

            if start_date is None:
                start_date = date.today() - timedelta(days=30)
            if end_date is None:
                end_date = date.today()

            updated_count = 0
            current_date = start_date

            while current_date <= end_date:
                for category in settings.news_categories:
                    for time_horizon in settings.time_horizons:
                        # Calculate weighted sentiment
                        weighted_score = self.calculate_weighted_sentiment(
                            current_date,
                            NewsCategory(category),
                            TimeHorizon(time_horizon)
                        )

                        if weighted_score is not None:
                            # Check if entry already exists
                            existing = db.query(WeightedSentiment).filter(
                                and_(
                                    WeightedSentiment.date == current_date,
                                    WeightedSentiment.category == category,
                                    WeightedSentiment.time_horizon == time_horizon
                                )
                            ).first()

                            if existing:
                                # Update existing
                                existing.weighted_score = weighted_score
                                db.commit()
                            else:
                                # Create new
                                sentiment_data = WeightedSentimentCreate(
                                    date=current_date,
                                    category=NewsCategory(category),
                                    weighted_score=weighted_score,
                                    time_horizon=TimeHorizon(time_horizon)
                                )
                                self.store_weighted_sentiment(sentiment_data)

                            updated_count += 1

                current_date += timedelta(days=1)

            self.logger.info(f"Updated {updated_count} weighted sentiment entries")
            return updated_count

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error updating weighted sentiments: {e}")
            return 0
        finally:
            db.close()

    # Price Prediction Management
    def store_price_prediction(self, prediction_data: PricePredictionCreate) -> Optional[PricePrediction]:
        """Store price prediction results."""
        try:
            db = SessionLocal()
            db_prediction = PricePrediction(**prediction_data.dict())
            db.add(db_prediction)
            db.commit()
            db.refresh(db_prediction)
            self.logger.info(f"Stored price prediction: {db_prediction.id} for {db_prediction.target_date}")
            return db_prediction
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing price prediction: {e}")
            return None
        finally:
            db.close()

    def get_price_predictions(
        self,
        target_date: Optional[date] = None,
        model_type: Optional[ModelType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[PricePrediction]:
        """Get price predictions."""
        try:
            db = SessionLocal()
            query = db.query(PricePrediction)

            if target_date:
                query = query.filter(PricePrediction.target_date == target_date)
            if model_type:
                query = query.filter(PricePrediction.model_type == model_type)

            return query.order_by(desc(PricePrediction.created_at)).offset(offset).limit(limit).all()

        except Exception as e:
            self.logger.error(f"Error getting price predictions: {e}")
            return []
        finally:
            db.close()

    # Gold Price Management
    def store_gold_price(self, price_data: GoldPriceCreate) -> Optional[GoldPrice]:
        """Store gold price data."""
        try:
            db = SessionLocal()
            db_price = GoldPrice(**price_data.dict())
            db.add(db_price)
            db.commit()
            db.refresh(db_price)
            self.logger.info(f"Stored gold price for {db_price.date}")
            return db_price
        except IntegrityError as e:
            db.rollback()
            self.logger.warning(f"Duplicate gold price entry: {e}")
            return None
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing gold price: {e}")
            return None
        finally:
            db.close()

    def get_gold_prices(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[GoldPrice]:
        """Get gold price data."""
        try:
            db = SessionLocal()
            query = db.query(GoldPrice)

            if start_date:
                query = query.filter(GoldPrice.date >= start_date)
            if end_date:
                query = query.filter(GoldPrice.date <= end_date)

            return query.order_by(desc(GoldPrice.date)).offset(offset).limit(limit).all()

        except Exception as e:
            self.logger.error(f"Error getting gold prices: {e}")
            return []
        finally:
            db.close()

    def get_latest_gold_price(self) -> Optional[GoldPrice]:
        """Get the latest gold price."""
        try:
            db = SessionLocal()
            return db.query(GoldPrice).order_by(desc(GoldPrice.date)).first()
        except Exception as e:
            self.logger.error(f"Error getting latest gold price: {e}")
            return None
        finally:
            db.close()

    # Cleanup Operations
    def cleanup_old_data(self, days_to_keep: int = 180) -> Dict[str, int]:
        """Clean up old data based on retention policy."""
        try:
            db = SessionLocal()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Clean up old news (keep sentiment analysis references)
            old_news_count = db.query(News).filter(News.created_at < cutoff_date).delete(synchronize_session=False)

            # Clean up old sentiment analysis without news references
            old_sentiment_count = db.query(SentimentAnalysis).filter(
                ~SentimentAnalysis.news_id.in_(db.query(News.id))
            ).delete(synchronize_session=False)

            db.commit()

            self.logger.info(f"Cleaned up {old_news_count} old news and {old_sentiment_count} orphaned sentiment analyses")

            return {
                "old_news_cleaned": old_news_count,
                "orphaned_sentiment_cleaned": old_sentiment_count
            }

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error cleaning up old data: {e}")
            return {}
        finally:
            db.close()

    # Analytics and Reporting
    def get_sentiment_summary(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get sentiment analysis summary."""
        try:
            db = SessionLocal()

            if start_date is None:
                start_date = date.today() - timedelta(days=30)
            if end_date is None:
                end_date = date.today()

            # Get sentiment statistics
            sentiment_stats = db.query(
                func.avg(SentimentAnalysis.bullish_score).label('avg_bullish'),
                func.avg(SentimentAnalysis.bearish_score).label('avg_bearish'),
                func.avg(SentimentAnalysis.attention_score).label('avg_attention'),
                func.avg(SentimentAnalysis.confidence).label('avg_confidence'),
                func.count(SentimentAnalysis.id).label('total_analyses')
            ).join(News).filter(
                News.published_at >= start_date
            ).first()

            # Get category distribution
            category_stats = db.query(
                News.category,
                func.count(News.id).label('count')
            ).filter(
                News.published_at >= start_date
            ).group_by(News.category).all()

            return {
                "period": {"start": start_date, "end": end_date},
                "sentiment_stats": {
                    "avg_bullish": float(sentiment_stats.avg_bullish or 0),
                    "avg_bearish": float(sentiment_stats.avg_bearish or 0),
                    "avg_attention": float(sentiment_stats.avg_attention or 0),
                    "avg_confidence": float(sentiment_stats.avg_confidence or 0),
                    "total_analyses": sentiment_stats.total_analyses or 0
                },
                "category_distribution": [
                    {"category": cat, "count": count} for cat, count in category_stats
                ]
            }

        except Exception as e:
            self.logger.error(f"Error getting sentiment summary: {e}")
            return {}
        finally:
            db.close()

    def get_news_sources_summary(self) -> Dict[str, Any]:
        """Get news sources summary."""
        try:
            db = SessionLocal()

            source_stats = db.query(
                News.source,
                func.count(News.id).label('count'),
                func.max(News.published_at).label('latest_news')
            ).group_by(News.source).all()

            return {
                "sources": [
                    {
                        "source": source,
                        "count": count,
                        "latest_news": latest.strftime('%Y-%m-%d %H:%M:%S') if latest else None
                    }
                    for source, count, latest in source_stats
                ]
            }

        except Exception as e:
            self.logger.error(f"Error getting news sources summary: {e}")
            return {}
        finally:
            db.close()
