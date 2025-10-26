"""
SQLAlchemy database models.
"""
import json
from sqlalchemy import Column, Integer, String, Text, DateTime, Date, DECIMAL, BigInteger, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, TEXT

from app.models.database import Base


class JSONColumn(TypeDecorator):
    """Custom JSON column type that works with both PostgreSQL and SQLite."""
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class News(Base):
    """News table model."""
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    url = Column(String(500))
    published_at = Column(DateTime, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    sentiment_analyses = relationship("SentimentAnalysis", back_populates="news")

    def __repr__(self):
        return f"<News(id={self.id}, title='{self.title[:50]}...', source='{self.source}')>"


class SentimentAnalysis(Base):
    """Sentiment analysis results table model."""
    __tablename__ = "sentiment_analysis"

    id = Column(Integer, primary_key=True, index=True)
    news_id = Column(Integer, ForeignKey("news.id"), nullable=False, index=True)
    analyzer_type = Column(String(20), nullable=False, index=True)
    bullish_score = Column(DECIMAL(5, 4))
    bearish_score = Column(DECIMAL(5, 4))
    attention_score = Column(DECIMAL(5, 4))
    confidence = Column(DECIMAL(5, 4))
    created_at = Column(DateTime, default=func.now())

    # Relationships
    news = relationship("News", back_populates="sentiment_analyses")

    def __repr__(self):
        return f"<SentimentAnalysis(id={self.id}, analyzer='{self.analyzer_type}', news_id={self.news_id})>"


class WeightedSentiment(Base):
    """Weighted sentiment index table model."""
    __tablename__ = "weighted_sentiment"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    weighted_score = Column(DECIMAL(8, 6), nullable=False)
    time_horizon = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())

    # Unique constraint
    __table_args__ = (
        UniqueConstraint('date', 'category', 'time_horizon', name='idx_weighted_sentiment_unique'),
    )

    def __repr__(self):
        return f"<WeightedSentiment(id={self.id}, date='{self.date}', category='{self.category}', score={self.weighted_score})>"


class PricePrediction(Base):
    """Price predictions table model."""
    __tablename__ = "price_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    target_date = Column(Date, nullable=False, index=True)
    model_type = Column(String(20), nullable=False, index=True)
    predicted_price = Column(DECIMAL(10, 2), nullable=False)
    confidence_interval_lower = Column(DECIMAL(10, 2))
    confidence_interval_upper = Column(DECIMAL(10, 2))
    feature_importance = Column(JSONColumn)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<PricePrediction(id={self.id}, model='{self.model_type}', price={self.predicted_price})>"


class GoldPrice(Base):
    """Gold prices historical data table model."""
    __tablename__ = "gold_prices"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    open_price = Column(DECIMAL(10, 2))
    high_price = Column(DECIMAL(10, 2))
    low_price = Column(DECIMAL(10, 2))
    close_price = Column(DECIMAL(10, 2))
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<GoldPrice(id={self.id}, date='{self.date}', close={self.close_price})>"
