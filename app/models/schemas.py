"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class NewsCategory(str, Enum):
    """News category enumeration."""
    MACRO_POLICY = "macro_policy"
    GEOPOLITICAL = "geopolitical"
    ECONOMIC_DATA = "economic_data"
    MARKET_SENTIMENT = "market_sentiment"
    CENTRAL_BANK = "central_bank"


class AnalyzerType(str, Enum):
    """Sentiment analyzer type enumeration."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    FINBERT = "finbert"


class TimeHorizon(str, Enum):
    """Time horizon enumeration."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ModelType(str, Enum):
    """Prediction model type enumeration."""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


class NewsSource(str, Enum):
    """News source enumeration."""
    NEWSAPI = "newsapi"
    FINNHUB = "finnhub"
    YAHOO = "yahoo"
    TEST = "test"  # For testing purposes


# News Schemas
class NewsBase(BaseModel):
    """Base news schema."""
    title: str = Field(..., max_length=500)
    content: Optional[str] = None
    category: str
    published_at: datetime
    source: str  # Changed from NewsSource to str to handle database strings
    url: Optional[str] = Field(None, max_length=500)
    language: str = "en"


class NewsCreate(NewsBase):
    """Schema for creating news."""
    pass


# Sentiment Analysis Schemas
class SentimentAnalysisBase(BaseModel):
    """Base sentiment analysis schema."""
    analyzer_type: AnalyzerType
    bullish_score: Optional[float] = Field(None, ge=0, le=1)
    bearish_score: Optional[float] = Field(None, ge=0, le=1)
    attention_score: Optional[float] = Field(None, ge=0, le=1)
    confidence: Optional[float] = Field(None, ge=0, le=1)


class SentimentAnalysisCreate(SentimentAnalysisBase):
    """Schema for creating sentiment analysis."""
    news_id: int


class SentimentAnalysisResponse(SentimentAnalysisBase):
    """Schema for sentiment analysis response."""
    id: int
    news_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Weighted Sentiment Schemas
class WeightedSentimentBase(BaseModel):
    """Base weighted sentiment schema."""
    date: date
    category: str
    weighted_score: float = Field(..., ge=-1, le=1)
    time_horizon: TimeHorizon

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WeightedSentimentCreate(WeightedSentimentBase):
    """Schema for creating weighted sentiment."""
    pass


class WeightedSentimentResponse(WeightedSentimentBase):
    """Schema for weighted sentiment response."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Price Predictions Schemas
class PricePredictionBase(BaseModel):
    """Base price prediction schema."""
    prediction_date: date
    target_date: date
    model_type: ModelType
    predicted_price: float = Field(..., gt=0)
    confidence_interval_lower: Optional[float] = Field(None, gt=0)
    confidence_interval_upper: Optional[float] = Field(None, gt=0)
    feature_importance: Optional[Dict[str, float]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PricePredictionCreate(PricePredictionBase):
    """Schema for creating price prediction."""
    pass


class PricePredictionResponse(PricePredictionBase):
    """Schema for price prediction response."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Gold Prices Schemas
class GoldPriceBase(BaseModel):
    """Base gold price schema."""
    date: date
    open_price: Optional[float] = Field(None, gt=0)
    high_price: Optional[float] = Field(None, gt=0)
    low_price: Optional[float] = Field(None, gt=0)
    close_price: Optional[float] = Field(None, gt=0)
    volume: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GoldPriceCreate(GoldPriceBase):
    """Schema for creating gold price."""
    pass


class GoldPriceResponse(GoldPriceBase):
    """Schema for gold price response."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# API Response Schemas
class NewsListResponse(BaseModel):
    """Paginated news list response."""
    data: List[Dict[str, Any]]
    total: int
    page: int
    per_page: int


class SentimentListResponse(BaseModel):
    """Sentiment data list response."""
    data: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """Price prediction response."""
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
