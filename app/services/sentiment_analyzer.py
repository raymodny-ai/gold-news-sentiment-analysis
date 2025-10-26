"""
Sentiment analysis service using multiple algorithms (VADER, TextBlob, FinBERT).
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
# Conditional imports for machine learning components
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class AutoTokenizer:
        pass
    class AutoModelForSequenceClassification:
        pass
    def pipeline(*args, **kwargs):
        return None

from app.core.config import settings
from app.models.schemas import AnalyzerType, NewsCategory, TimeHorizon


@dataclass
class SentimentResult:
    """Data class for sentiment analysis results."""
    analyzer_type: AnalyzerType
    bullish_score: float  # 0-1 scale (higher = more bullish)
    bearish_score: float  # 0-1 scale (higher = more bearish)
    attention_score: float  # 0-1 scale (higher = more attention-grabbing)
    confidence: float  # 0-1 scale (confidence in the analysis)


class BaseSentimentAnalyzer:
    """Base class for sentiment analyzers."""

    def __init__(self, name: AnalyzerType):
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of the given text."""
        raise NotImplementedError

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


class VaderSentimentAnalyzer(BaseSentimentAnalyzer):
    """VADER sentiment analyzer."""

    def __init__(self):
        super().__init__(AnalyzerType.VADER)
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER."""
        try:
            clean_text = self.preprocess_text(text)
            if not clean_text:
                return SentimentResult(
                    analyzer_type=self.name,
                    bullish_score=0.5,
                    bearish_score=0.5,
                    attention_score=0.5,
                    confidence=0.0
                )

            scores = self.analyzer.polarity_scores(clean_text)

            # Convert VADER compound score (-1 to 1) to bullish/bearish scores
            compound = scores['compound']
            confidence = abs(compound)  # Use absolute value as confidence

            if compound > 0:
                bullish_score = 0.5 + (compound * 0.5)  # Scale to 0.5-1.0
                bearish_score = 0.5 - (compound * 0.5)  # Scale to 0.5-0.0
            else:
                bullish_score = 0.5 + (compound * 0.5)  # Scale to 0.5-0.0
                bearish_score = 0.5 - (compound * 0.5)  # Scale to 0.5-1.0

            # Calculate attention score based on intensity of sentiment
            attention_score = 0.5 + (abs(compound) * 0.5)  # Scale based on sentiment intensity

            return SentimentResult(
                analyzer_type=self.name,
                bullish_score=max(0.0, min(1.0, bullish_score)),
                bearish_score=max(0.0, min(1.0, bearish_score)),
                attention_score=max(0.0, min(1.0, attention_score)),
                confidence=min(1.0, confidence)
            )

        except Exception as e:
            self.logger.error(f"Error in VADER analysis: {e}")
            return SentimentResult(
                analyzer_type=self.name,
                bullish_score=0.5,
                bearish_score=0.5,
                attention_score=0.5,
                confidence=0.0
            )


class TextBlobSentimentAnalyzer(BaseSentimentAnalyzer):
    """TextBlob sentiment analyzer."""

    def __init__(self):
        super().__init__(AnalyzerType.TEXTBLOB)

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using TextBlob."""
        try:
            clean_text = self.preprocess_text(text)
            if not clean_text:
                return SentimentResult(
                    analyzer_type=self.name,
                    bullish_score=0.5,
                    bearish_score=0.5,
                    attention_score=0.5,
                    confidence=0.0
                )

            blob = TextBlob(clean_text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            confidence = subjectivity  # Use subjectivity as confidence measure

            # Convert polarity to bullish/bearish scores
            if polarity > 0:
                bullish_score = 0.5 + (polarity * 0.5)  # Scale to 0.5-1.0
                bearish_score = 0.5 - (polarity * 0.5)  # Scale to 0.5-0.0
            else:
                bullish_score = 0.5 + (polarity * 0.5)  # Scale to 0.5-0.0
                bearish_score = 0.5 - (polarity * 0.5)  # Scale to 0.5-1.0

            # Calculate attention score based on sentiment intensity and subjectivity
            attention_score = 0.5 + (abs(polarity) * subjectivity * 0.5)

            return SentimentResult(
                analyzer_type=self.name,
                bullish_score=max(0.0, min(1.0, bullish_score)),
                bearish_score=max(0.0, min(1.0, bearish_score)),
                attention_score=max(0.0, min(1.0, attention_score)),
                confidence=min(1.0, confidence)
            )

        except Exception as e:
            self.logger.error(f"Error in TextBlob analysis: {e}")
            return SentimentResult(
                analyzer_type=self.name,
                bullish_score=0.5,
                bearish_score=0.5,
                attention_score=0.5,
                confidence=0.0
            )


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """FinBERT sentiment analyzer."""

    def __init__(self):
        super().__init__(AnalyzerType.FINBERT)
        self.model_name = settings.finbert_model_name
        self.max_length = settings.max_sequence_length
        self.batch_size = settings.batch_size

        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available. FinBERT analyzer will return neutral results.")
            self.pipeline = None
            return

        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {e}")
            self.pipeline = None

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using FinBERT."""
        try:
            clean_text = self.preprocess_text(text)
            if not clean_text or self.pipeline is None:
                return SentimentResult(
                    analyzer_type=self.name,
                    bullish_score=0.5,
                    bearish_score=0.5,
                    attention_score=0.5,
                    confidence=0.0
                )

            # Truncate text if too long
            if len(clean_text) > self.max_length:
                clean_text = clean_text[:self.max_length]

            result = self.pipeline(clean_text)[0]

            # FinBERT labels: 0 = negative, 1 = neutral, 2 = positive
            label = result['label']
            confidence = result['score']

            # Map FinBERT results to our scoring system
            if label == 'positive':
                bullish_score = 0.5 + (confidence * 0.5)  # Scale to 0.5-1.0
                bearish_score = 0.5 - (confidence * 0.5)  # Scale to 0.5-0.0
            elif label == 'negative':
                bullish_score = 0.5 - (confidence * 0.5)  # Scale to 0.5-0.0
                bearish_score = 0.5 + (confidence * 0.5)  # Scale to 0.5-1.0
            else:  # neutral
                bullish_score = 0.5
                bearish_score = 0.5

            # Calculate attention score based on confidence
            attention_score = 0.5 + (confidence * 0.5)

            return SentimentResult(
                analyzer_type=self.name,
                bullish_score=max(0.0, min(1.0, bullish_score)),
                bearish_score=max(0.0, min(1.0, bearish_score)),
                attention_score=max(0.0, min(1.0, attention_score)),
                confidence=min(1.0, confidence)
            )

        except Exception as e:
            self.logger.error(f"Error in FinBERT analysis: {e}")
            return SentimentResult(
                analyzer_type=self.name,
                bullish_score=0.5,
                bearish_score=0.5,
                attention_score=0.5,
                confidence=0.0
            )


class SentimentAnalysisService:
    """Main service for sentiment analysis using multiple analyzers."""

    def __init__(self):
        # Initialize available analyzers
        self.analyzers = {
            AnalyzerType.VADER: VaderSentimentAnalyzer(),
            AnalyzerType.TEXTBLOB: TextBlobSentimentAnalyzer(),
        }

        # Add FinBERT only if transformers is available
        if TRANSFORMERS_AVAILABLE:
            self.analyzers[AnalyzerType.FINBERT] = FinBERTSentimentAnalyzer()
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info("FinBERT analyzer loaded successfully")
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.warning("FinBERT analyzer not available (transformers not installed)")

        # Weights for combining analyzers (adjust based on available analyzers)
        self.weights = {
            AnalyzerType.VADER: settings.vader_weight,
            AnalyzerType.TEXTBLOB: settings.textblob_weight,
        }

        if TRANSFORMERS_AVAILABLE:
            self.weights[AnalyzerType.FINBERT] = settings.finbert_weight

    async def analyze_text(
        self,
        text: str,
        analyzers: Optional[List[AnalyzerType]] = None
    ) -> Dict[AnalyzerType, SentimentResult]:
        """Analyze text using specified analyzers (or all if not specified)."""
        if analyzers is None:
            analyzers = list(self.analyzers.keys())

        results = {}

        # Run analyzers concurrently
        tasks = []
        for analyzer_type in analyzers:
            if analyzer_type in self.analyzers:
                # Run each analyzer in a thread pool since some (like FinBERT) are CPU intensive
                task = asyncio.get_event_loop().run_in_executor(
                    None,
                    self.analyzers[analyzer_type].analyze,
                    text
                )
                tasks.append((analyzer_type, task))

        # Wait for all analyses to complete
        for analyzer_type, task in tasks:
            try:
                result = await task
                results[analyzer_type] = result
            except Exception as e:
                self.logger.error(f"Error running {analyzer_type} analyzer: {e}")
                # Provide default neutral result
                results[analyzer_type] = SentimentResult(
                    analyzer_type=analyzer_type,
                    bullish_score=0.5,
                    bearish_score=0.5,
                    attention_score=0.5,
                    confidence=0.0
                )

        return results

    def combine_sentiment_results(
        self,
        results: Dict[AnalyzerType, SentimentResult]
    ) -> Tuple[float, float, float, float]:
        """Combine results from multiple analyzers using weighted average."""
        if not results:
            return 0.5, 0.5, 0.5, 0.0

        total_weight = sum(self.weights.get(analyzer, 0) for analyzer in results.keys())
        if total_weight == 0:
            total_weight = len(results)  # Equal weights if no weights specified

        # Weighted averages
        weighted_bullish = sum(
            result.bullish_score * self.weights.get(analyzer, 1)
            for analyzer, result in results.items()
        ) / total_weight

        weighted_bearish = sum(
            result.bearish_score * self.weights.get(analyzer, 1)
            for analyzer, result in results.items()
        ) / total_weight

        weighted_attention = sum(
            result.attention_score * self.weights.get(analyzer, 1)
            for analyzer, result in results.items()
        ) / total_weight

        # Average confidence
        avg_confidence = sum(result.confidence for result in results.values()) / len(results)

        return (
            max(0.0, min(1.0, weighted_bullish)),
            max(0.0, min(1.0, weighted_bearish)),
            max(0.0, min(1.0, weighted_attention)),
            min(1.0, avg_confidence)
        )

    def apply_time_decay(
        self,
        score: float,
        published_at: datetime,
        current_time: Optional[datetime] = None,
        decay_factor: float = None
    ) -> float:
        """Apply time decay to sentiment scores."""
        if current_time is None:
            current_time = datetime.now()
        if decay_factor is None:
            decay_factor = settings.decay_factor

        # Calculate age in days
        age_days = (current_time - published_at).days

        # Apply exponential decay: score * (decay_factor ^ age_days)
        decayed_score = score * (decay_factor ** age_days)

        return max(0.0, min(1.0, decayed_score))

    async def analyze_batch(
        self,
        texts: List[str],
        analyzers: Optional[List[AnalyzerType]] = None
    ) -> List[Dict[AnalyzerType, SentimentResult]]:
        """Analyze a batch of texts concurrently."""
        if analyzers is None:
            analyzers = list(self.analyzers.keys())

        # Create tasks for all texts
        tasks = []
        for text in texts:
            task = self.analyze_text(text, analyzers)
            tasks.append(task)

        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error in batch analysis: {result}")
                # Provide default results for all analyzers
                default_results = {}
                for analyzer_type in analyzers:
                    default_results[analyzer_type] = SentimentResult(
                        analyzer_type=analyzer_type,
                        bullish_score=0.5,
                        bearish_score=0.5,
                        attention_score=0.5,
                        confidence=0.0
                    )
                final_results.append(default_results)
            else:
                final_results.append(result)

        return final_results

    def calculate_sentiment_scores(
        self,
        bullish_score: float,
        bearish_score: float,
        attention_score: float,
        confidence: float
    ) -> Dict[str, float]:
        """Calculate final sentiment scores with normalization."""
        # Normalize bullish and bearish scores
        total_sentiment = bullish_score + bearish_score
        if total_sentiment > 0:
            normalized_bullish = bullish_score / total_sentiment
            normalized_bearish = bearish_score / total_sentiment
        else:
            normalized_bullish = 0.5
            normalized_bearish = 0.5

        # Calculate overall sentiment score (-1 to 1)
        overall_score = normalized_bullish - normalized_bearish

        return {
            'bullish_score': round(normalized_bullish, 4),
            'bearish_score': round(normalized_bearish, 4),
            'attention_score': round(attention_score, 4),
            'confidence': round(confidence, 4),
            'overall_score': round(overall_score, 4)  # -1 (bearish) to 1 (bullish)
        }
