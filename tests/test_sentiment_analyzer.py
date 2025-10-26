"""
Tests for sentiment analysis service.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.sentiment_analyzer import (
    SentimentAnalysisService,
    VaderSentimentAnalyzer,
    TextBlobSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    SentimentResult
)
from app.models.schemas import AnalyzerType


class TestVaderSentimentAnalyzer:
    """Test VADER sentiment analyzer."""

    def test_analyze_positive_text(self):
        """Test analysis of positive sentiment text."""
        analyzer = VaderSentimentAnalyzer()

        result = analyzer.analyze("This is great! Gold prices are rising!")

        assert result.analyzer_type == AnalyzerType.VADER
        assert result.bullish_score > 0.5
        assert result.bearish_score < 0.5
        assert result.confidence > 0
        assert 0 <= result.bullish_score <= 1
        assert 0 <= result.bearish_score <= 1
        assert 0 <= result.confidence <= 1

    def test_analyze_negative_text(self):
        """Test analysis of negative sentiment text."""
        analyzer = VaderSentimentAnalyzer()

        result = analyzer.analyze("This is terrible! Gold prices are falling!")

        assert result.analyzer_type == AnalyzerType.VADER
        assert result.bullish_score < 0.5
        assert result.bearish_score > 0.5
        assert result.confidence > 0

    def test_analyze_neutral_text(self):
        """Test analysis of neutral sentiment text."""
        analyzer = VaderSentimentAnalyzer()

        result = analyzer.analyze("Gold prices are stable.")

        assert result.analyzer_type == AnalyzerType.VADER
        assert abs(result.bullish_score - 0.5) < 0.1
        assert abs(result.bearish_score - 0.5) < 0.1

    def test_analyze_empty_text(self):
        """Test analysis of empty text."""
        analyzer = VaderSentimentAnalyzer()

        result = analyzer.analyze("")

        assert result.analyzer_type == AnalyzerType.VADER
        assert result.bullish_score == 0.5
        assert result.bearish_score == 0.5
        assert result.confidence == 0.0

    def test_preprocess_text(self):
        """Test text preprocessing."""
        analyzer = VaderSentimentAnalyzer()

        # Test URL removal
        text = "Check this link: https://example.com and email: test@example.com"
        processed = analyzer.preprocess_text(text)
        assert "https://" not in processed
        assert "test@" not in processed


class TestTextBlobSentimentAnalyzer:
    """Test TextBlob sentiment analyzer."""

    def test_analyze_text(self):
        """Test basic TextBlob analysis."""
        analyzer = TextBlobSentimentAnalyzer()

        result = analyzer.analyze("Gold prices are increasing rapidly!")

        assert result.analyzer_type == AnalyzerType.TEXTBLOB
        assert 0 <= result.bullish_score <= 1
        assert 0 <= result.bearish_score <= 1
        assert 0 <= result.confidence <= 1


class TestFinBERTSentimentAnalyzer:
    """Test FinBERT sentiment analyzer."""

    @patch('app.services.sentiment_analyzer.pipeline')
    def test_analyze_with_mock(self, mock_pipeline):
        """Test FinBERT analysis with mocked pipeline."""
        # Mock the pipeline response
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.95}]

        analyzer = FinBERTSentimentAnalyzer()

        result = analyzer.analyze("Excellent gold market performance!")

        assert result.analyzer_type == AnalyzerType.FINBERT
        assert result.bullish_score > 0.5
        assert result.bearish_score < 0.5
        assert result.confidence == 0.95

    def test_analyze_without_pipeline(self):
        """Test FinBERT analysis when pipeline fails to load."""
        analyzer = FinBERTSentimentAnalyzer()
        analyzer.pipeline = None  # Simulate failed loading

        result = analyzer.analyze("Test text")

        assert result.analyzer_type == AnalyzerType.FINBERT
        assert result.bullish_score == 0.5
        assert result.bearish_score == 0.5
        assert result.confidence == 0.0


class TestSentimentAnalysisService:
    """Test main sentiment analysis service."""

    def test_combine_sentiment_results(self):
        """Test combining results from multiple analyzers."""
        service = SentimentAnalysisService()

        # Create mock results
        results = {
            AnalyzerType.VADER: SentimentResult(
                analyzer_type=AnalyzerType.VADER,
                bullish_score=0.8,
                bearish_score=0.2,
                attention_score=0.9,
                confidence=0.9
            ),
            AnalyzerType.TEXTBLOB: SentimentResult(
                analyzer_type=AnalyzerType.TEXTBLOB,
                bullish_score=0.7,
                bearish_score=0.3,
                attention_score=0.8,
                confidence=0.8
            )
        }

        combined = service.combine_sentiment_results(results)

        assert len(combined) == 4  # bullish, bearish, attention, confidence
        assert 0 <= combined[0] <= 1  # bullish_score
        assert 0 <= combined[1] <= 1  # bearish_score
        assert 0 <= combined[2] <= 1  # attention_score
        assert 0 <= combined[3] <= 1  # confidence

    def test_calculate_sentiment_scores(self):
        """Test final sentiment score calculation."""
        service = SentimentAnalysisService()

        scores = service.calculate_sentiment_scores(
            bullish_score=0.8,
            bearish_score=0.2,
            attention_score=0.9,
            confidence=0.85
        )

        assert 'bullish_score' in scores
        assert 'bearish_score' in scores
        assert 'attention_score' in scores
        assert 'confidence' in scores
        assert 'overall_score' in scores

        assert scores['bullish_score'] == 0.8
        assert scores['bearish_score'] == 0.2
        assert scores['overall_score'] == 0.6  # 0.8 - 0.2

    def test_apply_time_decay(self):
        """Test time decay functionality."""
        service = SentimentAnalysisService()

        # Test with recent date (should have minimal decay)
        recent_date = service.now() - timedelta(days=1)
        decayed_score = service.apply_time_decay(1.0, recent_date)
        assert decayed_score > 0.95  # Should decay very little

        # Test with old date (should decay significantly)
        old_date = service.now() - timedelta(days=30)
        decayed_score = service.apply_time_decay(1.0, old_date)
        assert decayed_score < 0.5  # Should decay significantly

    @pytest.mark.asyncio
    async def test_analyze_text(self):
        """Test async text analysis."""
        service = SentimentAnalysisService()

        results = await service.analyze_text("Gold prices are rising!")

        assert isinstance(results, dict)
        assert len(results) > 0

        for analyzer_type, result in results.items():
            assert isinstance(result, SentimentResult)
            assert result.analyzer_type == analyzer_type


# Note: In a real implementation, you would add more comprehensive tests
# including integration tests with actual APIs and database tests
