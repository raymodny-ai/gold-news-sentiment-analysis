"""
News collection service for gathering news from multiple sources.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests
import yfinance as yf
from bs4 import BeautifulSoup
import aiohttp

from app.core.config import settings
from app.models.schemas import NewsCreate, NewsCategory, NewsSource

# Check if Redis is available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class NewsArticle:
    """Data class for news article information."""
    title: str
    content: Optional[str]
    url: Optional[str]
    published_at: datetime
    source: str
    category: NewsCategory
    language: str = "en"


class BaseNewsCollector(ABC):
    """Base class for news collectors."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def collect_news(self, **kwargs) -> List[NewsArticle]:
        """Collect news articles from the source."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the source name."""
        pass

    def _classify_news(self, title: str, content: Optional[str] = None) -> NewsCategory:
        """Classify news into categories based on content."""
        text = f"{title} {content or ''}".lower()

        # Keywords for each category
        categories = {
            NewsCategory.MACRO_POLICY: [
                'fed', 'federal reserve', 'interest rate', 'monetary policy',
                'inflation', 'cpi', 'gdp', 'unemployment', 'jobs report'
            ],
            NewsCategory.GEOPOLITICAL: [
                'war', 'conflict', 'sanctions', 'trade war', 'geopolitics',
                'election', 'political', 'government', 'policy'
            ],
            NewsCategory.ECONOMIC_DATA: [
                'economic data', 'growth', 'recession', 'recovery', 'data',
                'report', 'statistics', 'indicator', 'forecast'
            ],
            NewsCategory.MARKET_SENTIMENT: [
                'market sentiment', 'investor', 'bullish', 'bearish', 'confidence',
                'mood', 'outlook', 'expectation', 'survey'
            ],
            NewsCategory.CENTRAL_BANK: [
                'central bank', 'ecb', 'boe', 'boj', 'pboc', 'bank of',
                'quantitative easing', 'rate decision', 'guidance'
            ]
        }

        # Find matching category
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category

        # Default to market sentiment if no match found
        return NewsCategory.MARKET_SENTIMENT

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text content."""
        if not text:
            return None

        # Remove extra whitespace and normalize
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        return text if text else None


class NewsAPICollector(BaseNewsCollector):
    """News collector for NewsAPI."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2"
        self.source_name = NewsSource.NEWSAPI

    def get_source_name(self) -> str:
        return self.source_name

    async def collect_news(
        self,
        query: str = "gold",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[NewsArticle]:
        """Collect news from NewsAPI."""
        if not self.api_key:
            self.logger.warning("NewsAPI key not provided")
            return []

        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)
        if to_date is None:
            to_date = datetime.now()

        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'pageSize': min(max_results, 100),
            'apiKey': self.api_key,
            'language': 'en'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/everything", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_newsapi_response(data)
                    else:
                        self.logger.error(f"NewsAPI request failed: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error collecting news from NewsAPI: {e}")
            return []

    def _parse_newsapi_response(self, data: Dict[str, Any]) -> List[NewsArticle]:
        """Parse NewsAPI response into NewsArticle objects."""
        articles = []

        for item in data.get('articles', []):
            try:
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=self._clean_text(item.get('description') or item.get('content')),
                    url=item.get('url'),
                    published_at=datetime.fromisoformat(item['publishedAt'].replace('Z', '+00:00')),
                    source=self.get_source_name(),
                    category=self._classify_news(item.get('title', ''), item.get('description'))
                )
                articles.append(article)

            except Exception as e:
                self.logger.warning(f"Error parsing NewsAPI article: {e}")
                continue

        return articles


class FinnhubCollector(BaseNewsCollector):
    """News collector for Finnhub API."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://finnhub.io/api/v1"
        self.source_name = NewsSource.FINNHUB

    def get_source_name(self) -> str:
        return self.source_name

    async def collect_news(
        self,
        symbol: str = "GC=F",  # Gold futures symbol
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[NewsArticle]:
        """Collect news from Finnhub."""
        if not self.api_key:
            self.logger.warning("Finnhub API key not provided")
            return []

        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)
        if to_date is None:
            to_date = datetime.now()

        params = {
            'symbol': symbol,
            'minId': 0,
            'token': self.api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/company-news", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_finnhub_response(data, from_date, to_date, max_results)
                    else:
                        self.logger.error(f"Finnhub request failed: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error collecting news from Finnhub: {e}")
            return []

    def _parse_finnhub_response(
        self,
        data: Dict[str, Any],
        from_date: datetime,
        to_date: datetime,
        max_results: int
    ) -> List[NewsArticle]:
        """Parse Finnhub response into NewsArticle objects."""
        articles = []

        # Finnhub response is typically keyed by symbol (e.g., "GC=F")
        for symbol_data in data.values():
            for item in symbol_data:
                try:
                    published_at = datetime.fromtimestamp(item.get('datetime', 0))
                    if not (from_date <= published_at <= to_date):
                        continue

                    article = NewsArticle(
                        title=item.get('headline', ''),
                        content=self._clean_text(item.get('summary')),
                        url=item.get('url'),
                        published_at=published_at,
                        source=self.get_source_name(),
                        category=self._classify_news(item.get('headline', ''), item.get('summary'))
                    )
                    articles.append(article)

                    if len(articles) >= max_results:
                        break

                except Exception as e:
                    self.logger.warning(f"Error parsing Finnhub article: {e}")
                    continue

        return articles


class YahooFinanceCollector(BaseNewsCollector):
    """News collector for Yahoo Finance."""

    def __init__(self):
        super().__init__(None)  # No API key needed for Yahoo Finance
        self.source_name = NewsSource.YAHOO

    def get_source_name(self) -> str:
        return self.source_name

    async def collect_news(
        self,
        symbol: str = "GC=F",
        max_results: int = 100
    ) -> List[NewsArticle]:
        """Collect news from Yahoo Finance."""
        try:
            # Note: yfinance doesn't have async support, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            news_data = await loop.run_in_executor(
                None,
                self._get_yahoo_news,
                symbol,
                max_results
            )
            return news_data

        except Exception as e:
            self.logger.error(f"Error collecting news from Yahoo Finance: {e}")
            return []

    def _get_yahoo_news(self, symbol: str, max_results: int) -> List[NewsArticle]:
        """Get news from Yahoo Finance (synchronous)."""
        try:
            ticker = yf.Ticker(symbol)
            news_data = []

            # Try to get news from the ticker
            if hasattr(ticker, 'news') and ticker.news is not None:
                for item in ticker.news[:max_results]:
                    try:
                        article = NewsArticle(
                            title=item.get('title', ''),
                            content=None,  # Yahoo Finance doesn't provide content in the basic API
                            url=item.get('link'),
                            published_at=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                            source=self.get_source_name(),
                            category=self._classify_news(item.get('title', ''))
                        )
                        news_data.append(article)

                    except Exception as e:
                        self.logger.warning(f"Error parsing Yahoo Finance article: {e}")
                        continue

            return news_data

        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance news: {e}")
            return []


class NewsCollectorService:
    """Main service for collecting news from multiple sources."""

    def __init__(self):
        self.collectors = {
            NewsSource.NEWSAPI: NewsAPICollector(settings.newsapi_key),
            NewsSource.FINNHUB: FinnhubCollector(settings.finnhub_api_key),
            NewsSource.YAHOO: YahooFinanceCollector()
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    async def collect_all_news(
        self,
        query: str = "gold",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results_per_source: int = 100
    ) -> List[NewsArticle]:
        """Collect news from all configured sources."""
        all_articles = []

        # Collect from all sources concurrently
        tasks = []
        for source, collector in self.collectors.items():
            if source == NewsSource.NEWSAPI and query:
                task = collector.collect_news(
                    query=query,
                    from_date=from_date,
                    to_date=to_date,
                    max_results=max_results_per_source
                )
            elif source == NewsSource.FINNHUB:
                task = collector.collect_news(
                    symbol="GC=F",
                    from_date=from_date,
                    to_date=to_date,
                    max_results=max_results_per_source
                )
            elif source == NewsSource.YAHOO:
                task = collector.collect_news(
                    symbol="GC=F",
                    max_results=max_results_per_source
                )
            else:
                continue

            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_name = list(self.collectors.keys())[i]
                self.logger.error(f"Error collecting from {source_name}: {result}")
                continue

            all_articles.extend(result)

        # Remove duplicates based on title and URL
        unique_articles = self._deduplicate_articles(all_articles)

        self.logger.info(f"Collected {len(unique_articles)} unique articles from {len(tasks)} sources")
        return unique_articles

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title and URL."""
        seen = set()
        unique_articles = []

        for article in articles:
            # Create a unique key based on title and URL
            key = (article.title.lower().strip(), article.url or "")

            if key not in seen:
                seen.add(key)
                unique_articles.append(article)

        return unique_articles

    def convert_to_news_create(self, article: NewsArticle) -> NewsCreate:
        """Convert NewsArticle to NewsCreate schema."""
        return NewsCreate(
            title=article.title,
            content=article.content,
            category=article.category,
            published_at=article.published_at,
            source=article.source,
            url=article.url,
            language=article.language
        )
