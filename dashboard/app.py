"""
Streamlit dashboard for gold news sentiment analysis system.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, date, timedelta
import time

# Import settings safely
try:
    from app.core.config import settings
except ImportError:
    # Fallback settings for when app is not available
    class Settings:
        api_host = "localhost"
        api_port = 8000
        api_version = "v1"
    settings = Settings()


# Configure page
st.set_page_config(
    page_title="Gold News Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}/api/{settings.api_version}"


def fetch_data(endpoint: str, params: dict = None) -> dict:
    """Fetch data from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data from {endpoint}: {e}")
        return {}


def main():
    """Main dashboard function."""
    st.title("📈 Gold News Sentiment Analysis Dashboard")
    st.markdown("Real-time sentiment analysis and gold price predictions")

    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "News Analysis", "Sentiment Trends", "Price Predictions", "Analytics"]
    )

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if page == "Overview":
        show_overview()
    elif page == "News Analysis":
        show_news_analysis()
    elif page == "Sentiment Trends":
        show_sentiment_trends()
    elif page == "Price Predictions":
        show_price_predictions()
    elif page == "Analytics":
        show_analytics()


def show_overview():
    """Show overview dashboard."""
    st.header("📋 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # News count
        news_data = fetch_data("/analytics/news-sources")
        total_news = sum(source.get('count', 0) for source in news_data.get('sources', []))
        st.metric("Total News Articles", total_news)

    with col2:
        # Sentiment summary
        sentiment_data = fetch_data("/analytics/sentiment-summary")
        avg_sentiment = sentiment_data.get('sentiment_stats', {}).get('avg_bullish', 0)
        st.metric("Avg Bullish Sentiment", f"{avg_sentiment:.3f}")

    with col3:
        # Latest gold price
        gold_data = fetch_data("/gold-prices", {"limit": 1})
        if gold_data.get('data'):
            latest_price = gold_data['data'][0].get('close_price')
            st.metric("Latest Gold Price", f"${latest_price}" if latest_price else "N/A")

    with col4:
        # Predictions count
        pred_data = fetch_data("/predictions")
        pred_count = len(pred_data.get('data', {}))
        st.metric("Active Predictions", pred_count)

    # Real-time charts
    st.subheader("📈 Real-time Data")

    tab1, tab2, tab3 = st.tabs(["Sentiment Overview", "News Sources", "Price Trends"])

    with tab1:
        show_sentiment_overview()

    with tab2:
        show_news_sources()

    with tab3:
        show_price_trends()


def show_sentiment_overview():
    """Show sentiment overview charts."""
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment by category
        sentiment_data = fetch_data("/sentiment", {"limit": 100})
        if sentiment_data.get('data'):
            df = pd.DataFrame(sentiment_data['data'])
            if not df.empty:
                fig = px.bar(
                    df.groupby('category')['weighted_score'].mean().reset_index(),
                    x='category',
                    y='weighted_score',
                    title='Average Sentiment by Category',
                    color='weighted_score',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sentiment over time
        if sentiment_data.get('data'):
            df['date'] = pd.to_datetime(df['date'])
            daily_sentiment = df.groupby(df['date'].dt.date)['weighted_score'].mean().reset_index()

            fig = px.line(
                daily_sentiment,
                x='date',
                y='weighted_score',
                title='Daily Sentiment Trend',
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)


def show_news_sources():
    """Show news sources distribution."""
    sources_data = fetch_data("/analytics/news-sources")

    if sources_data.get('sources'):
        df = pd.DataFrame(sources_data['sources'])

        fig = px.pie(
            df,
            values='count',
            names='source',
            title='News Distribution by Source',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

        # Source statistics table
        st.subheader("📊 Source Statistics")
        st.dataframe(df, use_container_width=True)


def show_price_trends():
    """Show gold price trends."""
    gold_data = fetch_data("/gold-prices", {"limit": 100})

    if gold_data.get('data'):
        df = pd.DataFrame(gold_data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Gold Price Trend', 'Trading Volume'),
            shared_xaxes=True
        )

        # Price chart
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close_price'],
                mode='lines',
                name='Close Price',
                line=dict(color='#FFD700', width=2)
            ),
            row=1, col=1
        )

        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name='Volume',
                marker_color='rgba(255, 215, 0, 0.3)'
            ),
            row=2, col=1
        )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_news_analysis():
    """Show news analysis page."""
    st.header("📰 News Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # News filters
        category = st.selectbox("Category", ["All"] + ["macro_policy", "geopolitical", "economic_data", "market_sentiment", "central_bank"])
        limit = st.slider("Number of articles", 10, 100, 20)

        # Fetch news
        params = {"limit": limit}
        if category != "All":
            params["category"] = category

        news_data = fetch_data("/news", params)

        if news_data.get('data'):
            st.subheader(f"📰 Recent News ({news_data['total']} total)")

            for article in news_data['data']:
                with st.expander(f"📄 {article['title'][:80]}..."):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"**Published:** {article['published_at']}")
                        st.write(f"**Category:** {article['category']}")
                        if article['content']:
                            st.write(f"**Content:** {article['content'][:200]}...")

                    with col2:
                        if st.button(f"Analyze Sentiment", key=f"analyze_{article['id']}"):
                            # Trigger sentiment analysis
                            response = requests.post(f"{API_BASE_URL}/sentiment/analyze", params={"news_id": article['id']})
                            if response.status_code == 200:
                                st.success("Sentiment analysis started!")
                                st.rerun()
                            else:
                                st.error("Failed to start analysis")

    with col2:
        st.subheader("🎯 Quick Actions")

        if st.button("🔄 Collect News", type="primary"):
            response = requests.post(f"{API_BASE_URL}/news/collect")
            if response.status_code == 200:
                st.success("News collection started!")
            else:
                st.error("Failed to start collection")

        if st.button("📊 Update Sentiment"):
            response = requests.post(f"{API_BASE_URL}/sentiment/update")
            if response.status_code == 200:
                st.success("Sentiment update started!")
            else:
                st.error("Failed to start update")

        if st.button("🧹 Cleanup Data"):
            response = requests.post(f"{API_BASE_URL}/cleanup")
            if response.status_code == 200:
                st.success("Cleanup started!")
            else:
                st.error("Failed to start cleanup")


def show_sentiment_trends():
    """Show sentiment trends page."""
    st.header("📈 Sentiment Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Time horizon filter
        time_horizon = st.selectbox("Time Horizon", ["short", "medium", "long"])
        category = st.selectbox("Category", ["macro_policy", "geopolitical", "economic_data", "market_sentiment", "central_bank"])

    with col2:
        # Date range
        end_date = st.date_input("End Date", date.today())
        start_date = st.date_input("Start Date", end_date - timedelta(days=30))

    # Fetch sentiment data
    params = {
        "time_horizon": time_horizon,
        "category": category,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "limit": 1000
    }

    sentiment_data = fetch_data("/sentiment", params)

    if sentiment_data.get('data'):
        df = pd.DataFrame(sentiment_data['data'])
        df['date'] = pd.to_datetime(df['date'])

        # Main sentiment chart
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Weighted Sentiment', 'Bullish vs Bearish', 'Attention Score'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        # Weighted sentiment
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['weighted_score'],
                mode='lines+markers',
                name='Weighted Score',
                line=dict(color='#FF6B6B')
            ),
            row=1, col=1
        )

        # Bullish vs Bearish (if available)
        # Note: This data might not be directly available in the API response
        # You might need to modify the API to include this

        # Attention score
        if 'attention_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['attention_score'],
                    mode='lines+markers',
                    name='Attention Score',
                    line=dict(color='#4ECDC4')
                ),
                row=3, col=1
            )

        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("📊 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_sentiment = df['weighted_score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")

        with col2:
            max_sentiment = df['weighted_score'].max()
            st.metric("Max Sentiment", f"{max_sentiment:.3f}")

        with col3:
            min_sentiment = df['weighted_score'].min()
            st.metric("Min Sentiment", f"{min_sentiment:.3f}")

        with col4:
            volatility = df['weighted_score'].std()
            st.metric("Volatility", f"{volatility:.3f}")

    else:
        st.info("No sentiment data available for the selected filters.")


def show_price_predictions():
    """Show price predictions page."""
    st.header("🔮 Price Predictions")

    col1, col2 = st.columns(2)

    with col1:
        target_date = st.date_input("Target Date", date.today() + timedelta(days=1))
        model_types = st.multiselect(
            "Model Types",
            ["lstm", "xgboost", "ensemble"],
            default=["ensemble"]
        )

    with col2:
        if st.button("🚀 Generate Predictions", type="primary"):
            # Trigger prediction generation
            response = requests.post(
                f"{API_BASE_URL}/predictions/predict",
                params={
                    "target_date": target_date.isoformat(),
                    "model_types": model_types
                }
            )

            if response.status_code == 200:
                st.success("Prediction generation started!")
                st.rerun()
            else:
                st.error("Failed to start predictions")

    # Display existing predictions
    predictions = fetch_data("/predictions", {"target_date": target_date.isoformat()})

    if predictions.get('data'):
        st.subheader("📊 Current Predictions")

        for model_name, pred_data in predictions['data'].items():
            with st.expander(f"🤖 {model_name.replace('_', ' ').title()}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicted Price", f"${pred_data['predicted_price']:.2f}")

                with col2:
                    confidence = pred_data.get('confidence_interval_lower', 0)
                    st.metric("Lower Bound", f"${confidence:.2f}" if confidence else "N/A")

                with col3:
                    confidence = pred_data.get('confidence_interval_upper', 0)
                    st.metric("Upper Bound", f"${confidence:.2f}" if confidence else "N/A")

                if pred_data.get('feature_importance'):
                    st.subheader("🎯 Feature Importance")
                    importance_df = pd.DataFrame([
                        {"feature": k, "importance": v}
                        for k, v in pred_data['feature_importance'].items()
                    ])
                    importance_df = importance_df.sort_values('importance', ascending=True)

                    fig = px.bar(
                        importance_df.tail(10),  # Top 10 features
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Training section
    st.subheader("🎓 Model Training")

    col1, col2 = st.columns(2)

    with col1:
        train_start = st.date_input("Training Start Date", date.today() - timedelta(days=365))
        train_end = st.date_input("Training End Date", date.today())

    with col2:
        train_models = st.multiselect(
            "Models to Train",
            ["lstm", "xgboost", "ensemble"],
            default=["xgboost"]
        )

    if st.button("⚡ Start Training", type="secondary"):
        response = requests.post(
            f"{API_BASE_URL}/predictions/train",
            params={
                "start_date": train_start.isoformat(),
                "end_date": train_end.isoformat(),
                "model_types": train_models
            }
        )

        if response.status_code == 200:
            st.success("Model training started!")
        else:
            st.error("Failed to start training")


def show_analytics():
    """Show analytics page."""
    st.header("📊 Analytics & Insights")

    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Performance", "Data Quality"])

    with tab1:
        st.subheader("🔍 Sentiment Analysis Insights")

        # Sentiment summary
        summary = fetch_data("/analytics/sentiment-summary")

        if summary:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Analyses",
                    summary.get('sentiment_stats', {}).get('total_analyses', 0)
                )

            with col2:
                st.metric(
                    "Avg Bullish",
                    f"{summary.get('sentiment_stats', {}).get('avg_bullish', 0):.3f}"
                )

            with col3:
                st.metric(
                    "Avg Bearish",
                    f"{summary.get('sentiment_stats', {}).get('avg_bearish', 0):.3f}"
                )

            with col4:
                st.metric(
                    "Avg Confidence",
                    f"{summary.get('sentiment_stats', {}).get('avg_confidence', 0):.3f}"
                )

            # Category distribution
            if summary.get('category_distribution'):
                st.subheader("📈 Category Distribution")
                cat_df = pd.DataFrame(summary['category_distribution'])

                fig = px.pie(
                    cat_df,
                    values='count',
                    names='category',
                    title='News Distribution by Category',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("⚡ Performance Metrics")

        # System health
        health = fetch_data("/health")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("System Status", health.get('status', 'unknown').upper())

        with col2:
            st.metric("API Version", health.get('version', 'unknown'))

        with col3:
            timestamp = health.get('timestamp', '')
            if timestamp:
                st.metric("Last Update", timestamp.split('T')[0])

    with tab3:
        st.subheader("🔍 Data Quality")

        # News sources
        sources = fetch_data("/analytics/news-sources")

        if sources.get('sources'):
            st.subheader("📰 News Sources Quality")
            sources_df = pd.DataFrame(sources['sources'])

            fig = px.bar(
                sources_df,
                x='source',
                y='count',
                title='Articles per Source',
                color='source'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sources table
            st.dataframe(sources_df, use_container_width=True)


if __name__ == "__main__":
    main()
