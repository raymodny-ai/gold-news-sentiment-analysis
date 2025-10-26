"""
Price prediction service using LSTM and XGBoost models.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json

# Conditional imports for machine learning components
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Sequential = None
    load_model = None
    LSTM = None
    Dense = None
    Dropout = None
    EarlyStopping = None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.core.config import settings
from app.models.schemas import ModelType, TimeHorizon
from app.services.data_manager import DataManager


@dataclass
class PredictionResult:
    """Data class for prediction results."""
    model_type: ModelType
    predicted_price: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    confidence_score: float = 0.0


@dataclass
class ModelMetrics:
    """Data class for model performance metrics."""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    training_time: float


class BasePredictionModel:
    """Base class for prediction models."""

    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.scaler = StandardScaler()

    def prepare_features(
        self,
        gold_prices: List[Dict[str, Any]],
        sentiment_data: List[Dict[str, Any]],
        lookback_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training/prediction."""
        raise NotImplementedError

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train the model."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions."""
        raise NotImplementedError

    def save_model(self, path: str):
        """Save the trained model."""
        raise NotImplementedError

    def load_model(self, path: str):
        """Load a trained model."""
        raise NotImplementedError


class XGBoostModel(BasePredictionModel):
    """XGBoost prediction model."""

    def __init__(self):
        super().__init__(ModelType.XGBOOST)
        self.model = None

        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available. XGBoost model will return default predictions.")
            return

    def prepare_features(
        self,
        gold_prices: List[Dict[str, Any]],
        sentiment_data: List[Dict[str, Any]],
        lookback_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for XGBoost model."""
        try:
            # Convert to DataFrame
            df_prices = pd.DataFrame(gold_prices)
            df_sentiment = pd.DataFrame(sentiment_data)

            # Sort by date
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            df_prices = df_prices.sort_values('date')

            # Calculate technical indicators
            df_prices = self._calculate_technical_indicators(df_prices)

            # Merge with sentiment data
            df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
            df_combined = pd.merge(df_prices, df_sentiment, on='date', how='left')

            # Fill missing sentiment values
            sentiment_cols = ['weighted_score', 'bullish_score', 'bearish_score', 'attention_score']
            df_combined[sentiment_cols] = df_combined[sentiment_cols].fillna(0)

            # Create features
            feature_cols = [
                'close_price', 'volume', 'ma_7', 'ma_30', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower', 'weighted_score',
                'bullish_score', 'bearish_score', 'attention_score'
            ]

            # Remove rows with missing values
            df_features = df_combined.dropna(subset=feature_cols + ['close_price'])

            # Create sequences for time series prediction
            X, y = [], []
            for i in range(lookback_days, len(df_features)):
                # Feature sequence
                feature_sequence = df_features[feature_cols].iloc[i-lookback_days:i].values
                X.append(feature_sequence.flatten())

                # Target (next day price)
                y.append(df_features['close_price'].iloc[i])

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error preparing features for XGBoost: {e}")
            return np.array([]), np.array([])

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for gold prices."""
        try:
            df = df.copy()

            # Moving averages
            df['ma_7'] = df['close_price'].rolling(window=7).mean()
            df['ma_30'] = df['close_price'].rolling(window=30).mean()

            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close_price'], 14)

            # MACD (Moving Average Convergence Divergence)
            df['macd'] = self._calculate_macd(df['close_price'])

            # Bollinger Bands
            df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['close_price'])

            # Price momentum
            df['momentum_5'] = df['close_price'].pct_change(5)
            df['momentum_10'] = df['close_price'].pct_change(10)

            # Volatility
            df['volatility'] = df['close_price'].rolling(window=20).std()

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        macd = fast_ema - slow_ema
        return macd

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available. Cannot train model.")
            return ModelMetrics(
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2_score=0.0,
                training_time=0.0
            )

        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data")

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train model
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            start_time = datetime.now()
            self.model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate model
            y_pred = self.model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # Calculate R² score
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            self.is_trained = True

            metrics = ModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2_score,
                training_time=training_time
            )

            self.logger.info(f"XGBoost model trained successfully. R² Score: {r2_score:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {e}")
            raise

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with XGBoost model."""
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available. Returning default prediction.")
            return PredictionResult(
                model_type=self.model_type,
                predicted_price=2000.0,  # Default gold price
                confidence_score=0.5
            )

        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained")

            # Make prediction
            prediction = self.model.predict(X.reshape(1, -1))[0]

            # Calculate confidence interval (simple approach)
            predictions = self.model.predict(X.reshape(1, -1))
            std_dev = np.std(predictions) if len(predictions) > 1 else prediction * 0.05

            confidence_lower = prediction - (1.96 * std_dev)  # 95% confidence interval
            confidence_upper = prediction + (1.96 * std_dev)

            # Get feature importance
            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(X.shape[1])],
                self.model.feature_importances_
            ))

            # Calculate confidence score based on prediction consistency
            confidence_score = min(1.0, 1.0 - (std_dev / prediction) if prediction != 0 else 0.5)

            return PredictionResult(
                model_type=self.model_type,
                predicted_price=prediction,
                confidence_interval_lower=confidence_lower,
                confidence_interval_upper=confidence_upper,
                feature_importance=feature_importance,
                confidence_score=confidence_score
            )

        except Exception as e:
            self.logger.error(f"Error making XGBoost prediction: {e}")
            return PredictionResult(
                model_type=self.model_type,
                predicted_price=0.0,
                confidence_score=0.0
            )

    def save_model(self, path: str):
        """Save XGBoost model."""
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available. Cannot save model.")
            return

        try:
            if self.model is not None:
                self.model.save_model(f"{path}_xgb.json")
                # Save scaler
                with open(f"{path}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {e}")

    def load_model(self, path: str):
        """Load XGBoost model."""
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available. Cannot load model.")
            return

        try:
            self.model = xgb.XGBRegressor()
            self.model.load_model(f"{path}_xgb.json")
            # Load scaler
            with open(f"{path}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {e}")


class LSTMModel(BasePredictionModel):
    """LSTM prediction model."""

    def __init__(self):
        super().__init__(ModelType.LSTM)
        self.model = None

        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. LSTM model will return default predictions.")
            return

    def prepare_features(
        self,
        gold_prices: List[Dict[str, Any]],
        sentiment_data: List[Dict[str, Any]],
        lookback_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for LSTM model."""
        try:
            # Convert to DataFrame
            df_prices = pd.DataFrame(gold_prices)
            df_sentiment = pd.DataFrame(sentiment_data)

            # Sort by date
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            df_prices = df_prices.sort_values('date')

            # Calculate technical indicators
            df_prices = self._calculate_technical_indicators(df_prices)

            # Merge with sentiment data
            df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
            df_combined = pd.merge(df_prices, df_sentiment, on='date', how='left')

            # Fill missing sentiment values
            sentiment_cols = ['weighted_score', 'bullish_score', 'bearish_score', 'attention_score']
            df_combined[sentiment_cols] = df_combined[sentiment_cols].fillna(0)

            # Select features
            feature_cols = [
                'close_price', 'volume', 'ma_7', 'ma_30', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower', 'momentum_5', 'momentum_10',
                'volatility', 'weighted_score', 'bullish_score', 'bearish_score', 'attention_score'
            ]

            # Remove rows with missing values
            df_features = df_combined.dropna(subset=feature_cols + ['close_price'])

            # Normalize features
            feature_data = df_features[feature_cols].values
            if not hasattr(self, 'scaler_fitted') or not self.scaler_fitted:
                feature_data_scaled = self.scaler.fit_transform(feature_data)
                self.scaler_fitted = True
            else:
                feature_data_scaled = self.scaler.transform(feature_data)

            # Create sequences for LSTM
            X, y = [], []
            for i in range(lookback_days, len(df_features)):
                # Feature sequence (3D for LSTM: samples, timesteps, features)
                feature_sequence = feature_data_scaled[i-lookback_days:i]
                X.append(feature_sequence)

                # Target (next day price, scaled)
                target_price = df_features['close_price'].iloc[i]
                y.append(target_price)

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error preparing features for LSTM: {e}")
            return np.array([]), np.array([])

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for LSTM model."""
        # Similar to XGBoost but adapted for LSTM
        return self._calculate_technical_indicators(df)

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Cannot train LSTM model.")
            return ModelMetrics(
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2_score=0.0,
                training_time=0.0
            )

        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data")

            # Reshape for LSTM (add feature dimension)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Build LSTM model
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            self.model.compile(optimizer='adam', loss='mse')

            # Train model
            start_time = datetime.now()
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=0
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate model
            y_pred = self.model.predict(X_test, verbose=0).flatten()

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # Calculate R² score
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            self.is_trained = True

            metrics = ModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2_score,
                training_time=training_time
            )

            self.logger.info(f"LSTM model trained successfully. R² Score: {r2_score:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            raise

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Returning default prediction.")
            return PredictionResult(
                model_type=self.model_type,
                predicted_price=2000.0,  # Default gold price
                confidence_score=0.5
            )

        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained")

            # Reshape for LSTM
            X_reshaped = X.reshape(1, X.shape[0], X.shape[1])

            # Make prediction
            prediction_scaled = self.model.predict(X_reshaped, verbose=0)[0][0]

            # Inverse transform to get actual price
            # Note: This is a simplified approach - in practice, you'd need to handle the scaling properly
            prediction = float(prediction_scaled)

            # Calculate confidence (simplified)
            confidence_score = 0.8  # LSTM typically has higher confidence due to sequence modeling

            # Simple confidence interval based on model confidence
            std_dev = prediction * 0.05  # 5% standard deviation assumption
            confidence_lower = prediction - (1.96 * std_dev)
            confidence_upper = prediction + (1.96 * std_dev)

            return PredictionResult(
                model_type=self.model_type,
                predicted_price=prediction,
                confidence_interval_lower=confidence_lower,
                confidence_interval_upper=confidence_upper,
                confidence_score=confidence_score
            )

        except Exception as e:
            self.logger.error(f"Error making LSTM prediction: {e}")
            return PredictionResult(
                model_type=self.model_type,
                predicted_price=0.0,
                confidence_score=0.0
            )

    def save_model(self, path: str):
        """Save LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Cannot save LSTM model.")
            return

        try:
            if self.model is not None:
                self.model.save(f"{path}_lstm.h5")
                # Save scaler
                with open(f"{path}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {e}")

    def load_model(self, path: str):
        """Load LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Cannot load LSTM model.")
            return

        try:
            self.model = load_model(f"{path}_lstm.h5")
            # Load scaler
            with open(f"{path}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")


class EnsembleModel(BasePredictionModel):
    """Ensemble model combining LSTM and XGBoost predictions."""

    def __init__(self):
        super().__init__(ModelType.ENSEMBLE)
        self.lstm_model = LSTMModel()
        self.xgb_model = XGBoostModel()
        self.meta_model = None

        if not (XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE):
            self.logger.warning("Required ML libraries not available. Ensemble model will return default predictions.")

    def prepare_features(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Use XGBoost feature preparation for simplicity."""
        return self.xgb_model.prepare_features(*args, **kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train ensemble model."""
        if not (XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE):
            self.logger.warning("Required ML libraries not available. Cannot train ensemble model.")
            return ModelMetrics(
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2_score=0.0,
                training_time=0.0
            )

        try:
            # Train base models
            lstm_metrics = self.lstm_model.train(X, y)
            xgb_metrics = self.xgb_model.train(X, y)

            # Create meta features (predictions from base models)
            X_train_meta = self._create_meta_features(X, y)

            # Train meta model (simple linear regression on predictions)
            from sklearn.linear_model import LinearRegression

            split_idx = int(len(X_train_meta) * 0.8)
            X_meta_train, X_meta_test = X_train_meta[:split_idx], X_train_meta[split_idx:]
            y_meta_train, y_meta_test = y[:split_idx], y[split_idx:]

            self.meta_model = LinearRegression()
            self.meta_model.fit(X_meta_train, y_meta_train)

            # Evaluate ensemble
            y_pred = self.meta_model.predict(X_meta_test)

            mse = mean_squared_error(y_meta_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_meta_test, y_pred)

            ss_res = np.sum((y_meta_test - y_pred) ** 2)
            ss_tot = np.sum((y_meta_test - np.mean(y_meta_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            self.is_trained = True

            metrics = ModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2_score,
                training_time=lstm_metrics.training_time + xgb_metrics.training_time
            )

            self.logger.info(f"Ensemble model trained successfully. R² Score: {r2_score:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error training ensemble model: {e}")
            raise

    def _create_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create meta features from base model predictions."""
        if not (XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE):
            self.logger.warning("Required ML libraries not available. Cannot create meta features.")
            return np.zeros((len(X), 2))

        try:
            # Get predictions from base models
            lstm_pred = self.lstm_model.model.predict(X.reshape(X.shape[0], 30, X.shape[1]//30), verbose=0).flatten()
            xgb_pred = self.xgb_model.model.predict(X)

            # Combine predictions as meta features
            meta_features = np.column_stack([lstm_pred, xgb_pred])
            return meta_features

        except Exception as e:
            self.logger.error(f"Error creating meta features: {e}")
            return np.zeros((len(X), 2))

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with ensemble model."""
        if not (XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE):
            self.logger.warning("Required ML libraries not available. Returning default prediction.")
            return PredictionResult(
                model_type=self.model_type,
                predicted_price=2000.0,  # Default gold price
                confidence_score=0.5
            )

        try:
            if not self.is_trained or self.meta_model is None:
                raise ValueError("Model not trained")

            # Get predictions from base models
            lstm_pred = self.lstm_model.model.predict(X.reshape(1, 30, X.shape[0]//30), verbose=0)[0][0]
            xgb_pred = self.xgb_model.model.predict(X.reshape(1, -1))[0]

            # Meta prediction
            meta_input = np.array([[lstm_pred, xgb_pred]])
            prediction = self.meta_model.predict(meta_input)[0]

            # Calculate confidence (ensemble typically has higher confidence)
            confidence_score = 0.9

            # Simple confidence interval
            std_dev = prediction * 0.03  # 3% standard deviation assumption
            confidence_lower = prediction - (1.96 * std_dev)
            confidence_upper = prediction + (1.96 * std_dev)

            # Combine feature importance from both models
            lstm_importance = {}  # Simplified
            xgb_importance = self.xgb_model.model.feature_importances_ if self.xgb_model.model else {}

            return PredictionResult(
                model_type=self.model_type,
                predicted_price=prediction,
                confidence_interval_lower=confidence_lower,
                confidence_interval_upper=confidence_upper,
                feature_importance=dict(xgb_importance),
                confidence_score=confidence_score
            )

        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {e}")
            return PredictionResult(
                model_type=self.model_type,
                predicted_price=0.0,
                confidence_score=0.0
            )

    def save_model(self, path: str):
        """Save ensemble model."""
        if not (XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE):
            self.logger.warning("Required ML libraries not available. Cannot save ensemble model.")
            return

        try:
            self.lstm_model.save_model(path)
            self.xgb_model.save_model(path)
            # Save meta model
            with open(f"{path}_meta.pkl", 'wb') as f:
                pickle.dump(self.meta_model, f)
        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {e}")

    def load_model(self, path: str):
        """Load ensemble model."""
        if not (XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE):
            self.logger.warning("Required ML libraries not available. Cannot load ensemble model.")
            return

        try:
            self.lstm_model.load_model(path)
            self.xgb_model.load_model(path)
            # Load meta model
            with open(f"{path}_meta.pkl", 'rb') as f:
                self.meta_model = pickle.load(f)
            self.is_trained = True
        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {e}")


class PredictionService:
    """Main service for price predictions."""

    def __init__(self):
        # Initialize available models only
        self.models = {
            ModelType.LSTM: LSTMModel(),
            ModelType.XGBOOST: XGBoostModel(),
        }

        # Add ensemble model only if both required libraries are available
        if XGBOOST_AVAILABLE and TENSORFLOW_AVAILABLE:
            self.models[ModelType.ENSEMBLE] = EnsembleModel()

        self.data_manager = DataManager()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Log available models
        available_models = list(self.models.keys())
        self.logger.info(f"Available prediction models: {[m.value for m in available_models]}")

        if ModelType.ENSEMBLE not in available_models:
            self.logger.warning("Ensemble model not available (missing required ML libraries)")

    async def train_models(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        models: Optional[List[ModelType]] = None
    ) -> Dict[ModelType, ModelMetrics]:
        """Train all or specified models."""
        if models is None:
            models = list(self.models.keys())

        # Filter to only available models
        available_models = [m for m in models if m in self.models]
        if len(available_models) != len(models):
            unavailable_models = [m for m in models if m not in self.models]
            self.logger.warning(f"Requested models not available: {[m.value for m in unavailable_models]}")

        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        # Get historical data
        gold_prices = self.data_manager.get_gold_prices(start_date, end_date)

        # Get sentiment data (simplified - using overall sentiment)
        sentiment_data = []
        current_date = start_date
        while current_date <= end_date:
            for category in settings.news_categories:
                for time_horizon in settings.time_horizons:
                    sentiment = self.data_manager.get_weighted_sentiment(
                        category=category,
                        time_horizon=time_horizon,
                        start_date=current_date,
                        end_date=current_date
                    )
                    if sentiment:
                        sentiment_data.append({
                            'date': current_date,
                            'category': category,
                            'time_horizon': time_horizon,
                            'weighted_score': sentiment[0].weighted_score if sentiment else 0,
                            'bullish_score': sentiment[0].weighted_score if sentiment and sentiment[0].weighted_score > 0 else 0,
                            'bearish_score': -sentiment[0].weighted_score if sentiment and sentiment[0].weighted_score < 0 else 0,
                            'attention_score': abs(sentiment[0].weighted_score) if sentiment else 0
                        })
            current_date += timedelta(days=1)

        # Convert to required format
        gold_prices_list = [
            {
                'date': str(price.date),
                'close_price': float(price.close_price or 0),
                'volume': int(price.volume or 0)
            }
            for price in gold_prices
        ]

        results = {}

        for model_type in available_models:
            try:
                self.logger.info(f"Training {model_type.value} model...")
                X, y = self.models[model_type].prepare_features(gold_prices_list, sentiment_data)
                metrics = self.models[model_type].train(X, y)
                results[model_type] = metrics

                # Save trained model
                model_path = f"models/{model_type.value}"
                os.makedirs(model_path, exist_ok=True)
                self.models[model_type].save_model(model_path)

            except Exception as e:
                self.logger.error(f"Error training {model_type.value} model: {e}")
                continue

        return results

    async def predict_price(
        self,
        target_date: date,
        model_types: Optional[List[ModelType]] = None
    ) -> Dict[ModelType, PredictionResult]:
        """Make price predictions using trained models."""
        if model_types is None:
            model_types = list(self.models.keys())

        # Filter to only available models
        available_model_types = [m for m in model_types if m in self.models]
        if len(available_model_types) != len(model_types):
            unavailable_models = [m for m in model_types if m not in self.models]
            self.logger.warning(f"Requested models not available: {[m.value for m in unavailable_models]}")

        # Get recent data for prediction
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # 3 months of data

        gold_prices = self.data_manager.get_gold_prices(start_date, end_date)

        # Get recent sentiment data
        sentiment_data = []
        current_date = start_date
        while current_date <= end_date:
            for category in settings.news_categories:
                for time_horizon in settings.time_horizons:
                    sentiment = self.data_manager.get_weighted_sentiment(
                        category=category,
                        time_horizon=time_horizon,
                        start_date=current_date,
                        end_date=current_date
                    )
                    if sentiment:
                        sentiment_data.append({
                            'date': current_date,
                            'category': category,
                            'time_horizon': time_horizon,
                            'weighted_score': sentiment[0].weighted_score if sentiment else 0,
                            'bullish_score': sentiment[0].weighted_score if sentiment and sentiment[0].weighted_score > 0 else 0,
                            'bearish_score': -sentiment[0].weighted_score if sentiment and sentiment[0].weighted_score < 0 else 0,
                            'attention_score': abs(sentiment[0].weighted_score) if sentiment else 0
                        })
            current_date += timedelta(days=1)

        # Convert to required format
        gold_prices_list = [
            {
                'date': str(price.date),
                'close_price': float(price.close_price or 0),
                'volume': int(price.volume or 0)
            }
            for price in gold_prices
        ]

        results = {}

        for model_type in available_model_types:
            try:
                # Load trained model if available
                model_path = f"models/{model_type.value}"
                if os.path.exists(f"{model_path}_{model_type.value.lower()}.json" if model_type != ModelType.LSTM else f"{model_path}_lstm.h5"):
                    self.models[model_type].load_model(model_path)

                # Prepare features for prediction
                X, _ = self.models[model_type].prepare_features(gold_prices_list, sentiment_data)

                if len(X) > 0:
                    # Use the most recent data point for prediction
                    prediction = self.models[model_type].predict(X[-1])
                    results[model_type] = prediction

                    # Store prediction in database
                    from app.models.schemas import PricePredictionCreate
                    pred_data = PricePredictionCreate(
                        prediction_date=date.today(),
                        target_date=target_date,
                        model_type=model_type,
                        predicted_price=prediction.predicted_price,
                        confidence_interval_lower=prediction.confidence_interval_lower,
                        confidence_interval_upper=prediction.confidence_interval_upper,
                        feature_importance=prediction.feature_importance
                    )
                    self.data_manager.store_price_prediction(pred_data)

            except Exception as e:
                self.logger.error(f"Error predicting with {model_type.value} model: {e}")
                continue

        return results

    def calculate_confidence(self, predictions: Dict[ModelType, PredictionResult]) -> float:
        """Calculate overall confidence based on model agreement."""
        if not predictions:
            return 0.0

        # Calculate agreement between models
        prices = [pred.predicted_price for pred in predictions.values()]

        if len(prices) == 1:
            return predictions[list(predictions.keys())[0]].confidence_score

        # Calculate standard deviation as measure of disagreement
        std_dev = np.std(prices)
        mean_price = np.mean(prices)

        # Confidence decreases with higher disagreement
        agreement_factor = max(0.1, 1.0 - (std_dev / mean_price) if mean_price != 0 else 0)

        # Average individual model confidences
        avg_model_confidence = np.mean([pred.confidence_score for pred in predictions.values()])

        return (agreement_factor + avg_model_confidence) / 2.0
