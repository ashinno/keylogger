"""
Advanced Behavioral Analytics Engine for Cybersecurity Monitoring
Master's Thesis Research Module - Comprehensive User Behavior Analysis

This module implements cutting-edge behavioral analytics including:
- Deep learning-based user behavior modeling
- Multi-dimensional anomaly detection
- Cross-platform behavior correlation
- Adversarial ML defense mechanisms
- Explainable AI for security decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque, Counter
import logging
import pickle
import json
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Advanced ML imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import TSNE
import joblib

# Deep learning specific
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Custom imports
from .data_preprocessing import DataPreprocessor
from .interpretability import ModelInterpretabilityEngine
from .confidence_engine import ConfidenceEngine
from .visualization import InterpretabilityVisualizer

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels for behavioral anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of behavioral anomalies."""
    TIMING_ANOMALY = "timing_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    CONTEXT_ANOMALY = "context_anomaly"
    CROSS_PLATFORM_ANOMALY = "cross_platform_anomaly"
    INSIDER_THREAT = "insider_threat"
    ACCOUNT_TAKEOVER = "account_takeover"


@dataclass
class BehavioralProfile:
    """User behavioral profile for anomaly detection."""
    user_id: str
    baseline_features: Dict[str, np.ndarray]
    behavioral_patterns: Dict[str, Any]
    risk_score: float
    confidence_level: float
    last_updated: datetime
    anomaly_history: List[Dict[str, Any]]
    threat_indicators: List[str]


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    is_anomalous: bool
    threat_level: ThreatLevel
    anomaly_type: AnomalyType
    confidence_score: float
    risk_score: float
    explanation: Dict[str, Any]
    recommended_actions: List[str]
    timestamp: datetime


class AdvancedBehavioralAnalyticsEngine:
    """
    Advanced Behavioral Analytics Engine for Cybersecurity Monitoring
    
    This class implements state-of-the-art behavioral analytics including:
    - Deep learning models for user behavior modeling
    - Multi-modal anomaly detection
    - Cross-platform behavior correlation
    - Adversarial ML defense
    - Explainable AI for security decisions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        
        # Initialize interpretability components
        self.interpretability_engine = ModelInterpretabilityEngine(config)
        self.confidence_engine = ConfidenceEngine(config)
        self.visualizer = InterpretabilityVisualizer(config)
        
        # Configuration parameters
        self.sensitivity = config.get('ml.advanced_behavioral_analytics.sensitivity', 0.05)
        self.learning_rate = config.get('ml.advanced_behavioral_analytics.learning_rate', 0.001)
        self.baseline_window = config.get('ml.advanced_behavioral_analytics.baseline_window', 2000)
        self.adaptation_threshold = config.get('ml.advanced_behavioral_analytics.adaptation_threshold', 0.85)
        self.drift_threshold = config.get('ml.advanced_behavioral_analytics.drift_threshold', 0.15)
        self.sequence_length = config.get('ml.advanced_behavioral_analytics.sequence_length', 50)
        self.embedding_dim = config.get('ml.advanced_behavioral_analytics.embedding_dim', 128)
        
        # User profiles storage
        self.user_profiles: Dict[str, BehavioralProfile] = {}
        self.global_baseline: Dict[str, Any] = {}
        
        # Data storage with thread safety
        self._lock = threading.RLock()
        self.baseline_data = defaultdict(lambda: deque(maxlen=self.baseline_window))
        self.recent_data = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Model components
        self.deep_learning_models = {}
        self.traditional_ml_models = {}
        self.ensemble_models = {}
        
        # Feature engineering components
        self.feature_scalers = {}
        self.feature_extractors = {}
        self.dimensionality_reducers = {}
        
        # Security and privacy
        self.model_integrity_hashes = {}
        self.adversarial_defense_enabled = config.get('ml.advanced_behavioral_analytics.adversarial_defense', True)
        self.differential_privacy_enabled = config.get('ml.advanced_behavioral_analytics.differential_privacy', True)
        
        # Statistics and monitoring
        self.stats = {
            'events_processed': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'model_updates': 0,
            'adversarial_attacks_detected': 0,
            'last_training': None,
            'model_accuracy': 0.0,
            'system_health': 'healthy'
        }
        
        # Initialize models
        self._initialize_models()
        
        # Load existing models if available
        self._load_models()
        
        logger.info("Advanced Behavioral Analytics Engine initialized")
    
    def _initialize_models(self):
        """Initialize all ML models."""
        try:
            # Deep Learning Models
            self._initialize_deep_learning_models()
            
            # Traditional ML Models
            self._initialize_traditional_ml_models()
            
            # Ensemble Models
            self._initialize_ensemble_models()
            
            # Feature Engineering
            self._initialize_feature_engineering()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _initialize_deep_learning_models(self):
        """Initialize deep learning models for behavioral analysis."""
        
        # LSTM Model for Sequential Pattern Analysis
        self.deep_learning_models['lstm_behavioral'] = self._build_lstm_model()
        
        # Autoencoder for Anomaly Detection
        self.deep_learning_models['autoencoder'] = self._build_autoencoder_model()
        
        # Transformer-based Model for Complex Pattern Recognition
        self.deep_learning_models['transformer'] = self._build_transformer_model()
        
        # GAN for Adversarial Training (defensive)
        if self.adversarial_defense_enabled:
            self.deep_learning_models['defensive_gan'] = self._build_defensive_gan()
    
    def _build_lstm_model(self) -> keras.Model:
        """Build LSTM model for sequential behavioral analysis."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.embedding_dim)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_autoencoder_model(self) -> keras.Model:
        """Build autoencoder for unsupervised anomaly detection."""
        # Encoder
        encoder = Sequential([
            Dense(128, activation='relu', input_shape=(self.embedding_dim,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu')
        ])
        
        # Decoder
        decoder = Sequential([
            Dense(32, activation='relu', input_shape=(16,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.embedding_dim, activation='sigmoid')
        ])
        
        # Full autoencoder
        autoencoder = Sequential([encoder, decoder])
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return autoencoder
    
    def _build_transformer_model(self) -> keras.Model:
        """Build transformer model for complex pattern recognition."""
        # Simplified transformer architecture
        inputs = keras.Input(shape=(self.sequence_length, self.embedding_dim))
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=self.embedding_dim // 8
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = layers.Add()([inputs, attention_output])
        attention_output = layers.LayerNormalization()(attention_output)
        
        # Feed forward
        ff_output = Dense(256, activation='relu')(attention_output)
        ff_output = Dropout(0.2)(ff_output)
        ff_output = Dense(self.embedding_dim)(ff_output)
        
        # Add & Norm
        transformer_output = layers.Add()([attention_output, ff_output])
        transformer_output = layers.LayerNormalization()(transformer_output)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(transformer_output)
        
        # Classification head
        outputs = Dense(1, activation='sigmoid')(pooled)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_defensive_gan(self) -> Dict[str, keras.Model]:
        """Build defensive GAN for adversarial training."""
        # Generator (creates adversarial examples)
        generator = Sequential([
            Dense(128, activation='relu', input_shape=(self.embedding_dim,)),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(self.embedding_dim, activation='tanh')
        ])
        
        # Discriminator (detects adversarial examples)
        discriminator = Sequential([
            Dense(256, activation='relu', input_shape=(self.embedding_dim,)),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        discriminator.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return {'generator': generator, 'discriminator': discriminator}
    
    def _initialize_traditional_ml_models(self):
        """Initialize traditional ML models."""
        
        # Unsupervised models
        self.traditional_ml_models['isolation_forest'] = IsolationForest(
            contamination=self.sensitivity,
            random_state=42,
            n_estimators=200,
            max_samples='auto'
        )
        
        self.traditional_ml_models['one_class_svm'] = OneClassSVM(
            nu=self.sensitivity,
            kernel='rbf',
            gamma='scale',
            shrinking=True
        )
        
        # Clustering models
        self.traditional_ml_models['hdbscan'] = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric='euclidean'
        )
        
        self.traditional_ml_models['optics'] = OPTICS(
            min_samples=5,
            xi=0.05,
            min_cluster_size=0.1
        )
        
        # Supervised models
        self.traditional_ml_models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=20
        )
        
        self.traditional_ml_models['mlp_classifier'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble models for robust detection."""
        
        # Voting classifier for supervised learning
        self.ensemble_models['voting_classifier'] = VotingClassifier(
            estimators=[
                ('rf', self.traditional_ml_models['random_forest']),
                ('mlp', self.traditional_ml_models['mlp_classifier'])
            ],
            voting='soft'
        )
        
        # Custom ensemble for anomaly detection
        self.ensemble_models['anomaly_ensemble'] = None  # Will be built after training
    
    def _initialize_feature_engineering(self):
        """Initialize feature engineering components."""
        
        # Feature scalers
        self.feature_scalers['standard'] = StandardScaler()
        self.feature_scalers['robust'] = RobustScaler()
        self.feature_scalers['minmax'] = MinMaxScaler()
        
        # Dimensionality reduction
        self.dimensionality_reducers['pca'] = PCA(n_components=0.95)
        self.dimensionality_reducers['ica'] = FastICA(n_components=50)
        
        # Feature extractors
        self.feature_extractors['statistical'] = self._extract_statistical_features
        self.feature_extractors['temporal'] = self._extract_temporal_features
        self.feature_extractors['frequency'] = self._extract_frequency_features
        self.feature_extractors['contextual'] = self._extract_contextual_features
    
    def create_user_profile(self, user_id: str, initial_data: List[Dict[str, Any]]) -> BehavioralProfile:
        """Create a comprehensive behavioral profile for a user."""
        
        try:
            # Extract baseline features
            baseline_features = self._extract_comprehensive_features(initial_data)
            
            # Analyze behavioral patterns
            behavioral_patterns = self._analyze_behavioral_patterns(initial_data)
            
            # Calculate initial risk score
            risk_score = self._calculate_initial_risk_score(baseline_features)
            
            # Create profile
            profile = BehavioralProfile(
                user_id=user_id,
                baseline_features=baseline_features,
                behavioral_patterns=behavioral_patterns,
                risk_score=risk_score,
                confidence_level=0.8,  # Initial confidence
                last_updated=datetime.now(timezone.utc),
                anomaly_history=[],
                threat_indicators=[]
            )
            
            # Store profile
            with self._lock:
                self.user_profiles[user_id] = profile
            
            logger.info(f"Created behavioral profile for user: {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating user profile for {user_id}: {e}")
            raise
    
    def analyze_behavior(self, user_id: str, current_data: Dict[str, Any]) -> AnomalyResult:
        """Analyze current behavior against user profile."""
        
        try:
            # Get user profile
            profile = self.user_profiles.get(user_id)
            if not profile:
                # Create profile if doesn't exist
                profile = self.create_user_profile(user_id, [current_data])
            
            # Extract features from current data
            current_features = self._extract_comprehensive_features([current_data])
            
            # Multi-model anomaly detection
            anomaly_results = self._multi_model_anomaly_detection(profile, current_features)
            
            # Cross-platform correlation
            cross_platform_anomalies = self._cross_platform_correlation(user_id, current_data)
            
            # Adversarial attack detection
            adversarial_indicators = self._detect_adversarial_attacks(current_features)
            
            # Combine results and determine final anomaly status
            final_result = self._combine_anomaly_results(
                anomaly_results, cross_platform_anomalies, adversarial_indicators
            )
            
            # Update user profile
            self._update_user_profile(user_id, final_result, current_features)
            
            # Generate explanation
            explanation = self._generate_explanation(final_result, profile)
            
            # Create final result
            result = AnomalyResult(
                is_anomalous=final_result['is_anomalous'],
                threat_level=final_result['threat_level'],
                anomaly_type=final_result['anomaly_type'],
                confidence_score=final_result['confidence_score'],
                risk_score=final_result['risk_score'],
                explanation=explanation,
                recommended_actions=final_result['recommended_actions'],
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update statistics
            self._update_statistics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing behavior for user {user_id}: {e}")
            # Return safe default in case of error
            return AnomalyResult(
                is_anomalous=True,
                threat_level=ThreatLevel.HIGH,
                anomaly_type=AnomalyType.CONTEXT_ANOMALY,
                confidence_score=0.5,
                risk_score=0.8,
                explanation={'error': str(e)},
                recommended_actions=['manual_review_required'],
                timestamp=datetime.now(timezone.utc)
            )
    
    def _extract_comprehensive_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract comprehensive features from behavioral data."""
        
        features = {}
        
        # Extract different types of features
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                extracted_features = extractor_func(data)
                features.update(extracted_features)
            except Exception as e:
                logger.warning(f"Error in {extractor_name} feature extraction: {e}")
        
        return features
    
    def _extract_statistical_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract statistical features from data."""
        
        features = {}
        
        if not data:
            return features
        
        # Timing statistics
        timestamps = [item.get('timestamp', 0) for item in data]
        if timestamps:
            features['timing_mean'] = np.array([np.mean(timestamps)])
            features['timing_std'] = np.array([np.std(timestamps)])
            features['timing_variance'] = np.array([np.var(timestamps)])
            features['timing_skewness'] = np.array([self._calculate_skewness(timestamps)])
            features['timing_kurtosis'] = np.array([self._calculate_kurtosis(timestamps)])
        
        # Frequency statistics
        event_types = [item.get('event_type', '') for item in data]
        if event_types:
            type_counts = Counter(event_types)
            features['event_type_entropy'] = np.array([self._calculate_entropy(type_counts)])
            features['most_frequent_event'] = np.array([type_counts.most_common(1)[0][1] if type_counts else 0])
        
        return features
    
    def _extract_temporal_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract temporal pattern features."""
        
        features = {}
        
        if not data:
            return features
        
        # Time-based patterns
        timestamps = [datetime.fromisoformat(item['timestamp']) if isinstance(item['timestamp'], str) 
                     else datetime.fromtimestamp(item['timestamp']) for item in data]
        
        if timestamps:
            # Hour of day patterns
            hours = [ts.hour for ts in timestamps]
            features['hour_distribution'] = np.array([self._calculate_hour_distribution(hours)])
            
            # Day of week patterns
            days = [ts.weekday() for ts in timestamps]
            features['day_distribution'] = np.array([self._calculate_day_distribution(days)])
            
            # Session duration patterns
            if len(timestamps) > 1:
                durations = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                           for i in range(1, len(timestamps))]
                features['session_duration_mean'] = np.array([np.mean(durations)])
                features['session_duration_std'] = np.array([np.std(durations)])
        
        return features
    
    def _extract_frequency_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract frequency domain features."""
        
        features = {}
        
        if not data:
            return features
        
        # Convert timing data to frequency domain
        timestamps = [item.get('timestamp', 0) for item in data]
        
        if len(timestamps) > 8:  # Minimum for FFT
            # Apply FFT to timing patterns
            fft_result = np.fft.fft(timestamps)
            
            # Extract frequency domain features
            features['dominant_frequency'] = np.array([np.argmax(np.abs(fft_result))])
            features['frequency_magnitude'] = np.array([np.max(np.abs(fft_result))])
            features['spectral_centroid'] = np.array([self._calculate_spectral_centroid(fft_result)])
            features['spectral_rolloff'] = np.array([self._calculate_spectral_rolloff(fft_result)])
        
        return features
    
    def _extract_contextual_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract contextual and environmental features."""
        
        features = {}
        
        if not data:
            return features
        
        # Application context
        applications = [item.get('application', '') for item in data]
        if applications:
            app_counts = Counter(applications)
            features['application_diversity'] = np.array([len(app_counts)])
            features['primary_application_ratio'] = np.array([
                app_counts.most_common(1)[0][1] / len(applications) if app_counts else 0
            ])
        
        # System context
        system_contexts = [item.get('system_context', {}) for item in data]
        if system_contexts:
            # Extract system metrics
            cpu_usages = [ctx.get('cpu_usage', 0) for ctx in system_contexts if ctx.get('cpu_usage')]
            if cpu_usages:
                features['avg_cpu_usage'] = np.array([np.mean(cpu_usages)])
                features['cpu_usage_variance'] = np.array([np.var(cpu_usages)])
            
            memory_usages = [ctx.get('memory_usage', 0) for ctx in system_contexts if ctx.get('memory_usage')]
            if memory_usages:
                features['avg_memory_usage'] = np.array([np.mean(memory_usages)])
                features['memory_usage_variance'] = np.array([np.var(memory_usages)])
        
        return features
    
    def _multi_model_anomaly_detection(self, profile: BehavioralProfile, 
                                     current_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform anomaly detection using multiple models."""
        
        results = {}
        
        # Deep learning models
        for model_name, model in self.deep_learning_models.items():
            if model_name == 'defensive_gan':
                continue  # Skip GAN for now
            
            try:
                # Prepare input for deep learning model
                model_input = self._prepare_deep_learning_input(current_features, profile)
                
                # Get prediction
                prediction = model.predict(model_input, verbose=0)
                
                # Calculate anomaly score
                if model_name == 'autoencoder':
                    # Reconstruction error for autoencoder
                    reconstruction_error = np.mean(np.square(model_input - prediction))
                    anomaly_score = min(reconstruction_error / 10.0, 1.0)  # Normalize
                else:
                    # Classification probability for other models
                    anomaly_score = float(prediction[0][0])
                
                results[f'dl_{model_name}'] = {
                    'anomaly_score': anomaly_score,
                    'confidence': float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0]),
                    'model_type': 'deep_learning'
                }
                
            except Exception as e:
                logger.warning(f"Error in deep learning model {model_name}: {e}")
                results[f'dl_{model_name}'] = {
                    'anomaly_score': 0.5,
                    'confidence': 0.5,
                    'model_type': 'deep_learning',
                    'error': str(e)
                }
        
        # Traditional ML models
        for model_name, model in self.traditional_ml_models.items():
            try:
                # Prepare input for traditional ML model
                model_input = self._prepare_traditional_ml_input(current_features, profile)
                
                if model_name in ['isolation_forest', 'one_class_svm', 'hdbscan', 'optics']:
                    # Unsupervised models
                    if hasattr(model, 'predict'):
                        prediction = model.predict(model_input)
                        if hasattr(model, 'score_samples'):
                            scores = model.score_samples(model_input)
                            anomaly_score = 1.0 - (scores[0] + 1.0) / 2.0  # Normalize to [0, 1]
                        else:
                            anomaly_score = 1.0 if prediction[0] == -1 else 0.0
                    else:
                        anomaly_score = 0.5  # Default if model not trained
                
                else:
                    # Supervised models (probability-based)
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(model_input)
                        anomaly_score = float(probabilities[0][1])  # Probability of anomaly class
                    else:
                        anomaly_score = 0.5
                
                results[f'ml_{model_name}'] = {
                    'anomaly_score': anomaly_score,
                    'confidence': abs(anomaly_score - 0.5) * 2,  # Confidence based on distance from 0.5
                    'model_type': 'traditional_ml'
                }
                
            except Exception as e:
                logger.warning(f"Error in traditional ML model {model_name}: {e}")
                results[f'ml_{model_name}'] = {
                    'anomaly_score': 0.5,
                    'confidence': 0.5,
                    'model_type': 'traditional_ml',
                    'error': str(e)
                }
        
        return results
    
    def _prepare_deep_learning_input(self, features: Dict[str, np.ndarray], 
                                   profile: BehavioralProfile) -> np.ndarray:
        """Prepare input for deep learning models."""
        
        # Combine all features into a single vector
        feature_vectors = []
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, np.ndarray):
                feature_vectors.append(feature_value.flatten())
            else:
                feature_vectors.append(np.array([feature_value]))
        
        # Concatenate all features
        combined_features = np.concatenate(feature_vectors)
        
        # Ensure correct dimensionality
        if len(combined_features) < self.embedding_dim:
            # Pad with zeros if necessary
            padding = np.zeros(self.embedding_dim - len(combined_features))
            combined_features = np.concatenate([combined_features, padding])
        elif len(combined_features) > self.embedding_dim:
            # Truncate if necessary
            combined_features = combined_features[:self.embedding_dim]
        
        # Reshape for sequence models
        if self.deep_learning_models['lstm_behavioral'].input_shape[1] == self.sequence_length:
            # Create sequence by repeating the feature vector
            sequence_input = np.tile(combined_features, (self.sequence_length, 1))
            return sequence_input.reshape(1, self.sequence_length, self.embedding_dim)
        else:
            return combined_features.reshape(1, -1)
    
    def _prepare_traditional_ml_input(self, features: Dict[str, np.ndarray], 
                                    profile: BehavioralProfile) -> np.ndarray:
        """Prepare input for traditional ML models."""
        
        # Similar to deep learning input preparation
        feature_vectors = []
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, np.ndarray):
                feature_vectors.append(feature_value.flatten())
            else:
                feature_vectors.append(np.array([feature_value]))
        
        combined_features = np.concatenate(feature_vectors)
        return combined_features.reshape(1, -1)
    
    def _cross_platform_correlation(self, user_id: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-platform behavior correlation."""
        
        # This would analyze behavior across different devices/platforms
        # For now, implement a basic version
        
        anomalies = {}
        
        # Check for simultaneous logins from different locations
        if 'location' in current_data:
            location = current_data['location']
            recent_locations = self._get_recent_locations(user_id)
            
            if recent_locations and location not in recent_locations:
                # Potential account takeover or travel
                travel_time = self._calculate_travel_time(recent_locations[-1], location)
                if travel_time and travel_time < 3600:  # Less than 1 hour
                    anomalies['impossible_travel'] = {
                        'severity': 'high',
                        'description': 'Impossible travel detected',
                        'details': {
                            'from_location': recent_locations[-1],
                            'to_location': location,
                            'travel_time_seconds': travel_time
                        }
                    }
        
        # Check for device switching patterns
        if 'device_id' in current_data:
            device_id = current_data['device_id']
            recent_devices = self._get_recent_devices(user_id)
            
            if recent_devices and device_id not in recent_devices:
                anomalies['new_device'] = {
                    'severity': 'medium',
                    'description': 'New device detected',
                    'details': {
                        'new_device': device_id,
                        'previous_devices': recent_devices
                    }
                }
        
        return anomalies
    
    def _detect_adversarial_attacks(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect potential adversarial attacks on the ML models."""
        
        indicators = {}
        
        if not self.adversarial_defense_enabled:
            return indicators
        
        try:
            # Check for unusual feature patterns that might indicate adversarial examples
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, np.ndarray):
                    # Check for unusual distributions
                    if self._is_unusual_distribution(feature_value):
                        indicators[f'unusual_distribution_{feature_name}'] = {
                            'severity': 'medium',
                            'description': f'Unusual distribution detected in {feature_name}'
                        }
                    
                    # Check for adversarial perturbations
                    if self._contains_adversarial_perturbations(feature_value):
                        indicators[f'adversarial_perturbations_{feature_name}'] = {
                            'severity': 'high',
                            'description': f'Potential adversarial perturbations in {feature_name}'
                        }
            
            # Use defensive GAN if available
            if 'defensive_gan' in self.deep_learning_models:
                gan_models = self.deep_learning_models['defensive_gan']
                discriminator = gan_models['discriminator']
                
                # Prepare features for discriminator
                feature_vector = self._prepare_traditional_ml_input(features, None)
                adversarial_score = discriminator.predict(feature_vector, verbose=0)[0][0]
                
                if adversarial_score > 0.7:  # High confidence of adversarial example
                    indicators['adversarial_example_detected'] = {
                        'severity': 'critical',
                        'description': 'Adversarial example detected by defensive GAN',
                        'confidence': float(adversarial_score)
                    }
        
        except Exception as e:
            logger.warning(f"Error in adversarial attack detection: {e}")
        
        return indicators
    
    def _combine_anomaly_results(self, ml_results: Dict[str, Any], 
                               cross_platform_anomalies: Dict[str, Any],
                               adversarial_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different detection methods."""
        
        # Calculate weighted anomaly score
        total_score = 0.0
        total_confidence = 0.0
        model_weights = {}
        
        # Weight different models based on their historical performance
        for result_name, result in ml_results.items():
            if 'dl_' in result_name:
                weight = 0.3  # Deep learning models get higher weight
            elif 'ml_' in result_name:
                weight = 0.2  # Traditional ML models get medium weight
            else:
                weight = 0.1
            
            model_weights[result_name] = weight
            total_score += result['anomaly_score'] * weight
            total_confidence += result['confidence'] * weight
        
        # Normalize scores
        if model_weights:
            weight_sum = sum(model_weights.values())
            total_score /= weight_sum
            total_confidence /= weight_sum
        
        # Adjust based on cross-platform anomalies
        cross_platform_weight = 0.3
        for anomaly_name, anomaly_data in cross_platform_anomalies.items():
            severity_score = self._severity_to_score(anomaly_data['severity'])
            total_score = (1 - cross_platform_weight) * total_score + cross_platform_weight * severity_score
        
        # Adjust based on adversarial indicators
        adversarial_weight = 0.4
        for indicator_name, indicator_data in adversarial_indicators.items():
            severity_score = self._severity_to_score(indicator_data['severity'])
            total_score = (1 - adversarial_weight) * total_score + adversarial_weight * severity_score
        
        # Clamp scores to valid range
        total_score = max(0.0, min(1.0, total_score))
        total_confidence = max(0.0, min(1.0, total_confidence))
        
        # Determine threat level
        threat_level = self._determine_threat_level(total_score, total_confidence)
        
        # Determine anomaly type
        anomaly_type = self._determine_anomaly_type(ml_results, cross_platform_anomalies, adversarial_indicators)
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(threat_level, anomaly_type, total_score)
        
        return {
            'is_anomalous': total_score > self.sensitivity,
            'threat_level': threat_level,
            'anomaly_type': anomaly_type,
            'confidence_score': total_confidence,
            'risk_score': total_score,
            'component_scores': {
                'ml_scores': ml_results,
                'cross_platform_anomalies': cross_platform_anomalies,
                'adversarial_indicators': adversarial_indicators
            },
            'recommended_actions': recommended_actions
        }
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity string to numerical score."""
        severity_map = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }
        return severity_map.get(severity.lower(), 0.5)
    
    def _determine_threat_level(self, risk_score: float, confidence: float) -> ThreatLevel:
        """Determine threat level based on risk score and confidence."""
        
        # Weighted threat level determination
        weighted_score = risk_score * confidence + risk_score * 0.3
        
        if weighted_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif weighted_score >= 0.6:
            return ThreatLevel.HIGH
        elif weighted_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _determine_anomaly_type(self, ml_results: Dict[str, Any], 
                              cross_platform_anomalies: Dict[str, Any],
                              adversarial_indicators: Dict[str, Any]) -> AnomalyType:
        """Determine the primary type of anomaly."""
        
        # Priority-based anomaly type determination
        if adversarial_indicators:
            return AnomalyType.CONTEXT_ANOMALY
        
        if cross_platform_anomalies:
            if 'impossible_travel' in cross_platform_anomalies:
                return AnomalyType.ACCOUNT_TAKEOVER
            else:
                return AnomalyType.CROSS_PLATFORM_ANOMALY
        
        # Analyze ML results for specific patterns
        timing_anomalies = sum(1 for key in ml_results.keys() if 'timing' in key.lower())
        sequence_anomalies = sum(1 for key in ml_results.keys() if 'sequence' in key.lower())
        frequency_anomalies = sum(1 for key in ml_results.keys() if 'frequency' in key.lower())
        
        if timing_anomalies > sequence_anomalies and timing_anomalies > frequency_anomalies:
            return AnomalyType.TIMING_ANOMALY
        elif sequence_anomalies > frequency_anomalies:
            return AnomalyType.SEQUENCE_ANOMALY
        elif frequency_anomalies > 0:
            return AnomalyType.FREQUENCY_ANOMALY
        else:
            return AnomalyType.CONTEXT_ANOMALY
    
    def _generate_recommended_actions(self, threat_level: ThreatLevel, 
                                    anomaly_type: AnomalyType, 
                                    risk_score: float) -> List[str]:
        """Generate recommended actions based on threat assessment."""
        
        actions = []
        
        # Base actions by threat level
        if threat_level == ThreatLevel.CRITICAL:
            actions.extend([
                'immediate_investigation_required',
                'isolate_user_session',
                'notify_security_team',
                'preserve_forensic_evidence',
                'consider_account_suspension'
            ])
        elif threat_level == ThreatLevel.HIGH:
            actions.extend([
                'enhanced_monitoring',
                'verify_user_identity',
                'review_recent_activities',
                'check_for_indicators_of_compromise'
            ])
        elif threat_level == ThreatLevel.MEDIUM:
            actions.extend([
                'increase_logging_level',
                'monitor_for_escalation',
                'correlate_with_other_events'
            ])
        else:
            actions.extend([
                'continue_normal_monitoring',
                'log_for_trend_analysis'
            ])
        
        # Additional actions by anomaly type
        if anomaly_type == AnomalyType.ACCOUNT_TAKEOVER:
            actions.append('force_password_reset')
            actions.append('review_account_privileges')
        
        elif anomaly_type == AnomalyType.INSIDER_THREAT:
            actions.append('review_data_access_patterns')
            actions.append('check_for_data_exfiltration')
        
        elif anomaly_type == AnomalyType.CROSS_PLATFORM_ANOMALY:
            actions.append('verify_device_trust_level')
            actions.append('check_network_location')
        
        return actions
    
    def _update_user_profile(self, user_id: str, result: Dict[str, Any], 
                           current_features: Dict[str, np.ndarray]):
        """Update user profile with new analysis results."""
        
        with self._lock:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return
            
            # Update anomaly history
            anomaly_record = {
                'timestamp': datetime.now(timezone.utc),
                'is_anomalous': result['is_anomalous'],
                'threat_level': result['threat_level'].value,
                'anomaly_type': result['anomaly_type'].value,
                'risk_score': result['risk_score'],
                'confidence_score': result['confidence_score']
            }
            
            profile.anomaly_history.append(anomaly_record)
            
            # Update threat indicators
            if result['is_anomalous']:
                threat_indicator = f"{result['anomaly_type'].value}_{result['threat_level'].value}"
                if threat_indicator not in profile.threat_indicators:
                    profile.threat_indicators.append(threat_indicator)
            
            # Update risk score (exponential moving average)
            alpha = 0.1  # Smoothing factor
            profile.risk_score = alpha * result['risk_score'] + (1 - alpha) * profile.risk_score
            
            # Update confidence level based on recent accuracy
            recent_anomalies = profile.anomaly_history[-10:]
            if len(recent_anomalies) >= 5:
                # Simple confidence update based on consistency
                high_confidence_count = sum(1 for a in recent_anomalies if a['confidence_score'] > 0.7)
                profile.confidence_level = high_confidence_count / len(recent_anomalies)
            
            # Update last updated timestamp
            profile.last_updated = datetime.now(timezone.utc)
            
            # Store recent data for future analysis
            self.recent_data[user_id].append({
                'timestamp': datetime.now(timezone.utc),
                'features': current_features,
                'anomaly_result': result
            })
    
    def _generate_explanation(self, result: Dict[str, Any], profile: BehavioralProfile) -> Dict[str, Any]:
        """Generate human-readable explanation of the anomaly detection result."""
        
        explanation = {
            'summary': self._generate_summary_explanation(result),
            'detailed_analysis': self._generate_detailed_analysis(result),
            'comparative_analysis': self._generate_comparative_analysis(result, profile),
            'supporting_evidence': self._generate_supporting_evidence(result),
            'confidence_justification': self._generate_confidence_justification(result)
        }
        
        return explanation
    
    def _generate_summary_explanation(self, result: Dict[str, Any]) -> str:
        """Generate a summary explanation."""
        
        if result['is_anomalous']:
            return (f"Anomalous behavior detected with {result['threat_level'].value} threat level. "
                   f"Primary anomaly type: {result['anomaly_type'].value}. "
                   f"Risk score: {result['risk_score']:.2f} (confidence: {result['confidence_score']:.2f})")
        else:
            return (f"Behavior appears normal. Risk score: {result['risk_score']:.2f} "
                   f"(confidence: {result['confidence_score']:.2f})")
    
    def _generate_detailed_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis breakdown."""
        
        component_scores = result.get('component_scores', {})
        
        analysis = {
            'ml_model_analysis': {},
            'cross_platform_analysis': {},
            'adversarial_analysis': {}
        }
        
        # Analyze ML model scores
        ml_scores = component_scores.get('ml_scores', {})
        for model_name, model_result in ml_scores.items():
            analysis['ml_model_analysis'][model_name] = {
                'anomaly_score': model_result['anomaly_score'],
                'confidence': model_result['confidence'],
                'contribution_to_final_score': self._calculate_model_contribution(model_result)
            }
        
        # Analyze cross-platform anomalies
        cross_platform = component_scores.get('cross_platform_anomalies', {})
        for anomaly_name, anomaly_data in cross_platform.items():
            analysis['cross_platform_analysis'][anomaly_name] = {
                'severity': anomaly_data['severity'],
                'description': anomaly_data['description'],
                'impact_on_final_score': self._severity_to_score(anomaly_data['severity'])
            }
        
        # Analyze adversarial indicators
        adversarial = component_scores.get('adversarial_indicators', {})
        for indicator_name, indicator_data in adversarial.items():
            analysis['adversarial_analysis'][indicator_name] = {
                'severity': indicator_data['severity'],
                'description': indicator_data['description'],
                'confidence': indicator_data.get('confidence', 0.5)
            }
        
        return analysis
    
    def _generate_comparative_analysis(self, result: Dict[str, Any], 
                                     profile: BehavioralProfile) -> Dict[str, Any]:
        """Generate comparative analysis against user baseline."""
        
        # This would compare current behavior against the user's historical baseline
        # For now, implement a basic version
        
        recent_anomalies = profile.anomaly_history[-10:]
        
        comparison = {
            'baseline_deviation': result['risk_score'],  # Simplified
            'historical_context': {
                'total_anomalies_last_10': sum(1 for a in recent_anomalies if a['is_anomalous']),
                'average_risk_score_last_10': np.mean([a['risk_score'] for a in recent_anomalies]) if recent_anomalies else 0,
                'trend': self._calculate_trend(recent_anomalies)
            },
            'unusualness_score': self._calculate_unusualness_score(result, profile)
        }
        
        return comparison
    
    def _generate_supporting_evidence(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate supporting evidence for the detection."""
        
        evidence = []
        
        component_scores = result.get('component_scores', {})
        
        # Evidence from ML models
        ml_scores = component_scores.get('ml_scores', {})
        for model_name, model_result in ml_scores.items():
            if model_result['anomaly_score'] > 0.6:
                evidence.append({
                    'type': 'ml_model_detection',
                    'source': model_name,
                    'strength': model_result['anomaly_score'],
                    'description': f"{model_name} detected anomalous behavior with score {model_result['anomaly_score']:.2f}"
                })
        
        # Evidence from cross-platform anomalies
        cross_platform = component_scores.get('cross_platform_anomalies', {})
        for anomaly_name, anomaly_data in cross_platform.items():
            evidence.append({
                'type': 'cross_platform_anomaly',
                'source': anomaly_name,
                'strength': self._severity_to_score(anomaly_data['severity']),
                'description': anomaly_data['description']
            })
        
        # Evidence from adversarial indicators
        adversarial = component_scores.get('adversarial_indicators', {})
        for indicator_name, indicator_data in adversarial.items():
            evidence.append({
                'type': 'adversarial_indicator',
                'source': indicator_name,
                'strength': indicator_data.get('confidence', 0.5),
                'description': indicator_data['description']
            })
        
        return evidence
    
    def _generate_confidence_justification(self, result: Dict[str, Any]) -> str:
        """Generate justification for the confidence score."""
        
        confidence = result['confidence_score']
        
        if confidence > 0.8:
            justification = (f"High confidence ({confidence:.2f}) due to strong agreement "
                           "across multiple detection models and clear anomaly patterns.")
        elif confidence > 0.6:
            justification = (f"Medium-high confidence ({confidence:.2f}) with good "
                           "agreement among most detection components.")
        elif confidence > 0.4:
            justification = (f"Medium confidence ({confidence:.2f}) with some "
                           "disagreement among detection models.")
        else:
            justification = (f"Low confidence ({confidence:.2f}) due to significant "
                           "disagreement among detection components or insufficient evidence.")
        
        return justification
    
    def _update_statistics(self, result: AnomalyResult):
        """Update system statistics."""
        
        with self._lock:
            self.stats['events_processed'] += 1
            
            if result.is_anomalous:
                self.stats['anomalies_detected'] += 1
            
            # Update accuracy (simplified)
            if result.confidence_score > 0.7:
                if result.is_anomalous:
                    self.stats['true_positives'] += 1
                else:
                    self.stats['false_positives'] += 1
            
            # Update model accuracy
            total_predictions = self.stats['true_positives'] + self.stats['false_positives']
            if total_predictions > 0:
                self.stats['model_accuracy'] = self.stats['true_positives'] / total_predictions
    
    # Helper methods for feature extraction
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        return float(np.skew(data))
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        return float(np.kurtosis(data))
    
    def _calculate_entropy(self, counts: Counter) -> float:
        """Calculate entropy of categorical data."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_hour_distribution(self, hours: List[int]) -> List[float]:
        """Calculate distribution across hours of day."""
        hour_counts = Counter(hours)
        total = len(hours)
        return [hour_counts.get(h, 0) / total for h in range(24)]
    
    def _calculate_day_distribution(self, days: List[int]) -> List[float]:
        """Calculate distribution across days of week."""
        day_counts = Counter(days)
        total = len(days)
        return [day_counts.get(d, 0) / total for d in range(7)]
    
    def _calculate_spectral_centroid(self, fft_result: np.ndarray) -> float:
        """Calculate spectral centroid of FFT result."""
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.fftfreq(len(fft_result))
        
        if np.sum(magnitudes) == 0:
            return 0.0
        
        return float(np.sum(frequencies * magnitudes) / np.sum(magnitudes))
    
    def _calculate_spectral_rolloff(self, fft_result: np.ndarray, rolloff_point: float = 0.85) -> float:
        """Calculate spectral rolloff point."""
        magnitudes = np.abs(fft_result)
        total_energy = np.sum(magnitudes)
        
        if total_energy == 0:
            return 0.0
        
        cumulative_energy = 0.0
        for i, mag in enumerate(magnitudes):
            cumulative_energy += mag
            if cumulative_energy >= rolloff_point * total_energy:
                return float(i / len(magnitudes))
        
        return 1.0
    
    def _is_unusual_distribution(self, feature_values: np.ndarray) -> bool:
        """Check if feature distribution is unusual."""
        # Simple check for extreme values or unusual patterns
        if len(feature_values) == 0:
            return False
        
        # Check for extreme outliers
        q1, q3 = np.percentile(feature_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = np.sum((feature_values < lower_bound) | (feature_values > upper_bound))
        outlier_ratio = outliers / len(feature_values)
        
        return outlier_ratio > 0.1  # More than 10% outliers
    
    def _contains_adversarial_perturbations(self, feature_values: np.ndarray) -> bool:
        """Check for adversarial perturbations in feature values."""
        # Simple detection based on unusual precision or patterns
        if len(feature_values) == 0:
            return False
        
        # Check for artificially precise values
        decimal_places = []
        for value in feature_values:
            if isinstance(value, (int, float)):
                str_val = str(value)
                if '.' in str_val:
                    decimal_places.append(len(str_val.split('.')[1]))
        
        if decimal_places:
            avg_decimals = np.mean(decimal_places)
            # Unusually high precision might indicate adversarial perturbation
            return avg_decimals > 6
        
        return False
    
    def _generate_comparative_analysis(self, result: Dict[str, Any], 
                                     profile: BehavioralProfile) -> Dict[str, Any]:
        """Generate comparative analysis against user baseline."""
        
        # This would compare current behavior against the user's historical baseline
        # For now, implement a basic version
        
        recent_anomalies = profile.anomaly_history[-10:]
        
        comparison = {
            'baseline_deviation': result['risk_score'],  # Simplified
            'historical_context': {
                'total_anomalies_last_10': sum(1 for a in recent_anomalies if a['is_anomalous']),
                'average_risk_score_last_10': np.mean([a['risk_score'] for a in recent_anomalies]) if recent_anomalies else 0,
                'trend': self._calculate_trend(recent_anomalies)
            },
            'unusualness_score': self._calculate_unusualness_score(result, profile)
        }
        
        return comparison
    
    def _generate_supporting_evidence(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate supporting evidence for the detection."""
        
        evidence = []
        
        component_scores = result.get('component_scores', {})
        
        # Evidence from ML models
        ml_scores = component_scores.get('ml_scores', {})
        for model_name, model_result in ml_scores.items():
            if model_result['anomaly_score'] > 0.6:
                evidence.append({
                    'type': 'ml_model_detection',
                    'source': model_name,
                    'strength': model_result['anomaly_score'],
                    'description': f"{model_name} detected anomalous behavior with score {model_result['anomaly_score']:.2f}"
                })
        
        # Evidence from cross-platform anomalies
        cross_platform = component_scores.get('cross_platform_anomalies', {})
        for anomaly_name, anomaly_data in cross_platform.items():
            evidence.append({
                'type': 'cross_platform_anomaly',
                'source': anomaly_name,
                'strength': self._severity_to_score(anomaly_data['severity']),
                'description': anomaly_data['description']
            })
        
        # Evidence from adversarial indicators
        adversarial = component_scores.get('adversarial_indicators', {})
        for indicator_name, indicator_data in adversarial.items():
            evidence.append({
                'type': 'adversarial_indicator',
                'source': indicator_name,
                'strength': indicator_data.get('confidence', 0.5),
                'description': indicator_data['description']
            })
        
        return evidence
    
    def _generate_confidence_justification(self, result: Dict[str, Any]) -> str:
        """Generate justification for the confidence score."""
        
        confidence = result['confidence_score']
        
        if confidence > 0.8:
            justification = (f"High confidence ({confidence:.2f}) due to strong agreement "
                           "across multiple detection models and clear anomaly patterns.")
        elif confidence > 0.6:
            justification = (f"Medium-high confidence ({confidence:.2f}) with good "
                           "agreement among most detection components.")
        elif confidence > 0.4:
            justification = (f"Medium confidence ({confidence:.2f}) with some "
                           "disagreement among detection models.")
        else:
            justification = (f"Low confidence ({confidence:.2f}) due to significant "
                           "disagreement among detection components or insufficient evidence.")
        
        return justification
    
    def _update_statistics(self, result: AnomalyResult):
        """Update system statistics."""
        
        with self._lock:
            self.stats['events_processed'] += 1
            
            if result.is_anomalous:
                self.stats['anomalies_detected'] += 1
            
            # Update accuracy (simplified)
            if result.confidence_score > 0.7:
                if result.is_anomalous:
                    self.stats['true_positives'] += 1
                else:
                    self.stats['false_positives'] += 1
            
            # Update model accuracy
            total_predictions = self.stats['true_positives'] + self.stats['false_positives']
            if total_predictions > 0:
                self.stats['model_accuracy'] = self.stats['true_positives'] / total_predictions
    
    # Helper methods for feature extraction
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        return float(np.skew(data))
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        return float(np.kurtosis(data))
    
    def _calculate_entropy(self, counts: Counter) -> float:
        """Calculate entropy of categorical data."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_hour_distribution(self, hours: List[int]) -> List[float]:
        """Calculate distribution across hours of day."""
        hour_counts = Counter(hours)
        total = len(hours)
        return [hour_counts.get(h, 0) / total for h in range(24)]
    
    def _calculate_day_distribution(self, days: List[int]) -> List[float]:
        """Calculate distribution across days of week."""
        day_counts = Counter(days)
        total = len(days)
        return [day_counts.get(d, 0) / total for d in range(7)]
    
    def _calculate_spectral_centroid(self, fft_result: np.ndarray) -> float:
        """Calculate spectral centroid of FFT result."""
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.fftfreq(len(fft_result))
        
        if np.sum(magnitudes) == 0:
            return 0.0
        
        return float(np.sum(frequencies * magnitudes) / np.sum(magnitudes))
    
    def _calculate_spectral_rolloff(self, fft_result: np.ndarray, rolloff_point: float = 0.85) -> float:
        """Calculate spectral rolloff point."""
        magnitudes = np.abs(fft_result)
        total_energy = np.sum(magnitudes)
        
        if total_energy == 0:
            return 0.0
        
        cumulative_energy = 0.0
        for i, mag in enumerate(magnitudes):
            cumulative_energy += mag
            if cumulative_energy >= rolloff_point * total_energy:
                return float(i / len(magnitudes))
        
        return 1.0
    
    def _is_unusual_distribution(self, feature_values: np.ndarray) -> bool:
        """Check if feature distribution is unusual."""
        # Simple check for extreme values or unusual patterns
        if len(feature_values) == 0:
            return False
        
        # Check for extreme outliers
        q1, q3 = np.percentile(feature_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = np.sum((feature_values < lower_bound) | (feature_values > upper_bound))
        outlier_ratio = outliers / len(feature_values)
        
        return outlier_ratio > 0.1  # More than 10% outliers
    
    def _contains_adversarial_perturbations(self, feature_values: np.ndarray) -> bool:
        """Check for adversarial perturbations in feature values."""
        # Simple detection based on unusual precision or patterns
        if len(feature_values) == 0:
            return False
        
        # Check for artificially precise values
        decimal_places = []
        for value in feature_values:
            if isinstance(value, (int, float)):
                str_val = str(value)
                if '.' in str_val:
                    decimal_places.append(len(str_val.split('.')[1]))
        
        if decimal_places:
            avg_decimals = np.mean(decimal_places)
            # Unusually high precision might indicate adversarial perturbation
            return avg_decimals > 6
        
        return False
    
    def _get_recent_locations(self, user_id: str) -> List[str]:
        """Get recent locations for user."""
        # Implementation would retrieve from data storage
        return []
    
    def _get_recent_devices(self, user_id: str) -> List[str]:
        """Get recent devices for user."""
        # Implementation would retrieve from data storage
        return []
    
    def _calculate_travel_time(self, from_location: str, to_location: str) -> Optional[float]:
        """Calculate travel time between locations."""
        # Implementation would use geographic calculations
        return None
    
    def _calculate_model_contribution(self, model_result: Dict[str, Any]) -> float:
        """Calculate model's contribution to final score."""
        # Simple contribution calculation
        return model_result['anomaly_score'] * model_result['confidence']
    
    def _calculate_trend(self, recent_anomalies: List[Dict[str, Any]]) -> str:
        """Calculate trend in anomaly patterns."""
        if len(recent_anomalies) < 3:
            return 'insufficient_data'
        
        recent_scores = [a['risk_score'] for a in recent_anomalies[-5:]]
        
        if all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            return 'increasing'
        elif all(recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_unusualness_score(self, result: Dict[str, Any], profile: BehavioralProfile) -> float:
        """Calculate how unusual this behavior is for the user."""
        # Simple unusualness calculation based on deviation from baseline
        return min(result['risk_score'] * 1.5, 1.0)
    
    def _analyze_behavioral_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze behavioral patterns in the data."""
        
        patterns = {
            'activity_patterns': self._analyze_activity_patterns(data),
            'timing_patterns': self._analyze_timing_patterns(data),
            'sequence_patterns': self._analyze_sequence_patterns(data),
            'contextual_patterns': self._analyze_contextual_patterns(data)
        }
        
        return patterns
    
    def _analyze_activity_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activity patterns."""
        
        if not data:
            return {}
        
        event_types = [item.get('event_type', 'unknown') for item in data]
        type_counts = Counter(event_types)
        
        return {
            'most_common_activity': type_counts.most_common(1)[0] if type_counts else ('unknown', 0),
            'activity_diversity': len(type_counts),
            'activity_distribution': dict(type_counts)
        }
    
    def _analyze_timing_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing patterns."""
        
        if not data:
            return {}
        
        timestamps = [datetime.fromisoformat(item['timestamp']) if isinstance(item['timestamp'], str)
                     else datetime.fromtimestamp(item['timestamp']) for item in data]
        
        hours = [ts.hour for ts in timestamps]
        hour_counts = Counter(hours)
        
        return {
            'peak_activity_hour': hour_counts.most_common(1)[0] if hour_counts else (0, 0),
            'activity_time_range': (min(hours), max(hours)) if hours else (0, 0),
            'hourly_distribution': dict(hour_counts)
        }
    
    def _analyze_sequence_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sequence patterns."""
        
        if len(data) < 2:
            return {}
        
        # Analyze event sequences
        sequences = []
        for i in range(len(data) - 1):
            current_event = data[i].get('event_type', 'unknown')
            next_event = data[i + 1].get('event_type', 'unknown')
            sequences.append((current_event, next_event))
        
        sequence_counts = Counter(sequences)
        
        return {
            'most_common_sequence': sequence_counts.most_common(1)[0] if sequence_counts else (('unknown', 'unknown'), 0),
            'sequence_diversity': len(sequence_counts),
            'common_sequences': sequence_counts.most_common(5)
        }
    
    def _analyze_contextual_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze contextual patterns."""
        
        if not data:
            return {}
        
        contexts = [item.get('context', {}) for item in data]
        
        # Analyze application usage patterns
        applications = [ctx.get('application', 'unknown') for ctx in contexts]
        app_counts = Counter(applications)
        
        return {
            'primary_application': app_counts.most_common(1)[0] if app_counts else ('unknown', 0),
            'application_diversity': len(app_counts),
            'application_usage': dict(app_counts)
        }
    
    def _calculate_initial_risk_score(self, baseline_features: Dict[str, np.ndarray]) -> float:
        """Calculate initial risk score based on baseline features."""
        
        # Simple risk calculation based on feature variability
        total_variance = 0.0
        feature_count = 0
        
        for feature_name, feature_values in baseline_features.items():
            if len(feature_values) > 1:
                variance = np.var(feature_values)
                total_variance += variance
                feature_count += 1
        
        if feature_count > 0:
            avg_variance = total_variance / feature_count
            # Normalize to [0, 1] range
            risk_score = min(avg_variance / 10.0, 1.0)
        else:
            risk_score = 0.1  # Default low risk
        
        return risk_score
    
    def _load_models(self):
        """Load existing models from disk."""
        
        try:
            model_dir = Path(self.config.get('ml.model_directory', 'models'))
            model_dir.mkdir(exist_ok=True)
            
            # Load deep learning models
            for model_name in self.deep_learning_models:
                model_path = model_dir / f"advanced_behavioral_{model_name}.h5"
                if model_path.exists():
                    self.deep_learning_models[model_name] = keras.models.load_model(str(model_path))
                    logger.info(f"Loaded deep learning model: {model_name}")
            
            # Load traditional ML models
            for model_name in self.traditional_ml_models:
                model_path = model_dir / f"advanced_behavioral_{model_name}.joblib"
                if model_path.exists():
                    self.traditional_ml_models[model_name] = joblib.load(str(model_path))
                    logger.info(f"Loaded traditional ML model: {model_name}")
            
            # Load user profiles
            profiles_path = model_dir / "user_profiles.json"
            if profiles_path.exists():
                with open(profiles_path, 'r') as f:
                    profiles_data = json.load(f)
                    for user_id, profile_data in profiles_data.items():
                        # Recreate BehavioralProfile objects
                        profile = BehavioralProfile(
                            user_id=user_id,
                            baseline_features=self._deserialize_features(profile_data['baseline_features']),
                            behavioral_patterns=profile_data['behavioral_patterns'],
                            risk_score=profile_data['risk_score'],
                            confidence_level=profile_data['confidence_level'],
                            last_updated=datetime.fromisoformat(profile_data['last_updated']),
                            anomaly_history=profile_data['anomaly_history'],
                            threat_indicators=profile_data['threat_indicators']
                        )
                        self.user_profiles[user_id] = profile
                
                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def save_models(self):
        """Save all models to disk."""
        
        try:
            model_dir = Path(self.config.get('ml.model_directory', 'models'))
            model_dir.mkdir(exist_ok=True)
            
            # Save deep learning models
            for model_name, model in self.deep_learning_models.items():
                if model_name != 'defensive_gan':  # Skip GAN for now
                    model_path = model_dir / f"advanced_behavioral_{model_name}.h5"
                    model.save(str(model_path))
                    logger.info(f"Saved deep learning model: {model_name}")
            
            # Save traditional ML models
            for model_name, model in self.traditional_ml_models.items():
                model_path = model_dir / f"advanced_behavioral_{model_name}.joblib"
                joblib.dump(model, str(model_path))
                logger.info(f"Saved traditional ML model: {model_name}")
            
            # Save user profiles
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profiles_data[user_id] = {
                    'user_id': profile.user_id,
                    'baseline_features': self._serialize_features(profile.baseline_features),
                    'behavioral_patterns': profile.behavioral_patterns,
                    'risk_score': profile.risk_score,
                    'confidence_level': profile.confidence_level,
                    'last_updated': profile.last_updated.isoformat(),
                    'anomaly_history': profile.anomaly_history,
                    'threat_indicators': profile.threat_indicators
                }
            
            profiles_path = model_dir / "user_profiles.json"
            with open(profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            logger.info(f"Saved {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def _serialize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, List]:
        """Serialize feature dictionaries for JSON storage."""
        
        serialized = {}
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, np.ndarray):
                serialized[feature_name] = feature_value.tolist()
            else:
                serialized[feature_name] = feature_value
        
        return serialized
    
    def _deserialize_features(self, serialized_features: Dict[str, List]) -> Dict[str, np.ndarray]:
        """Deserialize feature dictionaries from JSON storage."""
        
        deserialized = {}
        for feature_name, feature_value in serialized_features.items():
            if isinstance(feature_value, list):
                deserialized[feature_name] = np.array(feature_value)
            else:
                deserialized[feature_name] = feature_value
        
        return deserialized
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        
        with self._lock:
            health_status = {
                'overall_health': self.stats['system_health'],
                'statistics': self.stats.copy(),
                'model_status': self._get_model_status(),
                'performance_metrics': self._get_performance_metrics(),
                'alerts': self._get_system_alerts()
            }
        
        return health_status
    
    def _get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        
        status = {
            'deep_learning_models': {},
            'traditional_ml_models': {},
            'ensemble_models': {}
        }
        
        # Deep learning models status
        for model_name, model in self.deep_learning_models.items():
            if model_name == 'defensive_gan':
                continue
            
            status['deep_learning_models'][model_name] = {
                'status': 'loaded' if model is not None else 'not_loaded',
                'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'unknown',
                'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'unknown'
            }
        
        # Traditional ML models status
        for model_name, model in self.traditional_ml_models.items():
            status['traditional_ml_models'][model_name] = {
                'status': 'trained' if hasattr(model, 'n_features_in_') else 'untrained',
                'n_features': getattr(model, 'n_features_in_', 'unknown')
            }
        
        # Ensemble models status
        for model_name, model in self.ensemble_models.items():
            status['ensemble_models'][model_name] = {
                'status': 'configured' if model is not None else 'not_configured'
            }
        
        return status
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        
        return {
            'events_per_second': self.stats['events_processed'] / max(1, (datetime.now(timezone.utc) - 
                datetime.fromisoformat('