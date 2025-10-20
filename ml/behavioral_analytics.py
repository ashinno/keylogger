"""Behavioral Analytics Engine for user behavior pattern analysis and anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import logging
import pickle
import json
from pathlib import Path

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from .data_preprocessing import DataPreprocessor
from .interpretability import ModelInterpretabilityEngine
from .confidence_engine import ConfidenceEngine
from .visualization import InterpretabilityVisualizer

logger = logging.getLogger(__name__)


class BehavioralAnalyticsEngine:
    """Advanced behavioral analytics engine with supervised and unsupervised ML algorithms."""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        
        # Initialize interpretability components
        self.interpretability_engine = ModelInterpretabilityEngine(config)
        self.confidence_engine = ConfidenceEngine(config)
        self.visualizer = InterpretabilityVisualizer(config)
        
        # Model configuration
        self.sensitivity = config.get('ml.behavioral_analytics.sensitivity', 0.1)
        self.learning_rate = config.get('ml.behavioral_analytics.learning_rate', 0.01)
        self.baseline_window = config.get('ml.behavioral_analytics.baseline_window', 1000)
        self.adaptation_threshold = config.get('ml.behavioral_analytics.adaptation_threshold', 0.8)
        
        # Models
        self.unsupervised_models = {
            'isolation_forest': IsolationForest(
                contamination=self.sensitivity,
                random_state=42,
                n_estimators=100
            ),
            'one_class_svm': OneClassSVM(
                nu=self.sensitivity,
                kernel='rbf',
                gamma='scale'
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5
            )
        }
        
        self.supervised_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Data storage
        self.baseline_data = deque(maxlen=self.baseline_window)
        self.recent_data = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=500)
        
        # Model state
        self.models_trained = False
        self.baseline_established = False
        self.feature_scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        # Baseline manager for drift detection and cold-start handling
        self._drift_threshold = float(self.config.get('ml.behavioral_analytics.drift_threshold', 0.15))
        self._baseline_window = int(self.config.get('ml.behavioral_analytics.baseline_window', 1000))
        self._baseline_stats = defaultdict(lambda: {'sum': 0.0, 'sq': 0.0})
        self._baseline_samples = 0
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'anomalies_detected': 0,
            'baseline_samples': 0,
            'model_updates': 0,
            'last_training': None
        }
        
        # Load existing models if available
        self._load_models()
        
        # Setup interpretability explainers after models are loaded
        self._setup_interpretability_explainers()
        
        logger.info("BehavioralAnalyticsEngine initialized")
    
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event and return anomaly analysis results."""
        try:
            # Extract features
            features = self._extract_event_features(event)
            if not features:
                return {'anomaly_score': 0.0, 'is_anomaly': False}
            # Cold-start baseline accumulation
            if not self.baseline_established:
                for k, v in features.items():
                    try:
                        val = float(v) if isinstance(v, (int, float)) else 0.0
                        self._baseline_stats[k]['sum'] += val
                        self._baseline_stats[k]['sq'] += val * val
                    except Exception:
                        continue
                self._baseline_samples = min(self._baseline_samples + 1, self._baseline_window)
                if self._baseline_samples >= max(10, self._baseline_window // 4):
                    self.baseline_established = True
                # Conservative anomaly during cold-start
                return {'anomaly_score': 0.0, 'is_anomaly': False}
            
            # Add to recent data
            self.recent_data.append({
                'timestamp': event.get('timestamp'),
                'features': features,
                'event': event
            })
            
            # Update statistics
            self.stats['events_processed'] += 1
            
            # Drift detection: simple z-score across baseline means/vars
            try:
                drift_score = 0.0
                checked = 0
                for k, v in features.items():
                    try:
                        x = float(v) if isinstance(v, (int, float)) else 0.0
                        mean = self._baseline_stats[k]['sum'] / max(1, self._baseline_samples)
                        sq = self._baseline_stats[k]['sq'] / max(1, self._baseline_samples)
                        var = max(0.0, sq - mean * mean)
                        if var > 1e-8:
                            z = abs((x - mean) / (var ** 0.5))
                            drift_score += min(1.0, z / 5.0)
                            checked += 1
                    except Exception:
                        continue
                if checked:
                    avg_drift = drift_score / checked
                    if avg_drift > self._drift_threshold:
                        # Refresh baseline to adapt
                        self._baseline_stats = defaultdict(lambda: {'sum': 0.0, 'sq': 0.0})
                        self._baseline_samples = 0
            except Exception:
                pass

            # Analyze anomaly with interpretability
            result = self._analyze_anomaly(features, event)
            
            # Generate explanation if anomaly detected or requested
            if result.get('is_anomaly') or self.config.get('ml.interpretability.always_explain', False):
                explanation = self._generate_prediction_explanation(features, event, result)
                result['explanation'] = explanation
            
            # Update confidence assessment
            confidence_assessment = self._assess_prediction_confidence(features, result)
            result['confidence_assessment'] = confidence_assessment
            
            # Update baseline if needed
            self._update_baseline(features, result)
            
            # Adaptive learning
            if self.stats['events_processed'] % 100 == 0:
                self._adaptive_learning()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing event in behavioral analytics: {e}")
            return {'anomaly_score': 0.0, 'is_anomaly': False, 'error': str(e)}
    
    def _extract_event_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral features from a single event."""
        features = {}
        
        try:
            # Temporal features with timezone handling
            timestamp_str = event.get('timestamp', datetime.now(timezone.utc).isoformat())
            if isinstance(timestamp_str, str):
                # Handle both timezone-aware and naive timestamps
                if '+' in timestamp_str or 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            features.update({
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_weekend': timestamp.weekday() >= 5,
                'is_business_hours': 9 <= timestamp.hour <= 17
            })
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
            current_time = datetime.now(timezone.utc)
            features.update({
                'hour': current_time.hour,
                'day_of_week': current_time.weekday(),
                'is_weekend': current_time.weekday() >= 5,
                'is_business_hours': 9 <= current_time.hour <= 17
            })
        
        # Event type features
        event_type = event.get('type', 'unknown')
        features[f'event_type_{event_type}'] = 1.0
        
        # Event-specific features
        if event_type == 'keyboard':
            features.update(self._extract_keyboard_features(event))
        elif event_type == 'mouse':
            features.update(self._extract_mouse_features(event))
        elif event_type == 'clipboard':
            features.update(self._extract_clipboard_features(event))
        elif event_type == 'window':
            features.update(self._extract_window_features(event))
        
        # Context features from recent history
        features.update(self._extract_context_features())
        
        return features
    
    def _extract_keyboard_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract keyboard-specific features."""
        features = {}
        data = event.get('data', {})
        
        # Key characteristics
        key = data.get('key', '')
        features.update({
            'key_length': len(key),
            'is_special_key': key in ['shift', 'ctrl', 'alt', 'tab', 'enter', 'backspace'],
            'is_function_key': key.startswith('f') and key[1:].isdigit(),
            'is_number': key.isdigit(),
            'is_letter': key.isalpha()
        })
        
        # Timing features (if available)
        if 'dwell_time' in data:
            features['dwell_time'] = data['dwell_time']
        if 'flight_time' in data:
            features['flight_time'] = data['flight_time']
        
        return features
    
    def _extract_mouse_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mouse-specific features."""
        features = {}
        data = event.get('data', {})
        
        # Mouse action
        action = data.get('action', '')
        features[f'mouse_{action}'] = 1.0
        
        # Position features
        x, y = data.get('x', 0), data.get('y', 0)
        features.update({
            'mouse_x': x,
            'mouse_y': y,
            'mouse_quadrant': self._get_screen_quadrant(x, y)
        })
        
        # Movement features (if available)
        if len(self.recent_data) > 0:
            last_event = self.recent_data[-1]
            if last_event['event'].get('type') == 'mouse':
                last_data = last_event['event'].get('data', {})
                last_x, last_y = last_data.get('x', 0), last_data.get('y', 0)
                
                distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                features['mouse_distance'] = distance
                
                # Time between mouse events
                current_time = datetime.fromisoformat(event.get('timestamp'))
                last_time = datetime.fromisoformat(last_event['event'].get('timestamp'))
                time_diff = (current_time - last_time).total_seconds()
                features['mouse_time_diff'] = time_diff
                
                if time_diff > 0:
                    features['mouse_velocity'] = distance / time_diff
        
        return features
    
    def _extract_clipboard_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clipboard-specific features."""
        features = {}
        data = event.get('data', {})
        
        content = data.get('content', '')
        # Handle None content and non-string types safely
        if content is None:
            content = ''
        elif not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = ''
        
        features.update({
            'clipboard_length': len(content),
            'clipboard_words': len(content.split()) if content else 0,
            'clipboard_lines': content.count('\n') if content else 0,
            'clipboard_has_url': 'http' in content.lower() if content else False,
            'clipboard_has_email': '@' in content and '.' in content if content else False,
            'clipboard_numeric_ratio': sum(c.isdigit() for c in content) / max(len(content), 1) if content else 0.0
        })
        
        return features
    
    def _extract_window_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract window-specific features."""
        features = {}
        data = event.get('data', {})
        
        app_name = data.get('application', '')
        window_title = data.get('title', '')
        
        features.update({
            'app_name_length': len(app_name),
            'window_title_length': len(window_title),
            'is_browser': any(browser in app_name.lower() 
                            for browser in ['chrome', 'firefox', 'edge', 'safari']),
            'is_office': any(office in app_name.lower() 
                           for office in ['word', 'excel', 'powerpoint', 'outlook']),
            'is_dev_tool': any(dev in app_name.lower() 
                             for dev in ['code', 'studio', 'terminal', 'cmd'])
        })
        
        return features
    
    def _extract_context_features(self) -> Dict[str, Any]:
        """Extract features from recent event context."""
        features = {}
        
        if len(self.recent_data) < 2:
            return features
        
        # Recent activity patterns
        recent_events = list(self.recent_data)[-10:]  # Last 10 events
        
        # Event type distribution
        event_types = [e['event'].get('type', 'unknown') for e in recent_events]
        type_counts = {t: event_types.count(t) for t in set(event_types)}
        
        for event_type, count in type_counts.items():
            features[f'recent_{event_type}_ratio'] = count / max(len(recent_events), 1)
        
        try:
            # Timing patterns with timezone handling
            timestamps = []
            for e in recent_events:
                timestamp_str = e['event'].get('timestamp')
                if timestamp_str:
                    try:
                        if isinstance(timestamp_str, str):
                            if '+' in timestamp_str or 'Z' in timestamp_str:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
                            timestamps.append(timestamp)
                    except Exception:
                        continue
            
            if len(timestamps) > 1:
                intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                            for i in range(1, len(timestamps))]
                
                features.update({
                    'avg_interval': np.mean(intervals),
                    'interval_variance': np.var(intervals),
                    'activity_burst': sum(1 for interval in intervals if interval < 0.1)
                })
        except Exception as e:
            logger.warning(f"Error extracting timing patterns: {e}")
        
        return features
    
    def _get_screen_quadrant(self, x: int, y: int) -> int:
        """Get screen quadrant (0-3) for mouse position."""
        # Assume 1920x1080 screen (configurable)
        screen_width = self.config.get('screen.width', 1920)
        screen_height = self.config.get('screen.height', 1080)
        
        quad_x = 0 if x < screen_width // 2 else 1
        quad_y = 0 if y < screen_height // 2 else 1
        
        return quad_y * 2 + quad_x
    
    def _analyze_anomaly(self, features: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze features for anomalies using trained models."""
        if not self.models_trained or not features:
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'model_ready': False
            }
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return {'anomaly_score': 0.0, 'is_anomaly': False}
            
            # Get predictions from all models
            predictions = {}
            
            # Unsupervised models
            for model_name, model in self.unsupervised_models.items():
                if hasattr(model, 'decision_function'):
                    score = model.decision_function([feature_vector])[0]
                    predictions[model_name] = {
                        'score': score,
                        'is_anomaly': score < 0
                    }
                elif hasattr(model, 'fit_predict'):
                    # For DBSCAN
                    prediction = model.fit_predict([feature_vector])[0]
                    predictions[model_name] = {
                        'score': -1.0 if prediction == -1 else 1.0,
                        'is_anomaly': prediction == -1
                    }
            
            # Supervised model (if trained with labeled data)
            if hasattr(self.supervised_model, 'predict_proba'):
                try:
                    proba = self.supervised_model.predict_proba([feature_vector])[0]
                    anomaly_proba = proba[1] if len(proba) > 1 else 0.0
                    predictions['supervised'] = {
                        'score': anomaly_proba,
                        'is_anomaly': anomaly_proba > 0.5
                    }
                except:
                    pass
            
            # Ensemble scoring
            ensemble_result = self._ensemble_predictions(predictions)
            
            # Update anomaly history
            if ensemble_result['is_anomaly']:
                self.anomaly_history.append({
                    'timestamp': event.get('timestamp'),
                    'score': ensemble_result['anomaly_score'],
                    'features': features,
                    'event_type': event.get('type')
                })
                self.stats['anomalies_detected'] += 1
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in anomaly analysis: {e}")
            return {'anomaly_score': 0.0, 'is_anomaly': False, 'error': str(e)}
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for model input."""
        try:
            # Get baseline feature names
            if not hasattr(self, 'feature_names'):
                return None
            
            # Create vector with all features
            vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0.0)
                # Handle non-numeric values
                if isinstance(value, bool):
                    value = float(value)
                elif not isinstance(value, (int, float)):
                    value = 0.0
                vector.append(value)
            
            # Scale features
            vector = np.array(vector).reshape(1, -1)
            scaled_vector = self.feature_scaler.transform(vector)
            
            # Apply PCA if fitted
            if hasattr(self.pca, 'components_'):
                scaled_vector = self.pca.transform(scaled_vector)
            
            return scaled_vector[0]
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None
    
    def _ensemble_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions from multiple models."""
        if not predictions:
            return {'anomaly_score': 0.0, 'is_anomaly': False, 'confidence': 0.0}
        
        # Weighted ensemble
        weights = {
            'isolation_forest': 0.3,
            'one_class_svm': 0.3,
            'dbscan': 0.2,
            'supervised': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        anomaly_votes = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.1)
            total_score += pred['score'] * weight
            total_weight += weight
            
            if pred['is_anomaly']:
                anomaly_votes += weight
        
        if total_weight > 0:
            ensemble_score = total_score / total_weight
            is_anomaly = anomaly_votes / total_weight > 0.5
            confidence = min(abs(ensemble_score), 1.0)
        else:
            ensemble_score = 0.0
            is_anomaly = False
            confidence = 0.0
        
        return {
            'anomaly_score': ensemble_score,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'model_predictions': predictions,
            'ensemble_votes': anomaly_votes / total_weight if total_weight > 0 else 0
        }
    
    def _update_baseline(self, features: Dict[str, Any], result: Dict[str, Any]):
        """Update baseline behavior model."""
        # Only add to baseline if not anomalous
        if not result.get('is_anomaly', False):
            self.baseline_data.append(features)
            self.stats['baseline_samples'] += 1
            
            # Retrain models periodically
            if len(self.baseline_data) >= 50 and len(self.baseline_data) % 50 == 0:
                self._train_models()
    
    def _train_models(self):
        """Train/retrain anomaly detection models."""
        if len(self.baseline_data) < 20:
            return
        
        try:
            # Prepare training data
            training_data = list(self.baseline_data)
            
            # Get all unique feature names
            all_features = set()
            for data in training_data:
                all_features.update(data.keys())
            
            self.feature_names = sorted(list(all_features))
            
            # Create feature matrix
            X = []
            for data in training_data:
                vector = []
                for feature_name in self.feature_names:
                    value = data.get(feature_name, 0.0)
                    if isinstance(value, bool):
                        value = float(value)
                    elif not isinstance(value, (int, float)):
                        value = 0.0
                    vector.append(value)
                X.append(vector)
            
            X = np.array(X)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit scaler and PCA
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Apply PCA if we have enough features
            if X_scaled.shape[1] > 10:
                X_scaled = self.pca.fit_transform(X_scaled)
            
            # Train unsupervised models
            for model_name, model in self.unsupervised_models.items():
                try:
                    if model_name == 'dbscan':
                        # DBSCAN doesn't need explicit fitting
                        continue
                    else:
                        model.fit(X_scaled)
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
            
            # Train supervised model if we have anomaly labels
            if len(self.anomaly_history) > 10:
                self._train_supervised_model(X_scaled)
            
            self.models_trained = True
            self.baseline_established = True
            self.stats['model_updates'] += 1
            self.stats['last_training'] = datetime.now().isoformat()
            
            logger.info(f"Models trained with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _train_supervised_model(self, X_baseline: np.ndarray):
        """Train supervised model using baseline and anomaly data."""
        try:
            # Ensure X_baseline is a 2D array
            if X_baseline.ndim == 1:
                X_baseline = X_baseline.reshape(1, -1)
            
            # Create labels: 0 for normal, 1 for anomaly
            y_baseline = np.zeros(X_baseline.shape[0])
            
            # Get anomaly features
            anomaly_features = []
            for anomaly in self.anomaly_history:
                features = anomaly['features']
                vector = []
                for feature_name in self.feature_names:
                    value = features.get(feature_name, 0.0)
                    if isinstance(value, bool):
                        value = float(value)
                    elif not isinstance(value, (int, float)):
                        value = 0.0
                    vector.append(value)
                anomaly_features.append(vector)
            
            if anomaly_features:
                X_anomaly = np.array(anomaly_features)
                X_anomaly = np.nan_to_num(X_anomaly, nan=0.0, posinf=1e6, neginf=-1e6)
                X_anomaly_scaled = self.feature_scaler.transform(X_anomaly)
                
                if hasattr(self.pca, 'components_'):
                    X_anomaly_scaled = self.pca.transform(X_anomaly_scaled)
                
                y_anomaly = np.ones(len(X_anomaly))
                
                # Combine data
                X_combined = np.vstack([X_baseline, X_anomaly_scaled])
                y_combined = np.hstack([y_baseline, y_anomaly])
                
                # Train supervised model
                self.supervised_model.fit(X_combined, y_combined)
                
                logger.info(f"Supervised model trained with {len(X_combined)} samples")
            
        except Exception as e:
            logger.error(f"Error training supervised model: {e}")
    
    def _adaptive_learning(self):
        """Implement adaptive learning to adjust to user behavior changes."""
        if not self.baseline_established or len(self.recent_data) < 50:
            return
        
        try:
            # Analyze recent behavior patterns
            recent_features = [data['features'] for data in list(self.recent_data)[-50:]]
            
            # Calculate drift from baseline
            drift_score = self._calculate_behavior_drift(recent_features)
            
            # If significant drift detected, update baseline
            if drift_score > self.adaptation_threshold:
                logger.info(f"Behavior drift detected (score: {drift_score:.3f}), updating baseline")
                
                # Add recent normal behavior to baseline
                for data in list(self.recent_data)[-25:]:  # Add last 25 normal events
                    if not self._is_likely_anomaly(data['features']):
                        self.baseline_data.append(data['features'])
                
                # Retrain models
                self._train_models()
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
    
    def _calculate_behavior_drift(self, recent_features: List[Dict[str, Any]]) -> float:
        """Calculate behavioral drift score."""
        if not self.baseline_data or not recent_features:
            return 0.0
        
        try:
            # Calculate feature distributions for baseline and recent data
            baseline_features = list(self.baseline_data)[-200:]  # Last 200 baseline samples
            
            # Get common features
            common_features = set()
            for features in baseline_features + recent_features:
                common_features.update(features.keys())
            
            drift_scores = []
            
            for feature in common_features:
                # Get feature values
                baseline_values = [f.get(feature, 0.0) for f in baseline_features]
                recent_values = [f.get(feature, 0.0) for f in recent_features]
                
                # Convert to numeric
                baseline_values = [float(v) if isinstance(v, (int, float, bool)) else 0.0 
                                 for v in baseline_values]
                recent_values = [float(v) if isinstance(v, (int, float, bool)) else 0.0 
                               for v in recent_values]
                
                # Calculate statistical difference
                if baseline_values and recent_values:
                    baseline_mean = np.mean(baseline_values)
                    recent_mean = np.mean(recent_values)
                    baseline_std = np.std(baseline_values) + 1e-6  # Avoid division by zero
                    
                    # Normalized difference
                    drift = abs(recent_mean - baseline_mean) / baseline_std
                    drift_scores.append(drift)
            
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating behavior drift: {e}")
            return 0.0
    
    def _is_likely_anomaly(self, features: Dict[str, Any]) -> bool:
        """Quick check if features represent likely anomaly."""
        if not self.models_trained:
            return False
        
        try:
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return False
            
            # Use isolation forest for quick check
            if 'isolation_forest' in self.unsupervised_models:
                score = self.unsupervised_models['isolation_forest'].decision_function([feature_vector])[0]
                return score < 0
            
            return False
            
        except Exception:
            return False
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            models_dir.mkdir(exist_ok=True)
            
            # Save models
            model_data = {
                'unsupervised_models': self.unsupervised_models,
                'supervised_model': self.supervised_model,
                'feature_scaler': self.feature_scaler,
                'pca': self.pca,
                'feature_names': getattr(self, 'feature_names', []),
                'models_trained': self.models_trained,
                'baseline_established': self.baseline_established
            }
            
            with open(models_dir / 'behavioral_analytics.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Behavioral analytics models saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            model_file = models_dir / 'behavioral_analytics.pkl'
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.unsupervised_models = model_data.get('unsupervised_models', self.unsupervised_models)
                self.supervised_model = model_data.get('supervised_model', self.supervised_model)
                self.feature_scaler = model_data.get('feature_scaler', self.feature_scaler)
                self.pca = model_data.get('pca', self.pca)
                self.feature_names = model_data.get('feature_names', [])
                self.models_trained = model_data.get('models_trained', False)
                self.baseline_established = model_data.get('baseline_established', False)
                
                logger.info("Behavioral analytics models loaded")
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get summary of established baseline behavior."""
        if not self.baseline_established:
            return {'status': 'not_established'}
        
        baseline_features = list(self.baseline_data)
        
        # Calculate feature statistics
        feature_stats = {}
        all_features = set()
        for features in baseline_features:
            all_features.update(features.keys())
        
        for feature in all_features:
            values = [f.get(feature, 0.0) for f in baseline_features]
            numeric_values = [float(v) if isinstance(v, (int, float, bool)) else 0.0 for v in values]
            
            feature_stats[feature] = {
                'mean': np.mean(numeric_values),
                'std': np.std(numeric_values),
                'min': np.min(numeric_values),
                'max': np.max(numeric_values)
            }
        
        return {
            'status': 'established',
            'samples': len(baseline_features),
            'features': len(all_features),
            'feature_stats': feature_stats,
            'last_update': self.stats.get('last_training')
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get behavioral analytics statistics."""
        stats = self.stats.copy()
        stats.update({
            'baseline_samples': len(self.baseline_data),
            'recent_samples': len(self.recent_data),
            'anomaly_history': len(self.anomaly_history),
            'models_trained': self.models_trained,
            'baseline_established': self.baseline_established,
            'sensitivity': self.sensitivity,
            'adaptation_threshold': self.adaptation_threshold
        })
        
        # Recent anomaly rate
        if len(self.recent_data) > 0:
            recent_anomalies = sum(1 for data in self.recent_data 
                                 if self._is_likely_anomaly(data['features']))
            stats['recent_anomaly_rate'] = recent_anomalies / max(len(self.recent_data), 1)
        
        return stats
    
    def reset_baseline(self):
        """Reset baseline behavior model."""
        self.baseline_data.clear()
        self.anomaly_history.clear()
        self.models_trained = False
        self.baseline_established = False
        self.stats['baseline_samples'] = 0
        self.stats['model_updates'] = 0
        
        logger.info("Baseline behavior model reset")
    
    def __del__(self):
        """Save models on destruction."""
        if self.models_trained:
            self._save_models()
    
    def _setup_interpretability_explainers(self):
        """Setup interpretability explainers for trained models."""
        try:
            if not self.models_trained or not self.baseline_data:
                return
            
            # Prepare training data for explainer setup
            training_data = list(self.baseline_data)
            if len(training_data) < 10:
                return
            
            # Get feature names
            if hasattr(self, 'feature_names') and self.feature_names:
                feature_names = self.feature_names
            else:
                # Generate default feature names
                sample_features = training_data[0] if training_data else {}
                feature_names = sorted(list(sample_features.keys()))
            
            # Create feature matrix
            X_train = []
            for data in training_data:
                vector = []
                for feature_name in feature_names:
                    value = data.get(feature_name, 0.0)
                    if isinstance(value, bool):
                        value = float(value)
                    elif isinstance(value, str):
                        value = hash(value) % 1000 / 1000.0  # Simple string hashing
                    vector.append(float(value))
                X_train.append(vector)
            
            X_train = np.array(X_train)
            
            # Setup explainers for each model
            for model_name, model in self.unsupervised_models.items():
                if hasattr(model, 'predict'):
                    try:
                        explainer_info = self.interpretability_engine.setup_explainers(
                            model, X_train, feature_names, 'classifier'
                        )
                        logger.info(f"Setup explainers for {model_name}: {explainer_info}")
                    except Exception as e:
                        logger.warning(f"Failed to setup explainers for {model_name}: {e}")
            
            # Setup for supervised model if available
            if hasattr(self.supervised_model, 'predict'):
                try:
                    explainer_info = self.interpretability_engine.setup_explainers(
                        self.supervised_model, X_train, feature_names, 'classifier'
                    )
                    logger.info(f"Setup explainers for supervised model: {explainer_info}")
                except Exception as e:
                    logger.warning(f"Failed to setup explainers for supervised model: {e}")
            
        except Exception as e:
            logger.error(f"Error setting up interpretability explainers: {e}")
    
    def _generate_prediction_explanation(self, features: Dict[str, Any], 
                                       event: Dict[str, Any], 
                                       result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanation for the prediction."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return {'status': 'error', 'message': 'Could not prepare feature vector'}
            
            # Choose the best model for explanation (highest confidence)
            best_model = None
            best_score = 0
            
            # Check ensemble predictions for best model
            ensemble_predictions = result.get('ensemble_predictions', {})
            for model_name, prediction in ensemble_predictions.items():
                if prediction.get('score', 0) > best_score:
                    best_score = prediction['score']
                    if model_name in self.unsupervised_models:
                        best_model = self.unsupervised_models[model_name]
                    elif model_name == 'supervised' and hasattr(self.supervised_model, 'predict'):
                        best_model = self.supervised_model
            
            # Fallback to isolation forest if no best model found
            if best_model is None:
                best_model = self.unsupervised_models.get('isolation_forest')
            
            if best_model is None:
                return {'status': 'error', 'message': 'No suitable model for explanation'}
            
            # Generate explanation using interpretability engine
            explanation = self.interpretability_engine.explain_prediction(
                best_model, 
                feature_vector,
                explanation_types=['shap', 'lime', 'feature_importance', 'uncertainty']
            )
            
            # Add behavioral context
            explanation['behavioral_context'] = {
                'event_type': event.get('type'),
                'timestamp': event.get('timestamp'),
                'anomaly_score': result.get('anomaly_score', 0.0),
                'ensemble_agreement': self._calculate_ensemble_agreement(ensemble_predictions)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _assess_prediction_confidence(self, features: Dict[str, Any], 
                                    result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence for the current prediction."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return {'status': 'error', 'message': 'Could not prepare feature vector'}
            
            # Get ensemble predictions for uncertainty estimation
            ensemble_predictions = result.get('ensemble_predictions', {})
            ensemble_scores = [pred.get('score', 0) for pred in ensemble_predictions.values()]
            
            # Use the best performing model for confidence assessment
            best_model = None
            best_score = 0
            
            for model_name, prediction in ensemble_predictions.items():
                if prediction.get('score', 0) > best_score:
                    best_score = prediction['score']
                    if model_name in self.unsupervised_models:
                        best_model = self.unsupervised_models[model_name]
                    elif model_name == 'supervised' and hasattr(self.supervised_model, 'predict'):
                        best_model = self.supervised_model
            
            if best_model is None:
                best_model = self.unsupervised_models.get('isolation_forest')
            
            if best_model is None:
                return {'status': 'error', 'message': 'No suitable model for confidence assessment'}
            
            # Generate confidence assessment
            confidence_assessment = self.confidence_engine.assess_prediction_confidence(
                best_model,
                feature_vector,
                model_id=str(id(best_model)),
                ensemble_predictions=ensemble_scores
            )
            
            return confidence_assessment
            
        except Exception as e:
            logger.error(f"Error assessing prediction confidence: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_ensemble_agreement(self, ensemble_predictions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate agreement between ensemble model predictions."""
        try:
            if len(ensemble_predictions) < 2:
                return 1.0  # Perfect agreement if only one model
            
            # Get binary predictions (anomaly or not)
            predictions = []
            for pred in ensemble_predictions.values():
                predictions.append(1 if pred.get('is_anomaly', False) else 0)
            
            # Calculate agreement as ratio of models that agree with majority
            majority_vote = 1 if sum(predictions) > len(predictions) / 2 else 0
            agreement_count = sum(1 for pred in predictions if pred == majority_vote)
            
            return agreement_count / max(len(predictions), 1)
            
        except Exception as e:
            logger.error(f"Error calculating ensemble agreement: {e}")
            return 0.0
    
    def get_interpretability_summary(self) -> Dict[str, Any]:
        """Get comprehensive interpretability summary."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'interpretability_engine': self.interpretability_engine.get_explanation_summary(),
                'confidence_engine': self.confidence_engine.get_confidence_summary(),
                'model_status': {
                    'models_trained': self.models_trained,
                    'baseline_established': self.baseline_established,
                    'baseline_size': len(self.baseline_data),
                    'feature_count': len(self.feature_names) if hasattr(self, 'feature_names') else 0
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating interpretability summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_visualization(self, explanation_data: Dict[str, Any], 
                             visualization_type: str = 'feature_importance') -> Dict[str, Any]:
        """Generate visualization for explanation data."""
        try:
            if visualization_type == 'feature_importance':
                return self.visualizer.create_feature_importance_plot(explanation_data)
            elif visualization_type == 'confidence':
                return self.visualizer.create_confidence_indicator(explanation_data)
            elif visualization_type == 'decision_path':
                return self.visualizer.create_decision_path_visualization(explanation_data)
            elif visualization_type == 'shap_waterfall':
                return self.visualizer.create_shap_waterfall_plot(explanation_data)
            elif visualization_type == 'uncertainty':
                return self.visualizer.create_uncertainty_visualization(explanation_data)
            else:
                return {'status': 'error', 'message': f'Unknown visualization type: {visualization_type}'}
                
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def calibrate_models(self, X_cal: np.ndarray, y_cal: np.ndarray) -> Dict[str, Any]:
        """Calibrate models for better confidence estimation."""
        try:
            calibration_results = {}
            
            # Calibrate each model that supports probability prediction
            for model_name, model in self.unsupervised_models.items():
                if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                    try:
                        result = self.confidence_engine.calibrate_model(
                            model, X_cal, y_cal, model_id=f"{model_name}_{id(model)}"
                        )
                        calibration_results[model_name] = result
                    except Exception as e:
                        logger.warning(f"Failed to calibrate {model_name}: {e}")
                        calibration_results[model_name] = {'status': 'error', 'message': str(e)}
            
            # Calibrate supervised model if available
            if hasattr(self.supervised_model, 'predict_proba'):
                try:
                    result = self.confidence_engine.calibrate_model(
                        self.supervised_model, X_cal, y_cal, 
                        model_id=f"supervised_{id(self.supervised_model)}"
                    )
                    calibration_results['supervised'] = result
                except Exception as e:
                    logger.warning(f"Failed to calibrate supervised model: {e}")
                    calibration_results['supervised'] = {'status': 'error', 'message': str(e)}
            
            return {
                'status': 'success',
                'calibration_results': calibration_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calibrating models: {e}")
            return {'status': 'error', 'message': str(e)}