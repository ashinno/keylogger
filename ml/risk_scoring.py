"""Real-time Risk Scoring system with dynamic scoring and automated alerting.

Enhanced to support baseline drift detection, micro-batching, caching,
and unified score calibration with confidence metadata.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import logging
import pickle
import json
import time
import threading
from pathlib import Path

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from scipy import stats
from scipy.special import expit

from .data_preprocessing import DataPreprocessor
from .confidence_engine import ConfidenceEngine

logger = logging.getLogger(__name__)


class _ScoreCalibrator:
    """Lightweight score calibration to map arbitrary scores to [0,1]."""
    def __init__(self):
        self._min = 0.0
        self._max = 1.0

    def fit(self, values: List[float]) -> None:
        if not values:
            return
        self._min = float(min(values))
        self._max = float(max(values))
        if self._max <= self._min:
            self._max = self._min + 1e-6

    def to_unit_interval(self, x: float) -> float:
        try:
            # First, bound to [min,max]
            x = max(self._min, min(self._max, float(x)))
            # Min-max scale
            scaled = (x - self._min) / (self._max - self._min)
            # Smooth with sigmoid to avoid extremes
            return float(expit(6.0 * (scaled - 0.5)))
        except Exception:
            return float(expit(x))


class _BaselineManager:
    """Maintain rolling baseline statistics and detect drift."""
    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.15):
        self.window_size = int(max(10, window_size))
        self.drift_threshold = float(drift_threshold)
        self._feature_sums: Dict[str, float] = defaultdict(float)
        self._feature_sq_sums: Dict[str, float] = defaultdict(float)
        self._samples: deque = deque(maxlen=self.window_size)
        self.samples_collected = 0

    def add_baseline_sample(self, features: Dict[str, Any]) -> None:
        self._samples.append(features)
        self.samples_collected = min(self.samples_collected + 1, self.window_size)
        for k, v in features.items():
            try:
                val = float(v) if isinstance(v, (int, float)) else 0.0
                self._feature_sums[k] += val
                self._feature_sq_sums[k] += val * val
            except Exception:
                continue

    def _mean_var(self, k: str) -> Tuple[float, float]:
        n = max(1, self.samples_collected)
        mean = self._feature_sums.get(k, 0.0) / n
        sq = self._feature_sq_sums.get(k, 0.0) / n
        var = max(0.0, sq - mean * mean)
        return mean, var

    def detect_drift(self, features: Dict[str, Any]) -> bool:
        if self.samples_collected < max(10, self.window_size // 10):
            return False
        drift_score = 0.0
        checked = 0
        for k, v in features.items():
            try:
                x = float(v) if isinstance(v, (int, float)) else 0.0
                mean, var = self._mean_var(k)
                if var <= 1e-8:
                    continue
                # Z-score magnitude as simple drift proxy
                z = abs((x - mean) / (var ** 0.5))
                drift_score += min(1.0, z / 5.0)
                checked += 1
            except Exception:
                continue
        if checked == 0:
            return False
        avg_drift = drift_score / checked
        return avg_drift > self.drift_threshold

    def refresh_baseline(self) -> None:
        # Reset accumulators while keeping the latest samples to re-establish
        recent = list(self._samples)[-self.window_size // 5:]
        self._feature_sums.clear()
        self._feature_sq_sums.clear()
        self._samples.clear()
        self.samples_collected = 0
        for f in recent:
            self.add_baseline_sample(f)


class _MicroBatcher:
    """Accumulate events and release micro-batches under latency and size constraints."""
    def __init__(self, micro_batch_size: int = 32, max_latency_ms: int = 200):
        self.size = int(max(1, micro_batch_size))
        self.latency_ms = int(max(10, max_latency_ms))
        self._buffer: List[Dict[str, Any]] = []
        self._last_ts = time.time()

    def add(self, item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        self._buffer.append(item)
        now = time.time()
        if len(self._buffer) >= self.size or (now - self._last_ts) * 1000 >= self.latency_ms:
            batch = self._buffer
            self._buffer = []
            self._last_ts = now
            return batch
        return None

    def flush_if_ready(self) -> Optional[List[Dict[str, Any]]]:
        if not self._buffer:
            return None
        now = time.time()
        if (now - self._last_ts) * 1000 >= self.latency_ms:
            batch = self._buffer
            self._buffer = []
            self._last_ts = now
            return batch
        return None


class RealTimeRiskScorer:
    """Real-time risk scoring system with dynamic algorithms and automated alerting."""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.confidence_engine = ConfidenceEngine(config)
        
        # Configuration
        self.risk_threshold = config.get('ml.risk_scoring.threshold', 0.8)
        self.alert_threshold = config.get('ml.risk_scoring.alert_threshold', 0.9)
        self.decay_rate = config.get('ml.risk_scoring.decay_rate', 0.95)
        self.window_size = config.get('ml.risk_scoring.window_size', 100)
        self.update_interval = config.get('ml.risk_scoring.update_interval', 10)
        
        # Risk factor weights (configurable)
        self.risk_weights = {
            'behavioral_anomaly': config.get('ml.risk_scoring.weights.behavioral', 0.25),
            'keystroke_anomaly': config.get('ml.risk_scoring.weights.keystroke', 0.20),
            'insider_threat': config.get('ml.risk_scoring.weights.insider_threat', 0.30),
            'temporal_risk': config.get('ml.risk_scoring.weights.temporal', 0.15),
            'access_risk': config.get('ml.risk_scoring.weights.access', 0.10)
        }
        
        # Scoring models
        self.models = {
            'primary': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            'ensemble': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            ),
            'linear': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        # Risk state tracking
        self.current_risk_score = 0.0
        self.risk_history = deque(maxlen=1000)
        self.risk_factors = defaultdict(float)
        self.risk_trends = defaultdict(list)
        
        # Event processing
        self.event_buffer = deque(maxlen=self.window_size)
        self._cache_capacity = int(self.config.get('ml.risk_scoring.cache_capacity', 500) or 500)
        self._cache_order = deque(maxlen=self._cache_capacity)
        self._risk_cache = {}
        self.last_update = datetime.now()
        
        # Alerting system
        self.alert_callbacks = []
        self.alert_history = deque(maxlen=200)
        self.alert_suppression = defaultdict(datetime)
        self.alert_escalation = defaultdict(int)
        
        # Feature scaling / calibration
        self.scaler = StandardScaler()
        self.risk_scaler = MinMaxScaler()
        self.calibrator = _ScoreCalibrator()
        
        # Model state
        self.models_trained = False
        self.baseline_established = False
        self.baseline_manager = _BaselineManager(
            window_size=int(self.config.get('ml.risk_scoring.baseline_window', 1000) or 1000),
            drift_threshold=float(self.config.get('ml.risk_scoring.drift_threshold', 0.15) or 0.15)
        )
        self.micro_batcher = _MicroBatcher(
            micro_batch_size=int(self.config.get('ml.risk_scoring.micro_batch_size', 32) or 32),
            max_latency_ms=int(self.config.get('ml.risk_scoring.max_latency_ms', 200) or 200)
        )
        
        # Threading for continuous monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'risk_calculations': 0,
            'alerts_generated': 0,
            'high_risk_events': 0,
            'model_updates': 0,
            'last_training': None,
            'avg_risk_score': 0.0,
            'max_risk_score': 0.0
        }
        
        # Load existing models
        self._load_models()
        
        # Start continuous monitoring
        self._start_monitoring()
        
        logger.info("RealTimeRiskScorer initialized")
    
    def calculate_risk(self, event: Dict[str, Any]) -> float:
        """Calculate real-time risk score for an event.

        For extended metadata, use `calculate_risk_with_metadata`.
        """
        try:
            # Extract risk features
            risk_features = self._extract_risk_features(event)
            if not risk_features:
                return 0.0
            
            # Add to event buffer
            self.event_buffer.append({
                'timestamp': event.get('timestamp'),
                'features': risk_features,
                'event': event
            })
            # Micro-batching for downstream processing
            try:
                _ = self.micro_batcher.add({'event': event, 'features': risk_features})
            except Exception:
                pass
            
            self.stats['events_processed'] += 1
            
            # Cache lookup for repeated feature sets
            cache_key = self._make_cache_key(risk_features)
            cached = self._risk_cache.get(cache_key)
            if cached is not None:
                calibrated = self.calibrator.to_unit_interval(float(cached))
                # Update stats and state minimally
                self._update_risk_state(calibrated, {})
                return calibrated

            # Calculate multi-dimensional risk scores
            dimension_scores = self._calculate_dimensional_risks(risk_features, event)
            
            # Apply temporal weighting
            temporal_weight = self._calculate_temporal_weight(event)
            
            # Calculate base risk score
            base_risk = self._calculate_base_risk(dimension_scores, temporal_weight)
            
            # Apply dynamic adjustments
            adjusted_risk = self._apply_dynamic_adjustments(base_risk, event)

            # Baseline / cold-start gating and drift detection
            if not self.baseline_established:
                self.baseline_manager.add_baseline_sample(risk_features)
                # Establish baseline after collecting a fraction of window
                if self.baseline_manager.samples_collected >= max(10, self.baseline_manager.window_size // 4):
                    self.baseline_established = True
                # Conservative cap during cold-start
                adjusted_risk = min(adjusted_risk, 0.4)
            else:
                if self.baseline_manager.detect_drift(risk_features):
                    logger.info("RiskScorer: drift detected, refreshing baseline")
                    self.baseline_manager.refresh_baseline()
            
            # Update risk state with calibrated score
            calibrated_risk = self.calibrator.to_unit_interval(adjusted_risk)
            self._update_risk_state(calibrated_risk, dimension_scores)
            
            # Check for alerts
            self._check_alerts(calibrated_risk, event, dimension_scores)

            # Maintain LRU cache
            try:
                self._risk_cache[cache_key] = float(adjusted_risk)
                self._cache_order.append(cache_key)
                # Evict oldest if capacity exceeded
                if len(self._risk_cache) > self._cache_capacity:
                    old_key = self._cache_order.popleft()
                    if old_key in self._risk_cache:
                        del self._risk_cache[old_key]
            except Exception:
                pass
            
            self.stats['risk_calculations'] += 1
            
            return calibrated_risk
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.0

    def _make_cache_key(self, features: Dict[str, Any]) -> str:
        try:
            s = json.dumps(features, sort_keys=True, separators=(",", ":"))
            return str(hash(s))
        except Exception:
            return str(id(features))

    def calculate_risk_with_metadata(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score and return standardized metadata."""
        score = self.calculate_risk(event)
        try:
            confidence = float(self.confidence_engine.estimate_confidence({'risk_score': score}))
        except Exception:
            confidence = 0.5
        return {
            'risk_score': score,
            'confidence': confidence,
            'sources': list(self.risk_factors.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_risk_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features relevant to risk scoring."""
        features = {}

        try:
            # Basic event features
            timestamp_str = event.get('timestamp', datetime.now(timezone.utc).isoformat())
            if isinstance(timestamp_str, str):
                # Handle both timezone-aware and naive timestamps
                if '+' in timestamp_str or 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            # Ensure current time is timezone-aware
            current_time = datetime.now(timezone.utc)

            features.update({
                'event_type': event.get('type', 'unknown'),
                'timestamp': timestamp.isoformat(),
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday()
            })
        except Exception as e:
            logger.warning(f"Error extracting timestamp features: {e}")
            current_time = datetime.now(timezone.utc)
            features.update({
                'event_type': event.get('type', 'unknown'),
                'timestamp': current_time.isoformat(),
                'hour': current_time.hour,
                'day_of_week': current_time.weekday()
            })

        # Event-specific risk features
        event_type = event.get('type', '')
        data = event.get('data', {})
        
        if event_type == 'keyboard':
            features.update(self._extract_keyboard_risk_features(data))
        elif event_type == 'mouse':
            features.update(self._extract_mouse_risk_features(data))
        elif event_type == 'clipboard':
            features.update(self._extract_clipboard_risk_features(data))
        elif event_type == 'window':
            features.update(self._extract_window_risk_features(data))
        
        # Contextual risk features
        features.update(self._extract_contextual_risk_features())
        
        # Deterministic ordering to improve cache hits
        try:
            return {k: features[k] for k in sorted(features.keys())}
        except Exception:
            return features
    
    def _extract_keyboard_risk_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract keyboard-specific risk features."""
        features = {}
        
        key = data.get('key', '').lower()
        content = str(data.get('content', '')).lower()
        
        # Risk indicators
        features.update({
            'admin_key_usage': key in ['f1', 'f2', 'f12'] or 'ctrl+alt' in key,
            'system_commands': any(cmd in content for cmd in 
                                 ['cmd', 'powershell', 'terminal', 'sudo', 'admin']),
            'sensitive_content': any(sens in content for sens in 
                                   ['password', 'secret', 'key', 'token']),
            'file_operations': any(op in content for op in 
                                 ['delete', 'remove', 'copy', 'move']),
            'network_activity': any(net in content for net in 
                                  ['download', 'upload', 'ftp', 'ssh']),
            'rapid_typing': self._detect_rapid_typing(),
            'unusual_patterns': self._detect_unusual_typing_patterns()
        })
        
        return features
    
    def _extract_mouse_risk_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mouse-specific risk features."""
        features = {}
        
        action = data.get('action', '')
        x, y = data.get('x', 0), data.get('y', 0)
        
        features.update({
            'right_click_usage': action == 'right_click',
            'drag_operations': action == 'drag',
            'screen_edge_activity': self._is_screen_edge(x, y),
            'rapid_clicking': self._detect_rapid_clicking(),
            'unusual_mouse_patterns': self._detect_unusual_mouse_patterns(x, y)
        })
        
        return features
    
    def _extract_clipboard_risk_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clipboard-specific risk features."""
        features = {}
        
        content = data.get('content', '')
        
        features.update({
            'large_clipboard': len(content) > 10000,
            'sensitive_clipboard': self._contains_sensitive_data(content),
            'file_paths_clipboard': self._contains_file_paths(content),
            'urls_clipboard': self._contains_urls(content),
            'frequent_clipboard': self._detect_frequent_clipboard_usage(),
            'clipboard_size_anomaly': self._detect_clipboard_size_anomaly(len(content))
        })
        
        return features
    
    def _extract_window_risk_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract window-specific risk features."""
        features = {}
        
        app_name = data.get('application', '').lower()
        window_title = data.get('title', '').lower()
        
        features.update({
            'admin_application': any(admin in app_name for admin in 
                                   ['admin', 'elevated', 'administrator']),
            'system_application': any(sys in app_name for sys in 
                                    ['system', 'services', 'registry']),
            'security_application': any(sec in app_name for sec in 
                                      ['security', 'firewall', 'antivirus']),
            'development_tool': any(dev in app_name for dev in 
                                  ['code', 'studio', 'git', 'terminal']),
            'private_browsing': 'private' in window_title or 'incognito' in window_title,
            'rapid_app_switching': self._detect_rapid_app_switching(app_name)
        })
        
        return features
    
    def _extract_contextual_risk_features(self) -> Dict[str, Any]:
        """Extract contextual risk features from recent activity."""
        features = {}
        
        if len(self.event_buffer) < 5:
            return features
        
        recent_events = list(self.event_buffer)[-10:]
        
        # Activity intensity
        time_span = self._calculate_time_span(recent_events)
        features['activity_intensity'] = len(recent_events) / max(time_span, 1)
        
        # Event type diversity
        event_types = [e.get('type', 'unknown') for e in recent_events]
        unique_types = len(set(event_types))
        features['event_diversity'] = unique_types / max(len(event_types), 1)
        
        # Risk accumulation
        features['accumulated_risk'] = self._calculate_accumulated_risk()
        
        # Trend analysis
        features['risk_trend'] = self._calculate_risk_trend()
        
        return features
    
    def _calculate_dimensional_risks(self, features: Dict[str, Any], 
                                   event: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk scores for different dimensions."""
        dimension_scores = {}
        
        # Behavioral anomaly risk
        dimension_scores['behavioral_anomaly'] = self._calculate_behavioral_risk(features)
        
        # Keystroke dynamics risk
        dimension_scores['keystroke_anomaly'] = self._calculate_keystroke_risk(features, event)
        
        # Insider threat risk
        dimension_scores['insider_threat'] = self._calculate_insider_threat_risk(features)
        
        # Temporal risk
        dimension_scores['temporal_risk'] = self._calculate_temporal_risk(features)
        
        # Access pattern risk
        dimension_scores['access_risk'] = self._calculate_access_risk(features)
        
        return dimension_scores
    
    def _calculate_behavioral_risk(self, features: Dict[str, Any]) -> float:
        """Calculate behavioral anomaly risk score."""
        risk_score = 0.0
        
        # Activity intensity risk
        intensity = features.get('activity_intensity', 0)
        if intensity > 10:  # Very high activity
            risk_score += 0.3
        elif intensity > 5:
            risk_score += 0.1
        
        # Event diversity risk
        diversity = features.get('event_diversity', 0)
        if diversity > 0.8:  # Very diverse activity
            risk_score += 0.2
        
        # Accumulated risk
        accumulated = features.get('accumulated_risk', 0)
        risk_score += min(0.4, accumulated * 0.1)
        
        return min(1.0, risk_score)
    
    def _calculate_keystroke_risk(self, features: Dict[str, Any], event: Dict[str, Any]) -> float:
        """Calculate keystroke dynamics risk score."""
        if event.get('type') != 'keyboard':
            return 0.0
        
        risk_score = 0.0
        
        # Rapid typing risk
        if features.get('rapid_typing', False):
            risk_score += 0.2
        
        # Unusual patterns
        if features.get('unusual_patterns', False):
            risk_score += 0.3
        
        # Admin key usage
        if features.get('admin_key_usage', False):
            risk_score += 0.2
        
        # System commands
        if features.get('system_commands', False):
            risk_score += 0.4
        
        # Sensitive content
        if features.get('sensitive_content', False):
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _calculate_insider_threat_risk(self, features: Dict[str, Any]) -> float:
        """Calculate insider threat risk score."""
        risk_score = 0.0
        
        # File operations risk
        if features.get('file_operations', False):
            risk_score += 0.2
        
        # Network activity risk
        if features.get('network_activity', False):
            risk_score += 0.3
        
        # Large clipboard operations
        if features.get('large_clipboard', False):
            risk_score += 0.2
        
        # Sensitive clipboard content
        if features.get('sensitive_clipboard', False):
            risk_score += 0.4
        
        # Admin application usage
        if features.get('admin_application', False):
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _calculate_temporal_risk(self, features: Dict[str, Any]) -> float:
        """Calculate temporal risk score based on timing patterns."""
        risk_score = 0.0
        
        hour = features.get('hour', 12)
        day_of_week = features.get('day_of_week', 0)
        
        # After-hours risk
        if hour < 8 or hour > 18:
            risk_score += 0.3
        
        # Very late/early hours
        if hour < 6 or hour > 22:
            risk_score += 0.4
        
        # Weekend risk
        if day_of_week >= 5:
            risk_score += 0.2
        
        # Risk trend
        trend = features.get('risk_trend', 0)
        if trend > 0.1:  # Increasing risk
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _calculate_access_risk(self, features: Dict[str, Any]) -> float:
        """Calculate access pattern risk score."""
        risk_score = 0.0
        
        # System application access
        if features.get('system_application', False):
            risk_score += 0.4
        
        # Security application access
        if features.get('security_application', False):
            risk_score += 0.3
        
        # Private browsing
        if features.get('private_browsing', False):
            risk_score += 0.2
        
        # Rapid application switching
        if features.get('rapid_app_switching', False):
            risk_score += 0.2
        
        # File paths in clipboard
        if features.get('file_paths_clipboard', False):
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _calculate_temporal_weight(self, event: Dict[str, Any]) -> float:
        """Calculate temporal weighting factor."""
        try:
            timestamp_str = event.get('timestamp', datetime.now(timezone.utc).isoformat())
            if isinstance(timestamp_str, str):
                # Handle both timezone-aware and naive timestamps
                if '+' in timestamp_str or 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Ensure current time is timezone-aware
            current_time = datetime.now(timezone.utc)
            
            # Recent events have higher weight
            time_diff = (current_time - timestamp).total_seconds()
            
            # Exponential decay
            weight = np.exp(-time_diff / 3600)  # 1-hour half-life
            
            return max(0.1, min(1.0, weight))
        except Exception as e:
            logger.warning(f"Error calculating temporal weight: {e}")
            return 0.5  # Default weight
    
    def _calculate_base_risk(self, dimension_scores: Dict[str, float], 
                           temporal_weight: float) -> float:
        """Calculate base risk score from dimensional scores."""
        weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.risk_weights.get(dimension, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            base_risk = weighted_score / total_weight
        else:
            base_risk = 0.0
        
        # Apply temporal weighting
        return base_risk * temporal_weight
    
    def _apply_dynamic_adjustments(self, base_risk: float, event: Dict[str, Any]) -> float:
        """Apply dynamic adjustments to base risk score."""
        adjusted_risk = base_risk
        
        # Historical context adjustment
        if len(self.risk_history) > 10:
            recent_avg = np.mean([r['score'] for r in list(self.risk_history)[-10:]])
            if base_risk > recent_avg * 1.5:  # Significantly higher than recent average
                adjusted_risk *= 1.2  # 20% boost
        
        # Frequency adjustment
        event_type = event.get('type', '')
        recent_same_type = sum(1 for e in list(self.event_buffer)[-20:] 
                              if e['event'].get('type') == event_type)
        
        if recent_same_type > 15:  # High frequency of same event type
            adjusted_risk *= 1.1  # 10% boost
        
        # Model-based adjustment (if trained)
        if self.models_trained:
            model_adjustment = self._get_model_adjustment(event)
            adjusted_risk = (adjusted_risk + model_adjustment) / 2
        
        # Apply decay to prevent score inflation
        adjusted_risk *= self.decay_rate
        
        return min(1.0, max(0.0, adjusted_risk))
    
    def _get_model_adjustment(self, event: Dict[str, Any]) -> float:
        """Get risk adjustment from trained models."""
        try:
            # Prepare feature vector
            features = self._extract_risk_features(event)
            feature_vector = self._prepare_feature_vector(features)
            
            if feature_vector is None:
                return 0.0
            
            # Get predictions from models
            predictions = []
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict([feature_vector])[0]
                        predictions.append(max(0.0, min(1.0, pred)))
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_name}: {e}")
            
            # Return ensemble average
            return np.mean(predictions) if predictions else 0.0
            
        except Exception as e:
            logger.warning(f"Error in model adjustment: {e}")
            return 0.0
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for model input."""
        try:
            if not hasattr(self, 'feature_names'):
                return None
            
            vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0.0)
                if isinstance(value, bool):
                    value = float(value)
                elif isinstance(value, str):
                    value = hash(value) % 1000 / 1000.0  # Simple string hashing
                elif not isinstance(value, (int, float)):
                    value = 0.0
                vector.append(value)
            
            # Scale features
            vector = np.array(vector).reshape(1, -1)
            scaled_vector = self.scaler.transform(vector)
            
            return scaled_vector[0]
            
        except Exception as e:
            logger.warning(f"Error preparing feature vector: {e}")
            return None
    
    def _update_risk_state(self, risk_score: float, dimension_scores: Dict[str, float]):
        """Update internal risk state."""
        # Update current risk score
        self.current_risk_score = risk_score
        
        # Add to history
        self.risk_history.append({
            'timestamp': datetime.now().isoformat(),
            'score': risk_score,
            'dimensions': dimension_scores.copy()
        })
        
        # Update risk factors
        for dimension, score in dimension_scores.items():
            self.risk_factors[dimension] = score
            
            # Update trends
            if dimension not in self.risk_trends:
                self.risk_trends[dimension] = deque(maxlen=50)
            self.risk_trends[dimension].append(score)
        
        # Update statistics
        self.stats['avg_risk_score'] = np.mean([r['score'] for r in self.risk_history])
        self.stats['max_risk_score'] = max(self.stats['max_risk_score'], risk_score)
        
        if risk_score > self.risk_threshold:
            self.stats['high_risk_events'] += 1
    
    def _check_alerts(self, risk_score: float, event: Dict[str, Any], 
                     dimension_scores: Dict[str, float]):
        """Check if alerts should be generated."""
        if risk_score < self.alert_threshold:
            return
        
        # Check alert suppression
        alert_key = f"{event.get('type', 'unknown')}_{int(risk_score * 10)}"
        
        if alert_key in self.alert_suppression:
            last_alert = self.alert_suppression[alert_key]
            if (datetime.now() - last_alert).total_seconds() < 300:  # 5 minutes
                return
        
        # Generate alert
        alert = self._create_alert(risk_score, event, dimension_scores)
        
        # Record alert
        self.alert_history.append(alert)
        self.alert_suppression[alert_key] = datetime.now()
        self.stats['alerts_generated'] += 1
        
        # Escalate if needed
        self._handle_alert_escalation(alert)
        
        # Trigger callbacks
        self._trigger_alert_callbacks(alert)
        
        logger.warning(f"Risk alert generated: score={risk_score:.3f}, "
                      f"event={event.get('type')}, severity={alert['severity']}")
    
    def _create_alert(self, risk_score: float, event: Dict[str, Any], 
                     dimension_scores: Dict[str, float]) -> Dict[str, Any]:
        """Create alert object."""
        # Determine severity
        if risk_score >= 0.95:
            severity = 'critical'
        elif risk_score >= 0.9:
            severity = 'high'
        elif risk_score >= 0.8:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Get primary risk factors
        primary_factors = sorted(dimension_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        alert = {
            'id': f"alert_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_score,
            'severity': severity,
            'event': event,
            'dimension_scores': dimension_scores,
            'primary_factors': [f[0] for f in primary_factors],
            'description': self._generate_alert_description(risk_score, primary_factors),
            'recommendations': self._generate_recommendations(primary_factors)
        }
        
        return alert
    
    def _generate_alert_description(self, risk_score: float, 
                                  primary_factors: List[Tuple[str, float]]) -> str:
        """Generate human-readable alert description."""
        factor_names = {
            'behavioral_anomaly': 'unusual behavior patterns',
            'keystroke_anomaly': 'abnormal typing patterns',
            'insider_threat': 'potential insider threat indicators',
            'temporal_risk': 'suspicious timing patterns',
            'access_risk': 'risky access patterns'
        }
        
        primary_factor = primary_factors[0][0] if primary_factors else 'unknown'
        factor_desc = factor_names.get(primary_factor, primary_factor)
        
        return f"High risk activity detected (score: {risk_score:.2f}) - {factor_desc}"
    
    def _generate_recommendations(self, primary_factors: List[Tuple[str, float]]) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []
        
        factor_recommendations = {
            'behavioral_anomaly': [
                "Review recent user activity patterns",
                "Verify user identity through additional authentication",
                "Monitor for continued anomalous behavior"
            ],
            'keystroke_anomaly': [
                "Verify user identity through keystroke dynamics",
                "Check for potential unauthorized access",
                "Review typing pattern changes"
            ],
            'insider_threat': [
                "Investigate potential insider threat activity",
                "Review data access and transfer patterns",
                "Consider additional monitoring and restrictions"
            ],
            'temporal_risk': [
                "Verify authorization for after-hours access",
                "Review business justification for activity timing",
                "Consider implementing time-based access controls"
            ],
            'access_risk': [
                "Review application and system access patterns",
                "Verify authorization for sensitive system access",
                "Consider implementing additional access controls"
            ]
        }
        
        for factor, score in primary_factors:
            if score > 0.5 and factor in factor_recommendations:
                recommendations.extend(factor_recommendations[factor])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _handle_alert_escalation(self, alert: Dict[str, Any]):
        """Handle alert escalation logic."""
        severity = alert['severity']
        alert_id = alert['id']
        
        # Track escalation count
        self.alert_escalation[severity] += 1
        
        # Escalation thresholds
        escalation_thresholds = {
            'medium': 5,   # 5 medium alerts in window
            'high': 3,     # 3 high alerts in window
            'critical': 1  # 1 critical alert
        }
        
        if self.alert_escalation[severity] >= escalation_thresholds.get(severity, 10):
            # Create escalated alert
            escalated_alert = alert.copy()
            escalated_alert['escalated'] = True
            escalated_alert['escalation_reason'] = f"Multiple {severity} alerts"
            
            # Reset escalation counter
            self.alert_escalation[severity] = 0
            
            logger.critical(f"Alert escalated: {escalated_alert['description']}")
    
    def _trigger_alert_callbacks(self, alert: Dict[str, Any]):
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback function for alerts."""
        self.alert_callbacks.append(callback)
        logger.info(f"Alert callback registered: {callback.__name__}")
    
    def _start_monitoring(self):
        """Start continuous monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Risk monitoring thread started")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Periodic risk assessment
                self._periodic_risk_assessment()
                
                # Model updates
                if (datetime.now() - self.last_update).total_seconds() > self.update_interval:
                    self._periodic_model_update()
                    self.last_update = datetime.now()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _periodic_risk_assessment(self):
        """Perform periodic risk assessment."""
        if len(self.event_buffer) < 10:
            return
        
        # Calculate overall system risk
        recent_scores = [r['score'] for r in list(self.risk_history)[-20:]]
        
        if recent_scores:
            avg_risk = np.mean(recent_scores)
            risk_trend = self._calculate_risk_trend()
            
            # Check for sustained high risk
            if avg_risk > self.risk_threshold and risk_trend > 0.1:
                self._generate_system_alert(avg_risk, risk_trend)
    
    def _generate_system_alert(self, avg_risk: float, trend: float):
        """Generate system-level risk alert."""
        alert = {
            'id': f"system_alert_{int(time.time() * 1000)}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'system_risk',
            'avg_risk': avg_risk,
            'trend': trend,
            'description': f"Sustained high system risk detected (avg: {avg_risk:.2f}, trend: {trend:.2f})",
            'severity': 'high' if avg_risk > 0.9 else 'medium'
        }
        
        self.alert_history.append(alert)
        self._trigger_alert_callbacks(alert)
        
        logger.warning(f"System risk alert: {alert['description']}")
    
    def _periodic_model_update(self):
        """Perform periodic model updates."""
        if len(self.risk_history) > 100:
            # Retrain models with recent data
            self._train_models()
        # Attempt to flush micro-batches within latency bounds
        try:
            batch = self.micro_batcher.flush_if_ready()
            if batch:
                for item in batch:
                    try:
                        _ = self.calculate_risk(item['event'])
                    except Exception:
                        continue
        except Exception:
            pass
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues."""
        try:
            # Clean up alert suppression (older than 1 hour)
            current_time = datetime.now(timezone.utc)
            expired_keys = []
            
            for key, timestamp in self.alert_suppression.items():
                # Ensure timestamp is timezone-aware
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    if (current_time - timestamp).total_seconds() > 3600:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self.alert_suppression[key]
        except Exception as e:
            logger.warning(f"Error cleaning up old data: {e}")

        # Reset escalation counters periodically
        if len(self.alert_history) % 100 == 0:
            self.alert_escalation.clear()
    
    def _train_models(self):
        """Train risk scoring models."""
        if len(self.risk_history) < 50:
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 20:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Scale targets
            y_train_scaled = self.risk_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # Train models
            for model_name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train_scaled)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    logger.info(f"Trained {model_name} model - MSE: {mse:.4f}, R2: {r2:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
            
            self.models_trained = True
            self.stats['model_updates'] += 1
            self.stats['last_training'] = datetime.now().isoformat()
            
            logger.info(f"Risk scoring models trained with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from risk history."""
        # Get features and targets from event buffer
        X = []
        y = []
        
        # Collect feature names
        all_features = set()
        for event_data in self.event_buffer:
            all_features.update(event_data['features'].keys())
        
        self.feature_names = sorted(list(all_features))
        
        # Create feature vectors and targets
        for i, event_data in enumerate(self.event_buffer):
            features = event_data['features']
            
            # Create feature vector
            vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0.0)
                if isinstance(value, bool):
                    value = float(value)
                elif isinstance(value, str):
                    value = hash(value) % 1000 / 1000.0
                elif not isinstance(value, (int, float)):
                    value = 0.0
                vector.append(value)
            
            X.append(vector)
            
            # Get corresponding risk score from history
            if i < len(self.risk_history):
                y.append(self.risk_history[i]['score'])
            else:
                y.append(0.0)
        
        return np.array(X), np.array(y)
    
    # Helper methods for feature detection
    def _detect_rapid_typing(self) -> bool:
        """Detect rapid typing patterns."""
        if len(self.event_buffer) < 5:
            return False
        
        keyboard_events = [e for e in list(self.event_buffer)[-10:] 
                          if e['event'].get('type') == 'keyboard']
        
        if len(keyboard_events) < 5:
            return False
        
        # Calculate typing rate
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in keyboard_events]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        
        if time_span > 0:
            typing_rate = len(keyboard_events) / time_span
            return typing_rate > 10  # More than 10 keys per second
        
        return False
    
    def _detect_unusual_typing_patterns(self) -> bool:
        """Detect unusual typing patterns."""
        # Simplified implementation
        return False
    
    def _detect_rapid_clicking(self) -> bool:
        """Detect rapid mouse clicking."""
        if len(self.event_buffer) < 5:
            return False
        
        mouse_events = [e for e in list(self.event_buffer)[-10:] 
                       if e['event'].get('type') == 'mouse' and 
                       e['event'].get('data', {}).get('action') == 'click']
        
        if len(mouse_events) < 3:
            return False
        
        try:
            # Check for rapid succession
            timestamps = []
            for e in mouse_events:
                timestamp_str = e.get('timestamp', datetime.now(timezone.utc).isoformat())
                if isinstance(timestamp_str, str):
                    if '+' in timestamp_str or 'Z' in timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
                    timestamps.append(timestamp)
            
            if len(timestamps) < 2:
                return False
                
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
            
            return any(interval < 0.1 for interval in intervals)  # Less than 100ms
        except Exception as e:
            logger.warning(f"Error detecting rapid clicking: {e}")
            return False
    
    def _detect_unusual_mouse_patterns(self, x: int, y: int) -> bool:
        """Detect unusual mouse movement patterns."""
        # Simplified implementation
        return False
    
    def _is_screen_edge(self, x: int, y: int) -> bool:
        """Check if coordinates are near screen edge."""
        screen_width = self.config.get('screen.width', 1920)
        screen_height = self.config.get('screen.height', 1080)
        
        edge_threshold = 50  # pixels
        
        return (x < edge_threshold or x > screen_width - edge_threshold or
                y < edge_threshold or y > screen_height - edge_threshold)
    
    def _detect_frequent_clipboard_usage(self) -> bool:
        """Detect frequent clipboard usage."""
        if len(self.event_buffer) < 10:
            return False
        
        clipboard_events = sum(1 for e in list(self.event_buffer)[-20:] 
                             if e['event'].get('type') == 'clipboard')
        
        return clipboard_events > 5  # More than 5 clipboard events in recent activity
    
    def _detect_clipboard_size_anomaly(self, size: int) -> bool:
        """Detect clipboard size anomalies."""
        if len(self.event_buffer) < 10:
            return False
        
        # Get recent clipboard sizes
        recent_sizes = []
        for e in list(self.event_buffer)[-20:]:
            if e['event'].get('type') == 'clipboard':
                content = e['event'].get('data', {}).get('content', '')
                recent_sizes.append(len(content))
        
        if len(recent_sizes) < 3:
            return False
        
        # Check if current size is anomalous
        avg_size = np.mean(recent_sizes)
        std_size = np.std(recent_sizes)
        
        if std_size > 0:
            z_score = abs(size - avg_size) / std_size
            return z_score > 2.0  # More than 2 standard deviations
        
        return False
    
    def _detect_rapid_app_switching(self, app_name: str) -> bool:
        """Detect rapid application switching."""
        if len(self.event_buffer) < 5:
            return False
        
        window_events = [e for e in list(self.event_buffer)[-10:] 
                        if e['event'].get('type') == 'window']
        
        if len(window_events) < 3:
            return False
        
        # Count unique applications
        apps = set(e['event'].get('data', {}).get('application', '') 
                  for e in window_events)
        
        return len(apps) > 5  # More than 5 different apps
    
    def _contains_sensitive_data(self, content: str) -> bool:
        """Check if content contains sensitive data."""
        import re
        
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'password\s*[:=]\s*\S+',  # Password
            r'api[_\s]?key\s*[:=]\s*\S+',  # API key
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in sensitive_patterns)
    
    def _contains_file_paths(self, content: str) -> bool:
        """Check if content contains file paths."""
        import re
        
        path_patterns = [
            r'[A-Za-z]:\\[^\s]+',  # Windows paths
            r'/[^\s]+',  # Unix paths
        ]
        
        return any(re.search(pattern, content) for pattern in path_patterns)
    
    def _contains_urls(self, content: str) -> bool:
        """Check if content contains URLs."""
        import re
        
        url_pattern = r'https?://[^\s]+'
        return bool(re.search(url_pattern, content, re.IGNORECASE))
    
    def _calculate_time_span(self, events: List[Dict[str, Any]]) -> float:
        """Calculate time span of events in seconds."""
        if len(events) < 2:
            return 1.0
        
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in events]
        return (max(timestamps) - min(timestamps)).total_seconds()
    
    def _calculate_accumulated_risk(self) -> float:
        """Calculate accumulated risk from recent history."""
        if len(self.risk_history) < 5:
            return 0.0
        
        recent_scores = [r['score'] for r in list(self.risk_history)[-10:]]
        return np.mean(recent_scores)
    
    def _calculate_risk_trend(self) -> float:
        """Calculate risk trend (positive = increasing, negative = decreasing)."""
        if len(self.risk_history) < 10:
            return 0.0
        
        recent_scores = [r['score'] for r in list(self.risk_history)[-10:]]
        
        # Simple linear trend
        x = np.arange(len(recent_scores))
        slope, _, _, _, _ = stats.linregress(x, recent_scores)
        
        return slope
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        return {
            'current_score': self.current_risk_score,
            'risk_level': self._get_risk_level(self.current_risk_score),
            'risk_factors': dict(self.risk_factors),
            'trend': self._calculate_risk_trend(),
            'recent_alerts': len([a for a in self.alert_history 
                                if self._is_recent_alert(a['timestamp'])]),
            'system_status': self._get_system_status()
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level description."""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        elif score >= 0.3:
            return 'low'
        else:
            return 'minimal'
    
    def _is_recent_alert(self, timestamp_str: str, hours: int = 1) -> bool:
        """Check if alert is recent."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return (datetime.now() - timestamp).total_seconds() < hours * 3600
        except:
            return False
    
    def _get_system_status(self) -> str:
        """Get overall system status."""
        if self.current_risk_score >= 0.9:
            return 'critical'
        elif self.current_risk_score >= 0.7:
            return 'elevated'
        elif self.current_risk_score >= 0.5:
            return 'moderate'
        else:
            return 'normal'
    
    def _save_models(self):
        """Save trained models and configuration."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            models_dir.mkdir(exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'risk_scaler': self.risk_scaler,
                'risk_weights': self.risk_weights,
                'feature_names': getattr(self, 'feature_names', []),
                'models_trained': self.models_trained,
                'risk_threshold': self.risk_threshold,
                'alert_threshold': self.alert_threshold
            }
            
            with open(models_dir / 'risk_scoring.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Risk scoring models saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models and configuration."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            model_file = models_dir / 'risk_scoring.pkl'
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data.get('models', self.models)
                self.scaler = model_data.get('scaler', self.scaler)
                self.risk_scaler = model_data.get('risk_scaler', self.risk_scaler)
                self.risk_weights = model_data.get('risk_weights', self.risk_weights)
                self.feature_names = model_data.get('feature_names', [])
                self.models_trained = model_data.get('models_trained', False)
                self.risk_threshold = model_data.get('risk_threshold', self.risk_threshold)
                self.alert_threshold = model_data.get('alert_threshold', self.alert_threshold)
                
                logger.info("Risk scoring models loaded")
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk scoring statistics."""
        stats = self.stats.copy()
        stats.update({
            'current_risk_score': self.current_risk_score,
            'risk_history_size': len(self.risk_history),
            'event_buffer_size': len(self.event_buffer),
            'alert_history_size': len(self.alert_history),
            'models_trained': self.models_trained,
            'monitoring_active': self.monitoring_active,
            'risk_threshold': self.risk_threshold,
            'alert_threshold': self.alert_threshold
        })
        
        # Alert rate
        if self.stats['events_processed'] > 0:
            stats['alert_rate'] = self.stats['alerts_generated'] / self.stats['events_processed']
        
        return stats
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Risk monitoring stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()
        
        if self.models_trained:
            self._save_models()