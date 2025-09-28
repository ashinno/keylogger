"""Keystroke Dynamics Analysis for biometric authentication and typing pattern recognition."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import pickle
import json
from pathlib import Path
import hashlib

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import euclidean, cosine

from .data_preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class KeystrokeDynamicsAnalyzer:
    """Advanced keystroke dynamics analyzer for biometric authentication."""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        
        # Configuration
        self.min_samples_for_training = config.get('ml.keystroke_dynamics.min_samples', 100)
        self.authentication_threshold = config.get('ml.keystroke_dynamics.auth_threshold', 0.8)
        self.adaptation_rate = config.get('ml.keystroke_dynamics.adaptation_rate', 0.1)
        self.feature_window_size = config.get('ml.keystroke_dynamics.window_size', 50)
        
        # Models for different analysis types
        self.models = {
            'authentication': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),
            'verification': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'anomaly_detection': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        # Neural network for advanced pattern recognition
        self.neural_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=1000
        )
        
        # Data storage
        self.keystroke_buffer = deque(maxlen=self.feature_window_size)
        self.user_profiles = {}  # User typing profiles
        self.training_data = defaultdict(list)  # Training data per user
        self.feature_extractors = {}
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model state
        self.models_trained = False
        self.user_authenticated = None
        self.current_session_features = []
        
        # Statistics
        self.stats = {
            'keystrokes_processed': 0,
            'authentication_attempts': 0,
            'successful_authentications': 0,
            'failed_authentications': 0,
            'users_enrolled': 0,
            'model_updates': 0,
            'last_training': None
        }
        
        # Load existing models and profiles
        self._load_models()
        
        logger.info("KeystrokeDynamicsAnalyzer initialized")
    
    def process_keystroke(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a keystroke event and extract dynamics features."""
        try:
            # Extract keystroke features
            features = self._extract_keystroke_features(event)
            if not features:
                return {'status': 'no_features'}
            
            # Add to buffer
            self.keystroke_buffer.append({
                'timestamp': event.get('timestamp'),
                'features': features,
                'event': event
            })
            
            self.stats['keystrokes_processed'] += 1
            
            # Analyze typing pattern
            result = self._analyze_typing_pattern()
            
            # Update session features
            self.current_session_features.append(features)
            
            # Adaptive learning
            if len(self.current_session_features) % 50 == 0:
                self._adaptive_learning()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing keystroke: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_keystroke_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive keystroke dynamics features."""
        features = {}
        data = event.get('data', {})
        
        # Basic keystroke information
        key = data.get('key', '')
        timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
        
        # Timing features
        features.update(self._extract_timing_features(event, timestamp))
        
        # Pressure and force features (if available)
        features.update(self._extract_pressure_features(data))
        
        # Key-specific features
        features.update(self._extract_key_features(key))
        
        # Contextual features
        features.update(self._extract_contextual_features(key, timestamp))
        
        # Rhythm and pattern features
        features.update(self._extract_rhythm_features())
        
        return features
    
    def _extract_timing_features(self, event: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Extract timing-based features."""
        features = {}
        data = event.get('data', {})
        
        # Dwell time (key press to release)
        if 'dwell_time' in data:
            features['dwell_time'] = data['dwell_time']
        
        # Flight time (previous key release to current key press)
        if len(self.keystroke_buffer) > 0:
            last_keystroke = self.keystroke_buffer[-1]
            last_timestamp = datetime.fromisoformat(last_keystroke['timestamp'])
            flight_time = (timestamp - last_timestamp).total_seconds() * 1000  # milliseconds
            features['flight_time'] = flight_time
            
            # Inter-keystroke interval
            features['inter_keystroke_interval'] = flight_time
        
        # Typing speed indicators
        if len(self.keystroke_buffer) >= 5:
            recent_keystrokes = list(self.keystroke_buffer)[-5:]
            timestamps = [datetime.fromisoformat(ks['timestamp']) for ks in recent_keystrokes]
            timestamps.append(timestamp)
            
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() * 1000 
                        for i in range(1, len(timestamps))]
            
            features.update({
                'avg_interval_5': np.mean(intervals),
                'std_interval_5': np.std(intervals),
                'typing_speed_5': 5 / (sum(intervals) / 1000) if sum(intervals) > 0 else 0
            })
        
        return features
    
    def _extract_pressure_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pressure and force features (if available from hardware)."""
        features = {}
        
        # Simulated pressure features (would come from specialized hardware)
        if 'pressure' in data:
            features['key_pressure'] = data['pressure']
        
        if 'force' in data:
            features['key_force'] = data['force']
        
        # Pressure variation (if multiple pressure readings)
        if 'pressure_samples' in data:
            samples = data['pressure_samples']
            features.update({
                'pressure_mean': np.mean(samples),
                'pressure_std': np.std(samples),
                'pressure_max': np.max(samples),
                'pressure_min': np.min(samples)
            })
        
        return features
    
    def _extract_key_features(self, key: str) -> Dict[str, Any]:
        """Extract key-specific features."""
        features = {}
        
        # Key characteristics
        features.update({
            'key_length': len(key),
            'is_letter': key.isalpha(),
            'is_digit': key.isdigit(),
            'is_special': not key.isalnum(),
            'is_uppercase': key.isupper(),
            'is_lowercase': key.islower()
        })
        
        # Key position features (QWERTY layout)
        key_positions = self._get_key_position(key)
        features.update(key_positions)
        
        # Key frequency in typing
        features['key_frequency'] = self._get_key_frequency(key)
        
        return features
    
    def _extract_contextual_features(self, key: str, timestamp: datetime) -> Dict[str, Any]:
        """Extract contextual features based on surrounding keystrokes."""
        features = {}
        
        if len(self.keystroke_buffer) >= 2:
            # Previous key context
            prev_key = self.keystroke_buffer[-1]['event'].get('data', {}).get('key', '')
            features['prev_key_same'] = key == prev_key
            features['prev_key_adjacent'] = self._are_keys_adjacent(prev_key, key)
            
            # Bigram features
            bigram = f"{prev_key}_{key}"
            features['bigram_frequency'] = self._get_bigram_frequency(bigram)
        
        if len(self.keystroke_buffer) >= 3:
            # Trigram features
            prev2_key = self.keystroke_buffer[-2]['event'].get('data', {}).get('key', '')
            prev1_key = self.keystroke_buffer[-1]['event'].get('data', {}).get('key', '')
            trigram = f"{prev2_key}_{prev1_key}_{key}"
            features['trigram_frequency'] = self._get_trigram_frequency(trigram)
        
        # Temporal context
        features.update({
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_business_hours': 9 <= timestamp.hour <= 17
        })
        
        return features
    
    def _extract_rhythm_features(self) -> Dict[str, Any]:
        """Extract typing rhythm and pattern features."""
        features = {}
        
        if len(self.keystroke_buffer) < 10:
            return features
        
        # Get recent timing data
        recent_keystrokes = list(self.keystroke_buffer)[-10:]
        intervals = []
        
        for i in range(1, len(recent_keystrokes)):
            curr_time = datetime.fromisoformat(recent_keystrokes[i]['timestamp'])
            prev_time = datetime.fromisoformat(recent_keystrokes[i-1]['timestamp'])
            interval = (curr_time - prev_time).total_seconds() * 1000
            intervals.append(interval)
        
        if intervals:
            # Rhythm consistency
            features.update({
                'rhythm_mean': np.mean(intervals),
                'rhythm_std': np.std(intervals),
                'rhythm_cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0,
                'rhythm_entropy': self._calculate_entropy(intervals)
            })
            
            # Typing burst detection
            burst_threshold = np.mean(intervals) * 0.5
            bursts = sum(1 for interval in intervals if interval < burst_threshold)
            features['typing_bursts'] = bursts
            
            # Pause detection
            pause_threshold = np.mean(intervals) * 2.0
            pauses = sum(1 for interval in intervals if interval > pause_threshold)
            features['typing_pauses'] = pauses
        
        return features
    
    def _get_key_position(self, key: str) -> Dict[str, Any]:
        """Get key position features based on QWERTY layout."""
        # QWERTY layout mapping (row, column)
        qwerty_layout = {
            'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
            'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
            'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4),
            'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
            'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4),
            'n': (2, 5), 'm': (2, 6)
        }
        
        key_lower = key.lower()
        if key_lower in qwerty_layout:
            row, col = qwerty_layout[key_lower]
            return {
                'key_row': row,
                'key_col': col,
                'key_hand': 0 if col < 5 else 1,  # 0 for left hand, 1 for right hand
                'key_finger': self._get_finger_mapping(key_lower)
            }
        
        return {'key_row': -1, 'key_col': -1, 'key_hand': -1, 'key_finger': -1}
    
    def _get_finger_mapping(self, key: str) -> int:
        """Map key to finger (0-9 for 10 fingers)."""
        finger_mapping = {
            'q': 0, 'a': 0, 'z': 0,  # Left pinky
            'w': 1, 's': 1, 'x': 1,  # Left ring
            'e': 2, 'd': 2, 'c': 2,  # Left middle
            'r': 3, 'f': 3, 'v': 3, 't': 3, 'g': 3, 'b': 3,  # Left index
            'y': 6, 'h': 6, 'n': 6, 'u': 6, 'j': 6, 'm': 6,  # Right index
            'i': 7, 'k': 7,  # Right middle
            'o': 8, 'l': 8,  # Right ring
            'p': 9  # Right pinky
        }
        
        return finger_mapping.get(key, -1)
    
    def _are_keys_adjacent(self, key1: str, key2: str) -> bool:
        """Check if two keys are adjacent on the keyboard."""
        pos1 = self._get_key_position(key1)
        pos2 = self._get_key_position(key2)
        
        if pos1['key_row'] == -1 or pos2['key_row'] == -1:
            return False
        
        row_diff = abs(pos1['key_row'] - pos2['key_row'])
        col_diff = abs(pos1['key_col'] - pos2['key_col'])
        
        return row_diff <= 1 and col_diff <= 1
    
    def _get_key_frequency(self, key: str) -> float:
        """Get frequency of key in recent typing."""
        if len(self.keystroke_buffer) < 10:
            return 0.0
        
        recent_keys = [ks['event'].get('data', {}).get('key', '') 
                      for ks in list(self.keystroke_buffer)[-50:]]
        
        return recent_keys.count(key) / max(len(recent_keys), 1)
    
    def _get_bigram_frequency(self, bigram: str) -> float:
        """Get frequency of key bigram in recent typing."""
        if len(self.keystroke_buffer) < 10:
            return 0.0
        
        recent_keys = [ks['event'].get('data', {}).get('key', '') 
                      for ks in list(self.keystroke_buffer)[-50:]]
        
        bigrams = [f"{recent_keys[i]}_{recent_keys[i+1]}" 
                  for i in range(len(recent_keys) - 1)]
        
        return bigrams.count(bigram) / len(bigrams) if bigrams else 0.0
    
    def _get_trigram_frequency(self, trigram: str) -> float:
        """Get frequency of key trigram in recent typing."""
        if len(self.keystroke_buffer) < 10:
            return 0.0
        
        recent_keys = [ks['event'].get('data', {}).get('key', '') 
                      for ks in list(self.keystroke_buffer)[-50:]]
        
        trigrams = [f"{recent_keys[i]}_{recent_keys[i+1]}_{recent_keys[i+2]}" 
                   for i in range(len(recent_keys) - 2)]
        
        return trigrams.count(trigram) / len(trigrams) if trigrams else 0.0
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy of timing intervals."""
        if not values:
            return 0.0
        
        # Discretize values into bins
        hist, _ = np.histogram(values, bins=10)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _analyze_typing_pattern(self) -> Dict[str, Any]:
        """Analyze current typing pattern for authentication and anomalies."""
        if len(self.keystroke_buffer) < self.min_samples_for_training:
            return {
                'status': 'insufficient_data',
                'samples_needed': self.min_samples_for_training - len(self.keystroke_buffer)
            }
        
        # Extract features from current session
        session_features = self._extract_session_features()
        
        result = {
            'status': 'analyzed',
            'session_features': session_features,
            'authentication': None,
            'anomaly_detection': None,
            'user_verification': None
        }
        
        # Authentication analysis
        if self.models_trained:
            result['authentication'] = self._authenticate_user(session_features)
            result['anomaly_detection'] = self._detect_typing_anomalies(session_features)
            
            if self.user_authenticated:
                result['user_verification'] = self._verify_user_identity(session_features)
        
        return result
    
    def _extract_session_features(self) -> Dict[str, Any]:
        """Extract features from current keystroke session."""
        if len(self.keystroke_buffer) < 10:
            return {}
        
        session_data = list(self.keystroke_buffer)
        features = {}
        
        # Aggregate timing features
        dwell_times = []
        flight_times = []
        
        for i, keystroke in enumerate(session_data):
            ks_features = keystroke['features']
            
            if 'dwell_time' in ks_features:
                dwell_times.append(ks_features['dwell_time'])
            
            if 'flight_time' in ks_features:
                flight_times.append(ks_features['flight_time'])
        
        # Statistical features
        if dwell_times:
            features.update({
                'dwell_mean': np.mean(dwell_times),
                'dwell_std': np.std(dwell_times),
                'dwell_median': np.median(dwell_times),
                'dwell_q25': np.percentile(dwell_times, 25),
                'dwell_q75': np.percentile(dwell_times, 75),
                'dwell_skew': stats.skew(dwell_times),
                'dwell_kurtosis': stats.kurtosis(dwell_times)
            })
        
        if flight_times:
            features.update({
                'flight_mean': np.mean(flight_times),
                'flight_std': np.std(flight_times),
                'flight_median': np.median(flight_times),
                'flight_q25': np.percentile(flight_times, 25),
                'flight_q75': np.percentile(flight_times, 75),
                'flight_skew': stats.skew(flight_times),
                'flight_kurtosis': stats.kurtosis(flight_times)
            })
        
        # Rhythm features
        if len(session_data) >= 5:
            intervals = []
            for i in range(1, len(session_data)):
                curr_time = datetime.fromisoformat(session_data[i]['timestamp'])
                prev_time = datetime.fromisoformat(session_data[i-1]['timestamp'])
                interval = (curr_time - prev_time).total_seconds() * 1000
                intervals.append(interval)
            
            if intervals:
                features.update({
                    'rhythm_consistency': 1.0 / (1.0 + np.std(intervals)),
                    'typing_speed': len(session_data) / (sum(intervals) / 1000) if sum(intervals) > 0 else 0,
                    'rhythm_entropy': self._calculate_entropy(intervals)
                })
        
        # Key usage patterns
        keys_used = [ks['event'].get('data', {}).get('key', '') for ks in session_data]
        unique_keys = set(keys_used)
        
        features.update({
            'unique_keys': len(unique_keys),
            'key_diversity': len(unique_keys) / max(len(keys_used), 1) if keys_used else 0,
            'most_common_key_freq': max([keys_used.count(k) for k in unique_keys]) / max(len(keys_used), 1) if unique_keys else 0
        })
        
        return features
    
    def _authenticate_user(self, session_features: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user based on typing patterns."""
        if not session_features or not self.models_trained:
            return {'status': 'not_ready'}
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(session_features)
            if feature_vector is None:
                return {'status': 'feature_error'}
            
            # Get predictions from authentication model
            auth_model = self.models['authentication']
            
            if hasattr(auth_model, 'predict_proba'):
                probabilities = auth_model.predict_proba([feature_vector])[0]
                predicted_user_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_user_idx]
                
                # Get user ID from label encoder
                if hasattr(self, 'user_labels'):
                    predicted_user = self.label_encoder.inverse_transform([predicted_user_idx])[0]
                else:
                    predicted_user = f"user_{predicted_user_idx}"
                
                is_authenticated = confidence >= self.authentication_threshold
                
                self.stats['authentication_attempts'] += 1
                if is_authenticated:
                    self.stats['successful_authentications'] += 1
                    self.user_authenticated = predicted_user
                else:
                    self.stats['failed_authentications'] += 1
                
                return {
                    'status': 'completed',
                    'authenticated': is_authenticated,
                    'user_id': predicted_user if is_authenticated else None,
                    'confidence': confidence,
                    'threshold': self.authentication_threshold,
                    'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities)) if hasattr(self, 'user_labels') else {}
                }
            
            return {'status': 'model_error'}
            
        except Exception as e:
            logger.error(f"Error in user authentication: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _detect_typing_anomalies(self, session_features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in typing patterns."""
        if not session_features or not self.models_trained:
            return {'status': 'not_ready'}
        
        try:
            feature_vector = self._prepare_feature_vector(session_features)
            if feature_vector is None:
                return {'status': 'feature_error'}
            
            anomaly_model = self.models['anomaly_detection']
            
            # Get anomaly score
            if hasattr(anomaly_model, 'decision_function'):
                anomaly_score = anomaly_model.decision_function([feature_vector])[0]
                is_anomaly = anomaly_score < 0
            elif hasattr(anomaly_model, 'predict_proba'):
                probabilities = anomaly_model.predict_proba([feature_vector])[0]
                anomaly_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                is_anomaly = anomaly_score > 0.5
            else:
                prediction = anomaly_model.predict([feature_vector])[0]
                is_anomaly = prediction == 1
                anomaly_score = 1.0 if is_anomaly else 0.0
            
            return {
                'status': 'completed',
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'severity': 'high' if anomaly_score > 0.8 else 'medium' if anomaly_score > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _verify_user_identity(self, session_features: Dict[str, Any]) -> Dict[str, Any]:
        """Verify identity of authenticated user."""
        if not self.user_authenticated or not session_features:
            return {'status': 'no_authenticated_user'}
        
        try:
            # Compare with user profile
            user_profile = self.user_profiles.get(self.user_authenticated)
            if not user_profile:
                return {'status': 'no_user_profile'}
            
            # Calculate similarity to user profile
            similarity_score = self._calculate_profile_similarity(session_features, user_profile)
            
            is_verified = similarity_score >= self.authentication_threshold
            
            return {
                'status': 'completed',
                'verified': is_verified,
                'similarity_score': similarity_score,
                'user_id': self.user_authenticated,
                'threshold': self.authentication_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in user verification: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_profile_similarity(self, session_features: Dict[str, Any], 
                                    user_profile: Dict[str, Any]) -> float:
        """Calculate similarity between session features and user profile."""
        try:
            # Get common features
            common_features = set(session_features.keys()) & set(user_profile['mean_features'].keys())
            
            if not common_features:
                return 0.0
            
            # Calculate normalized distances
            distances = []
            
            for feature in common_features:
                session_value = session_features[feature]
                profile_mean = user_profile['mean_features'][feature]
                profile_std = user_profile['std_features'].get(feature, 1.0)
                
                # Normalized distance
                if profile_std > 0:
                    normalized_distance = abs(session_value - profile_mean) / profile_std
                else:
                    normalized_distance = abs(session_value - profile_mean)
                
                distances.append(normalized_distance)
            
            # Convert distance to similarity (0-1 scale)
            avg_distance = np.mean(distances)
            similarity = 1.0 / (1.0 + avg_distance)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating profile similarity: {e}")
            return 0.0
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for model input."""
        try:
            if not hasattr(self, 'feature_names'):
                return None
            
            # Create vector with all expected features
            vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0.0)
                if isinstance(value, bool):
                    value = float(value)
                elif not isinstance(value, (int, float)):
                    value = 0.0
                vector.append(value)
            
            # Scale features
            vector = np.array(vector).reshape(1, -1)
            scaled_vector = self.scaler.transform(vector)
            
            return scaled_vector[0]
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None
    
    def enroll_user(self, user_id: str, typing_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enroll a new user with their typing samples."""
        try:
            if len(typing_samples) < self.min_samples_for_training:
                return {
                    'status': 'insufficient_samples',
                    'required': self.min_samples_for_training,
                    'provided': len(typing_samples)
                }
            
            # Extract features from all samples
            user_features = []
            for sample in typing_samples:
                features = self._extract_keystroke_features(sample)
                if features:
                    user_features.append(features)
            
            if not user_features:
                return {'status': 'no_valid_features'}
            
            # Create user profile
            user_profile = self._create_user_profile(user_features)
            self.user_profiles[user_id] = user_profile
            
            # Add to training data
            self.training_data[user_id].extend(user_features)
            
            # Update statistics
            self.stats['users_enrolled'] += 1
            
            # Retrain models if we have enough users
            if len(self.user_profiles) >= 2:
                self._train_models()
            
            logger.info(f"User {user_id} enrolled with {len(user_features)} samples")
            
            return {
                'status': 'success',
                'user_id': user_id,
                'samples_processed': len(user_features),
                'profile_created': True
            }
            
        except Exception as e:
            logger.error(f"Error enrolling user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _create_user_profile(self, user_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a user profile from typing features."""
        # Get all feature names
        all_features = set()
        for features in user_features:
            all_features.update(features.keys())
        
        # Calculate statistics for each feature
        mean_features = {}
        std_features = {}
        
        for feature in all_features:
            values = [f.get(feature, 0.0) for f in user_features]
            numeric_values = [float(v) if isinstance(v, (int, float, bool)) else 0.0 for v in values]
            
            mean_features[feature] = np.mean(numeric_values)
            std_features[feature] = np.std(numeric_values)
        
        return {
            'mean_features': mean_features,
            'std_features': std_features,
            'sample_count': len(user_features),
            'created_at': datetime.now().isoformat()
        }
    
    def _train_models(self):
        """Train keystroke dynamics models."""
        if len(self.training_data) < 2:
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 10:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Train models
            for model_name, model in self.models.items():
                try:
                    if model_name == 'authentication':
                        model.fit(X_train_scaled, y_train_encoded)
                    elif model_name == 'anomaly_detection':
                        # Create anomaly labels (0 for normal, 1 for anomaly)
                        # For now, treat all training data as normal
                        anomaly_labels = np.zeros(len(X_train_scaled))
                        model.fit(X_train_scaled, anomaly_labels)
                    else:
                        model.fit(X_train_scaled, y_train_encoded)
                    
                    logger.info(f"Trained {model_name} model")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
            
            # Train neural network
            try:
                self.neural_model.fit(X_train_scaled, y_train_encoded)
                logger.info("Trained neural network model")
            except Exception as e:
                logger.warning(f"Failed to train neural network: {e}")
            
            # Evaluate models
            self._evaluate_models(X_test_scaled, y_test_encoded)
            
            self.models_trained = True
            self.stats['model_updates'] += 1
            self.stats['last_training'] = datetime.now().isoformat()
            
            logger.info(f"Models trained with {len(X)} samples from {len(set(y))} users")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from user samples."""
        X = []
        y = []
        
        # Get all feature names
        all_features = set()
        for user_id, features_list in self.training_data.items():
            for features in features_list:
                all_features.update(features.keys())
        
        self.feature_names = sorted(list(all_features))
        
        # Create feature vectors
        for user_id, features_list in self.training_data.items():
            for features in features_list:
                vector = []
                for feature_name in self.feature_names:
                    value = features.get(feature_name, 0.0)
                    if isinstance(value, bool):
                        value = float(value)
                    elif not isinstance(value, (int, float)):
                        value = 0.0
                    vector.append(value)
                
                X.append(vector)
                y.append(user_id)
        
        return np.array(X), np.array(y)
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate trained models."""
        try:
            for model_name, model in self.models.items():
                if model_name == 'authentication':
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    logger.info(f"{model_name} accuracy: {accuracy:.3f}")
            
            # Evaluate neural network
            if hasattr(self.neural_model, 'predict'):
                nn_predictions = self.neural_model.predict(X_test)
                nn_accuracy = accuracy_score(y_test, nn_predictions)
                logger.info(f"Neural network accuracy: {nn_accuracy:.3f}")
                
        except Exception as e:
            logger.warning(f"Error evaluating models: {e}")
    
    def _adaptive_learning(self):
        """Implement adaptive learning for user pattern evolution."""
        if not self.user_authenticated or len(self.current_session_features) < 20:
            return
        
        try:
            # Update user profile with recent typing patterns
            user_profile = self.user_profiles.get(self.user_authenticated)
            if not user_profile:
                return
            
            # Get recent features
            recent_features = self.current_session_features[-20:]
            
            # Update profile with adaptive learning rate
            for features in recent_features:
                for feature_name, value in features.items():
                    if isinstance(value, (int, float)):
                        current_mean = user_profile['mean_features'].get(feature_name, value)
                        # Exponential moving average
                        new_mean = (1 - self.adaptation_rate) * current_mean + self.adaptation_rate * value
                        user_profile['mean_features'][feature_name] = new_mean
            
            logger.info(f"Updated profile for user {self.user_authenticated}")
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
    
    def _save_models(self):
        """Save trained models and user profiles."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            models_dir.mkdir(exist_ok=True)
            
            # Save models and data
            model_data = {
                'models': self.models,
                'neural_model': self.neural_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'user_profiles': self.user_profiles,
                'training_data': dict(self.training_data),
                'feature_names': getattr(self, 'feature_names', []),
                'models_trained': self.models_trained
            }
            
            with open(models_dir / 'keystroke_dynamics.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Keystroke dynamics models saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models and user profiles."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            model_file = models_dir / 'keystroke_dynamics.pkl'
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data.get('models', self.models)
                self.neural_model = model_data.get('neural_model', self.neural_model)
                self.scaler = model_data.get('scaler', self.scaler)
                self.label_encoder = model_data.get('label_encoder', self.label_encoder)
                self.user_profiles = model_data.get('user_profiles', {})
                self.training_data = defaultdict(list, model_data.get('training_data', {}))
                self.feature_names = model_data.get('feature_names', [])
                self.models_trained = model_data.get('models_trained', False)
                
                self.stats['users_enrolled'] = len(self.user_profiles)
                
                logger.info(f"Keystroke dynamics models loaded ({len(self.user_profiles)} users)")
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get keystroke dynamics statistics."""
        stats = self.stats.copy()
        stats.update({
            'keystroke_buffer_size': len(self.keystroke_buffer),
            'current_session_features': len(self.current_session_features),
            'user_profiles': len(self.user_profiles),
            'models_trained': self.models_trained,
            'authenticated_user': self.user_authenticated,
            'authentication_threshold': self.authentication_threshold
        })
        
        # Authentication success rate
        if self.stats['authentication_attempts'] > 0:
            stats['authentication_success_rate'] = (
                self.stats['successful_authentications'] / self.stats['authentication_attempts']
            )
        
        return stats
    
    def reset_session(self):
        """Reset current session data."""
        self.keystroke_buffer.clear()
        self.current_session_features.clear()
        self.user_authenticated = None
        
        logger.info("Keystroke dynamics session reset")
    
    def __del__(self):
        """Save models on destruction."""
        if self.models_trained:
            self._save_models()