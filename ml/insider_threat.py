"""Insider Threat Detection system with specialized ML models for behavior pattern analysis."""

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
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, classification_report
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from scipy import stats
from scipy.spatial.distance import mahalanobis

from .data_preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class InsiderThreatDetector:
    """Advanced insider threat detection system with multi-dimensional behavior analysis."""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        
        # Configuration
        self.threat_threshold = config.get('ml.insider_threat.threshold', 0.7)
        self.baseline_window = config.get('ml.insider_threat.baseline_window', 1000)
        self.analysis_window = config.get('ml.insider_threat.analysis_window', 100)
        self.correlation_threshold = config.get('ml.insider_threat.correlation_threshold', 0.8)
        
        # Multi-dimensional analysis models
        self.models = {
            'access_patterns': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            ),
            'data_usage': OneClassSVM(
                nu=0.1,
                kernel='rbf',
                gamma='scale'
            ),
            'temporal_behavior': DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            'privilege_escalation': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),
            'data_exfiltration': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        # Ensemble model for final threat scoring
        self.ensemble_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        )
        
        # Data storage
        self.baseline_behaviors = deque(maxlen=self.baseline_window)
        self.recent_activities = deque(maxlen=self.analysis_window)
        self.threat_incidents = deque(maxlen=500)
        self.user_baselines = defaultdict(dict)
        
        # Feature processing
        self.scalers = {
            'access': StandardScaler(),
            'usage': RobustScaler(),
            'temporal': StandardScaler(),
            'privilege': StandardScaler(),
            'exfiltration': StandardScaler()
        }
        
        self.feature_selectors = {
            'access': SelectKBest(f_classif, k=20),
            'usage': SelectKBest(f_classif, k=15),
            'temporal': SelectKBest(f_classif, k=10)
        }
        
        # Dimensionality reduction
        self.pca = PCA(n_components=0.95)
        self.ica = FastICA(n_components=10, random_state=42)
        
        # Model state
        self.models_trained = False
        self.baseline_established = False
        
        # Threat indicators tracking
        self.threat_indicators = {
            'unusual_access_times': 0,
            'excessive_data_access': 0,
            'privilege_attempts': 0,
            'suspicious_applications': 0,
            'data_transfer_anomalies': 0,
            'behavioral_deviations': 0
        }
        
        # Statistics
        self.stats = {
            'events_analyzed': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'baseline_samples': 0,
            'model_updates': 0,
            'last_training': None,
            'threat_categories': defaultdict(int)
        }
        
        # Load existing models
        self._load_models()
        
        logger.info("InsiderThreatDetector initialized")
    
    def analyze_event(self, event: Dict[str, Any]) -> float:
        """Analyze a single event for insider threat indicators."""
        try:
            # Extract threat-specific features
            threat_features = self._extract_threat_features(event)
            if not threat_features:
                return 0.0
            
            # Add to recent activities
            ts_value = event.get('timestamp')
            if isinstance(ts_value, str) and ts_value:
                timestamp_str = ts_value
            elif isinstance(ts_value, datetime):
                timestamp_str = ts_value.isoformat()
            else:
                timestamp_str = datetime.now().isoformat()

            self.recent_activities.append({
                'timestamp': timestamp_str,
                'features': threat_features,
                'event': event
            })
            
            self.stats['events_analyzed'] += 1
            
            # Multi-dimensional threat analysis
            threat_scores = self._multi_dimensional_analysis(threat_features, event)
            
            # Correlation analysis
            correlation_score = self._correlation_analysis()
            
            # Ensemble threat scoring
            final_threat_score = self._ensemble_threat_scoring(threat_scores, correlation_score)
            
            # Update baseline if not threatening
            if final_threat_score < self.threat_threshold:
                self._update_baseline(threat_features, event)
            else:
                self._record_threat_incident(event, final_threat_score, threat_scores)
            
            # Adaptive learning
            if self.stats['events_analyzed'] % 100 == 0:
                self._adaptive_learning()
            
            return final_threat_score
            
        except Exception as e:
            logger.error(f"Error analyzing event for insider threats: {e}")
            return 0.0
    
    def _extract_threat_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features specifically relevant to insider threat detection."""
        features = {}
        
        # Temporal features
        features.update(self._extract_temporal_threat_features(event))
        
        # Access pattern features
        features.update(self._extract_access_pattern_features(event))
        
        # Data usage features
        features.update(self._extract_data_usage_features(event))
        
        # Privilege escalation features
        features.update(self._extract_privilege_features(event))
        
        # Data exfiltration features
        features.update(self._extract_exfiltration_features(event))
        
        # Behavioral deviation features
        features.update(self._extract_behavioral_deviation_features(event))
        
        return features
    
    def _extract_temporal_threat_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features that may indicate insider threats."""
        features = {}
        
        timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
        
        # Time-based risk indicators
        features.update({
            'is_after_hours': timestamp.hour < 8 or timestamp.hour > 18,
            'is_weekend': timestamp.weekday() >= 5,
            'is_holiday': self._is_holiday(timestamp),
            'hour_risk_score': self._calculate_hour_risk(timestamp.hour),
            'day_risk_score': self._calculate_day_risk(timestamp.weekday())
        })
        
        # Activity timing patterns
        if len(self.recent_activities) > 0:
            last_activity = self.recent_activities[-1]
            last_timestamp = self._safe_parse_iso(last_activity.get('timestamp'))
            if last_timestamp is not None:
                time_gap = (timestamp - last_timestamp).total_seconds()
                features.update({
                    'time_gap': time_gap,
                    'rapid_activity': time_gap < 1.0,  # Less than 1 second
                    'extended_gap': time_gap > 3600.0,  # More than 1 hour
                    'burst_activity': self._detect_activity_burst(timestamp)
                })
            else:
                # Fallback defaults when previous timestamp is invalid
                features.update({
                    'time_gap': 0.0,
                    'rapid_activity': False,
                    'extended_gap': False,
                    'burst_activity': self._detect_activity_burst(timestamp)
                })
        
        return features
    
    def _extract_access_pattern_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract access pattern features for threat detection."""
        features = {}
        event_type = event.get('type', '')
        data = event.get('data', {})
        
        # File and application access patterns
        if event_type == 'window':
            app_name = data.get('application', '').lower()
            window_title = data.get('title', '').lower()
            
            # Suspicious application indicators
            features.update({
                'admin_tool_access': any(tool in app_name for tool in 
                                       ['cmd', 'powershell', 'regedit', 'services', 'taskmgr']),
                'file_manager_access': any(fm in app_name for fm in 
                                         ['explorer', 'finder', 'nautilus']),
                'network_tool_access': any(tool in app_name for tool in 
                                         ['wireshark', 'netstat', 'tcpdump']),
                'dev_tool_access': any(tool in app_name for tool in 
                                     ['studio', 'code', 'git', 'terminal']),
                'browser_private_mode': 'private' in window_title or 'incognito' in window_title
            })
        
        # Keyboard activity patterns
        elif event_type == 'keyboard':
            key = data.get('key', '').lower()
            content = str(data.get('content', '')).lower()
            
            # Suspicious keyboard patterns
            features.update({
                'admin_commands': any(cmd in content for cmd in 
                                    ['sudo', 'runas', 'net user', 'chmod', 'chown']),
                'file_operations': any(op in content for op in 
                                     ['copy', 'move', 'delete', 'rm', 'cp', 'mv']),
                'network_commands': any(cmd in content for cmd in 
                                      ['ping', 'nslookup', 'netstat', 'ssh', 'ftp']),
                'system_info_commands': any(cmd in content for cmd in 
                                          ['systeminfo', 'whoami', 'id', 'ps', 'top']),
                'shortcut_usage': key in ['ctrl+c', 'ctrl+v', 'ctrl+x', 'ctrl+a']
            })
        
        # USB and external device access
        elif 'usb' in event_type.lower() or 'device' in str(data).lower():
            features.update({
                'usb_access': True,
                'external_device_access': True,
                'device_connection_time': self._get_device_connection_time()
            })
        
        return features
    
    def _extract_data_usage_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data usage patterns that may indicate threats."""
        features = {}
        event_type = event.get('type', '')
        data = event.get('data', {})
        
        # Clipboard usage patterns
        if event_type == 'clipboard':
            raw_content = data.get('content', '')
            # Normalize content to a safe string
            if raw_content is None:
                content = ''
            elif isinstance(raw_content, str):
                content = raw_content
            else:
                try:
                    content = str(raw_content)
                except Exception:
                    content = ''
            
            features.update({
                'clipboard_size': len(content),
                'large_clipboard_operation': len(content) > 10000,
                'contains_sensitive_data': self._contains_sensitive_data(content),
                'contains_credentials': self._contains_credentials(content),
                'contains_file_paths': self._contains_file_paths(content),
                'contains_urls': self._contains_urls(content),
                'clipboard_frequency': self._get_clipboard_frequency()
            })
        
        # File access patterns
        content = str(data.get('content', '') or '').lower()
        features.update({
            'database_access': any(db in content for db in 
                                 ['database', 'sql', 'mysql', 'postgres', 'oracle']),
            'config_file_access': any(cfg in content for cfg in 
                                    ['config', '.conf', '.ini', '.xml', '.json']),
            'log_file_access': any(log in content for log in 
                                 ['.log', 'logs', 'audit', 'event']),
            'backup_access': any(bak in content for bak in 
                               ['backup', '.bak', '.zip', '.tar']),
            'sensitive_directory': any(dir in content for dir in 
                                     ['system32', 'etc', 'var', 'tmp', 'temp'])
        })
        
        return features
    
    def _extract_privilege_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract privilege escalation indicators."""
        features = {}
        data = event.get('data', {})
        content = str(data.get('content', '')).lower()
        
        # Privilege escalation attempts
        features.update({
            'sudo_usage': 'sudo' in content,
            'runas_usage': 'runas' in content,
            'admin_elevation': any(elev in content for elev in 
                                 ['administrator', 'admin', 'root', 'elevation']),
            'user_management': any(user in content for user in 
                                 ['net user', 'useradd', 'usermod', 'passwd']),
            'group_management': any(group in content for group in 
                                  ['net group', 'groupadd', 'groupmod']),
            'service_management': any(svc in content for svc in 
                                    ['sc create', 'systemctl', 'service']),
            'registry_access': any(reg in content for reg in 
                                 ['regedit', 'reg add', 'reg delete']),
            'policy_changes': any(pol in content for pol in 
                                ['gpedit', 'secpol', 'policy'])
        })
        
        return features
    
    def _extract_exfiltration_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data exfiltration indicators."""
        features = {}
        data = event.get('data', {})
        content = str(data.get('content', '')).lower()
        
        # Data transfer indicators
        features.update({
            'file_compression': any(comp in content for comp in 
                                  ['zip', 'rar', '7z', 'tar', 'gzip']),
            'encryption_tools': any(enc in content for enc in 
                                  ['encrypt', 'gpg', 'openssl', 'cipher']),
            'cloud_services': any(cloud in content for cloud in 
                                ['dropbox', 'google drive', 'onedrive', 'icloud']),
            'file_transfer': any(transfer in content for transfer in 
                               ['ftp', 'sftp', 'scp', 'rsync', 'wget', 'curl']),
            'email_attachment': any(email in content for email in 
                                  ['attachment', 'send', 'forward', 'mailto']),
            'usb_transfer': self._detect_usb_transfer(),
            'network_upload': self._detect_network_upload(content),
            'large_file_operations': self._detect_large_file_operations(content)
        })
        
        return features
    
    def _extract_behavioral_deviation_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral deviation indicators."""
        features = {}
        
        if not self.baseline_established:
            return features
        
        # Compare current behavior to established baseline
        current_behavior = self._get_current_behavior_profile()
        baseline_behavior = self._get_baseline_behavior_profile()
        
        # Calculate deviations
        for metric, current_value in current_behavior.items():
            baseline_value = baseline_behavior.get(metric, current_value)
            if baseline_value != 0:
                deviation = abs(current_value - baseline_value) / baseline_value
                features[f'{metric}_deviation'] = deviation
                features[f'{metric}_anomaly'] = deviation > 2.0  # 2 standard deviations
        
        return features
    
    def _multi_dimensional_analysis(self, features: Dict[str, Any], 
                                  event: Dict[str, Any]) -> Dict[str, float]:
        """Perform multi-dimensional threat analysis."""
        threat_scores = {}
        
        if not self.models_trained:
            return {model_name: 0.0 for model_name in self.models.keys()}
        
        try:
            # Prepare feature vectors for each dimension
            feature_vectors = self._prepare_dimensional_features(features)
            
            # Analyze each dimension
            for dimension, model in self.models.items():
                if dimension in feature_vectors:
                    vector = feature_vectors[dimension]
                    score = self._get_model_score(model, vector, dimension)
                    threat_scores[dimension] = score
                else:
                    threat_scores[dimension] = 0.0
            
            return threat_scores
            
        except Exception as e:
            logger.error(f"Error in multi-dimensional analysis: {e}")
            return {model_name: 0.0 for model_name in self.models.keys()}
    
    def _prepare_dimensional_features(self, features: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Prepare feature vectors for different analysis dimensions."""
        dimensional_features = {
            'access_patterns': [],
            'data_usage': [],
            'temporal_behavior': [],
            'privilege_escalation': [],
            'data_exfiltration': []
        }
        
        # Map features to dimensions
        access_features = [k for k in features.keys() if any(term in k for term in 
                          ['access', 'application', 'window', 'file'])]
        usage_features = [k for k in features.keys() if any(term in k for term in 
                         ['clipboard', 'data', 'usage', 'size'])]
        temporal_features = [k for k in features.keys() if any(term in k for term in 
                           ['time', 'hour', 'day', 'gap', 'burst'])]
        privilege_features = [k for k in features.keys() if any(term in k for term in 
                            ['sudo', 'admin', 'elevation', 'privilege'])]
        exfiltration_features = [k for k in features.keys() if any(term in k for term in 
                               ['transfer', 'compression', 'encryption', 'cloud'])]
        
        # Create feature vectors
        for feature_set, dimension in [
            (access_features, 'access_patterns'),
            (usage_features, 'data_usage'),
            (temporal_features, 'temporal_behavior'),
            (privilege_features, 'privilege_escalation'),
            (exfiltration_features, 'data_exfiltration')
        ]:
            if feature_set:
                vector = [float(features.get(f, 0.0)) if isinstance(features.get(f, 0.0), (int, float, bool)) 
                         else 0.0 for f in feature_set]
                if vector:
                    dimensional_features[dimension] = np.array(vector)
        
        return dimensional_features
    
    def _get_model_score(self, model, feature_vector: np.ndarray, dimension: str) -> float:
        """Get threat score from a specific model."""
        try:
            if len(feature_vector) == 0:
                return 0.0
            
            # Ensure consistent feature dimensions
            if hasattr(self, 'all_feature_names'):
                expected_features = len(self.all_feature_names)
                if len(feature_vector) != expected_features:
                    # Pad or truncate features to match expected dimensions
                    if len(feature_vector) < expected_features:
                        padding = np.zeros(expected_features - len(feature_vector))
                        feature_vector = np.hstack([feature_vector, padding])
                    else:
                        feature_vector = feature_vector[:expected_features]
            
            # Initialize scaled_vector with default value
            scaled_vector = feature_vector.reshape(1, -1)
            
            # Scale features if scaler is available and fitted
            if dimension in self.scalers:
                scaler = self.scalers[dimension]
                try:
                    # Check if scaler is fitted by looking for fitted attributes
                    if hasattr(scaler, 'mean_') or hasattr(scaler, 'scale_') or hasattr(scaler, 'center_'):
                        scaled_vector = scaler.transform(feature_vector.reshape(1, -1))
                    else:
                        logger.debug(f"Scaler for {dimension} not fitted yet, using unscaled features")
                except Exception as e:
                    logger.warning(f"Error scaling features for {dimension}: {e}")
                    # scaled_vector already set to default above
            
            # Check if model is fitted (avoid triggering properties on unfitted models)
            if hasattr(model, 'fit_predict'):
                # Unsupervised clustering (e.g., DBSCAN) - fit per sample window
                prediction = model.fit_predict(scaled_vector)[0]
                return 1.0 if prediction == -1 else 0.0
            else:
                try:
                    check_is_fitted(model)
                except Exception:
                    logger.warning(f"{dimension} model is not fitted yet")
                    return 0.0

            # Get score based on model type
            if hasattr(model, 'decision_function'):
                score = model.decision_function(scaled_vector)[0]
                # Convert to 0-1 scale (negative scores indicate anomalies)
                return max(0.0, min(1.0, (-score + 1) / 2))
            
            elif hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled_vector)[0]
                return proba[1] if len(proba) > 1 else proba[0]
            
            else:
                prediction = model.predict(scaled_vector)[0]
                return float(prediction)
            
        except Exception as e:
            logger.warning(f"Error getting score from {dimension} model: {e}")
            return 0.0
    
    def _correlation_analysis(self) -> float:
        """Analyze correlations between different behavioral indicators."""
        if len(self.recent_activities) < 10:
            return 0.0
        
        try:
            # Get recent threat indicators
            recent_indicators = []
            for activity in list(self.recent_activities)[-10:]:
                indicators = []
                features = activity['features']
                
                # Count threat indicators
                for category in ['access', 'usage', 'temporal', 'privilege', 'exfiltration']:
                    category_indicators = sum(1 for k, v in features.items() 
                                            if category in k and v)
                    indicators.append(category_indicators)
                
                recent_indicators.append(indicators)
            
            if len(recent_indicators) < 5:
                return 0.0
            
            # Guard against constant or invalid columns
            indicators_array = np.array(recent_indicators).T
            if indicators_array.ndim != 2 or indicators_array.shape[0] < 2:
                return 0.0
            
            # Compute correlation with robust NaN/Inf handling
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation_matrix = np.corrcoef(indicators_array)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get average correlation (excluding diagonal)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            if correlation_matrix.size == 0 or np.count_nonzero(mask) == 0:
                return 0.0
            avg_correlation = np.mean(np.abs(correlation_matrix[mask]))
            
            # High correlation indicates coordinated suspicious activity
            return float(min(1.0, max(0.0, avg_correlation)))
            
        except Exception as e:
            logger.warning(f"Error in correlation analysis: {e}")
            return 0.0
    
    def _ensemble_threat_scoring(self, threat_scores: Dict[str, float], 
                               correlation_score: float) -> float:
        """Combine multiple threat scores into final ensemble score."""
        if not threat_scores:
            return 0.0
        
        # Weighted ensemble
        weights = {
            'access_patterns': 0.2,
            'data_usage': 0.25,
            'temporal_behavior': 0.15,
            'privilege_escalation': 0.25,
            'data_exfiltration': 0.3
        }
        
        # Calculate weighted average
        weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, score in threat_scores.items():
            weight = weights.get(dimension, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = 0.0
        
        # Apply correlation boost
        correlation_boost = correlation_score * 0.2  # Up to 20% boost
        final_score = min(1.0, base_score + correlation_boost)
        
        # Apply threat indicator multipliers
        multiplier = self._calculate_threat_multiplier()
        final_score = min(1.0, final_score * multiplier)
        
        return final_score
    
    def _calculate_threat_multiplier(self) -> float:
        """Calculate threat multiplier based on recent indicators."""
        multiplier = 1.0
        
        # Recent threat indicator counts
        recent_count = sum(self.threat_indicators.values())
        
        if recent_count > 10:
            multiplier += 0.3  # 30% increase
        elif recent_count > 5:
            multiplier += 0.15  # 15% increase
        
        # Time-based multipliers
        current_time = datetime.now()
        if current_time.hour < 8 or current_time.hour > 18:
            multiplier += 0.1  # After hours
        
        if current_time.weekday() >= 5:
            multiplier += 0.1  # Weekend
        
        return multiplier
    
    def _update_baseline(self, features: Dict[str, Any], event: Dict[str, Any]):
        """Update baseline behavior model."""
        self.baseline_behaviors.append({
            'timestamp': event.get('timestamp'),
            'features': features,
            'event_type': event.get('type')
        })
        
        self.stats['baseline_samples'] += 1
        
        # Retrain models periodically
        if len(self.baseline_behaviors) >= 100 and len(self.baseline_behaviors) % 100 == 0:
            self._train_models()
    
    def _record_threat_incident(self, event: Dict[str, Any], threat_score: float, 
                              dimension_scores: Dict[str, float]):
        """Record a threat incident for analysis."""
        incident = {
            'timestamp': event.get('timestamp'),
            'threat_score': threat_score,
            'dimension_scores': dimension_scores,
            'event': event,
            'severity': self._calculate_severity(threat_score),
            'primary_indicators': self._get_primary_indicators(dimension_scores)
        }
        
        self.threat_incidents.append(incident)
        self.stats['threats_detected'] += 1
        
        # Update threat category statistics
        primary_category = max(dimension_scores.items(), key=lambda x: x[1])[0]
        self.stats['threat_categories'][primary_category] += 1
        
        # Update threat indicators
        self._update_threat_indicators(dimension_scores)
        
        logger.warning(f"Insider threat detected: score={threat_score:.3f}, "
                      f"primary={primary_category}, event={event.get('type')}")
    
    def _calculate_severity(self, threat_score: float) -> str:
        """Calculate threat severity level."""
        if threat_score >= 0.9:
            return 'critical'
        elif threat_score >= 0.8:
            return 'high'
        elif threat_score >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _get_primary_indicators(self, dimension_scores: Dict[str, float]) -> List[str]:
        """Get primary threat indicators."""
        # Sort dimensions by score
        sorted_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top indicators above threshold
        primary = [dim for dim, score in sorted_dimensions if score > 0.5]
        return primary[:3]  # Top 3 indicators
    
    def _update_threat_indicators(self, dimension_scores: Dict[str, float]):
        """Update threat indicator counters."""
        for dimension, score in dimension_scores.items():
            if score > 0.5:
                if 'access' in dimension:
                    self.threat_indicators['unusual_access_times'] += 1
                elif 'usage' in dimension:
                    self.threat_indicators['excessive_data_access'] += 1
                elif 'privilege' in dimension:
                    self.threat_indicators['privilege_attempts'] += 1
                elif 'exfiltration' in dimension:
                    self.threat_indicators['data_transfer_anomalies'] += 1
                else:
                    self.threat_indicators['behavioral_deviations'] += 1
    
    def _train_models(self):
        """Train insider threat detection models."""
        if len(self.baseline_behaviors) < 50:
            return
        
        try:
            # Prepare training data
            X_normal, X_threat = self._prepare_training_data()
            
            if len(X_normal) < 20:
                return
            
            # Initialize feature names if not already done
            if not hasattr(self, 'all_feature_names'):
                self._initialize_feature_names()
            
            # Ensure consistent feature dimensions
            expected_features = len(self.all_feature_names)
            
            # Train each dimensional model
            for dimension, model in self.models.items():
                try:
                    if dimension in ['access_patterns', 'data_usage', 'temporal_behavior']:
                        # Unsupervised models - train on normal data only
                        if len(X_normal) > 0:
                            # Ensure consistent feature dimensions for training
                            if X_normal.shape[1] != expected_features:
                                # Pad or truncate features to match expected dimensions
                                if X_normal.shape[1] < expected_features:
                                    padding = np.zeros((X_normal.shape[0], expected_features - X_normal.shape[1]))
                                    X_normal_padded = np.hstack([X_normal, padding])
                                else:
                                    X_normal_padded = X_normal[:, :expected_features]
                            else:
                                X_normal_padded = X_normal
                            
                            if dimension in self.scalers:
                                X_scaled = self.scalers[dimension].fit_transform(X_normal_padded)
                            else:
                                # Initialize scaler for this dimension
                                self.scalers[dimension] = StandardScaler()
                                X_scaled = self.scalers[dimension].fit_transform(X_normal_padded)
                            
                            if hasattr(model, 'fit'):
                                model.fit(X_scaled)
                    
                    else:
                        # Supervised models - need both normal and threat data
                        # For now, create synthetic threat data if we don't have enough real threat examples
                        if len(X_threat) < 5:
                            # Generate synthetic threat examples by adding noise to normal data
                            np.random.seed(42)
                            synthetic_threat = X_normal_padded + np.random.normal(0, 0.5, X_normal_padded.shape)
                            # Take a small subset as synthetic threats
                            n_synthetic = min(10, len(X_normal_padded) // 10)
                            X_threat_synthetic = synthetic_threat[:n_synthetic]
                            
                            X_combined = np.vstack([X_normal_padded, X_threat_synthetic])
                            y_combined = np.hstack([
                                np.zeros(len(X_normal_padded)),
                                np.ones(len(X_threat_synthetic))
                            ])
                        else:
                            # Use real threat data
                            # Ensure consistent feature dimensions for threat data
                            if X_threat.shape[1] != expected_features:
                                if X_threat.shape[1] < expected_features:
                                    padding = np.zeros((X_threat.shape[0], expected_features - X_threat.shape[1]))
                                    X_threat_padded = np.hstack([X_threat, padding])
                                else:
                                    X_threat_padded = X_threat[:, :expected_features]
                            else:
                                X_threat_padded = X_threat
                            
                            X_combined = np.vstack([X_normal_padded, X_threat_padded])
                            y_combined = np.hstack([
                                np.zeros(len(X_normal_padded)),
                                np.ones(len(X_threat_padded))
                            ])
                        
                        if dimension in self.scalers:
                            X_scaled = self.scalers[dimension].fit_transform(X_combined)
                        else:
                            # Initialize scaler for this dimension
                            self.scalers[dimension] = StandardScaler()
                            X_scaled = self.scalers[dimension].fit_transform(X_combined)
                        
                        model.fit(X_scaled, y_combined)
                    
                    logger.info(f"Trained {dimension} model")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {dimension} model: {e}")
            
            self.models_trained = True
            self.baseline_established = True
            self.stats['model_updates'] += 1
            self.stats['last_training'] = datetime.now().isoformat()
            
            logger.info(f"Insider threat models trained with {len(X_normal)} normal samples")
            
        except Exception as e:
            logger.error(f"Error training insider threat models: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from baseline and threat incidents."""
        # Normal behavior data
        normal_features = []
        for behavior in self.baseline_behaviors:
            features = behavior['features']
            # Convert to feature vector
            vector = self._features_to_vector(features)
            if vector is not None:
                normal_features.append(vector)
        
        # Threat behavior data
        threat_features = []
        for incident in self.threat_incidents:
            event = incident['event']
            features = self._extract_threat_features(event)
            vector = self._features_to_vector(features)
            if vector is not None:
                threat_features.append(vector)
        
        X_normal = np.array(normal_features) if normal_features else np.array([]).reshape(0, 0)
        X_threat = np.array(threat_features) if threat_features else np.array([]).reshape(0, 0)
        
        return X_normal, X_threat
    
    def _features_to_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert feature dictionary to vector."""
        try:
            # Get all possible feature names
            if not hasattr(self, 'all_feature_names'):
                self._initialize_feature_names()
            
            vector = []
            for feature_name in self.all_feature_names:
                value = features.get(feature_name, 0.0)
                if isinstance(value, bool):
                    value = float(value)
                elif not isinstance(value, (int, float)):
                    value = 0.0
                vector.append(value)
            
            return np.array(vector)
            
        except Exception as e:
            logger.warning(f"Error converting features to vector: {e}")
            return None
    
    def _initialize_feature_names(self):
        """Initialize the list of all possible feature names."""
        all_features = set()
        
        # Collect feature names from baseline behaviors
        for behavior in self.baseline_behaviors:
            all_features.update(behavior['features'].keys())
        
        # Collect feature names from threat incidents
        for incident in self.threat_incidents:
            event = incident['event']
            features = self._extract_threat_features(event)
            all_features.update(features.keys())
        
        self.all_feature_names = sorted(list(all_features))
    
    def _adaptive_learning(self):
        """Implement adaptive learning for evolving threat patterns."""
        if not self.baseline_established:
            return
        
        try:
            # Analyze recent threat patterns
            recent_threats = [incident for incident in self.threat_incidents 
                            if self._is_recent(incident['timestamp'])]
            
            if len(recent_threats) > 5:
                # Update threat thresholds based on recent patterns
                recent_scores = [t['threat_score'] for t in recent_threats]
                avg_threat_score = np.mean(recent_scores)
                
                # Adaptive threshold adjustment
                if avg_threat_score > self.threat_threshold + 0.1:
                    # Many high-scoring threats - might need to be more sensitive
                    self.threat_threshold = max(0.5, self.threat_threshold - 0.05)
                elif avg_threat_score < self.threat_threshold - 0.1:
                    # Few threats or low scores - might be too sensitive
                    self.threat_threshold = min(0.9, self.threat_threshold + 0.05)
                
                logger.info(f"Adaptive threshold adjustment: {self.threat_threshold:.3f}")
            
            # Retrain models with recent data
            if len(recent_threats) > 10:
                self._train_models()
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
    
    def _is_recent(self, timestamp_str: str, hours: int = 24) -> bool:
        """Check if timestamp is within recent hours."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return (datetime.now() - timestamp).total_seconds() < hours * 3600
        except:
            return False
    
    # Helper methods for feature extraction
    def _is_holiday(self, timestamp: datetime) -> bool:
        """Check if date is a holiday (simplified implementation)."""
        # This is a simplified implementation
        # In production, you'd use a proper holiday calendar
        return False
    
    def _calculate_hour_risk(self, hour: int) -> float:
        """Calculate risk score based on hour of day."""
        # Higher risk for unusual hours
        if 2 <= hour <= 6:  # Very early morning
            return 0.9
        elif hour <= 8 or hour >= 20:  # Early morning or late evening
            return 0.7
        elif 9 <= hour <= 17:  # Business hours
            return 0.1
        else:
            return 0.5
    
    def _calculate_day_risk(self, weekday: int) -> float:
        """Calculate risk score based on day of week."""
        if weekday >= 5:  # Weekend
            return 0.8
        else:  # Weekday
            return 0.2
    
    def _detect_activity_burst(self, current_time: datetime) -> bool:
        """Detect if current activity is part of a burst."""
        if len(self.recent_activities) < 5:
            return False
        
        recent_times = []
        for a in list(self.recent_activities)[-5:]:
            ts = self._safe_parse_iso(a.get('timestamp'))
            if ts is not None:
                recent_times.append(ts)
        recent_times.append(current_time)
        
        # Check if all activities are within a short time window
        if not recent_times:
            return False
        time_span = (max(recent_times) - min(recent_times)).total_seconds()
        return time_span < 60  # 1 minute
    
    def _contains_sensitive_data(self, content: str) -> bool:
        """Check if content contains sensitive data patterns."""
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'password\s*[:=]\s*\S+',  # Password
            r'api[_\s]?key\s*[:=]\s*\S+',  # API key
        ]
        
        import re
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in sensitive_patterns)
    
    def _contains_credentials(self, content: str) -> bool:
        """Check if content contains credential patterns."""
        credential_keywords = ['password', 'passwd', 'pwd', 'token', 'key', 'secret', 'auth']
        return any(keyword in content.lower() for keyword in credential_keywords)
    
    def _contains_file_paths(self, content: str) -> bool:
        """Check if content contains file path patterns."""
        import re
        path_patterns = [
            r'[A-Za-z]:\\[^\s]+',  # Windows paths
            r'/[^\s]+',  # Unix paths
        ]
        return any(re.search(pattern, content) for pattern in path_patterns)
    
    def _contains_urls(self, content: str) -> bool:
        """Check if content contains URL patterns."""
        import re
        url_pattern = r'https?://[^\s]+'
        return bool(re.search(url_pattern, content, re.IGNORECASE))
    
    def _get_clipboard_frequency(self) -> float:
        """Get clipboard usage frequency in recent activities."""
        if len(self.recent_activities) < 10:
            return 0.0
        
        clipboard_events = sum(1 for a in self.recent_activities 
                             if a['event'].get('type') == 'clipboard')
        return clipboard_events / max(len(self.recent_activities), 1)
    
    def _get_device_connection_time(self) -> float:
        """Get time since last device connection."""
        # Simplified implementation
        return 0.0
    
    def _detect_usb_transfer(self) -> bool:
        """Detect USB transfer activity."""
        # Check recent activities for USB-related events
        recent_usb = any('usb' in str(a['event']).lower() 
                        for a in list(self.recent_activities)[-10:])
        return recent_usb
    
    def _detect_network_upload(self, content: str) -> bool:
        """Detect network upload activity."""
        upload_keywords = ['upload', 'send', 'post', 'put', 'transfer']
        return any(keyword in content.lower() for keyword in upload_keywords)
    
    def _detect_large_file_operations(self, content: str) -> bool:
        """Detect large file operations."""
        large_file_keywords = ['gb', 'gigabyte', 'tb', 'terabyte', 'large', 'big']
        return any(keyword in content.lower() for keyword in large_file_keywords)
    
    def _get_current_behavior_profile(self) -> Dict[str, float]:
        """Get current behavior profile metrics."""
        if len(self.recent_activities) < 10:
            return {}
        
        recent = list(self.recent_activities)[-10:]
        
        # Calculate behavior metrics
        profile = {
            'activity_rate': len(recent) / 600,  # per 10 minutes
            'clipboard_usage': sum(1 for a in recent if a['event'].get('type') == 'clipboard'),
            'application_switches': len(set(a['event'].get('data', {}).get('application', '') 
                                          for a in recent if a['event'].get('type') == 'window')),
            'after_hours_activity': sum(1 for a in recent 
                                      if self._is_after_hours(a['timestamp']))
        }
        
        return profile
    
    def _get_baseline_behavior_profile(self) -> Dict[str, float]:
        """Get baseline behavior profile metrics."""
        if len(self.baseline_behaviors) < 50:
            return {}
        
        baseline = list(self.baseline_behaviors)[-100:]  # Last 100 baseline samples
        
        # Calculate baseline metrics
        profile = {
            'activity_rate': len(baseline) / (len(baseline) * 10),  # Normalized
            'clipboard_usage': sum(1 for b in baseline 
                                 if b['event_type'] == 'clipboard') / max(len(baseline), 1),
            'application_switches': 0.1,  # Simplified
            'after_hours_activity': sum(1 for b in baseline 
                                      if self._is_after_hours(b['timestamp'])) / max(len(baseline), 1)
        }
        
        return profile
    
    def _is_after_hours(self, timestamp_str: str) -> bool:
        """Check if timestamp is after business hours."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return timestamp.hour < 8 or timestamp.hour > 18
        except:
            return False

    def _safe_parse_iso(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Safely parse ISO timestamp strings, returning None if invalid."""
        try:
            if not timestamp_str:
                return None
            return datetime.fromisoformat(timestamp_str)
        except Exception:
            return None
    
    def _save_models(self):
        """Save trained models and data."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            models_dir.mkdir(exist_ok=True)
            
            model_data = {
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'user_baselines': dict(self.user_baselines),
                'threat_indicators': self.threat_indicators,
                'all_feature_names': getattr(self, 'all_feature_names', []),
                'models_trained': self.models_trained,
                'baseline_established': self.baseline_established,
                'threat_threshold': self.threat_threshold
            }
            
            with open(models_dir / 'insider_threat.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Insider threat models saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models and data."""
        try:
            models_dir = Path(self.config.get('ml.models_dir', 'models'))
            model_file = models_dir / 'insider_threat.pkl'
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data.get('models', self.models)
                self.ensemble_model = model_data.get('ensemble_model', self.ensemble_model)
                self.scalers = model_data.get('scalers', self.scalers)
                self.feature_selectors = model_data.get('feature_selectors', self.feature_selectors)
                self.user_baselines = defaultdict(dict, model_data.get('user_baselines', {}))
                self.threat_indicators = model_data.get('threat_indicators', self.threat_indicators)
                self.all_feature_names = model_data.get('all_feature_names', [])
                self.models_trained = model_data.get('models_trained', False)
                self.baseline_established = model_data.get('baseline_established', False)
                self.threat_threshold = model_data.get('threat_threshold', self.threat_threshold)
                
                logger.info("Insider threat models loaded")
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of threat detection results."""
        recent_threats = [t for t in self.threat_incidents 
                         if self._is_recent(t['timestamp'], 24)]
        
        return {
            'total_threats': len(self.threat_incidents),
            'recent_threats_24h': len(recent_threats),
            'threat_categories': dict(self.stats['threat_categories']),
            'threat_indicators': self.threat_indicators.copy(),
            'current_threshold': self.threat_threshold,
            'severity_distribution': self._get_severity_distribution(recent_threats)
        }
    
    def _get_severity_distribution(self, threats: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of threat severities."""
        distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for threat in threats:
            severity = threat.get('severity', 'low')
            distribution[severity] += 1
        
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get insider threat detection statistics."""
        stats = self.stats.copy()
        stats.update({
            'baseline_samples': len(self.baseline_behaviors),
            'recent_activities': len(self.recent_activities),
            'threat_incidents': len(self.threat_incidents),
            'models_trained': self.models_trained,
            'baseline_established': self.baseline_established,
            'threat_threshold': self.threat_threshold,
            'threat_indicators': self.threat_indicators.copy()
        })
        
        # Detection rate
        if self.stats['events_analyzed'] > 0:
            stats['threat_detection_rate'] = self.stats['threats_detected'] / self.stats['events_analyzed']
        
        return stats
    
    def reset_baseline(self):
        """Reset baseline behavior model."""
        self.baseline_behaviors.clear()
        self.user_baselines.clear()
        self.threat_indicators = {key: 0 for key in self.threat_indicators.keys()}
        self.models_trained = False
        self.baseline_established = False
        
        logger.info("Insider threat baseline reset")
    
    def __del__(self):
        """Save models on destruction."""
        if self.models_trained:
            self._save_models()