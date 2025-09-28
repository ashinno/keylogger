"""Data preprocessing and feature extraction for ML anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses raw keylogger data for ML analysis."""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.vectorizers = {}
        self.feature_cache = defaultdict(deque)
        self.window_size = config.get('ml.preprocessing.window_size', 100)
        self.cache_size = config.get('ml.preprocessing.cache_size', 1000)
        
        # Initialize scalers
        self.timing_scaler = StandardScaler()
        self.frequency_scaler = MinMaxScaler()
        
        # Text vectorizer for content analysis
        self.text_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logger.info("DataPreprocessor initialized")
    
    def extract_keystroke_features(self, keystroke_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from keystroke events for dynamics analysis."""
        if not keystroke_events:
            return {}
        
        features = {}
        
        # Timing features
        dwell_times = []
        flight_times = []
        
        for i, event in enumerate(keystroke_events):
            if event.get('type') == 'key_press':
                # Dwell time (key press to release)
                press_time = event.get('timestamp', 0)
                release_event = self._find_release_event(keystroke_events, i, event.get('key'))
                if release_event:
                    dwell_time = release_event.get('timestamp', 0) - press_time
                    dwell_times.append(dwell_time)
                
                # Flight time (release to next press)
                if i > 0:
                    prev_release = self._find_previous_release(keystroke_events, i)
                    if prev_release:
                        flight_time = press_time - prev_release.get('timestamp', 0)
                        flight_times.append(flight_time)
        
        # Statistical features
        if dwell_times:
            features.update({
                'dwell_mean': np.mean(dwell_times),
                'dwell_std': np.std(dwell_times),
                'dwell_median': np.median(dwell_times),
                'dwell_min': np.min(dwell_times),
                'dwell_max': np.max(dwell_times)
            })
        
        if flight_times:
            features.update({
                'flight_mean': np.mean(flight_times),
                'flight_std': np.std(flight_times),
                'flight_median': np.median(flight_times),
                'flight_min': np.min(flight_times),
                'flight_max': np.max(flight_times)
            })
        
        # Rhythm features
        features.update(self._extract_rhythm_features(keystroke_events))
        
        # Key frequency features
        features.update(self._extract_key_frequency_features(keystroke_events))
        
        return features
    
    def extract_behavioral_features(self, events: List[Dict[str, Any]], 
                                  time_window: timedelta = timedelta(minutes=10)) -> Dict[str, Any]:
        """Extract behavioral features from mixed event types."""
        if not events:
            return {}
        
        features = {}
        
        # Time-based features
        features.update(self._extract_temporal_features(events, time_window))
        
        # Activity patterns
        features.update(self._extract_activity_patterns(events))
        
        # Application usage patterns
        features.update(self._extract_application_features(events))
        
        # Mouse behavior patterns
        features.update(self._extract_mouse_features(events))
        
        # Clipboard usage patterns
        features.update(self._extract_clipboard_features(events))
        
        return features
    
    def extract_insider_threat_features(self, events: List[Dict[str, Any]], 
                                      user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract features specifically for insider threat detection."""
        features = {}
        
        # Access pattern anomalies
        features.update(self._extract_access_patterns(events))
        
        # Data exfiltration indicators
        features.update(self._extract_exfiltration_indicators(events))
        
        # Privilege escalation indicators
        features.update(self._extract_privilege_indicators(events))
        
        # Behavioral deviation metrics
        features.update(self._extract_deviation_metrics(events, user_context))
        
        # Network activity correlation
        features.update(self._extract_network_correlation(events))
        
        return features
    
    def _extract_rhythm_features(self, keystroke_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract typing rhythm features."""
        features = {}
        
        # Inter-keystroke intervals
        intervals = []
        timestamps = [e.get('timestamp', 0) for e in keystroke_events if e.get('type') == 'key_press']
        
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
        
        if intervals:
            # Rhythm consistency
            features['rhythm_consistency'] = 1.0 / (1.0 + np.std(intervals))
            
            # Typing speed (WPM approximation)
            total_time = sum(intervals)
            if total_time > 0:
                features['typing_speed'] = len(keystroke_events) / total_time * 60
            
            # Rhythm entropy
            features['rhythm_entropy'] = self._calculate_entropy(intervals)
        
        return features
    
    def _extract_key_frequency_features(self, keystroke_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key frequency and pattern features."""
        features = {}
        
        # Key frequency distribution
        key_counts = defaultdict(int)
        for event in keystroke_events:
            if event.get('type') == 'key_press':
                key = event.get('key', '')
                key_counts[key] += 1
        
        if key_counts:
            total_keys = sum(key_counts.values())
            
            # Most common keys
            sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
            features['most_common_key_freq'] = sorted_keys[0][1] / total_keys if sorted_keys else 0
            
            # Key diversity (entropy)
            frequencies = [count / total_keys for count in key_counts.values()]
            features['key_diversity'] = self._calculate_entropy(frequencies)
            
            # Special key usage
            special_keys = ['shift', 'ctrl', 'alt', 'tab', 'enter', 'backspace', 'delete']
            special_count = sum(key_counts.get(key, 0) for key in special_keys)
            features['special_key_ratio'] = special_count / total_keys
        
        return features
    
    def _extract_temporal_features(self, events: List[Dict[str, Any]], 
                                 time_window: timedelta) -> Dict[str, Any]:
        """Extract time-based behavioral features."""
        features = {}
        
        if not events:
            return features
        
        # Activity intensity over time
        timestamps = [datetime.fromisoformat(e.get('timestamp', '')) for e in events 
                     if e.get('timestamp')]
        
        if timestamps:
            # Activity rate
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span > 0:
                features['activity_rate'] = len(events) / time_span
            
            # Peak activity periods
            hourly_activity = defaultdict(int)
            for ts in timestamps:
                hourly_activity[ts.hour] += 1
            
            if hourly_activity:
                features['peak_hour'] = max(hourly_activity.items(), key=lambda x: x[1])[0]
                features['activity_variance'] = np.var(list(hourly_activity.values()))
        
        return features
    
    def _extract_activity_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract activity pattern features."""
        features = {}
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in events:
            event_types[event.get('type', 'unknown')] += 1
        
        total_events = len(events)
        if total_events > 0:
            features['keyboard_ratio'] = event_types.get('keyboard', 0) / total_events
            features['mouse_ratio'] = event_types.get('mouse', 0) / total_events
            features['clipboard_ratio'] = event_types.get('clipboard', 0) / total_events
            features['window_ratio'] = event_types.get('window', 0) / total_events
        
        # Activity bursts
        features['activity_bursts'] = self._detect_activity_bursts(events)
        
        return features
    
    def _extract_application_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract application usage features."""
        features = {}
        
        # Application switching patterns
        app_switches = 0
        current_app = None
        app_usage = defaultdict(int)
        
        for event in events:
            if event.get('type') == 'window':
                app_name = event.get('data', {}).get('application', '')
                if app_name:
                    app_usage[app_name] += 1
                    if current_app and current_app != app_name:
                        app_switches += 1
                    current_app = app_name
        
        features['app_switches'] = app_switches
        features['unique_apps'] = len(app_usage)
        
        if app_usage:
            # Application diversity
            total_usage = sum(app_usage.values())
            app_frequencies = [count / total_usage for count in app_usage.values()]
            features['app_diversity'] = self._calculate_entropy(app_frequencies)
        
        return features
    
    def _extract_mouse_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract mouse behavior features."""
        features = {}
        
        mouse_events = [e for e in events if e.get('type') == 'mouse']
        
        if mouse_events:
            # Click patterns
            clicks = [e for e in mouse_events if e.get('data', {}).get('action') == 'click']
            features['click_rate'] = len(clicks) / max(len(mouse_events), 1) if mouse_events else 0
            
            # Movement patterns
            movements = [e for e in mouse_events if e.get('data', {}).get('action') == 'move']
            if movements:
                positions = [(e.get('data', {}).get('x', 0), e.get('data', {}).get('y', 0)) 
                           for e in movements]
                
                # Calculate movement distances
                distances = []
                for i in range(1, len(positions)):
                    dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                 (positions[i][1] - positions[i-1][1])**2)
                    distances.append(dist)
                
                if distances:
                    features['avg_mouse_distance'] = np.mean(distances)
                    features['mouse_movement_variance'] = np.var(distances)
        
        return features
    
    def _extract_clipboard_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract clipboard usage features."""
        features = {}
        
        clipboard_events = [e for e in events if e.get('type') == 'clipboard']
        
        if clipboard_events:
            # Clipboard usage frequency
            features['clipboard_frequency'] = len(clipboard_events)
            
            # Content analysis
            content_lengths = []
            for event in clipboard_events:
                content = event.get('data', {}).get('content', '')
                content_lengths.append(len(content))
            
            if content_lengths:
                features['avg_clipboard_length'] = np.mean(content_lengths)
                features['clipboard_length_variance'] = np.var(content_lengths)
        
        return features
    
    def _extract_access_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract access pattern features for insider threat detection."""
        features = {}
        
        # File access patterns
        file_accesses = [e for e in events if 'file' in e.get('data', {}).get('content', '').lower()]
        features['file_access_frequency'] = len(file_accesses)
        
        # Unusual time access
        timestamps = [datetime.fromisoformat(e.get('timestamp', '')) for e in events 
                     if e.get('timestamp')]
        
        if timestamps:
            # After-hours activity (outside 9-17)
            after_hours = sum(1 for ts in timestamps if ts.hour < 9 or ts.hour > 17)
            features['after_hours_ratio'] = after_hours / max(len(timestamps), 1)
        
        # Weekend activity
        weekend_activity = sum(1 for ts in timestamps if ts.weekday() >= 5)
        features['weekend_ratio'] = weekend_activity / max(len(timestamps), 1)
        
        return features
    
    def _extract_exfiltration_indicators(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data exfiltration indicators."""
        features = {}
        
        # Large clipboard operations
        clipboard_events = [e for e in events if e.get('type') == 'clipboard']
        large_clipboard_ops = 0
        
        for event in clipboard_events:
            content = event.get('data', {}).get('content', '')
            if len(content) > 1000:  # Configurable threshold
                large_clipboard_ops += 1
        
        features['large_clipboard_ops'] = large_clipboard_ops
        
        # USB activity correlation
        usb_events = [e for e in events if 'usb' in str(e.get('data', {})).lower()]
        features['usb_activity'] = len(usb_events)
        
        return features
    
    def _extract_privilege_indicators(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract privilege escalation indicators."""
        features = {}
        
        # Admin/sudo commands
        admin_keywords = ['sudo', 'admin', 'administrator', 'runas', 'elevation']
        admin_events = 0
        
        for event in events:
            content = str(event.get('data', {})).lower()
            if any(keyword in content for keyword in admin_keywords):
                admin_events += 1
        
        features['admin_activity'] = admin_events
        
        return features
    
    def _extract_deviation_metrics(self, events: List[Dict[str, Any]], 
                                 user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract behavioral deviation metrics."""
        features = {}
        
        if not user_context:
            return features
        
        # Compare current behavior to baseline
        current_features = self.extract_behavioral_features(events)
        baseline = user_context.get('baseline_behavior', {})
        
        for key, current_value in current_features.items():
            baseline_value = baseline.get(key, current_value)
            if baseline_value != 0:
                deviation = abs(current_value - baseline_value) / baseline_value
                features[f'{key}_deviation'] = deviation
        
        return features
    
    def _extract_network_correlation(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract network activity correlation features."""
        features = {}
        
        # Network-related keywords in content
        network_keywords = ['http', 'ftp', 'ssh', 'download', 'upload', 'transfer']
        network_events = 0
        
        for event in events:
            content = str(event.get('data', {})).lower()
            if any(keyword in content for keyword in network_keywords):
                network_events += 1
        
        features['network_activity'] = network_events
        
        return features
    
    def _find_release_event(self, events: List[Dict[str, Any]], start_idx: int, key: str) -> Optional[Dict[str, Any]]:
        """Find the corresponding key release event."""
        for i in range(start_idx + 1, len(events)):
            event = events[i]
            if (event.get('type') == 'key_release' and 
                event.get('key') == key):
                return event
        return None
    
    def _find_previous_release(self, events: List[Dict[str, Any]], current_idx: int) -> Optional[Dict[str, Any]]:
        """Find the previous key release event."""
        for i in range(current_idx - 1, -1, -1):
            event = events[i]
            if event.get('type') == 'key_release':
                return event
        return None
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy of a distribution."""
        if not values:
            return 0.0
        
        # Normalize to probabilities
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values if v > 0]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _detect_activity_bursts(self, events: List[Dict[str, Any]]) -> int:
        """Detect activity bursts in event stream."""
        if len(events) < 10:
            return 0
        
        # Calculate event rate in sliding windows
        window_size = 10
        burst_threshold = 2.0  # 2x average rate
        
        rates = []
        for i in range(len(events) - window_size + 1):
            window_events = events[i:i + window_size]
            timestamps = [datetime.fromisoformat(e.get('timestamp', '')) 
                         for e in window_events if e.get('timestamp')]
            
            if len(timestamps) >= 2:
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                if time_span > 0:
                    rate = len(window_events) / time_span
                    rates.append(rate)
        
        if not rates:
            return 0
        
        avg_rate = np.mean(rates)
        bursts = sum(1 for rate in rates if rate > burst_threshold * avg_rate)
        
        return bursts
    
    def normalize_features(self, features: Dict[str, Any], feature_type: str = 'behavioral') -> Dict[str, Any]:
        """Normalize features using appropriate scalers."""
        if not features:
            return features
        
        # Get or create scaler for this feature type
        if feature_type not in self.scalers:
            self.scalers[feature_type] = StandardScaler()
        
        scaler = self.scalers[feature_type]
        
        # Convert to array format
        feature_names = list(features.keys())
        feature_values = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Fit and transform (in production, you'd fit on training data)
        try:
            normalized_values = scaler.fit_transform(feature_values)[0]
            normalized_features = dict(zip(feature_names, normalized_values))
            return normalized_features
        except Exception as e:
            logger.warning(f"Feature normalization failed: {e}")
            return features
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return {
            'cache_sizes': {key: len(cache) for key, cache in self.feature_cache.items()},
            'scalers_fitted': list(self.scalers.keys()),
            'window_size': self.window_size,
            'cache_size': self.cache_size
        }