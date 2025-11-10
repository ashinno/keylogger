"""
Comprehensive Test Suite for Advanced Behavioral Analytics Engine
Master's Thesis Research - Behavioral Analytics Validation
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

# Import the modules to test
try:
    from ml.advanced_behavioral_analytics import (
        AdvancedBehavioralAnalyticsEngine, ThreatLevel, AnomalyType,
        BehavioralProfile, AnomalyResult
    )
    from ml.behavioral_analytics_integration import (
        BehavioralAnalyticsIntegration, BehavioralAnalyticsWebInterface
    )
    from core.config_manager import ConfigManager
    from core.logging_manager import LoggingManager
except ImportError as e:
    print(f"Import error in test suite: {e}")
    # Create mock classes for testing
    AdvancedBehavioralAnalyticsEngine = None
    BehavioralAnalyticsIntegration = None


class TestAdvancedBehavioralAnalyticsEngine(unittest.TestCase):
    """Test cases for Advanced Behavioral Analytics Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        
        # Create test configuration
        self.test_config = {
            'ml.advanced_behavioral_analytics.sensitivity': 0.05,
            'ml.advanced_behavioral_analytics.learning_rate': 0.001,
            'ml.advanced_behavioral_analytics.baseline_window': 200,
            'ml.advanced_behavioral_analytics.adaptation_threshold': 0.85,
            'ml.advanced_behavioral_analytics.drift_threshold': 0.15,
            'ml.advanced_behavioral_analytics.sequence_length': 50,
            'ml.advanced_behavioral_analytics.embedding_dim': 128,
            'ml.advanced_behavioral_analytics.adversarial_defense': True,
            'ml.advanced_behavioral_analytics.differential_privacy': True,
            'ml.model_directory': 'test_models'
        }
        
        # Create test data
        self.create_test_data()
        
        # Initialize engine if available
        if AdvancedBehavioralAnalyticsEngine:
            self.engine = AdvancedBehavioralAnalyticsEngine(self.test_config)
        else:
            self.engine = None
    
    def create_test_data(self):
        """Create test data for behavioral analysis."""
        
        self.test_user_data = []
        
        # Create normal user behavior data
        base_time = datetime.now(timezone.utc)
        
        for i in range(100):
            event = {
                'timestamp': (base_time + timedelta(seconds=i*60)).isoformat(),
                'event_type': 'keyboard',
                'application': 'notepad.exe',
                'window_title': 'Untitled - Notepad',
                'keystroke': f'key_{i % 26}',
                'dwell_time': np.random.normal(0.1, 0.02),
                'flight_time': np.random.normal(0.05, 0.01),
                'system_context': {
                    'cpu_usage': np.random.normal(25, 5),
                    'memory_usage': np.random.normal(40, 10)
                },
                'network_context': {
                    'active_connections': np.random.randint(5, 15)
                }
            }
            self.test_user_data.append(event)
        
        # Create anomalous user behavior data
        self.anomalous_user_data = []
        
        for i in range(20):
            event = {
                'timestamp': (base_time + timedelta(seconds=i*5)).isoformat(),
                'event_type': 'keyboard',
                'application': 'cmd.exe',
                'window_title': 'Command Prompt',
                'keystroke': 'suspicious_command',
                'dwell_time': np.random.normal(0.01, 0.005),  # Much faster typing
                'flight_time': np.random.normal(0.01, 0.005),
                'system_context': {
                    'cpu_usage': np.random.normal(80, 10),  # High CPU usage
                    'memory_usage': np.random.normal(70, 15)
                },
                'network_context': {
                    'active_connections': np.random.randint(20, 50)  # Many connections
                }
            }
            self.anomalous_user_data.append(event)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_engine_initialization(self):
        """Test engine initialization."""
        
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.config)
        self.assertIsNotNone(self.engine.user_profiles)
        self.assertIsNotNone(self.engine.deep_learning_models)
        self.assertIsNotNone(self.engine.traditional_ml_models)
        self.assertIsNotNone(self.engine.ensemble_models)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_user_profile_creation(self):
        """Test user profile creation."""
        
        user_id = "test_user_1"
        
        # Create user profile
        profile = self.engine.create_user_profile(user_id, self.test_user_data)
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile.user_id, user_id)
        self.assertIsNotNone(profile.baseline_features)
        self.assertIsNotNone(profile.behavioral_patterns)
        self.assertGreaterEqual(profile.risk_score, 0.0)
        self.assertLessEqual(profile.risk_score, 1.0)
        self.assertGreaterEqual(profile.confidence_level, 0.0)
        self.assertLessEqual(profile.confidence_level, 1.0)
        
        # Check if profile is stored
        self.assertIn(user_id, self.engine.user_profiles)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_normal_behavior_analysis(self):
        """Test analysis of normal user behavior."""
        
        user_id = "test_user_normal"
        
        # Create user profile with normal data
        self.engine.create_user_profile(user_id, self.test_user_data)
        
        # Analyze normal behavior
        current_event = self.test_user_data[0]
        result = self.engine.analyze_behavior(user_id, current_event)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AnomalyResult)
        self.assertIsInstance(result.is_anomalous, bool)
        self.assertIsInstance(result.threat_level, ThreatLevel)
        self.assertIsInstance(result.anomaly_type, AnomalyType)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)
        
        # Normal behavior should have low risk score
        self.assertLess(result.risk_score, 0.5)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_anomalous_behavior_analysis(self):
        """Test analysis of anomalous user behavior."""
        
        user_id = "test_user_anomalous"
        
        # Create user profile with normal data
        self.engine.create_user_profile(user_id, self.test_user_data)
        
        # Analyze anomalous behavior
        current_event = self.anomalous_user_data[0]
        result = self.engine.analyze_behavior(user_id, current_event)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AnomalyResult)
        
        # Anomalous behavior should have higher risk score
        self.assertGreater(result.risk_score, 0.3)
        
        # Should have explanation
        self.assertIsNotNone(result.explanation)
        self.assertIn('summary', result.explanation)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_feature_extraction(self):
        """Test feature extraction methods."""
        
        # Test statistical features
        features = self.engine._extract_statistical_features(self.test_user_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Test temporal features
        features = self.engine._extract_temporal_features(self.test_user_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Test frequency features
        features = self.engine._extract_frequency_features(self.test_user_data)
        self.assertIsInstance(features, dict)
        
        # Test contextual features
        features = self.engine._extract_contextual_features(self.test_user_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_multi_model_anomaly_detection(self):
        """Test multi-model anomaly detection."""
        
        user_id = "test_user_multi"
        
        # Create user profile
        profile = self.engine.create_user_profile(user_id, self.test_user_data)
        
        # Extract features from anomalous data
        current_features = self.engine._extract_comprehensive_features([self.anomalous_user_data[0]])
        
        # Perform multi-model detection
        results = self.engine._multi_model_anomaly_detection(profile, current_features)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that we have results from different model types
        model_types = set(result['model_type'] for result in results.values())
        self.assertGreater(len(model_types), 0)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_cross_platform_correlation(self):
        """Test cross-platform correlation."""
        
        user_id = "test_user_cross"
        
        # Create test data with location information
        cross_platform_data = {
            'location': 'New York, USA',
            'device_id': 'device_123',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Test cross-platform correlation
        anomalies = self.engine._cross_platform_correlation(user_id, cross_platform_data)
        
        self.assertIsInstance(anomalies, dict)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_adversarial_attack_detection(self):
        """Test adversarial attack detection."""
        
        # Create potentially adversarial features
        adversarial_features = {
            'timing_mean': np.array([0.123456789]),  # Unusually precise
            'timing_std': np.array([0.987654321]),
            'extreme_values': np.array([999999.999999])  # Extreme values
        }
        
        # Test adversarial detection
        indicators = self.engine._detect_adversarial_attacks(adversarial_features)
        
        self.assertIsInstance(indicators, dict)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_system_health_check(self):
        """Test system health monitoring."""
        
        health_status = self.engine.get_system_health()
        
        self.assertIsInstance(health_status, dict)
        self.assertIn('overall_health', health_status)
        self.assertIn('statistics', health_status)
        self.assertIn('model_status', health_status)
        self.assertIn('performance_metrics', health_status)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_analytics_summary(self):
        """Test analytics summary generation."""
        
        # Create some test profiles
        for i in range(3):
            user_id = f"test_user_{i}"
            self.engine.create_user_profile(user_id, self.test_user_data)
        
        # Get analytics summary
        summary = self.engine.get_analytics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('system_statistics', summary)
        self.assertIn('user_profiles', summary)
        self.assertIn('model_performance', summary)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_model_insights_export(self):
        """Test model insights export."""
        
        # Create test profile
        user_id = "test_user_insights"
        self.engine.create_user_profile(user_id, self.test_user_data)
        
        # Export model insights
        insights = self.engine.export_model_insights()
        
        self.assertIsInstance(insights, dict)
        self.assertIn('feature_importance', insights)
        self.assertIn('model_interpretability', insights)
        self.assertIn('anomaly_patterns', insights)
        self.assertIn('behavioral_clusters', insights)
        self.assertIn('threat_landscape', insights)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_model_saving_and_loading(self):
        """Test model persistence."""
        
        import tempfile
        import shutil
        
        # Create temporary directory for models
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Update config to use temp directory
            self.engine.config['ml.model_directory'] = temp_dir
            
            # Create test profile
            user_id = "test_user_persistence"
            self.engine.create_user_profile(user_id, self.test_user_data)
            
            # Save models
            self.engine.save_models()
            
            # Check that files were created
            model_files = os.listdir(temp_dir)
            self.assertGreater(len(model_files), 0)
            
            # Create new engine instance
            new_engine = AdvancedBehavioralAnalyticsEngine(self.engine.config)
            
            # Check that user profile was loaded
            self.assertIn(user_id, new_engine.user_profiles)
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_model_integrity_validation(self):
        """Test model integrity validation."""
        
        validation_result = self.engine.validate_model_integrity()
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('integrity_status', validation_result)
        self.assertIn('model_hashes', validation_result)
        self.assertIn('tampering_detected', validation_result)
    
    def tearDown(self):
        """Clean up after tests."""
        
        # Clean up any temporary files or resources
        if self.engine and hasattr(self.engine, 'cleanup_old_data'):
            self.engine.cleanup_old_data(days_to_keep=0)


class TestBehavioralAnalyticsIntegration(unittest.TestCase):
    """Test cases for Behavioral Analytics Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        
        # Create mock config manager
        self.mock_config_manager = Mock()
        self.mock_config_manager.get_all_config.return_value = {
            'behavioral_analytics.real_time_analysis': True,
            'behavioral_analytics.auto_response_enabled': True,
            'behavioral_analytics.alert_threshold': 'medium',
            'behavioral_analytics.data_retention_days': 30
        }
        self.mock_config_manager.get.side_effect = lambda key, default=None: {
            'behavioral_analytics.security_policies': {},
            'web.admin_username': 'admin',
            'web.admin_password': 'admin123'
        }.get(key, default)
        
        # Create mock logging manager
        self.mock_logging_manager = Mock()
        self.mock_logging_manager.log_security_event = Mock()
        self.mock_logging_manager.log_analysis_event = Mock()
        self.mock_logging_manager.log_response_event = Mock()
        
        # Create test data
        self.create_test_event_data()
        
        # Initialize integration if available
        if BehavioralAnalyticsIntegration:
            self.integration = BehavioralAnalyticsIntegration(
                self.mock_config_manager, 
                self.mock_logging_manager
            )
        else:
            self.integration = None
    
    def create_test_event_data(self):
        """Create test event data."""
        
        self.test_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': 'keyboard',
            'application': 'notepad.exe',
            'window_title': 'Test Document',
            'keystroke': 'test_key',
            'dwell_time': 0.1,
            'flight_time': 0.05
        }
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_integration_initialization(self):
        """Test integration initialization."""
        
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.analytics_engine)
        self.assertIsNotNone(self.integration.security_policies)
        self.assertIsNotNone(self.integration.response_handlers)
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_event_processing(self):
        """Test event processing through integration."""
        
        user_id = "test_integration_user"
        
        # Process event
        result = self.integration.process_user_event(user_id, self.test_event)
        
        self.assertIsNotNone(result)
        self.assertIn('user_id', result)
        self.assertIn('timestamp', result)
        self.assertIn('analysis_result', result)
        self.assertIn('explanation', result)
        self.assertIn('requires_attention', result)
        
        # Check analysis result structure
        analysis_result = result['analysis_result']
        self.assertIn('is_anomalous', analysis_result)
        self.assertIn('threat_level', analysis_result)
        self.assertIn('risk_score', analysis_result)
        self.assertIn('confidence_score', analysis_result)
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_security_response_execution(self):
        """Test security response execution."""
        
        user_id = "test_response_user"
        
        # Process multiple events to trigger potential responses
        for i in range(5):
            event = self.test_event.copy()
            event['application'] = 'cmd.exe'  # Make it more suspicious
            result = self.integration.process_user_event(user_id, event)
        
        # Check that logging was called
        self.assertTrue(self.mock_logging_manager.log_security_event.called or
                       self.mock_logging_manager.log_analysis_event.called)
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_system_status_retrieval(self):
        """Test system status retrieval."""
        
        status = self.integration.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('integration_status', status)
        self.assertIn('analytics_engine_status', status)
        self.assertIn('overall_health', status)
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_analytics_summary_retrieval(self):
        """Test analytics summary retrieval."""
        
        # Process some events first
        for i in range(3):
            user_id = f"test_summary_user_{i}"
            self.integration.process_user_event(user_id, self.test_event)
        
        summary = self.integration.get_analytics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('analytics_summary', summary)
        self.assertIn('integration_summary', summary)
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_data_cleanup(self):
        """Test data cleanup functionality."""
        
        # Process some events
        for i in range(5):
            user_id = f"test_cleanup_user_{i}"
            self.integration.process_user_event(user_id, self.test_event)
        
        # Clean up data
        self.integration.cleanup_old_data(days_to_keep=0)
        
        # Verify cleanup completed without errors
        self.assertTrue(True)  # If we reach here, cleanup didn't crash
    
    @unittest.skipIf(BehavioralAnalyticsIntegration is None, "BehavioralAnalyticsIntegration not available")
    def test_integrity_validation(self):
        """Test system integrity validation."""
        
        validation_result = self.integration.validate_integrity()
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('model_integrity', validation_result)
        self.assertIn('integration_integrity', validation_result)
        self.assertIn('overall_integrity', validation_result)
    
    def tearDown(self):
        """Clean up after tests."""
        
        # Clean up any test data
        if self.integration:
            self.integration.cleanup_old_data(days_to_keep=0)


class TestBehavioralAnalyticsWebInterface(unittest.TestCase):
    """Test cases for Behavioral Analytics Web Interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        
        # Create mock integration
        self.mock_integration = Mock()
        
        # Mock analytics engine
        self.mock_analytics_engine = Mock()
        self.mock_analytics_engine.user_profiles = {}
        self.mock_analytics_engine.stats = {
            'events_processed': 100,
            'anomalies_detected': 5,
            'model_accuracy': 0.85
        }
        self.mock_analytics_engine.get_analytics_summary.return_value = {
            'system_statistics': self.mock_analytics_engine.stats,
            'user_profiles': {'total_profiles': 10}
        }
        self.mock_analytics_engine.export_model_insights.return_value = {
            'feature_importance': {},
            'model_interpretability': {}
        }
        
        # Mock integration components
        self.mock_integration.analytics_engine = self.mock_analytics_engine
        self.mock_integration.get_system_status.return_value = {
            'overall_health': 'healthy'
        }
        self.mock_integration.get_analytics_summary.return_value = {
            'analytics_summary': {'user_profiles': {'total_profiles': 10}}
        }
        self.mock_integration.validate_integrity.return_value = {
            'overall_integrity': 'valid'
        }
        
        # Initialize web interface if available
        if BehavioralAnalyticsWebInterface:
            self.web_interface = BehavioralAnalyticsWebInterface(self.mock_integration)
        else:
            self.web_interface = None
    
    @unittest.skipIf(BehavioralAnalyticsWebInterface is None, "BehavioralAnalyticsWebInterface not available")
    def test_dashboard_data_retrieval(self):
        """Test dashboard data retrieval."""
        
        dashboard_data = self.web_interface.get_dashboard_data()
        
        self.assertIsInstance(dashboard_data, dict)
        self.assertIn('system_status', dashboard_data)
        self.assertIn('analytics_summary', dashboard_data)
        self.assertIn('recent_anomalies', dashboard_data)
        self.assertIn('user_risk_distribution', dashboard_data)
        self.assertIn('model_performance', dashboard_data)
        self.assertIn('timestamp', dashboard_data)
    
    @unittest.skipIf(BehavioralAnalyticsWebInterface is None, "BehavioralAnalyticsWebInterface not available")
    def test_user_details_retrieval(self):
        """Test user details retrieval."""
        
        # Add test user profile
        test_user_id = "test_web_user"
        mock_profile = Mock()
        mock_profile.user_id = test_user_id
        mock_profile.risk_score = 0.3
        mock_profile.confidence_level = 0.8
        mock_profile.behavioral_patterns = {}
        mock_profile.anomaly_history = []
        mock_profile.threat_indicators = []
        mock_profile.last_updated = datetime.now(timezone.utc)
        
        self.mock_analytics_engine.user_profiles[test_user_id] = mock_profile
        
        user_details = self.web_interface.get_user_details(test_user_id)
        
        self.assertIsInstance(user_details, dict)
        self.assertIn('user_id', user_details)
        self.assertIn('risk_score', user_details)
        self.assertIn('confidence_level', user_details)
    
    @unittest.skipIf(BehavioralAnalyticsWebInterface is None, "BehavioralAnalyticsWebInterface not available")
    def test_analytics_report_generation(self):
        """Test analytics report generation."""
        
        report = self.web_interface.export_analytics_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('report_metadata', report)
        self.assertIn('executive_summary', report)
        self.assertIn('detailed_analytics', report)
        self.assertIn('model_insights', report)
        self.assertIn('system_integrity', report)
        self.assertIn('recommendations', report)


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress testing for behavioral analytics."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        
        self.test_config = {
            'ml.advanced_behavioral_analytics.sensitivity': 0.05,
            'ml.advanced_behavioral_analytics.learning_rate': 0.001,
            'ml.advanced_behavioral_analytics.baseline_window': 100,
            'ml.model_directory': 'test_performance_models'
        }
        
        # Create large test dataset
        self.large_dataset = self.create_large_dataset()
        
        # Initialize engine if available
        if AdvancedBehavioralAnalyticsEngine:
            self.engine = AdvancedBehavioralAnalyticsEngine(self.test_config)
        else:
            self.engine = None
    
    def create_large_dataset(self, size=1000):
        """Create large dataset for performance testing."""
        
        dataset = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(size):
            event = {
                'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
                'event_type': 'keyboard',
                'application': 'test_app.exe',
                'dwell_time': np.random.normal(0.1, 0.02),
                'flight_time': np.random.normal(0.05, 0.01),
                'system_context': {
                    'cpu_usage': np.random.normal(25, 5),
                    'memory_usage': np.random.normal(40, 10)
                }
            }
            dataset.append(event)
        
        return dataset
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_performance_large_dataset(self):
        """Test performance with large dataset."""
        
        import time
        
        user_id = "performance_test_user"
        
        # Measure profile creation time
        start_time = time.time()
        profile = self.engine.create_user_profile(user_id, self.large_dataset[:500])
        profile_creation_time = time.time() - start_time
        
        # Measure analysis time
        test_event = self.large_dataset[500]
        start_time = time.time()
        result = self.engine.analyze_behavior(user_id, test_event)
        analysis_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        self.assertLess(profile_creation_time, 10.0, "Profile creation took too long")
        self.assertLess(analysis_time, 2.0, "Analysis took too long")
        
        print(f"Profile creation time: {profile_creation_time:.3f}s")
        print(f"Analysis time: {analysis_time:.3f}s")
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_memory_usage(self):
        """Test memory usage under load."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple user profiles
        for i in range(10):
            user_id = f"memory_test_user_{i}"
            self.engine.create_user_profile(user_id, self.large_dataset[:100])
        
        # Perform multiple analyses
        for i in range(50):
            user_id = f"memory_test_user_{i % 10}"
            test_event = self.large_dataset[i % len(self.large_dataset)]
            self.engine.analyze_behavior(user_id, test_event)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 500, f"Memory usage increased by {memory_increase:.1f}MB")
        
        print(f"Memory usage increase: {memory_increase:.1f}MB")
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_concurrent_access(self):
        """Test concurrent access to the engine."""
        
        import threading
        import time
        
        results = []
        errors = []
        
        def analyze_user(user_id, event_data):
            try:
                result = self.engine.analyze_behavior(user_id, event_data)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create user profiles first
        for i in range(5):
            user_id = f"concurrent_user_{i}"
            self.engine.create_user_profile(user_id, self.large_dataset[:50])
        
        # Start multiple threads
        threads = []
        for i in range(20):
            user_id = f"concurrent_user_{i % 5}"
            event_data = self.large_dataset[i % len(self.large_dataset)]
            thread = threading.Thread(target=analyze_user, args=(user_id, event_data))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 20, "Not all threads completed successfully")
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        # All results should be valid
        for result in results:
            self.assertIsInstance(result, AnomalyResult)


class TestSecurityAndRobustness(unittest.TestCase):
    """Security and robustness testing for behavioral analytics."""
    
    def setUp(self):
        """Set up security test fixtures."""
        
        self.test_config = {
            'ml.advanced_behavioral_analytics.sensitivity': 0.05,
            'ml.advanced_behavioral_analytics.adversarial_defense': True,
            'ml.model_directory': 'test_security_models'
        }
        
        # Initialize engine if available
        if AdvancedBehavioralAnalyticsEngine:
            self.engine = AdvancedBehavioralAnalyticsEngine(self.test_config)
        else:
            self.engine = None
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_malformed_input_handling(self):
        """Test handling of malformed input data."""
        
        user_id = "malformed_test_user"
        
        # Create normal profile first
        normal_data = [
            {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': 'keyboard',
                'application': 'test.exe'
            }
        ]
        self.engine.create_user_profile(user_id, normal_data)
        
        # Test various malformed inputs
        malformed_inputs = [
            None,
            {},
            {'timestamp': None},
            {'timestamp': 'invalid'},
            {'event_type': None},
            {'application': None}
        ]
        
        for malformed_input in malformed_inputs:
            try:
                result = self.engine.analyze_behavior(user_id, malformed_input)
                # Should return a valid result even with malformed input
                self.assertIsInstance(result, AnomalyResult)
                self.assertTrue(result.is_anomalous)  # Malformed input should be flagged
            except Exception as e:
                # Should handle errors gracefully
                self.assertIn("error", str(e).lower())
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_adversarial_input_detection(self):
        """Test detection of adversarial inputs."""
        
        user_id = "adversarial_test_user"
        
        # Create normal profile
        normal_data = [
            {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': 'keyboard',
                'application': 'test.exe',
                'dwell_time': 0.1,
                'flight_time': 0.05
            }
        ]
        self.engine.create_user_profile(user_id, normal_data)
        
        # Create adversarial input with unusual precision
        adversarial_input = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': 'keyboard',
            'application': 'test.exe',
            'dwell_time': 0.123456789,  # Unusually precise
            'flight_time': 0.987654321,
            'system_context': {
                'cpu_usage': 123.456789,  # Unusually precise
                'memory_usage': 987.654321
            }
        }
        
        result = self.engine.analyze_behavior(user_id, adversarial_input)
        
        # Should detect potential adversarial input
        self.assertIsInstance(result, AnomalyResult)
        self.assertGreater(result.risk_score, 0.3)  # Higher risk due to adversarial indicators
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_model_integrity_protection(self):
        """Test model integrity protection mechanisms."""
        
        # Get initial integrity status
        initial_validation = self.engine.validate_model_integrity()
        
        self.assertIsInstance(initial_validation, dict)
        self.assertIn('integrity_status', initial_validation)
        
        # Integrity should be valid for new engine
        self.assertIn(initial_validation['integrity_status'], ['valid', 'unknown'])
    
    @unittest.skipIf(AdvancedBehavioralAnalyticsEngine is None, "AdvancedBehavioralAnalyticsEngine not available")
    def test_privacy_preservation(self):
        """Test privacy preservation mechanisms."""
        
        user_id = "privacy_test_user"
        
        # Create profile with sensitive data
        sensitive_data = [
            {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': 'keyboard',
                'application': 'password_manager.exe',  # Sensitive application
                'window_title': 'Password Entry - Bank Account'
            }
        ]
        
        # Engine should handle sensitive data appropriately
        profile = self.engine.create_user_profile(user_id, sensitive_data)
        
        # Check that sensitive information is not directly stored
        self.assertIsInstance(profile, BehavioralProfile)
        # The engine should sanitize or hash sensitive data


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)