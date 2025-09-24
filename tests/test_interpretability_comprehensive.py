"""Comprehensive test suite for ML interpretability features."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ml.interpretability import ModelInterpretabilityEngine
    from ml.confidence_engine import ConfidenceEngine
    from ml.visualization import InterpretabilityVisualizer
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.datasets import make_classification
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class TestModelInterpretabilityEngine(unittest.TestCase):
    """Test cases for ModelInterpretabilityEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ml': {
                'interpretability': {
                    'enabled': True,
                    'shap_enabled': True,
                    'lime_enabled': True,
                    'feature_importance_enabled': True,
                    'decision_paths_enabled': True
                }
            }
        }
        self.engine = ModelInterpretabilityEngine(self.config)
        
        # Create test data
        self.X_test, self.y_test = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Create test model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_test, self.y_test)
    
    def test_engine_initialization(self):
        """Test interpretability engine initialization."""
        self.assertTrue(self.engine.enabled)
        self.assertIsInstance(self.engine.explanations_cache, dict)
        self.assertIsInstance(self.engine.stats, dict)
        self.assertEqual(self.engine.stats['explanations_generated'], 0)
    
    def test_setup_explainers(self):
        """Test explainer setup for models."""
        result = self.engine.setup_explainers(
            self.model, self.X_test, self.feature_names, 'classifier'
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('model_id', result)
        self.assertIn('explainers', result)
        self.assertEqual(result['feature_count'], len(self.feature_names))
    
    def test_explain_prediction(self):
        """Test prediction explanation generation."""
        # Setup explainers first
        self.engine.setup_explainers(
            self.model, self.X_test, self.feature_names, 'classifier'
        )
        
        # Test single instance explanation
        instance = self.X_test[0]
        explanation = self.engine.explain_prediction(
            self.model, instance, ['feature_importance']
        )
        
        self.assertIn('timestamp', explanation)
        self.assertIn('prediction', explanation)
        self.assertIn('confidence', explanation)
        self.assertIn('explanations', explanation)
    
    def test_global_explanations(self):
        """Test global explanation generation."""
        # Setup explainers first
        self.engine.setup_explainers(
            self.model, self.X_test, self.feature_names, 'classifier'
        )
        
        global_explanations = self.engine.generate_global_explanations(
            self.model, self.X_test[:50], self.y_test[:50]
        )
        
        self.assertIn('timestamp', global_explanations)
        self.assertIn('sample_size', global_explanations)
        self.assertIn('explanations', global_explanations)
        self.assertEqual(global_explanations['sample_size'], 50)
    
    def test_explanation_summary(self):
        """Test explanation summary generation."""
        summary = self.engine.get_explanation_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('statistics', summary)
        self.assertIn('configuration', summary)
        self.assertIn('available_explainers', summary)
    
    def test_disabled_engine(self):
        """Test behavior when engine is disabled."""
        disabled_config = {
            'ml': {
                'interpretability': {
                    'enabled': False
                }
            }
        }
        disabled_engine = ModelInterpretabilityEngine(disabled_config)
        
        result = disabled_engine.setup_explainers(
            self.model, self.X_test, self.feature_names
        )
        self.assertEqual(result['status'], 'disabled')
        
        explanation = disabled_engine.explain_prediction(self.model, self.X_test[0])
        self.assertEqual(explanation['status'], 'disabled')


class TestConfidenceEngine(unittest.TestCase):
    """Test cases for ConfidenceEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ml': {
                'confidence': {
                    'enabled': True,
                    'calibration_enabled': True,
                    'confidence_threshold': 0.8,
                    'calibration_method': 'isotonic'
                }
            }
        }
        self.engine = ConfidenceEngine(self.config)
        
        # Create test data
        self.X_test, self.y_test = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        # Create test model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_test, self.y_test)
    
    def test_engine_initialization(self):
        """Test confidence engine initialization."""
        self.assertTrue(self.engine.enabled)
        self.assertTrue(self.engine.calibration_enabled)
        self.assertEqual(self.engine.confidence_threshold, 0.8)
        self.assertIsInstance(self.engine.confidence_statistics, dict)
    
    def test_model_calibration(self):
        """Test model calibration functionality."""
        result = self.engine.calibrate_model(
            self.model, self.X_test, self.y_test, 'test_model'
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('model_id', result)
        self.assertIn('calibration_method', result)
        self.assertIn('metrics', result)
        self.assertIn('calibration_size', result)
    
    def test_prediction_confidence_assessment(self):
        """Test prediction confidence assessment."""
        # Calibrate model first
        self.engine.calibrate_model(
            self.model, self.X_test, self.y_test, 'test_model'
        )
        
        # Test confidence assessment
        instance = self.X_test[0]
        assessment = self.engine.assess_prediction_confidence(
            self.model, instance, 'test_model'
        )
        
        self.assertIn('timestamp', assessment)
        self.assertIn('prediction', assessment)
        self.assertIn('confidence_metrics', assessment)
        self.assertIn('uncertainty_metrics', assessment)
        self.assertIn('confidence_level', assessment)
        self.assertIn('reliability_score', assessment)
    
    def test_confidence_drift_detection(self):
        """Test confidence drift detection."""
        # Generate some predictions to build history
        for i in range(150):
            instance = self.X_test[i % len(self.X_test)]
            self.engine.assess_prediction_confidence(
                self.model, instance, 'test_model'
            )
        
        drift_analysis = self.engine.detect_confidence_drift(window_size=50)
        
        if drift_analysis['status'] == 'success':
            self.assertIn('older_window_stats', drift_analysis)
            self.assertIn('recent_window_stats', drift_analysis)
            self.assertIn('drift_metrics', drift_analysis)
            self.assertIn('drift_severity', drift_analysis)
    
    def test_recalibration_recommendation(self):
        """Test recalibration recommendation system."""
        recommendation = self.engine.recommend_recalibration('test_model')
        
        self.assertIn('timestamp', recommendation)
        self.assertIn('recalibration_needed', recommendation)
        self.assertIn('reasons', recommendation)
        self.assertIn('priority', recommendation)
        self.assertIn('confidence_score', recommendation)
    
    def test_confidence_summary(self):
        """Test confidence summary generation."""
        summary = self.engine.get_confidence_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('statistics', summary)
        self.assertIn('configuration', summary)
        self.assertIn('calibrated_models', summary)
    
    def test_disabled_engine(self):
        """Test behavior when engine is disabled."""
        disabled_config = {
            'ml': {
                'confidence': {
                    'enabled': False
                }
            }
        }
        disabled_engine = ConfidenceEngine(disabled_config)
        
        result = disabled_engine.calibrate_model(
            self.model, self.X_test, self.y_test
        )
        self.assertEqual(result['status'], 'disabled')
        
        assessment = disabled_engine.assess_prediction_confidence(
            self.model, self.X_test[0]
        )
        self.assertEqual(assessment['status'], 'disabled')


class TestInterpretabilityVisualizer(unittest.TestCase):
    """Test cases for InterpretabilityVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ml': {
                'visualization': {
                    'enabled': True,
                    'use_plotly': True,
                    'use_matplotlib': True,
                    'figure_size': (10, 6),
                    'dpi': 100
                }
            }
        }
        self.visualizer = InterpretabilityVisualizer(self.config)
        
        # Create mock explanation data
        self.explanation_data = {
            'prediction': 1,
            'confidence': 0.85,
            'confidence_level': 'high',
            'reliability_score': 0.78,
            'explanations': {
                'shap': {
                    'feature_importance': [
                        {'feature': 'feature_1', 'shap_value': 0.3, 'abs_importance': 0.3, 'rank': 1},
                        {'feature': 'feature_2', 'shap_value': -0.2, 'abs_importance': 0.2, 'rank': 2},
                        {'feature': 'feature_3', 'shap_value': 0.1, 'abs_importance': 0.1, 'rank': 3}
                    ],
                    'total_impact': 0.6,
                    'base_value': 0.5
                },
                'lime': {
                    'feature_importance': [
                        {'feature': 'feature_1', 'lime_value': 0.25, 'abs_importance': 0.25, 'rank': 1},
                        {'feature': 'feature_2', 'lime_value': -0.15, 'abs_importance': 0.15, 'rank': 2}
                    ]
                },
                'uncertainty': {
                    'entropy': 0.3,
                    'confidence': 0.85,
                    'uncertainty': 0.15,
                    'margin': 0.7
                }
            }
        }
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertTrue(self.visualizer.enabled)
        self.assertTrue(self.visualizer.use_plotly)
        self.assertTrue(self.visualizer.use_matplotlib)
        self.assertEqual(self.visualizer.figure_size, (10, 6))
    
    def test_feature_importance_plot(self):
        """Test feature importance plot generation."""
        result = self.visualizer.create_feature_importance_plot(
            self.explanation_data, plot_type='bar', top_n=10
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('plots', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['plot_type'], 'bar')
    
    def test_confidence_indicator(self):
        """Test confidence indicator generation."""
        result = self.visualizer.create_confidence_indicator(self.explanation_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('plots', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['confidence'], 0.85)
    
    def test_shap_waterfall_plot(self):
        """Test SHAP waterfall plot generation."""
        result = self.visualizer.create_shap_waterfall_plot(
            self.explanation_data, top_n=10
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('plots', result)
        self.assertIn('metadata', result)
    
    def test_uncertainty_visualization(self):
        """Test uncertainty visualization generation."""
        result = self.visualizer.create_uncertainty_visualization(self.explanation_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('plots', result)
        self.assertIn('metadata', result)
    
    def test_decision_path_visualization(self):
        """Test decision path visualization."""
        # Create mock decision path data
        decision_path_data = {
            'explanations': {
                'decision_path': {
                    'path': [
                        {
                            'node_id': 0,
                            'feature': 'feature_1',
                            'threshold': 0.5,
                            'threshold_sign': '<=',
                            'value': 0.3
                        },
                        {
                            'node_id': 1,
                            'feature': 'feature_2',
                            'threshold': 0.8,
                            'threshold_sign': '>',
                            'value': 0.9
                        }
                    ],
                    'leaf_id': 5
                }
            }
        }
        
        result = self.visualizer.create_decision_path_visualization(decision_path_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('plots', result)
        self.assertIn('metadata', result)
    
    def test_disabled_visualizer(self):
        """Test behavior when visualizer is disabled."""
        disabled_config = {
            'ml': {
                'visualization': {
                    'enabled': False
                }
            }
        }
        disabled_visualizer = InterpretabilityVisualizer(disabled_config)
        
        result = disabled_visualizer.create_feature_importance_plot(self.explanation_data)
        self.assertEqual(result['status'], 'disabled')
    
    def test_no_data_scenarios(self):
        """Test handling of missing or invalid data."""
        empty_data = {'explanations': {}}
        
        result = self.visualizer.create_feature_importance_plot(empty_data)
        self.assertEqual(result['status'], 'no_data')
        
        result = self.visualizer.create_shap_waterfall_plot(empty_data)
        self.assertEqual(result['status'], 'no_data')


class TestInterpretabilityIntegration(unittest.TestCase):
    """Integration tests for interpretability components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ml': {
                'interpretability': {
                    'enabled': True,
                    'shap_enabled': True,
                    'lime_enabled': True,
                    'feature_importance_enabled': True
                },
                'confidence': {
                    'enabled': True,
                    'calibration_enabled': True,
                    'confidence_threshold': 0.8
                },
                'visualization': {
                    'enabled': True,
                    'use_plotly': True,
                    'use_matplotlib': True
                }
            }
        }
        
        # Initialize all components
        self.interpretability_engine = ModelInterpretabilityEngine(self.config)
        self.confidence_engine = ConfidenceEngine(self.config)
        self.visualizer = InterpretabilityVisualizer(self.config)
        
        # Create test data and model
        self.X_test, self.y_test = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.feature_names = [f'feature_{i}' for i in range(10)]
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_test, self.y_test)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end interpretability workflow."""
        # Step 1: Setup explainers
        explainer_result = self.interpretability_engine.setup_explainers(
            self.model, self.X_test, self.feature_names, 'classifier'
        )
        self.assertEqual(explainer_result['status'], 'success')
        
        # Step 2: Calibrate model
        calibration_result = self.confidence_engine.calibrate_model(
            self.model, self.X_test, self.y_test, 'test_model'
        )
        self.assertEqual(calibration_result['status'], 'success')
        
        # Step 3: Generate explanation for a prediction
        instance = self.X_test[0]
        explanation = self.interpretability_engine.explain_prediction(
            self.model, instance, ['shap', 'lime', 'feature_importance']
        )
        self.assertIn('explanations', explanation)
        
        # Step 4: Assess confidence
        confidence_assessment = self.confidence_engine.assess_prediction_confidence(
            self.model, instance, 'test_model'
        )
        self.assertIn('confidence_level', confidence_assessment)
        
        # Step 5: Generate visualizations
        viz_result = self.visualizer.create_feature_importance_plot(explanation)
        self.assertEqual(viz_result['status'], 'success')
        
        confidence_viz = self.visualizer.create_confidence_indicator(confidence_assessment)
        self.assertEqual(confidence_viz['status'], 'success')
    
    def test_component_interaction(self):
        """Test interaction between different interpretability components."""
        # Setup explainers
        self.interpretability_engine.setup_explainers(
            self.model, self.X_test, self.feature_names, 'classifier'
        )
        
        # Generate explanation
        instance = self.X_test[0]
        explanation = self.interpretability_engine.explain_prediction(
            self.model, instance
        )
        
        # Use explanation data for visualization
        if explanation.get('explanations'):
            viz_result = self.visualizer.create_feature_importance_plot(explanation)
            self.assertIn('status', viz_result)
    
    def test_error_handling(self):
        """Test error handling across components."""
        # Test with invalid model
        invalid_model = Mock()
        invalid_model.predict = Mock(side_effect=Exception("Model error"))
        
        result = self.interpretability_engine.explain_prediction(
            invalid_model, self.X_test[0]
        )
        self.assertIn('error', result)
        
        # Test with invalid data
        result = self.visualizer.create_feature_importance_plot({})
        self.assertEqual(result['status'], 'no_data')
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        # Create larger dataset
        X_large, y_large = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=42
        )
        feature_names_large = [f'feature_{i}' for i in range(20)]
        
        # Train model
        large_model = RandomForestClassifier(n_estimators=10, random_state=42)
        large_model.fit(X_large, y_large)
        
        # Test setup time
        start_time = datetime.now()
        result = self.interpretability_engine.setup_explainers(
            large_model, X_large[:100], feature_names_large, 'classifier'
        )
        setup_time = (datetime.now() - start_time).total_seconds()
        
        self.assertEqual(result['status'], 'success')
        self.assertLess(setup_time, 30)  # Should complete within 30 seconds
        
        # Test explanation generation time
        start_time = datetime.now()
        explanation = self.interpretability_engine.explain_prediction(
            large_model, X_large[0], ['feature_importance']
        )
        explanation_time = (datetime.now() - start_time).total_seconds()
        
        self.assertIn('explanations', explanation)
        self.assertLess(explanation_time, 10)  # Should complete within 10 seconds


class TestInterpretabilityConfiguration(unittest.TestCase):
    """Test configuration handling for interpretability components."""
    
    def test_default_configuration(self):
        """Test default configuration handling."""
        minimal_config = {}
        
        # Components should work with minimal config
        engine = ModelInterpretabilityEngine(minimal_config)
        confidence = ConfidenceEngine(minimal_config)
        visualizer = InterpretabilityVisualizer(minimal_config)
        
        # Check default values are applied
        self.assertTrue(engine.enabled)
        self.assertTrue(confidence.enabled)
        self.assertTrue(visualizer.enabled)
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        invalid_config = {
            'ml': {
                'interpretability': {
                    'enabled': 'invalid_boolean'
                }
            }
        }
        
        # Should handle invalid configuration gracefully
        engine = ModelInterpretabilityEngine(invalid_config)
        self.assertIsInstance(engine.enabled, bool)
    
    def test_feature_toggles(self):
        """Test individual feature toggles."""
        config = {
            'ml': {
                'interpretability': {
                    'enabled': True,
                    'shap_enabled': False,
                    'lime_enabled': True,
                    'feature_importance_enabled': True
                }
            }
        }
        
        engine = ModelInterpretabilityEngine(config)
        self.assertFalse(engine.shap_enabled)
        self.assertTrue(engine.lime_enabled)
        self.assertTrue(engine.feature_importance_enabled)


class TestInterpretabilityUtilities(unittest.TestCase):
    """Test utility functions and helper methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ml': {
                'interpretability': {'enabled': True},
                'confidence': {'enabled': True},
                'visualization': {'enabled': True}
            }
        }
    
    def test_data_validation(self):
        """Test data validation utilities."""
        engine = ModelInterpretabilityEngine(self.config)
        
        # Test with valid data
        valid_data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertIsNotNone(valid_data)
        
        # Test with invalid data
        invalid_data = None
        self.assertIsNone(invalid_data)
    
    def test_feature_name_handling(self):
        """Test feature name handling and validation."""
        engine = ModelInterpretabilityEngine(self.config)
        
        # Test with valid feature names
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.assertEqual(len(feature_names), 3)
        self.assertTrue(all(isinstance(name, str) for name in feature_names))
        
        # Test with empty feature names
        empty_names = []
        self.assertEqual(len(empty_names), 0)
    
    def test_model_compatibility(self):
        """Test model compatibility checking."""
        # Test with sklearn model
        sklearn_model = RandomForestClassifier()
        self.assertTrue(hasattr(sklearn_model, 'fit'))
        self.assertTrue(hasattr(sklearn_model, 'predict'))
        
        # Test with mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1])
        self.assertTrue(hasattr(mock_model, 'predict'))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestModelInterpretabilityEngine))
    test_suite.addTest(unittest.makeSuite(TestConfidenceEngine))
    test_suite.addTest(unittest.makeSuite(TestInterpretabilityVisualizer))
    test_suite.addTest(unittest.makeSuite(TestInterpretabilityIntegration))
    test_suite.addTest(unittest.makeSuite(TestInterpretabilityConfiguration))
    test_suite.addTest(unittest.makeSuite(TestInterpretabilityUtilities))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)