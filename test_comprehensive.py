#!/usr/bin/env python3
"""Comprehensive test suite for the keylogger project."""

import sys
import os
import logging
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")

    try:
        # Core modules
        from core.config_manager import ConfigManager
        from core.encryption_manager import EncryptionManager
        from core.logging_manager import LoggingManager
        from core.keylogger import KeyloggerCore

        # Listeners
        from listeners.keyboard_listener import KeyboardListener
        from listeners.mouse_listener import MouseListener
        from listeners.clipboard_listener import ClipboardListener

        # Utils
        from utils.screenshot_monitor import ScreenshotMonitor
        from utils.window_monitor import WindowMonitor
        from utils.performance_monitor import PerformanceMonitor

        # ML modules
        from ml.behavioral_analytics import BehavioralAnalyticsEngine
        from ml.insider_threat import InsiderThreatDetector
        from ml.keystroke_dynamics import KeystrokeDynamicsAnalyzer
        from ml.risk_scoring import RealTimeRiskScorer
        from ml.data_preprocessing import DataPreprocessor
        from ml.model_manager import ModelManager
        from ml.interpretability import ModelInterpretabilityEngine
        from ml.confidence_engine import ConfidenceEngine
        from ml.visualization import InterpretabilityVisualizer

        # Web interface
        from web.app import create_web_app

        # Parsers
        from parsers.log_parser import LogParser

        logger.info("✓ All modules imported successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_engines():
    """Test ML engines initialization and basic functionality."""
    logger.info("Testing ML engines...")

    try:
        from ml.behavioral_analytics import BehavioralAnalyticsEngine
        from ml.insider_threat import InsiderThreatDetector
        from ml.keystroke_dynamics import KeystrokeDynamicsAnalyzer
        from ml.risk_scoring import RealTimeRiskScorer

        # Mock configuration
        config = {
            'ml.behavioral_analytics.sensitivity': 0.1,
            'ml.behavioral_analytics.learning_rate': 0.01,
            'ml.behavioral_analytics.baseline_window': 1000,
            'ml.insider_threat.threshold': 0.7,
            'ml.keystroke_dynamics.threshold': 0.8,
            'ml.risk_scoring.threshold': 0.8,
        }

        # Test sample event
        sample_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'keyboard',
            'data': {
                'key': 'a',
                'action': 'press'
            }
        }

        # Test Behavioral Analytics
        logger.info("  Testing BehavioralAnalyticsEngine...")
        behavioral_engine = BehavioralAnalyticsEngine(config)
        result = behavioral_engine.process_event(sample_event)
        assert 'anomaly_score' in result or 'is_anomaly' in result
        logger.info(f"    ✓ Behavioral Analytics: anomaly_score={result.get('anomaly_score', 'N/A')}")

        # Test Insider Threat
        logger.info("  Testing InsiderThreatDetector...")
        threat_detector = InsiderThreatDetector(config)
        threat_score = threat_detector.analyze_event(sample_event)
        assert isinstance(threat_score, (int, float))
        logger.info(f"    ✓ Insider Threat: score={threat_score}")

        # Test Keystroke Dynamics
        logger.info("  Testing KeystrokeDynamicsAnalyzer...")
        keystroke_analyzer = KeystrokeDynamicsAnalyzer(config)
        keystroke_result = keystroke_analyzer.process_keystroke(sample_event)
        logger.info(f"    ✓ Keystroke Dynamics: result={keystroke_result}")

        # Test Risk Scoring
        logger.info("  Testing RealTimeRiskScorer...")
        risk_scorer = RealTimeRiskScorer(config)
        risk_score = risk_scorer.calculate_risk(sample_event)
        assert isinstance(risk_score, (int, float))
        assert 0 <= risk_score <= 1
        logger.info(f"    ✓ Risk Scoring: score={risk_score:.4f}")

        # Stop monitoring thread
        risk_scorer.stop_monitoring()

        logger.info("✓ All ML engines tested successfully")
        return True

    except Exception as e:
        logger.error(f"✗ ML engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases...")

    try:
        from ml.behavioral_analytics import BehavioralAnalyticsEngine
        from ml.risk_scoring import RealTimeRiskScorer

        config = {}

        # Test with None clipboard content
        logger.info("  Testing None clipboard content...")
        engine = BehavioralAnalyticsEngine(config)
        clipboard_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'clipboard',
            'data': {
                'content': None
            }
        }
        result = engine.process_event(clipboard_event)
        logger.info("    ✓ None clipboard handled")

        # Test with empty data
        logger.info("  Testing empty event data...")
        empty_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'keyboard',
            'data': {}
        }
        result = engine.process_event(empty_event)
        logger.info("    ✓ Empty data handled")

        # Test with malformed timestamp
        logger.info("  Testing malformed timestamp...")
        risk_scorer = RealTimeRiskScorer(config)
        bad_event = {
            'timestamp': 'invalid',
            'type': 'mouse',
            'data': {'x': 100, 'y': 200}
        }
        risk_score = risk_scorer.calculate_risk(bad_event)
        risk_scorer.stop_monitoring()
        logger.info("    ✓ Malformed timestamp handled")

        logger.info("✓ All edge cases handled correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_manager():
    """Test configuration manager."""
    logger.info("Testing ConfigManager...")

    try:
        from core.config_manager import ConfigManager

        # Test with default config
        config_manager = ConfigManager('config.json')

        # Test getting values
        log_level = config_manager.get('logging.level', 'INFO')
        assert isinstance(log_level, str)

        ml_enabled = config_manager.get('ml.enabled', False)
        assert isinstance(ml_enabled, bool)

        logger.info("✓ ConfigManager working correctly")
        return True

    except Exception as e:
        logger.error(f"✗ ConfigManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE TEST SUITE")
    logger.info("=" * 70)

    tests = [
        ("Module Imports", test_imports),
        ("ML Engines", test_ml_engines),
        ("Edge Cases", test_edge_cases),
        ("Config Manager", test_config_manager),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info("")
        logger.info(f"Running: {test_name}")
        logger.info("-" * 70)
        success = test_func()
        results.append((test_name, success))
        logger.info("")

    logger.info("=" * 70)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if not success:
            all_passed = False

    logger.info("=" * 70)

    if all_passed:
        logger.info("✓ ALL TESTS PASSED!")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
