#!/usr/bin/env python3
"""Test script to validate ML model fixes."""

import sys
import os
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_ml_models():
    """Test all ML models to ensure they work without errors."""
    try:
        # Import ML modules
        from ml.behavioral_analytics import BehavioralAnalyticsEngine
        from ml.insider_threat import InsiderThreatDetector
        from ml.keystroke_dynamics import KeystrokeDynamicsAnalyzer
        from ml.risk_scoring import RealTimeRiskScorer
        
        # Mock configuration
        config = {
            'ml.behavioral_analytics.sensitivity': 0.1,
            'ml.insider_threat.threshold': 0.7,
            'ml.keystroke_dynamics.threshold': 0.8,
            'ml.risk_scoring.threshold': 0.8,
        }
        
        logger.info("Testing Behavioral Analytics Engine...")
        behavioral_engine = BehavioralAnalyticsEngine(config)
        
        # Test with sample event
        sample_event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'keyboard',
            'data': {
                'key': 'a',
                'action': 'press'
            }
        }
        
        result = behavioral_engine.process_event(sample_event)
        logger.info(f"Behavioral Analytics result: {result.get('anomaly_score', 'N/A')}")
        
        logger.info("Testing Insider Threat Detector...")
        threat_detector = InsiderThreatDetector(config)
        
        threat_score = threat_detector.analyze_event(sample_event)
        logger.info(f"Insider Threat score: {threat_score}")
        
        logger.info("Testing Keystroke Dynamics Analyzer...")
        keystroke_analyzer = KeystrokeDynamicsAnalyzer(config)
        
        keystroke_result = keystroke_analyzer.process_keystroke(sample_event)
        logger.info(f"Keystroke Dynamics result: {keystroke_result.get('anomaly_score', 'N/A')}")
        
        logger.info("Testing Risk Scoring Engine...")
        risk_scorer = RealTimeRiskScorer(config)
        
        risk_score = risk_scorer.calculate_risk(sample_event)
        logger.info(f"Risk Score: {risk_score}")
        
        logger.info("All ML models tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing ML models: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clipboard_handling():
    """Test clipboard handling with None values."""
    try:
        from ml.behavioral_analytics import BehavioralAnalyticsEngine
        
        config = {'ml.behavioral_analytics.sensitivity': 0.1}
        engine = BehavioralAnalyticsEngine(config)
        
        # Test with None clipboard content
        clipboard_event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'clipboard',
            'data': {
                'content': None
            }
        }
        
        result = engine.process_event(clipboard_event)
        logger.info(f"Clipboard None handling test passed: {result.get('anomaly_score', 'N/A')}")
        
        # Test with empty string
        clipboard_event['data']['content'] = ''
        result = engine.process_event(clipboard_event)
        logger.info(f"Clipboard empty string test passed: {result.get('anomaly_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing clipboard handling: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting ML model validation tests...")
    
    success = True
    
    # Test basic ML functionality
    if not test_ml_models():
        success = False
    
    # Test clipboard handling fixes
    if not test_clipboard_handling():
        success = False
    
    if success:
        logger.info("All tests passed! ML models are working correctly.")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)