#!/usr/bin/env python3
"""
Advanced Behavioral Analytics Integration Test
Demonstrates the complete advanced behavioral analytics system
"""

import sys
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ml.advanced_behavioral_analytics import (
        AdvancedBehavioralAnalyticsEngine, ThreatLevel, AnomalyType
    )
    from ml.behavioral_analytics_integration import BehavioralAnalyticsIntegration
       from core.config_manager import ConfigManager
    from core.logging_manager import LoggingManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AdvancedBehavioralAnalyticsDemo:
    """Demonstration class for advanced behavioral analytics."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.logging_manager = LoggingManager(self.config_manager.get_all_config())
        
        # Initialize advanced behavioral analytics
        self.analytics_integration = BehavioralAnalyticsIntegration(
            self.config_manager, self.logging_manager
        )
        
        logger.info("Advanced Behavioral Analytics Demo initialized")
    
    def generate_normal_user_data(self, user_id: str, num_events: int = 100) -> list:
        """Generate normal user behavior data."""
        
        normal_data = []
        base_time = datetime.now(timezone.utc)
        
        # Normal typing patterns
        for i in range(num_events):
            event = {
                'timestamp': (base_time + timedelta(seconds=i*60)).isoformat(),
                'event_type': 'keyboard',
                'application': 'notepad.exe',
                'window_title': 'Document - Notepad',
                'keystroke': chr(ord('a') + (i % 26)),
                'dwell_time': np.random.normal(0.12, 0.03),  # Normal typing speed
                'flight_time': np.random.normal(0.08, 0.02),
                'system_context': {
                    'cpu_usage': np.random.normal(25, 8),
                    'memory_usage': np.random.normal(40, 12),
                    'active_processes': np.random.randint(8, 15)
                },
                'network_context': {
                    'active_connections': np.random.randint(3, 8),
                    'bandwidth_usage': np.random.normal(100, 50)
                },
                'location': 'New York, USA',
                'device_id': 'laptop_001'
            }
            normal_data.append(event)
        
        return normal_data
    
    def generate_anomalous_user_data(self, user_id: str, num_events: int = 20) -> list:
        """Generate anomalous user behavior data."""
        
        anomalous_data = []
        base_time = datetime.now(timezone.utc)
        
        # Suspicious behavior patterns
        for i in range(num_events):
            event = {
                'timestamp': (base_time + timedelta(seconds=i*5)).isoformat(),
                'event_type': 'keyboard',
                'application': 'cmd.exe',
                'window_title': 'Administrator: Command Prompt',
                'keystroke': 'suspicious_command_' + str(i),
                'dwell_time': np.random.normal(0.02, 0.005),  # Very fast typing
                'flight_time': np.random.normal(0.015, 0.003),
                'system_context': {
                    'cpu_usage': np.random.normal(85, 10),  # High CPU usage
                    'memory_usage': np.random.normal(75, 15),
                    'active_processes': np.random.randint(20, 35)
                },
                'network_context': {
                    'active_connections': np.random.randint(25, 50),  # Many connections
                    'bandwidth_usage': np.random.normal(500, 200)
                },
                'location': 'Unknown Location',  # Different location
                'device_id': 'unknown_device'
            }
            anomalous_data.append(event)
        
        return anomalous_data
    
    def generate_adversarial_user_data(self, user_id: str, num_events: int = 15) -> list:
        """Generate adversarial user behavior data."""
        
        adversarial_data = []
        base_time = datetime.now(timezone.utc)
        
        # Adversarial patterns designed to evade detection
        for i in range(num_events):
            event = {
                'timestamp': (base_time + timedelta(seconds=i*30)).isoformat(),
                'event_type': 'keyboard',
                'application': 'powershell.exe',
                'window_title': 'Windows PowerShell',
                'keystroke': 'encoded_command_' + str(i),
                'dwell_time': 0.123456789,  # Unusually precise timing
                'flight_time': 0.987654321,
                'system_context': {
                    'cpu_usage': 50.000000001,  # Artificially precise values
                    'memory_usage': 60.000000002,
                    'active_processes': 12
                },
                'network_context': {
                    'active_connections': 10.000000003,
                    'bandwidth_usage': 200.000000004
                },
                'location': 'Suspicious Location',
                'device_id': 'adversarial_device'
            }
            adversarial_data.append(event)
        
        return adversarial_data
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of the system."""
        
        print("\n" + "="*80)
        print("ADVANCED BEHAVIORAL ANALYTICS SYSTEM DEMONSTRATION")
        print("="*80 + "\n")
        
        # Test Case 1: Normal User Behavior
        print("1. TESTING NORMAL USER BEHAVIOR")
        print("-" * 40)
        
        normal_user_id = "demo_user_normal"
        normal_data = self.generate_normal_user_data(normal_user_id, 50)
        
        # Process normal events
        normal_results = []
        for i, event in enumerate(normal_data[:10]):  # Process first 10 events
            result = self.analytics_integration.process_user_event(normal_user_id, event)
            normal_results.append(result)
            
            if i == 0:  # Show first result in detail
                print(f"First event analysis:")
                print(f"  - Anomalous: {result['analysis_result']['is_anomalous']}")
                print(f"  - Threat Level: {result['analysis_result']['threat_level']}")
                print(f"  - Risk Score: {result['analysis_result']['risk_score']:.3f}")
                print(f"  - Confidence: {result['analysis_result']['confidence_score']:.3f}")
        
        # Summary statistics
        anomalous_count = sum(1 for r in normal_results if r['analysis_result']['is_anomalous'])
        avg_risk_score = np.mean([r['analysis_result']['risk_score'] for r in normal_results])
        
        print(f"\nNormal behavior summary:")
        print(f"  - Events processed: {len(normal_results)}")
        print(f"  - Anomalies detected: {anomalous_count}")
        print(f"  - Average risk score: {avg_risk_score:.3f}")
        print(f"  - False positive rate: {anomalous_count/len(normal_results):.1%}")
        
        # Test Case 2: Anomalous User Behavior
        print("\n\n2. TESTING ANOMALOUS USER BEHAVIOR")
        print("-" * 40)
        
        anomalous_user_id = "demo_user_anomalous"
        anomalous_data = self.generate_anomalous_user_data(anomalous_user_id, 10)
        
        # Create baseline for anomalous user with normal data first
        normal_baseline = self.generate_normal_user_data(anomalous_user_id, 30)
        for event in normal_baseline[:20]:  # Create baseline with normal behavior
            self.analytics_integration.process_user_event(anomalous_user_id, event)
        
        # Now process anomalous events
        anomalous_results = []
        for i, event in enumerate(anomalous_data):
            result = self.analytics_integration.process_user_event(anomalous_user_id, event)
            anomalous_results.append(result)
            
            if i == 0:  # Show first anomalous result
                print(f"First anomalous event analysis:")
                print(f"  - Anomalous: {result['analysis_result']['is_anomalous']}")
                print(f"  - Threat Level: {result['analysis_result']['threat_level']}")
                print(f"  - Anomaly Type: {result['analysis_result']['anomaly_type']}")
                print(f"  - Risk Score: {result['analysis_result']['risk_score']:.3f}")
                print(f"  - Confidence: {result['analysis_result']['confidence_score']:.3f}")
                print(f"  - Recommended Actions: {result['analysis_result']['recommended_actions']}")
        
        # Summary statistics
        anomalous_detected = sum(1 for r in anomalous_results if r['analysis_result']['is_anomalous'])
        avg_risk_score_anomalous = np.mean([r['analysis_result']['risk_score'] for r in anomalous_results])
        
        print(f"\nAnomalous behavior summary:")
        print(f"  - Events processed: {len(anomalous_results)}")
        print(f"  - Anomalies detected: {anomalous_detected}")
        print(f"  - Detection rate: {anomalous_detected/len(anomalous_results):.1%}")
        print(f"  - Average risk score: {avg_risk_score_anomalous:.3f}")
        
        # Test Case 3: System Health and Analytics
        print("\n\n3. SYSTEM HEALTH AND ANALYTICS")
        print("-" * 40)
        
        # Get system status
        system_status = self.analytics_integration.get_system_status()
        print(f"System Health: {system_status['overall_health']}")
        print(f"Active Users: {system_status['integration_status']['active_users']}")
        print(f"Total Events Processed: {system_status['analytics_engine_status']['statistics']['events_processed']}")
        print(f"Anomalies Detected: {system_status['analytics_engine_status']['statistics']['anomalies_detected']}")
        
        # Get analytics summary
        analytics_summary = self.analytics_integration.get_analytics_summary()
        print(f"\nAnalytics Summary:")
        print(f"  - Total User Profiles: {analytics_summary['analytics_summary']['user_profiles']['total_profiles']}")
        print(f"  - High Risk Users: {analytics_summary['analytics_summary']['user_profiles']['high_risk_users']}")
        print(f"  - Model Accuracy: {analytics_summary['analytics_summary']['model_performance']['model_accuracy']:.3f}")
        
        # Test Case 4: Advanced Features
        print("\n\n4. ADVANCED FEATURES DEMONSTRATION")
        print("-" * 40)
        
        # Model interpretability
        if hasattr(self.analytics_integration.analytics_engine, 'export_model_insights'):
            try:
                insights = self.analytics_integration.analytics_engine.export_model_insights()
                print(f"Model Insights Available:")
                print(f"  - Feature Importance: {'Available' if 'feature_importance' in insights else 'Not Available'}")
                print(f"  - Behavioral Clusters: {'Available' if 'behavioral_clusters' in insights else 'Not Available'}")
                print(f"  - Threat Landscape: {'Available' if 'threat_landscape' in insights else 'Not Available'}")
            except Exception as e:
                print(f"Model insights error: {e}")
        
        # System integrity validation
        try:
            integrity_check = self.analytics_integration.validate_integrity()
            print(f"\nSystem Integrity:")
            print(f"  - Overall Integrity: {integrity_check['overall_integrity']}")
            print(f"  - Model Integrity: {integrity_check['model_integrity']['integrity_status']}")
            print(f"  - Integration Integrity: {'Valid' if all(integrity_check['integration_integrity'].values()) else 'Issues Detected'}")
        except Exception as e:
            print(f"Integrity check error: {e}")
        
        # Test Case 5: Performance Metrics
        print("\n\n5. PERFORMANCE METRICS")
        print("-" * 40)
        
        # Measure processing time for batch of events
        import time
        
        test_user = "performance_test_user"
        test_events = self.generate_normal_user_data(test_user, 20)
        
        start_time = time.time()
        for event in test_events:
            self.analytics_integration.process_user_event(test_user, event)
        processing_time = time.time() - start_time
        
        print(f"Performance Test Results:")
        print(f"  - Events Processed: {len(test_events)}")
        print(f"  - Total Processing Time: {processing_time:.3f} seconds")
        print(f"  - Average Time per Event: {processing_time/len(test_events)*1000:.1f} ms")
        print(f"  - Processing Rate: {len(test_events)/processing_time:.1f} events/second")
        
        # Test Case 6: Error Handling and Edge Cases
        print("\n\n6. ERROR HANDLING AND EDGE CASES")
        print("-" * 40)
        
        # Test with malformed data
        malformed_events = [
            None,
            {},
            {'timestamp': None},
            {'timestamp': 'invalid', 'event_type': None},
            {'application': 'test.exe'}  # Missing required fields
        ]
        
        print("Testing malformed data handling:")
        for i, malformed_event in enumerate(malformed_events):
            try:
                result = self.analytics_integration.process_user_event(f"malformed_user_{i}", malformed_event)
                print(f"  - Malformed event {i+1}: Handled gracefully")
            except Exception as e:
                print(f"  - Malformed event {i+1}: Error - {str(e)[:50]}...")
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        # Save demonstration results
        self.save_demo_results({
            'normal_user_results': normal_results,
            'anomalous_user_results': anomalous_results,
            'system_status': system_status,
            'analytics_summary': analytics_summary,
            'performance_metrics': {
                'events_processed': len(test_events),
                'processing_time': processing_time,
                'avg_time_per_event_ms': processing_time/len(test_events)*1000,
                'processing_rate': len(test_events)/processing_time
            }
        })
    
    def save_demo_results(self, results: dict):
        """Save demonstration results to file."""
        
        try:
            # Create results directory
            results_dir = Path("demo_results")
            results_dir.mkdir(exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save results
            results_file = results_dir / f"behavioral_analytics_demo_{timestamp}.json"
            
            # Convert datetime objects to strings for JSON serialization
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=serialize_datetime)
            
            print(f"Demo results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving demo results: {e}")
    
    def run_stress_test(self, num_users: int = 10, events_per_user: int = 50):
        """Run stress test with multiple users."""
        
        print(f"\nRunning stress test with {num_users} users, {events_per_user} events each...")
        
        import time
        start_time = time.time()
        
        results = []
        for user_idx in range(num_users):
            user_id = f"stress_test_user_{user_idx}"
            
            # Generate mixed behavior data
            if user_idx % 3 == 0:
                user_data = self.generate_normal_user_data(user_id, events_per_user)
            elif user_idx % 3 == 1:
                user_data = self.generate_anomalous_user_data(user_id, events_per_user // 2)
            else:
                user_data = self.generate_normal_user_data(user_id, events_per_user // 2) + \
                           self.generate_anomalous_user_data(user_id, events_per_user // 2)
            
            # Process events for this user
            user_results = []
            for event in user_data:
                result = self.analytics_integration.process_user_event(user_id, event)
                user_results.append(result)
            
            results.extend(user_results)
            
            if (user_idx + 1) % 5 == 0:
                print(f"  - Processed {user_idx + 1}/{num_users} users...")
        
        total_time = time.time() - start_time
        total_events = len(results)
        
        # Calculate statistics
        anomalous_events = sum(1 for r in results if r['analysis_result']['is_anomalous'])
        avg_risk_score = np.mean([r['analysis_result']['risk_score'] for r in results])
        
        print(f"\nStress Test Results:")
        print(f"  - Total Users: {num_users}")
        print(f"  - Total Events: {total_events}")
        print(f"  - Total Time: {total_time:.2f} seconds")
        print(f"  - Processing Rate: {total_events/total_time:.1f} events/second")
        print(f"  - Anomalies Detected: {anomalous_events} ({anomalous_events/total_events:.1%})")
        print(f"  - Average Risk Score: {avg_risk_score:.3f}")
        
        return {
            'num_users': num_users,
            'total_events': total_events,
            'total_time': total_time,
            'processing_rate': total_events/total_time,
            'anomalies_detected': anomalous_events,
            'anomaly_rate': anomalous_events/total_events,
            'avg_risk_score': avg_risk_score
        }


def main():
    """Main demonstration function."""
    
    print("Advanced Behavioral Analytics System Demonstration")
    print("=" * 60)
    print("This demo showcases the comprehensive behavioral analytics capabilities")
    print("including deep learning models, anomaly detection, and security responses.")
    print()
    
    try:
        # Initialize demo
        demo = AdvancedBehavioralAnalyticsDemo()
        
        # Run comprehensive demonstration
        demo.run_comprehensive_demo()
        
        # Optional: Run stress test
        print("\nWould you like to run a stress test? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            stress_results = demo.run_stress_test(num_users=20, events_per_user=30)
            
            # Save stress test results
            demo.save_demo_results({'stress_test_results': stress_results})
        
        print("\nDemo completed successfully!")
        print("Check the demo_results directory for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()