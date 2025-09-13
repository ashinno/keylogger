"""Comprehensive integration tests for the Enhanced Keylogger system.

These tests verify full system workflows and component interactions,
ensuring proper integration between all system components.
"""

import unittest
import tempfile
import os
import json
import time
import threading
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.keylogger import KeyloggerCore
from core.config_manager import ConfigManager
from core.encryption_manager import EncryptionManager
from core.logging_manager import LoggingManager


class TestSystemInitialization(unittest.TestCase):
    """Test complete system initialization workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        # Create minimal test configuration
        self.test_config = {
            "logging": {
                "log_file": os.path.join(self.test_dir, "test.log"),
                "buffer_size": 10,
                "flush_interval": 1
            },
            "encryption": {
                "enabled": True,
                "key_file": os.path.join(self.test_dir, "test.key")
            },
            "features": {
                "keyboard": True,
                "mouse": True,
                "clipboard": False,
                "window_tracking": True,
                "screenshots": False,
                "camera": False
            },
            "ml": {
                "enabled": False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_system_initialization(self):
        """Test complete system initialization workflow."""
        # Initialize keylogger core
        keylogger = KeyloggerCore(self.config_file)
        
        # Verify core components are initialized
        self.assertIsNotNone(keylogger.config_manager)
        self.assertIsNotNone(keylogger.encryption_manager)
        self.assertIsNotNone(keylogger.logging_manager)
        
        # Verify configuration is loaded
        self.assertEqual(keylogger.config_manager.get('logging.buffer_size'), 10)
        
        # Verify session ID is generated
        self.assertIsNotNone(keylogger.session_id)
        self.assertTrue(keylogger.session_id.startswith('session_'))
    
    def test_component_dependency_chain(self):
        """Test proper component dependency initialization."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Verify dependency chain: Config -> Encryption -> Logging
        self.assertIsNotNone(keylogger.config_manager)
        self.assertIsNotNone(keylogger.encryption_manager)
        self.assertIsNotNone(keylogger.logging_manager)
        
        # Verify encryption manager has config reference
        self.assertEqual(keylogger.encryption_manager.config_manager, keylogger.config_manager)
        
        # Verify logging manager has both config and encryption references
        self.assertEqual(keylogger.logging_manager.config_manager, keylogger.config_manager)
        self.assertEqual(keylogger.logging_manager.encryption_manager, keylogger.encryption_manager)
    
    @patch('core.keylogger.KeyboardListener')
    @patch('core.keylogger.MouseListener')
    def test_listener_initialization_workflow(self, mock_mouse, mock_keyboard):
        """Test listener initialization based on configuration."""
        # Mock listener classes
        mock_keyboard_instance = Mock()
        mock_mouse_instance = Mock()
        mock_keyboard.return_value = mock_keyboard_instance
        mock_mouse.return_value = mock_mouse_instance
        
        keylogger = KeyloggerCore(self.config_file)
        
        # Verify listeners are initialized based on config
        mock_keyboard.assert_called_once_with(keylogger)
        mock_mouse.assert_called_once_with(keylogger)
        
        # Verify listener instances are stored
        self.assertEqual(keylogger.keyboard_listener, mock_keyboard_instance)
        self.assertEqual(keylogger.mouse_listener, mock_mouse_instance)
    
    def test_configuration_injection_workflow(self):
        """Test initialization with injected configuration manager."""
        # Create separate config manager
        config_manager = ConfigManager(self.config_file)
        config_manager.load_config()
        
        # Initialize keylogger with injected config
        keylogger = KeyloggerCore(config_manager)
        
        # Verify injected config is used
        self.assertEqual(keylogger.config_manager, config_manager)
        self.assertIsNotNone(keylogger.encryption_manager)
        self.assertIsNotNone(keylogger.logging_manager)


class TestSystemLifecycle(unittest.TestCase):
    """Test complete system lifecycle workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        self.test_config = {
            "logging": {
                "log_file": os.path.join(self.test_dir, "test.log"),
                "buffer_size": 5,
                "flush_interval": 1
            },
            "encryption": {
                "enabled": True,
                "key_file": os.path.join(self.test_dir, "test.key")
            },
            "features": {
                "keyboard": True,
                "mouse": True,
                "clipboard": False,
                "window_tracking": False,
                "screenshots": False,
                "camera": False
            },
            "ml": {"enabled": False}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('core.keylogger.KeyboardListener')
    @patch('core.keylogger.MouseListener')
    def test_complete_start_stop_workflow(self, mock_mouse, mock_keyboard):
        """Test complete system start and stop workflow."""
        # Mock listener instances
        mock_keyboard_instance = Mock()
        mock_mouse_instance = Mock()
        mock_keyboard.return_value = mock_keyboard_instance
        mock_mouse.return_value = mock_mouse_instance
        
        keylogger = KeyloggerCore(self.config_file)
        
        # Test start workflow
        self.assertTrue(keylogger.start())
        self.assertTrue(keylogger.is_running())
        self.assertIsNotNone(keylogger.start_time)
        
        # Verify listeners are started
        mock_keyboard_instance.start.assert_called_once()
        mock_mouse_instance.start.assert_called_once()
        
        # Test stop workflow
        self.assertTrue(keylogger.stop())
        self.assertFalse(keylogger.is_running())
        
        # Verify listeners are stopped
        mock_keyboard_instance.stop.assert_called_once()
        mock_mouse_instance.stop.assert_called_once()
    
    @patch('core.keylogger.KeyboardListener')
    def test_event_logging_workflow(self, mock_keyboard):
        """Test complete event logging workflow."""
        mock_keyboard_instance = Mock()
        mock_keyboard.return_value = mock_keyboard_instance
        
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Test event logging
        success = keylogger.log_event('keyboard', 'test_key', 'Test Window')
        self.assertTrue(success)
        
        # Verify stats are updated
        stats = keylogger.get_stats()
        self.assertEqual(stats['total_events'], 1)
        self.assertEqual(stats['keyboard_events'], 1)
        
        keylogger.stop()
    
    def test_session_statistics_workflow(self):
        """Test session statistics tracking workflow."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Test initial session stats
        session_stats = keylogger.get_session_stats()
        self.assertIsNotNone(session_stats['session_id'])
        self.assertFalse(session_stats['running'])
        
        # Start keylogger and test running stats
        keylogger.start()
        session_stats = keylogger.get_session_stats()
        self.assertTrue(session_stats['running'])
        self.assertIsNotNone(session_stats['start_time'])
        
        keylogger.stop()


class TestComponentInteractions(unittest.TestCase):
    """Test interactions between system components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        self.test_config = {
            "logging": {
                "log_file": os.path.join(self.test_dir, "test.log"),
                "buffer_size": 3,
                "flush_interval": 1
            },
            "encryption": {
                "enabled": True,
                "key_file": os.path.join(self.test_dir, "test.key")
            },
            "features": {
                "keyboard": True,
                "mouse": True,
                "clipboard": True,
                "window_tracking": True,
                "screenshots": False,
                "camera": False
            },
            "ml": {"enabled": False}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_encryption_logging_interaction(self):
        """Test interaction between config, encryption, and logging components."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Test configuration propagation
        buffer_size = keylogger.config_manager.get('logging.buffer_size')
        self.assertEqual(buffer_size, 3)
        
        # Test encryption key generation
        key_file = keylogger.config_manager.get('encryption.key_file')
        self.assertTrue(os.path.exists(key_file))
        
        # Test logging with encryption
        keylogger.start()
        success = keylogger.log_event('test', 'encrypted_data')
        self.assertTrue(success)
        keylogger.stop()
        
        # Verify log file exists
        log_file = keylogger.config_manager.get('logging.log_file')
        self.assertTrue(os.path.exists(log_file))
    
    @patch('core.keylogger.WindowMonitor')
    def test_window_monitor_integration(self, mock_window_monitor):
        """Test window monitor integration with core system."""
        mock_monitor_instance = Mock()
        mock_window_monitor.return_value = mock_monitor_instance
        
        keylogger = KeyloggerCore(self.config_file)
        
        # Verify window monitor is initialized
        mock_window_monitor.assert_called_once_with(keylogger)
        self.assertEqual(keylogger.window_monitor, mock_monitor_instance)
        
        # Test window update callback
        keylogger.update_active_window('Test Application', 'test.exe')
        
        # Verify session stats are updated
        session_stats = keylogger.get_session_stats()
        self.assertEqual(session_stats['active_window'], 'Test Application')
        self.assertEqual(session_stats['active_application'], 'test.exe')
    
    def test_event_flow_through_components(self):
        """Test complete event flow through all components."""
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Simulate event flow: Listener -> Core -> Logging -> Encryption
        test_events = [
            ('keyboard', 'a', 'Notepad'),
            ('mouse', {'x': 100, 'y': 200, 'button': 'left'}, 'Browser'),
            ('clipboard', 'copied text', 'Editor')
        ]
        
        for event_type, content, window in test_events:
            success = keylogger.log_event(event_type, content, window)
            self.assertTrue(success)
        
        # Verify all events are tracked
        stats = keylogger.get_stats()
        self.assertEqual(stats['total_events'], 3)
        self.assertEqual(stats['keyboard_events'], 1)
        self.assertEqual(stats['mouse_events'], 1)
        self.assertEqual(stats['clipboard_events'], 1)
        
        keylogger.stop()


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across component interactions."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_missing_config_file_handling(self):
        """Test handling of missing configuration file."""
        non_existent_config = os.path.join(self.test_dir, 'missing.json')
        
        # Should handle missing config gracefully
        keylogger = KeyloggerCore(non_existent_config)
        
        # Should still initialize with defaults
        self.assertIsNotNone(keylogger.config_manager)
        self.assertIsNotNone(keylogger.session_id)
    
    def test_component_initialization_failure_handling(self):
        """Test handling of component initialization failures."""
        # Create invalid config
        invalid_config = {
            "logging": {
                "log_file": "/invalid/path/test.log",  # Invalid path
                "buffer_size": "invalid"  # Invalid type
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        # Should handle initialization errors gracefully
        keylogger = KeyloggerCore(self.config_file)
        
        # Core should still be initialized
        self.assertIsNotNone(keylogger.config_manager)
        self.assertIsNotNone(keylogger.session_id)
    
    @patch('core.keylogger.KeyboardListener')
    def test_listener_failure_handling(self, mock_keyboard):
        """Test handling of listener failures."""
        # Mock listener that fails to start
        mock_keyboard_instance = Mock()
        mock_keyboard_instance.start.side_effect = Exception("Listener failed")
        mock_keyboard.return_value = mock_keyboard_instance
        
        # Create minimal config
        test_config = {
            "features": {"keyboard": True, "mouse": False, "clipboard": False},
            "ml": {"enabled": False}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        keylogger = KeyloggerCore(self.config_file)
        
        # Should handle listener start failure gracefully
        result = keylogger.start()
        self.assertTrue(result)  # Should still report success for core start
        
        keylogger.stop()


class TestConcurrencyIntegration(unittest.TestCase):
    """Test concurrent operations and thread safety."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        self.test_config = {
            "logging": {
                "log_file": os.path.join(self.test_dir, "test.log"),
                "buffer_size": 10,
                "flush_interval": 1
            },
            "encryption": {"enabled": True},
            "features": {
                "keyboard": True,
                "mouse": True,
                "clipboard": False,
                "window_tracking": False,
                "screenshots": False,
                "camera": False
            },
            "ml": {"enabled": False}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_concurrent_event_logging(self):
        """Test concurrent event logging from multiple threads."""
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Function to log events from multiple threads
        def log_events(thread_id, event_count):
            for i in range(event_count):
                keylogger.log_event('test', f'thread_{thread_id}_event_{i}')
        
        # Create multiple threads
        threads = []
        events_per_thread = 5
        thread_count = 3
        
        for i in range(thread_count):
            thread = threading.Thread(target=log_events, args=(i, events_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all events were logged
        stats = keylogger.get_stats()
        expected_events = thread_count * events_per_thread
        self.assertEqual(stats['total_events'], expected_events)
        
        keylogger.stop()
    
    def test_start_stop_race_conditions(self):
        """Test start/stop operations under race conditions."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Function to repeatedly start/stop
        def start_stop_cycle():
            for _ in range(5):
                keylogger.start()
                time.sleep(0.01)
                keylogger.stop()
                time.sleep(0.01)
        
        # Run multiple start/stop cycles concurrently
        threads = []
        for _ in range(2):
            thread = threading.Thread(target=start_stop_cycle)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # System should be in a consistent state
        self.assertFalse(keylogger.is_running())


class TestMLIntegrationWorkflows(unittest.TestCase):
    """Test ML component integration workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        self.test_config = {
            "logging": {
                "log_file": os.path.join(self.test_dir, "test.log"),
                "buffer_size": 5
            },
            "encryption": {"enabled": False},  # Disable for simpler testing
            "features": {"keyboard": True, "mouse": True, "clipboard": False},
            "ml": {
                "enabled": True,
                "behavioral_analytics": {"enabled": True},
                "keystroke_dynamics": {"enabled": True},
                "insider_threat": {"enabled": True, "threshold": 0.7},
                "risk_scoring": {"enabled": True, "threshold": 0.8}
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('core.keylogger.BehavioralAnalyticsEngine')
    @patch('core.keylogger.KeystrokeDynamicsAnalyzer')
    def test_ml_event_processing_workflow(self, mock_kd, mock_ba):
        """Test ML event processing workflow."""
        # Mock ML components
        mock_ba_instance = Mock()
        mock_kd_instance = Mock()
        mock_ba.return_value = mock_ba_instance
        mock_kd.return_value = mock_kd_instance
        
        # Configure ML responses
        mock_ba_instance.process_event.return_value = {'is_anomaly': False}
        mock_kd_instance.process_keystroke.return_value = {
            'status': 'completed',
            'authentication': {'authenticated': True}
        }
        
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Log events that should trigger ML processing
        keylogger.log_event('keyboard', 'test_key')
        keylogger.log_event('mouse', {'x': 100, 'y': 200})
        
        # Verify ML components were called
        self.assertEqual(mock_ba_instance.process_event.call_count, 2)
        mock_kd_instance.process_keystroke.assert_called_once()
        
        keylogger.stop()
    
    def test_ml_disabled_workflow(self):
        """Test workflow when ML is disabled."""
        # Disable ML in config
        self.test_config['ml']['enabled'] = False
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Events should be logged normally without ML processing
        success = keylogger.log_event('keyboard', 'test_key')
        self.assertTrue(success)
        
        stats = keylogger.get_stats()
        self.assertEqual(stats['total_events'], 1)
        
        keylogger.stop()


if __name__ == '__main__':
    # Configure test logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSystemInitialization,
        TestSystemLifecycle,
        TestComponentInteractions,
        TestErrorHandlingIntegration,
        TestConcurrencyIntegration,
        TestMLIntegrationWorkflows
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    print(f"\n{'='*50}")