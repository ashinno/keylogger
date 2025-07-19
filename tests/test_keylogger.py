"""Comprehensive test suite for the enhanced keylogger."""

import os
import sys
import time
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.config_manager import ConfigManager
    from core.encryption_manager import EncryptionManager
    from core.logging_manager import LoggingManager, LogEntry
    from core.keylogger import KeyloggerCore
    from listeners.keyboard_listener import KeyboardListener
    from listeners.mouse_listener import MouseListener
    from listeners.clipboard_listener import ClipboardListener
    from utils.window_monitor import WindowMonitor
    from utils.performance_monitor import PerformanceMonitor
    from parsers.log_parser import LogParser, ParsedEvent
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create test config
        self.test_config = {
            'logging': {
                'level': 'INFO',
                'file_path': 'test.log',
                'max_size_mb': 10
            },
            'encryption': {
                'enabled': True,
                'key_file': 'test.key'
            },
            'features': {
                'keyboard': True,
                'mouse': True,
                'clipboard': False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config(self):
        """Test configuration loading."""
        config_manager = ConfigManager(self.config_file)
        
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        self.assertEqual(config_manager.get('logging.file_path'), 'test.log')
        self.assertTrue(config_manager.get('encryption.enabled'))
        self.assertTrue(config_manager.get('features.keyboard'))
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager(self.config_file)
        
        # Test valid config
        self.assertTrue(config_manager.validate_config())
        
        # Test invalid config
        config_manager.config['logging']['level'] = 'INVALID'
        self.assertFalse(config_manager.validate_config())
    
    def test_feature_enabled(self):
        """Test feature enablement checks."""
        config_manager = ConfigManager(self.config_file)
        
        self.assertTrue(config_manager.is_feature_enabled('keyboard'))
        self.assertTrue(config_manager.is_feature_enabled('mouse'))
        self.assertFalse(config_manager.is_feature_enabled('clipboard'))
        self.assertFalse(config_manager.is_feature_enabled('nonexistent'))
    
    def test_config_update(self):
        """Test configuration updates."""
        config_manager = ConfigManager(self.config_file)
        
        # Update config
        config_manager.update_config({'features.clipboard': True})
        self.assertTrue(config_manager.get('features.clipboard'))
        
        # Save and reload
        config_manager.save_config()
        new_config_manager = ConfigManager(self.config_file)
        self.assertTrue(new_config_manager.get('features.clipboard'))


class TestEncryptionManager(unittest.TestCase):
    """Test cases for EncryptionManager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.key_file = os.path.join(self.temp_dir, 'test.key')
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'encryption.key_file': self.key_file,
            'encryption.algorithm': 'AES-256-GCM'
        }.get(key, default)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_key_generation(self):
        """Test encryption key generation."""
        encryption_manager = EncryptionManager(self.mock_config)
        
        # Generate key from password
        key = encryption_manager.generate_key_from_password('test_password')
        self.assertIsNotNone(key)
        self.assertEqual(len(key), 32)  # 256 bits
        
        # Generate random key
        random_key = encryption_manager.generate_random_key()
        self.assertIsNotNone(random_key)
        self.assertEqual(len(random_key), 32)
        self.assertNotEqual(key, random_key)
    
    def test_key_save_load(self):
        """Test key saving and loading."""
        encryption_manager = EncryptionManager(self.mock_config)
        
        # Generate and save key
        original_key = encryption_manager.generate_random_key()
        encryption_manager.save_key(original_key)
        
        # Load key
        loaded_key = encryption_manager.load_key()
        self.assertEqual(original_key, loaded_key)
    
    def test_data_encryption(self):
        """Test data encryption and decryption."""
        encryption_manager = EncryptionManager(self.mock_config)
        
        # Set up key
        key = encryption_manager.generate_random_key()
        encryption_manager.key = key
        
        # Test data
        test_data = "This is test data for encryption"
        
        # Encrypt
        encrypted_data = encryption_manager.encrypt_data(test_data)
        self.assertIsNotNone(encrypted_data)
        self.assertNotEqual(test_data.encode(), encrypted_data)
        
        # Decrypt
        decrypted_data = encryption_manager.decrypt_data(encrypted_data)
        self.assertEqual(test_data, decrypted_data)
    
    def test_file_encryption(self):
        """Test file encryption and decryption."""
        encryption_manager = EncryptionManager(self.mock_config)
        
        # Set up key
        key = encryption_manager.generate_random_key()
        encryption_manager.key = key
        
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.txt')
        test_content = "This is test file content"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Encrypt file
        encrypted_file = test_file + '.enc'
        success = encryption_manager.encrypt_file(test_file, encrypted_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(encrypted_file))
        
        # Decrypt file
        decrypted_file = test_file + '.dec'
        success = encryption_manager.decrypt_file(encrypted_file, decrypted_file)
        self.assertTrue(success)
        
        # Verify content
        with open(decrypted_file, 'r') as f:
            decrypted_content = f.read()
        
        self.assertEqual(test_content, decrypted_content)


class TestLoggingManager(unittest.TestCase):
    """Test cases for LoggingManager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'logging.file_path': self.log_file,
            'logging.buffer_size': 10,
            'logging.flush_interval': 1.0,
            'logging.max_size_mb': 1,
            'logging.enable_encryption': False
        }.get(key, default)
        
        # Mock encryption manager
        self.mock_encryption = Mock()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_entry_creation(self):
        """Test log entry creation."""
        entry = LogEntry(
            timestamp=time.time(),
            event_type="Test Event",
            content="Test content",
            window_name="Test Window"
        )
        
        self.assertIsNotNone(entry.timestamp)
        self.assertEqual(entry.event_type, "Test Event")
        self.assertEqual(entry.content, "Test content")
        self.assertEqual(entry.window_name, "Test Window")
    
    def test_logging_manager_initialization(self):
        """Test logging manager initialization."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        
        self.assertIsNotNone(logging_manager)
        self.assertEqual(logging_manager.log_file_path, self.log_file)
        self.assertEqual(logging_manager.buffer_size, 10)
    
    def test_event_logging(self):
        """Test event logging."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        
        # Log an event
        logging_manager.log_event(
            "Key Press",
            "test key",
            "Test Window",
            metadata={'test': 'data'}
        )
        
        # Check buffer
        self.assertEqual(len(logging_manager.buffer), 1)
        
        # Flush buffer
        logging_manager.flush_buffer()
        
        # Check file exists
        self.assertTrue(os.path.exists(self.log_file))
    
    def test_buffer_auto_flush(self):
        """Test automatic buffer flushing."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        
        # Fill buffer beyond capacity
        for i in range(15):
            logging_manager.log_event(
                "Test Event",
                f"content {i}",
                "Test Window"
            )
        
        # Buffer should have been flushed
        self.assertLess(len(logging_manager.buffer), 15)
        self.assertTrue(os.path.exists(self.log_file))


class TestLogParser(unittest.TestCase):
    """Test cases for LogParser."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Create test log content
        self.test_log_content = [
            '[2024-01-01 12:00:00] Key Press: hello (window: Notepad)',
            '[2024-01-01 12:00:01] Key Press: world (window: Notepad)',
            '[2024-01-01 12:00:02] Mouse Click: (100, 200) (window: Browser)',
            '{"timestamp": 1704110403, "event_type": "Clipboard", "content": "copied text", "window_name": "Browser", "metadata": {}}'
        ]
        
        with open(self.test_log_file, 'w') as f:
            f.write('\n'.join(self.test_log_content))
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'parser.enable_content_analysis': True,
            'parser.enable_pattern_detection': True,
            'parser.privacy_mode': False
        }.get(key, default)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_parsing(self):
        """Test log file parsing."""
        parser = LogParser(self.mock_config)
        events = parser.parse_log_file(self.test_log_file)
        
        self.assertEqual(len(events), 4)
        
        # Check first event (legacy format)
        first_event = events[0]
        self.assertEqual(first_event.event_type, 'Key Press')
        self.assertEqual(first_event.content, 'hello')
        self.assertEqual(first_event.window_name, 'Notepad')
        
        # Check last event (JSON format)
        last_event = events[3]
        self.assertEqual(last_event.event_type, 'Clipboard')
        self.assertEqual(last_event.content, 'copied text')
        self.assertEqual(last_event.window_name, 'Browser')
    
    def test_content_processing(self):
        """Test content processing."""
        parser = LogParser(self.mock_config)
        
        # Test keyboard content processing
        processed = parser._process_content('<backspace>hello<enter>', 'Key Press')
        self.assertIn('⌫', processed)
        self.assertIn('↵', processed)
        
        # Test content length limiting
        long_content = 'a' * 2000
        processed = parser._process_content(long_content, 'Test')
        self.assertLess(len(processed), len(long_content))
        self.assertTrue(processed.endswith('...'))
    
    def test_pattern_detection(self):
        """Test pattern detection."""
        parser = LogParser(self.mock_config)
        
        # Test email detection
        tags = parser._detect_content_tags('Contact me at test@example.com')
        self.assertIn('email', tags)
        
        # Test URL detection
        tags = parser._detect_content_tags('Visit https://example.com')
        self.assertIn('url', tags)
        
        # Test phone detection
        tags = parser._detect_content_tags('Call me at (555) 123-4567')
        self.assertIn('phone', tags)
    
    def test_report_generation(self):
        """Test readable report generation."""
        parser = LogParser(self.mock_config)
        events = parser.parse_log_file(self.test_log_file)
        
        report_file = os.path.join(self.temp_dir, 'test_report.txt')
        report = parser.generate_readable_report(events, report_file)
        
        self.assertIsNotNone(report)
        self.assertTrue(os.path.exists(report_file))
        
        # Check report content
        with open(report_file, 'r') as f:
            report_content = f.read()
        
        self.assertIn('KEYLOGGER ACTIVITY REPORT', report_content)
        self.assertIn('STATISTICS', report_content)
    
    def test_export_formats(self):
        """Test different export formats."""
        parser = LogParser(self.mock_config)
        events = parser.parse_log_file(self.test_log_file)
        
        # Test CSV export
        csv_file = os.path.join(self.temp_dir, 'test.csv')
        parser.export_to_csv(events, csv_file)
        self.assertTrue(os.path.exists(csv_file))
        
        # Test JSON export
        json_file = os.path.join(self.temp_dir, 'test.json')
        parser.export_to_json(events, json_file)
        self.assertTrue(os.path.exists(json_file))
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        self.assertIn('events', json_data)
        self.assertIn('statistics', json_data)
        self.assertEqual(len(json_data['events']), 4)


class TestKeyloggerCore(unittest.TestCase):
    """Test cases for KeyloggerCore."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config manager
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'logging.file_path': os.path.join(self.temp_dir, 'test.log'),
            'encryption.enabled': False,
            'features.keyboard': True,
            'features.mouse': True,
            'features.clipboard': False,
            'features.window_tracking': True,
            'features.performance_monitoring': False
        }.get(key, default)
        
        self.mock_config.is_feature_enabled.side_effect = lambda feature: {
            'keyboard': True,
            'mouse': True,
            'clipboard': False,
            'window_tracking': True,
            'performance_monitoring': False
        }.get(feature, False)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('core.keylogger.KeyboardListener')
    @patch('core.keylogger.MouseListener')
    @patch('core.keylogger.WindowMonitor')
    def test_keylogger_initialization(self, mock_window, mock_mouse, mock_keyboard):
        """Test keylogger core initialization."""
        keylogger = KeyloggerCore(self.mock_config)
        
        self.assertIsNotNone(keylogger.config)
        self.assertIsNotNone(keylogger.encryption_manager)
        self.assertIsNotNone(keylogger.logging_manager)
        
        # Check that enabled components are initialized
        mock_keyboard.assert_called_once()
        mock_mouse.assert_called_once()
        mock_window.assert_called_once()
    
    @patch('core.keylogger.KeyboardListener')
    @patch('core.keylogger.MouseListener')
    def test_keylogger_start_stop(self, mock_mouse, mock_keyboard):
        """Test keylogger start and stop functionality."""
        # Mock listener instances
        mock_kb_instance = Mock()
        mock_mouse_instance = Mock()
        mock_keyboard.return_value = mock_kb_instance
        mock_mouse.return_value = mock_mouse_instance
        
        keylogger = KeyloggerCore(self.mock_config)
        
        # Test start
        self.assertFalse(keylogger.is_running())
        
        # Test stop
        keylogger.stop()
        self.assertFalse(keylogger.is_running())
    
    def test_event_logging(self):
        """Test event logging functionality."""
        keylogger = KeyloggerCore(self.mock_config)
        
        # Log an event
        keylogger.log_event(
            "Test Event",
            "test content",
            "Test Window",
            metadata={'test': 'data'}
        )
        
        # Check that event was logged
        stats = keylogger.get_stats()
        self.assertGreater(stats.get('total_events', 0), 0)
    
    def test_config_reload(self):
        """Test configuration reloading."""
        keylogger = KeyloggerCore(self.mock_config)
        
        # Reload config
        keylogger.reload_config()
        
        # Verify config manager reload was called
        self.mock_config.reload_config.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create minimal config
        config_data = {
            'logging': {
                'file_path': os.path.join(self.temp_dir, 'keylog.txt'),
                'level': 'INFO',
                'buffer_size': 5,
                'flush_interval': 0.1
            },
            'encryption': {
                'enabled': False
            },
            'features': {
                'keyboard': False,  # Disable to avoid actual input capture
                'mouse': False,
                'clipboard': False,
                'window_tracking': False,
                'performance_monitoring': False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_system_integration(self):
        """Test complete system integration."""
        # Initialize config manager
        config_manager = ConfigManager(self.config_file)
        
        # Initialize keylogger core
        keylogger = KeyloggerCore(config_manager)
        
        # Log some test events
        for i in range(10):
            keylogger.log_event(
                "Test Event",
                f"test content {i}",
                "Test Window",
                metadata={'index': i}
            )
        
        # Force flush
        keylogger.logging_manager.flush_buffer()
        
        # Verify log file was created
        log_file = config_manager.get('logging.file_path')
        self.assertTrue(os.path.exists(log_file))
        
        # Parse the log file
        parser = LogParser(config_manager)
        events = parser.parse_log_file(log_file)
        
        # Verify events were parsed correctly
        self.assertEqual(len(events), 10)
        
        for i, event in enumerate(events):
            self.assertEqual(event.event_type, "Test Event")
            self.assertEqual(event.content, f"test content {i}")
            self.assertEqual(event.window_name, "Test Window")
        
        # Generate report
        report_file = os.path.join(self.temp_dir, 'integration_report.txt')
        report = parser.generate_readable_report(events, report_file)
        
        self.assertIsNotNone(report)
        self.assertTrue(os.path.exists(report_file))
        
        # Verify statistics
        stats = parser.get_statistics()
        self.assertEqual(stats['total_events'], 10)
        self.assertGreater(stats['total_characters'], 0)


class TestPerformance(unittest.TestCase):
    """Performance tests for the keylogger."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config for performance testing
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'logging.file_path': os.path.join(self.temp_dir, 'perf_test.log'),
            'logging.buffer_size': 1000,
            'encryption.enabled': False
        }.get(key, default)
        
        self.mock_encryption = Mock()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logging_performance(self):
        """Test logging performance with many events."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        
        # Measure time for logging many events
        start_time = time.time()
        
        for i in range(10000):
            logging_manager.log_event(
                "Performance Test",
                f"content {i}",
                "Test Window",
                metadata={'index': i}
            )
        
        # Force flush
        logging_manager.flush_buffer()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be able to log 10k events in reasonable time
        self.assertLess(duration, 5.0)  # Less than 5 seconds
        
        # Verify all events were logged
        log_file = self.mock_config.get('logging.file_path')
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 10000)
    
    def test_parsing_performance(self):
        """Test parsing performance with large log files."""
        # Create large test log file
        log_file = os.path.join(self.temp_dir, 'large_test.log')
        
        with open(log_file, 'w') as f:
            for i in range(5000):
                f.write(f'[2024-01-01 12:00:{i%60:02d}] Key Press: test{i} (window: TestWindow)\n')
        
        # Measure parsing time
        parser = LogParser()
        
        start_time = time.time()
        events = parser.parse_log_file(log_file)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should parse 5k events in reasonable time
        self.assertLess(duration, 10.0)  # Less than 10 seconds
        self.assertEqual(len(events), 5000)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfigManager,
        TestEncryptionManager,
        TestLoggingManager,
        TestLogParser,
        TestKeyloggerCore,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)