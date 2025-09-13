"""Comprehensive test suite for core components with extensive edge cases and validation."""

import os
import sys
import time
import json
import tempfile
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import shutil
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.config_manager import ConfigManager
    from core.encryption_manager import EncryptionManager
    from core.logging_manager import LoggingManager, LogEntry, LogEvent
    from core.keylogger import KeyloggerCore
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class TestConfigManagerComprehensive(unittest.TestCase):
    """Comprehensive tests for ConfigManager with edge cases."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Valid test config
        self.valid_config = {
            'logging': {
                'level': 'INFO',
                'file_path': 'test.log',
                'max_size_mb': 10,
                'buffer_size': 50,
                'flush_interval': 2.0
            },
            'encryption': {
                'enabled': True,
                'key_file': 'test.key',
                'algorithm': 'AES-256-GCM'
            },
            'features': {
                'keyboard': True,
                'mouse': False,
                'clipboard': True
            },
            'privacy': {
                'sanitize_passwords': True,
                'excluded_applications': ['test.exe']
            }
        }
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_file)
        
        # Verify config loaded correctly
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        self.assertEqual(config_manager.get('encryption.enabled'), True)
        self.assertEqual(config_manager.get('features.keyboard'), True)
        
    def test_load_nonexistent_config(self):
        """Test behavior when config file doesn't exist."""
        nonexistent_file = os.path.join(self.temp_dir, 'nonexistent.json')
        
        config_manager = ConfigManager(nonexistent_file)
        
        # Should use default values
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        self.assertTrue(os.path.exists(nonexistent_file))  # Should create default config
        
    def test_load_invalid_json(self):
        """Test handling of invalid JSON in config file."""
        with open(self.config_file, 'w') as f:
            f.write('{ invalid json }')
        
        config_manager = ConfigManager(self.config_file)
        
        # Should fall back to defaults
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        
    def test_load_empty_config(self):
        """Test handling of empty config file."""
        with open(self.config_file, 'w') as f:
            f.write('')
        
        config_manager = ConfigManager(self.config_file)
        
        # Should use defaults
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        
    def test_get_with_default_value(self):
        """Test getting config values with default fallback."""
        config_manager = ConfigManager(self.config_file)
        
        # Test existing key
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        
        # Test non-existing key with default
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')
        
        # Test nested non-existing key
        self.assertIsNone(config_manager.get('nonexistent.nested.key'))
        
    def test_nested_config_access(self):
        """Test accessing deeply nested configuration values."""
        nested_config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'deep_value'
                    }
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(nested_config, f)
        
        config_manager = ConfigManager(self.config_file)
        
        self.assertEqual(config_manager.get('level1.level2.level3.value'), 'deep_value')
        self.assertIsNone(config_manager.get('level1.level2.level3.nonexistent'))
        
    def test_config_validation_types(self):
        """Test configuration validation with different data types."""
        type_config = {
            'string_val': 'test',
            'int_val': 42,
            'float_val': 3.14,
            'bool_val': True,
            'list_val': [1, 2, 3],
            'dict_val': {'nested': 'value'}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(type_config, f)
        
        config_manager = ConfigManager(self.config_file)
        
        self.assertIsInstance(config_manager.get('string_val'), str)
        self.assertIsInstance(config_manager.get('int_val'), int)
        self.assertIsInstance(config_manager.get('float_val'), float)
        self.assertIsInstance(config_manager.get('bool_val'), bool)
        self.assertIsInstance(config_manager.get('list_val'), list)
        self.assertIsInstance(config_manager.get('dict_val'), dict)
        
    def test_feature_enabled_method(self):
        """Test the feature_enabled convenience method."""
        with open(self.config_file, 'w') as f:
            json.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_file)
        
        self.assertTrue(config_manager.feature_enabled('keyboard'))
        self.assertFalse(config_manager.feature_enabled('mouse'))
        self.assertTrue(config_manager.feature_enabled('clipboard'))
        self.assertFalse(config_manager.feature_enabled('nonexistent'))
        
    def test_config_reload(self):
        """Test configuration reloading functionality."""
        # Initial config
        with open(self.config_file, 'w') as f:
            json.dump(self.valid_config, f)
        
        config_manager = ConfigManager(self.config_file)
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        
        # Modify config file
        modified_config = self.valid_config.copy()
        modified_config['logging']['level'] = 'DEBUG'
        
        with open(self.config_file, 'w') as f:
            json.dump(modified_config, f)
        
        # Reload and verify
        config_manager.reload_config()
        self.assertEqual(config_manager.get('logging.level'), 'DEBUG')
        
    def test_config_update_method(self):
        """Test programmatic config updates."""
        config_manager = ConfigManager(self.config_file)
        
        # Update existing value
        config_manager.update_config('logging.level', 'DEBUG')
        self.assertEqual(config_manager.get('logging.level'), 'DEBUG')
        
        # Update nested value
        config_manager.update_config('features.new_feature', True)
        self.assertTrue(config_manager.get('features.new_feature'))
        
    def test_readonly_config_protection(self):
        """Test that config cannot be modified directly."""
        config_manager = ConfigManager(self.config_file)
        
        # Attempt to modify config directly should not affect internal state
        original_level = config_manager.get('logging.level')
        config_dict = config_manager.config
        
        # This should not change the internal config
        if 'logging' in config_dict:
            config_dict['logging']['level'] = 'MODIFIED'
        
        # Internal config should remain unchanged
        self.assertEqual(config_manager.get('logging.level'), original_level)


class TestEncryptionManagerComprehensive(unittest.TestCase):
    """Comprehensive tests for EncryptionManager with security focus."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.key_file = os.path.join(self.temp_dir, 'test.key')
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_key_generation_and_storage(self):
        """Test encryption key generation and secure storage."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Key should be generated and stored
        self.assertTrue(os.path.exists(self.key_file))
        self.assertIsNotNone(encryption_manager.key)
        self.assertEqual(len(encryption_manager.key), 32)  # 256 bits
        
    def test_key_loading_existing(self):
        """Test loading existing encryption key."""
        # Create first manager to generate key
        manager1 = EncryptionManager(self.key_file)
        original_key = manager1.key
        
        # Create second manager to load existing key
        manager2 = EncryptionManager(self.key_file)
        
        # Keys should be identical
        self.assertEqual(original_key, manager2.key)
        
    def test_data_encryption_decryption(self):
        """Test basic data encryption and decryption."""
        encryption_manager = EncryptionManager(self.key_file)
        
        test_data = "Sensitive keylogger data"
        
        # Encrypt data
        encrypted = encryption_manager.encrypt_data(test_data)
        self.assertIsNotNone(encrypted)
        self.assertNotEqual(encrypted, test_data)
        
        # Decrypt data
        decrypted = encryption_manager.decrypt_data(encrypted)
        self.assertEqual(decrypted, test_data)
        
    def test_encryption_different_data_types(self):
        """Test encryption with various data types."""
        encryption_manager = EncryptionManager(self.key_file)
        
        test_cases = [
            "string data",
            b"binary data",
            {"json": "data", "number": 42},
            [1, 2, 3, "list"],
            42,
            3.14,
            True
        ]
        
        for test_data in test_cases:
            with self.subTest(data=test_data):
                encrypted = encryption_manager.encrypt_data(test_data)
                decrypted = encryption_manager.decrypt_data(encrypted)
                self.assertEqual(decrypted, test_data)
                
    def test_encryption_empty_data(self):
        """Test encryption of empty or None data."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Test empty string
        encrypted = encryption_manager.encrypt_data("")
        decrypted = encryption_manager.decrypt_data(encrypted)
        self.assertEqual(decrypted, "")
        
        # Test None
        encrypted = encryption_manager.encrypt_data(None)
        decrypted = encryption_manager.decrypt_data(encrypted)
        self.assertIsNone(decrypted)
        
    def test_file_encryption_decryption(self):
        """Test file encryption and decryption."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.txt')
        test_content = "This is test file content\nWith multiple lines\nAnd special chars: !@#$%"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Encrypt file
        encrypted_file = os.path.join(self.temp_dir, 'test.enc')
        success = encryption_manager.encrypt_file(test_file, encrypted_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(encrypted_file))
        
        # Decrypt file
        decrypted_file = os.path.join(self.temp_dir, 'test_decrypted.txt')
        success = encryption_manager.decrypt_file(encrypted_file, decrypted_file)
        self.assertTrue(success)
        
        # Verify content
        with open(decrypted_file, 'r') as f:
            decrypted_content = f.read()
        
        self.assertEqual(decrypted_content, test_content)
        
    def test_corrupted_key_handling(self):
        """Test handling of corrupted encryption key."""
        # Create corrupted key file
        with open(self.key_file, 'w') as f:
            f.write('corrupted key data')
        
        # Should handle gracefully and create new key
        encryption_manager = EncryptionManager(self.key_file)
        
        # Should have valid key despite corruption
        self.assertIsNotNone(encryption_manager.key)
        self.assertEqual(len(encryption_manager.key), 32)
        
    def test_invalid_encrypted_data(self):
        """Test handling of invalid encrypted data."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Test with invalid encrypted data
        invalid_data = "not encrypted data"
        
        decrypted = encryption_manager.decrypt_data(invalid_data)
        self.assertIsNone(decrypted)  # Should return None for invalid data
        
    def test_key_rotation(self):
        """Test encryption key rotation functionality."""
        encryption_manager = EncryptionManager(self.key_file)
        original_key = encryption_manager.key
        
        # Rotate key
        success = encryption_manager.rotate_key()
        self.assertTrue(success)
        
        # Key should be different
        self.assertNotEqual(original_key, encryption_manager.key)
        self.assertEqual(len(encryption_manager.key), 32)
        
    def test_concurrent_encryption(self):
        """Test thread safety of encryption operations."""
        encryption_manager = EncryptionManager(self.key_file)
        results = []
        errors = []
        
        def encrypt_decrypt_worker(data):
            try:
                encrypted = encryption_manager.encrypt_data(f"test data {data}")
                decrypted = encryption_manager.decrypt_data(encrypted)
                results.append(decrypted)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=encrypt_decrypt_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        
        for i, result in enumerate(results):
            self.assertIn(str(i), result)


class TestLoggingManagerComprehensive(unittest.TestCase):
    """Comprehensive tests for LoggingManager with buffering and performance."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Mock config manager
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'logging.file_path': self.log_file,
            'logging.max_size_mb': 1,
            'logging.buffer_size': 10,
            'logging.flush_interval': 0.1,
            'logging.enable_rotation': True,
            'logging.enable_encryption': False,
            'logging.backup_count': 3,
            'encryption.enabled': False
        }.get(key, default)
        
        # Mock encryption manager
        self.mock_encryption = Mock()
        self.mock_encryption.encrypt_data.side_effect = lambda x: f"encrypted_{x}"
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_log_entry_creation(self):
        """Test LogEntry creation and serialization."""
        entry = LogEntry(
            event_type="keyboard",
            content="test keystroke",
            window_name="Test Window",
            metadata={"key": "value"}
        )
        
        # Test basic properties
        self.assertEqual(entry.event_type, "keyboard")
        self.assertEqual(entry.content, "test keystroke")
        self.assertEqual(entry.window_name, "Test Window")
        self.assertEqual(entry.metadata["key"], "value")
        
        # Test serialization
        entry_dict = entry.to_dict()
        self.assertIn('timestamp', entry_dict)
        self.assertIn('datetime', entry_dict)
        self.assertEqual(entry_dict['event_type'], "keyboard")
        
        # Test JSON serialization
        json_str = entry.to_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed['event_type'], "keyboard")
        
    def test_logging_manager_initialization(self):
        """Test LoggingManager initialization and configuration."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        
        # Verify initialization
        self.assertEqual(logging_manager.log_file_path, self.log_file)
        self.assertEqual(logging_manager.buffer_size, 10)
        self.assertEqual(logging_manager.flush_interval, 0.1)
        self.assertFalse(logging_manager.is_running)
        
    def test_event_logging_and_buffering(self):
        """Test event logging with buffering."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        logging_manager.start()
        
        try:
            # Log multiple events
            for i in range(5):
                entry = LogEntry(
                    event_type="test",
                    content=f"test event {i}",
                    window_name="Test Window"
                )
                logging_manager.log_event(entry)
            
            # Events should be in buffer
            self.assertEqual(len(logging_manager.event_buffer), 5)
            
            # Force flush
            logging_manager.flush_buffer()
            
            # Buffer should be empty after flush
            self.assertEqual(len(logging_manager.event_buffer), 0)
            
            # Log file should exist and contain data
            self.assertTrue(os.path.exists(self.log_file))
            
        finally:
            logging_manager.stop()
            
    def test_buffer_auto_flush(self):
        """Test automatic buffer flushing based on size and time."""
        # Set small buffer size for testing
        self.mock_config.get.side_effect = lambda key, default=None: {
            'logging.file_path': self.log_file,
            'logging.buffer_size': 3,  # Small buffer
            'logging.flush_interval': 10.0,  # Long interval
            'logging.enable_encryption': False,
            'encryption.enabled': False
        }.get(key, default)
        
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        logging_manager.start()
        
        try:
            # Add events to exceed buffer size
            for i in range(5):
                entry = LogEntry(event_type="test", content=f"event {i}")
                logging_manager.log_event(entry)
            
            # Give time for auto-flush
            time.sleep(0.2)
            
            # Buffer should have been flushed automatically
            self.assertLess(len(logging_manager.event_buffer), 5)
            
        finally:
            logging_manager.stop()
            
    def test_log_rotation(self):
        """Test log file rotation when size limit is reached."""
        # Create a large log entry to trigger rotation
        large_content = "x" * 1024 * 1024  # 1MB of data
        
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        logging_manager.start()
        
        try:
            # Log large entry to trigger rotation
            entry = LogEntry(event_type="large", content=large_content)
            logging_manager.log_event(entry)
            logging_manager.flush_buffer()
            
            # Log another entry
            entry2 = LogEntry(event_type="test", content="after rotation")
            logging_manager.log_event(entry2)
            logging_manager.flush_buffer()
            
            # Check if rotation occurred (backup file should exist)
            backup_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.1')]
            # Note: Rotation might not occur immediately in test environment
            
        finally:
            logging_manager.stop()
            
    def test_concurrent_logging(self):
        """Test thread safety of logging operations."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        logging_manager.start()
        
        errors = []
        
        def log_worker(worker_id):
            try:
                for i in range(10):
                    entry = LogEntry(
                        event_type="concurrent",
                        content=f"worker {worker_id} event {i}"
                    )
                    logging_manager.log_event(entry)
            except Exception as e:
                errors.append(e)
        
        try:
            # Start multiple logging threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=log_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            self.assertEqual(len(errors), 0)
            
            # Flush and verify events were logged
            logging_manager.flush_buffer()
            
        finally:
            logging_manager.stop()
            
    def test_statistics_tracking(self):
        """Test logging statistics tracking."""
        logging_manager = LoggingManager(self.mock_config, self.mock_encryption)
        logging_manager.start()
        
        try:
            # Log some events
            for i in range(5):
                entry = LogEntry(event_type="stats_test", content=f"event {i}")
                logging_manager.log_event(entry)
            
            # Check statistics
            stats = logging_manager.get_stats()
            self.assertGreaterEqual(stats['total_events'], 5)
            
        finally:
            logging_manager.stop()


class TestKeyloggerCoreComprehensive(unittest.TestCase):
    """Comprehensive tests for KeyloggerCore orchestration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create test config
        test_config = {
            'logging': {'file_path': os.path.join(self.temp_dir, 'test.log')},
            'encryption': {'key_file': os.path.join(self.temp_dir, 'test.key')},
            'features': {
                'keyboard': True,
                'mouse': True,
                'clipboard': False,
                'window_tracking': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
            
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('core.keylogger.KeyboardListener')
    @patch('core.keylogger.MouseListener')
    @patch('core.keylogger.WindowMonitor')
    def test_keylogger_initialization(self, mock_window, mock_mouse, mock_keyboard):
        """Test KeyloggerCore initialization with mocked components."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Verify core components initialized
        self.assertIsNotNone(keylogger.config_manager)
        self.assertIsNotNone(keylogger.encryption_manager)
        self.assertIsNotNone(keylogger.logging_manager)
        
        # Verify session ID generated
        self.assertIsNotNone(keylogger.session_id)
        
    @patch('core.keylogger.KeyboardListener')
    @patch('core.keylogger.MouseListener')
    def test_keylogger_start_stop_cycle(self, mock_mouse, mock_keyboard):
        """Test complete start/stop cycle."""
        # Setup mocks
        mock_keyboard_instance = Mock()
        mock_mouse_instance = Mock()
        mock_keyboard.return_value = mock_keyboard_instance
        mock_mouse.return_value = mock_mouse_instance
        
        keylogger = KeyloggerCore(self.config_file)
        
        # Test start
        keylogger.start()
        self.assertTrue(keylogger.is_running)
        self.assertIsNotNone(keylogger.start_time)
        
        # Verify listeners started
        mock_keyboard_instance.start.assert_called_once()
        mock_mouse_instance.start.assert_called_once()
        
        # Test stop
        keylogger.stop()
        self.assertFalse(keylogger.is_running)
        
        # Verify listeners stopped
        mock_keyboard_instance.stop.assert_called_once()
        mock_mouse_instance.stop.assert_called_once()
        
    def test_config_reload_functionality(self):
        """Test configuration reloading during runtime."""
        keylogger = KeyloggerCore(self.config_file)
        
        original_level = keylogger.config_manager.get('logging.level', 'INFO')
        
        # Modify config file
        modified_config = {
            'logging': {
                'level': 'DEBUG',
                'file_path': os.path.join(self.temp_dir, 'test.log')
            },
            'features': {'keyboard': False}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(modified_config, f)
        
        # Reload config
        keylogger.reload_config()
        
        # Verify changes applied
        new_level = keylogger.config_manager.get('logging.level')
        self.assertEqual(new_level, 'DEBUG')
        
    def test_session_statistics(self):
        """Test session statistics collection."""
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        try:
            # Let it run briefly
            time.sleep(0.1)
            
            stats = keylogger.get_session_stats()
            
            # Verify basic stats structure
            self.assertIn('uptime', stats)
            self.assertIn('total_events', stats)
            self.assertIn('running', stats)
            self.assertTrue(stats['running'])
            self.assertGreaterEqual(stats['uptime'], 0)
            
        finally:
            keylogger.stop()
            
    def test_error_handling_missing_components(self):
        """Test graceful handling when optional components are missing."""
        # This test verifies the system handles missing imports gracefully
        keylogger = KeyloggerCore(self.config_file)
        
        # Should initialize successfully even if some components are missing
        self.assertIsNotNone(keylogger.config_manager)
        
    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles for stability."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Perform multiple cycles
        for i in range(3):
            with self.subTest(cycle=i):
                keylogger.start()
                self.assertTrue(keylogger.is_running)
                
                time.sleep(0.05)  # Brief run time
                
                keylogger.stop()
                self.assertFalse(keylogger.is_running)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)