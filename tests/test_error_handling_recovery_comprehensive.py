"""Comprehensive error handling and recovery tests for the Enhanced Keylogger system.
This module tests various failure scenarios and recovery mechanisms including:
- Configuration errors and validation failures
- Encryption key corruption and recovery
- File system errors and permission issues
- Component initialization failures
- Network connectivity issues
- Resource exhaustion scenarios
- Graceful degradation and recovery
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.keylogger import KeyloggerCore
from core.config_manager import ConfigManager
from core.encryption_manager import EncryptionManager
from core.logging_manager import LoggingManager


class TestConfigurationErrorHandling(unittest.TestCase):
    """Test configuration-related error handling and recovery."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_missing_config_file_recovery(self):
        """Test recovery when config file is missing."""
        non_existent_config = os.path.join(self.test_dir, 'missing.json')
        
        # Should create default config and continue
        config_manager = ConfigManager(non_existent_config)
        
        # Should have default values
        self.assertIsNotNone(config_manager.get('logging.level'))
        self.assertIsNotNone(config_manager.get('features.keyboard'))
        
    def test_corrupted_config_file_recovery(self):
        """Test recovery from corrupted JSON config file."""
        # Create corrupted JSON file
        with open(self.config_file, 'w') as f:
            f.write('{"logging": {"level": "INFO", invalid json}')
        
        # Should handle gracefully and use defaults
        config_manager = ConfigManager(self.config_file)
        
        # Should fall back to defaults
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        
    def test_invalid_config_values_handling(self):
        """Test handling of invalid configuration values."""
        invalid_config = {
            'logging': {
                'level': 'INVALID_LEVEL',
                'buffer_size': -1,  # Invalid negative value
                'flush_interval': 'not_a_number'
            },
            'encryption': {
                'enabled': 'not_boolean'
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.config_file)
        
        # Should validate and use safe defaults
        self.assertFalse(config_manager.validate_config())
        
        # Should provide fallback values
        level = config_manager.get('logging.level')
        self.assertIn(level, ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        
    def test_readonly_config_file_handling(self):
        """Test handling when config file is read-only."""
        # Create valid config
        config = {'logging': {'level': 'INFO'}}
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        # Make file read-only
        os.chmod(self.config_file, 0o444)
        
        try:
            config_manager = ConfigManager(self.config_file)
            
            # Should load successfully
            self.assertEqual(config_manager.get('logging.level'), 'INFO')
            
            # Update should handle read-only gracefully
            result = config_manager.update_config({'logging.level': 'DEBUG'})
            # Should either succeed or fail gracefully without crashing
            self.assertIsInstance(result, bool)
            
        finally:
            # Restore permissions for cleanup
            os.chmod(self.config_file, 0o644)
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        edge_cases = [
            {},  # Empty config
            {'logging': None},  # Null section
            {'logging': {'level': None}},  # Null value
            {'logging': {'buffer_size': 0}},  # Zero value
            {'logging': {'buffer_size': float('inf')}},  # Infinite value
        ]
        
        for i, config in enumerate(edge_cases):
            with self.subTest(case=i):
                test_file = os.path.join(self.test_dir, f'edge_case_{i}.json')
                with open(test_file, 'w') as f:
                    json.dump(config, f)
                
                # Should handle gracefully
                config_manager = ConfigManager(test_file)
                self.assertIsNotNone(config_manager)


class TestEncryptionErrorHandling(unittest.TestCase):
    """Test encryption-related error handling and recovery."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.key_file = os.path.join(self.test_dir, 'test.key')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_missing_key_file_recovery(self):
        """Test recovery when encryption key file is missing."""
        non_existent_key = os.path.join(self.test_dir, 'missing.key')
        
        # Should generate new key and continue
        encryption_manager = EncryptionManager(non_existent_key)
        
        # Should have valid key
        self.assertIsNotNone(encryption_manager.key)
        self.assertEqual(len(encryption_manager.key), 32)  # AES-256 key
        
        # Key file should be created
        self.assertTrue(os.path.exists(non_existent_key))
    
    def test_corrupted_key_file_recovery(self):
        """Test recovery from corrupted key file."""
        # Create corrupted key file
        with open(self.key_file, 'w') as f:
            f.write('corrupted key data that is not valid')
        
        # Should handle gracefully and generate new key
        encryption_manager = EncryptionManager(self.key_file)
        
        # Should have valid key despite corruption
        self.assertIsNotNone(encryption_manager.key)
        self.assertEqual(len(encryption_manager.key), 32)
    
    def test_permission_denied_key_file(self):
        """Test handling when key file directory has no write permissions."""
        restricted_dir = os.path.join(self.test_dir, 'restricted')
        os.makedirs(restricted_dir)
        
        # Make directory read-only
        os.chmod(restricted_dir, 0o444)
        
        try:
            restricted_key = os.path.join(restricted_dir, 'test.key')
            
            # Should handle permission error gracefully
            encryption_manager = EncryptionManager(restricted_key)
            
            # Should still have a working key (in memory)
            self.assertIsNotNone(encryption_manager.key)
            
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_dir, 0o755)
    
    def test_encryption_decryption_error_handling(self):
        """Test error handling in encryption/decryption operations."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Test encryption with invalid data types
        invalid_inputs = [None, 123, [], {}, object()]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=type(invalid_input).__name__):
                result = encryption_manager.encrypt_data(invalid_input)
                # Should handle gracefully (return None or raise specific exception)
                self.assertTrue(result is None or isinstance(result, (bytes, str)))
        
        # Test decryption with invalid data
        invalid_encrypted_data = [
            b'',  # Empty data
            b'too_short',  # Too short for valid encrypted data
            b'\x00' * 100,  # Invalid encrypted data
            'not_bytes',  # Wrong type
        ]
        
        for invalid_data in invalid_encrypted_data:
            with self.subTest(data=str(invalid_data)[:20]):
                result = encryption_manager.decrypt_data(invalid_data)
                # Should return None for invalid data
                self.assertIsNone(result)
    
    def test_key_rotation_error_handling(self):
        """Test error handling during key rotation."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Encrypt some data with original key
        original_data = "test data for rotation"
        encrypted_data = encryption_manager.encrypt_data(original_data)
        
        # Simulate key rotation failure by making key file read-only
        os.chmod(self.key_file, 0o444)
        
        try:
            # Attempt key rotation
            if hasattr(encryption_manager, 'rotate_key'):
                result = encryption_manager.rotate_key()
                # Should handle gracefully
                self.assertIsInstance(result, bool)
            
            # Should still be able to decrypt with original key
            decrypted_data = encryption_manager.decrypt_data(encrypted_data)
            self.assertEqual(decrypted_data, original_data)
            
        finally:
            os.chmod(self.key_file, 0o644)


class TestLoggingErrorHandling(unittest.TestCase):
    """Test logging-related error handling and recovery."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        self.key_file = os.path.join(self.test_dir, 'test.key')
        self.log_file = os.path.join(self.test_dir, 'test.log')
        
        # Create test config
        self.config = {
            'logging': {
                'log_file': self.log_file,
                'buffer_size': 10,
                'flush_interval': 1.0
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_log_directory_creation_failure(self):
        """Test handling when log directory cannot be created."""
        # Try to create log in a restricted location
        if os.name == 'nt':  # Windows
            restricted_log = 'C:\\Windows\\System32\\restricted.log'
        else:  # Unix-like
            restricted_log = '/root/restricted.log'
        
        config = {
            'logging': {
                'log_file': restricted_log,
                'buffer_size': 10
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        config_manager = ConfigManager(self.config_file)
        encryption_manager = EncryptionManager(self.key_file)
        
        # Should handle gracefully
        logging_manager = LoggingManager(config_manager, encryption_manager)
        
        # Should still be functional (may use fallback location)
        self.assertIsNotNone(logging_manager)
    
    def test_disk_space_exhaustion_handling(self):
        """Test handling when disk space is exhausted."""
        config_manager = ConfigManager(self.config_file)
        encryption_manager = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config_manager, encryption_manager)
        
        # Mock disk space exhaustion
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            # Should handle gracefully
            result = logging_manager.log_event('test', 'data')
            # Should return False or handle gracefully
            self.assertIsInstance(result, bool)
    
    def test_log_file_corruption_recovery(self):
        """Test recovery from log file corruption."""
        config_manager = ConfigManager(self.config_file)
        encryption_manager = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config_manager, encryption_manager)
        
        # Create corrupted log file
        with open(self.log_file, 'wb') as f:
            f.write(b'\x00\x01\x02corrupted data\xff\xfe')
        
        # Should handle corruption gracefully
        logging_manager.start()
        result = logging_manager.log_event('test', 'recovery_test')
        
        # Should continue logging despite corruption
        self.assertTrue(result)
        
        logging_manager.stop()
    
    def test_concurrent_logging_error_handling(self):
        """Test error handling in concurrent logging scenarios."""
        config_manager = ConfigManager(self.config_file)
        encryption_manager = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config_manager, encryption_manager)
        
        logging_manager.start()
        
        errors = []
        
        def log_with_errors(thread_id):
            """Log events and capture any errors."""
            try:
                for i in range(10):
                    # Simulate various error conditions
                    if i % 3 == 0:
                        # Invalid data type
                        result = logging_manager.log_event('test', None)
                    elif i % 3 == 1:
                        # Very large data
                        result = logging_manager.log_event('test', 'x' * 1000000)
                    else:
                        # Normal data
                        result = logging_manager.log_event('test', f'thread_{thread_id}_event_{i}')
                    
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_with_errors, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        logging_manager.stop()
        
        # Should handle concurrent access without major errors
        self.assertLess(len(errors), 5)  # Allow some errors but not complete failure


class TestComponentInitializationErrors(unittest.TestCase):
    """Test error handling during component initialization."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_missing_dependencies_handling(self):
        """Test handling when optional dependencies are missing."""
        # Create minimal config
        config = {'logging': {'log_file': 'test.log'}}
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        # Mock missing dependencies
        with patch.dict('sys.modules', {
            'listeners.keyboard_listener': None,
            'listeners.mouse_listener': None,
            'utils.window_monitor': None
        }):
            # Should initialize core components despite missing optional ones
            keylogger = KeyloggerCore(self.config_file)
            
            # Core should be initialized
            self.assertIsNotNone(keylogger.config_manager)
            self.assertIsNotNone(keylogger.session_id)
    
    def test_component_initialization_partial_failure(self):
        """Test handling when some components fail to initialize."""
        config = {
            'logging': {
                'log_file': '/invalid/path/test.log',  # Invalid path
                'buffer_size': 'invalid'  # Invalid type
            },
            'encryption': {
                'key_file': '/invalid/path/test.key'  # Invalid path
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        # Should handle initialization errors gracefully
        keylogger = KeyloggerCore(self.config_file)
        
        # Core should still be initialized
        self.assertIsNotNone(keylogger.config_manager)
        self.assertIsNotNone(keylogger.session_id)
    
    def test_listener_initialization_failure_recovery(self):
        """Test recovery when listeners fail to initialize."""
        config = {'features': {'keyboard': True, 'mouse': True}}
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        # Mock listener initialization failure
        with patch('listeners.keyboard_listener.KeyboardListener', side_effect=Exception("Listener init failed")):
            keylogger = KeyloggerCore(self.config_file)
            
            # Should continue without the failed listener
            self.assertIsNotNone(keylogger)
            self.assertIsNone(keylogger.keyboard_listener)


class TestSystemResourceErrors(unittest.TestCase):
    """Test handling of system resource-related errors."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        config = {
            'logging': {
                'log_file': os.path.join(self.test_dir, 'test.log'),
                'buffer_size': 10
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion scenarios."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Mock memory error
        with patch.object(keylogger.logging_manager, 'log_event', side_effect=MemoryError("Out of memory")):
            # Should handle gracefully
            result = keylogger.log_event('test', 'data')
            self.assertIsInstance(result, bool)
    
    def test_thread_creation_failure_handling(self):
        """Test handling when thread creation fails."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Mock thread creation failure
        with patch('threading.Thread', side_effect=RuntimeError("Cannot create thread")):
            # Should handle gracefully
            result = keylogger.start()
            # Should either succeed or fail gracefully
            self.assertIsInstance(result, bool)
    
    def test_file_handle_exhaustion(self):
        """Test handling when file handles are exhausted."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Mock file handle exhaustion
        with patch('builtins.open', side_effect=OSError("Too many open files")):
            # Should handle gracefully
            result = keylogger.log_event('test', 'data')
            self.assertIsInstance(result, bool)


class TestGracefulDegradationAndRecovery(unittest.TestCase):
    """Test graceful degradation and recovery mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        config = {
            'logging': {
                'log_file': os.path.join(self.test_dir, 'test.log'),
                'buffer_size': 5
            },
            'features': {
                'keyboard': True,
                'mouse': True,
                'clipboard': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_partial_system_failure_degradation(self):
        """Test graceful degradation when parts of the system fail."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Start system
        keylogger.start()
        
        # Simulate partial failure (logging manager fails)
        keylogger.logging_manager = None
        
        # System should continue operating in degraded mode
        self.assertTrue(keylogger.is_running())
        
        # Should handle log events gracefully even without logging manager
        result = keylogger.log_event('test', 'data')
        self.assertIsInstance(result, bool)
        
        keylogger.stop()
    
    def test_recovery_after_temporary_failure(self):
        """Test recovery after temporary system failures."""
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Simulate temporary failure
        original_log_event = keylogger.logging_manager.log_event
        failure_count = 0
        
        def failing_log_event(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 attempts
                raise Exception("Temporary failure")
            return original_log_event(*args, **kwargs)
        
        keylogger.logging_manager.log_event = failing_log_event
        
        # Should eventually recover
        results = []
        for i in range(5):
            try:
                result = keylogger.log_event('test', f'data_{i}')
                results.append(result)
            except Exception:
                results.append(False)
            time.sleep(0.1)
        
        # Should have some successful attempts after recovery
        self.assertTrue(any(results[-2:]))  # Last 2 attempts should succeed
        
        keylogger.stop()
    
    def test_shutdown_error_handling(self):
        """Test error handling during system shutdown."""
        keylogger = KeyloggerCore(self.config_file)
        keylogger.start()
        
        # Mock shutdown errors
        with patch.object(keylogger.logging_manager, 'stop', side_effect=Exception("Shutdown error")):
            # Should handle shutdown errors gracefully
            result = keylogger.stop()
            # Should complete shutdown despite errors
            self.assertIsInstance(result, bool)
            self.assertFalse(keylogger.is_running())
    
    def test_signal_handling_errors(self):
        """Test error handling in signal handlers."""
        keylogger = KeyloggerCore(self.config_file)
        
        # Mock signal handler registration failure
        with patch('signal.signal', side_effect=OSError("Signal handling not supported")):
            # Should continue without signal handling
            result = keylogger.start()
            self.assertIsInstance(result, bool)
            
            if keylogger.is_running():
                keylogger.stop()


class TestNetworkAndExternalServiceErrors(unittest.TestCase):
    """Test handling of network and external service errors."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_remote_logging_failure_handling(self):
        """Test handling when remote logging fails."""
        config = {
            'logging': {
                'log_file': os.path.join(self.test_dir, 'test.log'),
                'remote_logging': {
                    'enabled': True,
                    'url': 'http://invalid-server.example.com/logs'
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        keylogger = KeyloggerCore(self.config_file)
        
        # Should handle remote logging failure gracefully
        # and continue with local logging
        result = keylogger.log_event('test', 'data')
        self.assertIsInstance(result, bool)
    
    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        # Mock network operations with timeout
        with patch('requests.post', side_effect=Exception("Connection timeout")):
            keylogger = KeyloggerCore(self.config_file)
            
            # Should handle network timeouts gracefully
            result = keylogger.log_event('test', 'data')
            self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main(verbosity=2)