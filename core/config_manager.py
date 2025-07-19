"""Configuration management with validation and security features."""

import json
import os
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration with validation and security."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config: Dict[str, Any] = {}
        self._default_config = self._get_default_config()
        self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            "logging": {
                "level": "INFO",
                "file_path": "keylog.txt",
                "max_file_size_mb": 100,
                "backup_count": 5,
                "buffer_size": 10,
                "flush_interval_seconds": 10
            },
            "encryption": {
                "enabled": True,
                "key_file": "encryption.key",
                "algorithm": "fernet"
            },
            "features": {
                "key_logging": True,
                "mouse_logging": True,
                "clipboard_logging": True,
                "screenshots": True,
                "usb_monitoring": True,
                "network_logging": False,
                "window_tracking": True
            },
            "privacy": {
                "sanitize_passwords": True,
                "hash_sensitive_data": True,
                "max_clipboard_length": 1000,
                "excluded_applications": [],
                "sensitive_keywords": ["password", "ssn", "credit card"]
            },
            "performance": {
                "screenshot_interval_seconds": 30,
                "window_check_interval_seconds": 5,
                "clipboard_check_interval_seconds": 3,
                "max_memory_usage_mb": 500,
                "thread_pool_size": 4
            }
        }
    
    def load_config(self) -> None:
        """Load configuration from file with fallback to defaults."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._config = self._merge_configs(self._default_config, file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self._config = self._default_config.copy()
                self.save_config()
                logger.info("Created default configuration file")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = self._default_config.copy()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.get(f"features.{feature}", False)
    
    def is_application_excluded(self, app_name: str) -> bool:
        """Check if an application should be excluded from logging."""
        excluded_apps = self.get("privacy.excluded_applications", [])
        return any(excluded_app.lower() in app_name.lower() for excluded_app in excluded_apps)
    
    def contains_sensitive_data(self, text: str) -> bool:
        """Check if text contains sensitive keywords."""
        if not self.get("privacy.sanitize_passwords", True):
            return False
        
        sensitive_keywords = self.get("privacy.sensitive_keywords", [])
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in sensitive_keywords)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data if privacy settings require it."""
        if self.get("privacy.hash_sensitive_data", True):
            return hashlib.sha256(data.encode()).hexdigest()[:16]
        return data
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate required sections
        required_sections = ['logging', 'encryption', 'features', 'privacy', 'performance']
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        # Validate numeric values
        numeric_validations = [
            ('logging.max_file_size_mb', 1, 1000),
            ('logging.buffer_size', 1, 1000),
            ('performance.screenshot_interval_seconds', 1, 3600),
            ('performance.thread_pool_size', 1, 20)
        ]
        
        for key, min_val, max_val in numeric_validations:
            value = self.get(key)
            if value is not None and not (min_val <= value <= max_val):
                errors.append(f"{key} must be between {min_val} and {max_val}")
        
        return errors
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get read-only copy of configuration."""
        return self._config.copy()