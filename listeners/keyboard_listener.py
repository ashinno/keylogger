"""Keyboard listener with security and performance optimizations."""

import time
import threading
import logging
from typing import Optional, Dict, Any, Set
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import re

logger = logging.getLogger(__name__)


class KeyboardListener:
    """Enhanced keyboard listener with security and performance features."""
    
    def __init__(self, keylogger_core):
        self.keylogger = keylogger_core
        self.config = keylogger_core.config
        self.listener: Optional[keyboard.Listener] = None
        self.is_running = False
        self.current_text = ""
        self.last_key_time = 0
        self.key_sequence = []
        self.modifier_keys: Set[Key] = set()
        
        # Security patterns
        self.password_patterns = [
            r'password',
            r'passwd',
            r'pwd',
            r'pin',
            r'secret',
            r'token',
            r'key',
            r'auth'
        ]
        
        # Performance settings
        self.text_flush_interval = self.config.get('performance.text_flush_interval', 5.0)
        self.max_sequence_length = self.config.get('performance.max_key_sequence', 100)
        self.typing_timeout = self.config.get('performance.typing_timeout', 2.0)
        
        # Statistics
        self.stats = {
            'keys_pressed': 0,
            'keys_released': 0,
            'shortcuts_detected': 0,
            'text_entries': 0,
            'sensitive_data_filtered': 0
        }
        
        # Setup flush timer
        self._setup_flush_timer()
    
    def _setup_flush_timer(self) -> None:
        """Setup periodic text flushing."""
        def flush_timer():
            while self.is_running:
                time.sleep(self.text_flush_interval)
                if self.current_text.strip():
                    self._flush_current_text()
        
        self.flush_thread = threading.Thread(target=flush_timer, daemon=True)
    
    def start(self) -> None:
        """Start the keyboard listener."""
        if self.is_running:
            logger.warning("Keyboard listener is already running")
            return
        
        try:
            self.is_running = True
            self.listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.listener.start()
            self.flush_thread.start()
            logger.info("Keyboard listener started")
            
        except Exception as e:
            logger.error(f"Failed to start keyboard listener: {e}")
            self.is_running = False
            raise
    
    def stop(self) -> None:
        """Stop the keyboard listener."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            if self.listener:
                self.listener.stop()
            
            # Flush any remaining text
            if self.current_text.strip():
                self._flush_current_text()
            
            logger.info("Keyboard listener stopped")
            logger.info(f"Keyboard stats: {self.stats}")
            
        except Exception as e:
            logger.error(f"Error stopping keyboard listener: {e}")
    
    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        try:
            if not self.is_running:
                return
            
            current_time = time.time()
            self.last_key_time = current_time
            self.stats['keys_pressed'] += 1
            
            # Handle modifier keys
            if self._is_modifier_key(key):
                self.modifier_keys.add(key)
                return
            
            # Check for shortcuts
            if self.modifier_keys and self._is_shortcut(key):
                self._handle_shortcut(key)
                return
            
            # Handle special keys
            if isinstance(key, Key):
                self._handle_special_key(key)
            elif isinstance(key, KeyCode):
                self._handle_character_key(key)
            
            # Manage sequence length
            self._manage_key_sequence(key)
            
        except Exception as e:
            logger.error(f"Error in key press handler: {e}")
    
    def _on_key_release(self, key) -> None:
        """Handle key release events."""
        try:
            if not self.is_running:
                return
            
            self.stats['keys_released'] += 1
            
            # Remove modifier keys
            if self._is_modifier_key(key):
                self.modifier_keys.discard(key)
            
        except Exception as e:
            logger.error(f"Error in key release handler: {e}")
    
    def _is_modifier_key(self, key) -> bool:
        """Check if key is a modifier key."""
        modifier_keys = {
            Key.ctrl, Key.ctrl_l, Key.ctrl_r,
            Key.alt, Key.alt_l, Key.alt_r, Key.alt_gr,
            Key.shift, Key.shift_l, Key.shift_r,
            Key.cmd, Key.cmd_l, Key.cmd_r
        }
        return key in modifier_keys
    
    def _is_shortcut(self, key) -> bool:
        """Check if current key combination is a shortcut."""
        # Common shortcut patterns
        if Key.ctrl in self.modifier_keys or Key.cmd in self.modifier_keys:
            return True
        if Key.alt in self.modifier_keys and len(self.modifier_keys) > 1:
            return True
        return False
    
    def _handle_shortcut(self, key) -> None:
        """Handle keyboard shortcuts."""
        try:
            modifiers = "+".join([self._key_to_string(mod) for mod in sorted(self.modifier_keys)])
            key_str = self._key_to_string(key)
            shortcut = f"{modifiers}+{key_str}"
            
            self.stats['shortcuts_detected'] += 1
            
            # Flush current text before logging shortcut
            if self.current_text.strip():
                self._flush_current_text()
            
            window_name = self.keylogger.session_stats.get('active_window', 'Unknown')
            self.keylogger.log_event(
                "Keyboard Shortcut",
                shortcut,
                window_name,
                metadata={'timestamp': time.time()}
            )
            
        except Exception as e:
            logger.error(f"Error handling shortcut: {e}")
    
    def _handle_special_key(self, key: Key) -> None:
        """Handle special keys like Enter, Backspace, etc."""
        try:
            if key == Key.enter:
                self.current_text += "\n"
                self._flush_current_text()
            elif key == Key.tab:
                self.current_text += "\t"
            elif key == Key.space:
                self.current_text += " "
            elif key == Key.backspace:
                if self.current_text:
                    self.current_text = self.current_text[:-1]
            elif key in [Key.delete, Key.home, Key.end, Key.page_up, Key.page_down]:
                # These keys might indicate text manipulation
                if self.current_text.strip():
                    self._flush_current_text()
            
        except Exception as e:
            logger.error(f"Error handling special key: {e}")
    
    def _handle_character_key(self, key: KeyCode) -> None:
        """Handle character keys."""
        try:
            if hasattr(key, 'char') and key.char:
                self.current_text += key.char
                
                # Check for typing timeout
                if (time.time() - self.last_key_time > self.typing_timeout and 
                    self.current_text.strip()):
                    self._flush_current_text()
            
        except Exception as e:
            logger.error(f"Error handling character key: {e}")
    
    def _flush_current_text(self) -> None:
        """Flush current text buffer to log."""
        try:
            if not self.current_text.strip():
                return
            
            text = self.current_text.strip()
            self.current_text = ""
            
            # Apply privacy filters
            if self._is_sensitive_data(text):
                if self.config.get('privacy.sanitize_passwords', True):
                    text = self._sanitize_sensitive_data(text)
                    self.stats['sensitive_data_filtered'] += 1
            
            # Check text length limits
            max_length = self.config.get('privacy.max_text_length', 1000)
            if len(text) > max_length:
                text = text[:max_length] + "...[truncated]"
            
            window_name = self.keylogger.session_stats.get('active_window', 'Unknown')
            
            # Skip if application is excluded
            if self.config.is_excluded_application(window_name):
                return
            
            self.keylogger.log_event(
                "Text Input",
                text,
                window_name,
                metadata={
                    'length': len(text),
                    'timestamp': time.time(),
                    'sanitized': self.stats['sensitive_data_filtered'] > 0
                }
            )
            
            self.stats['text_entries'] += 1
            
        except Exception as e:
            logger.error(f"Error flushing text: {e}")
    
    def _is_sensitive_data(self, text: str) -> bool:
        """Check if text contains sensitive data."""
        try:
            text_lower = text.lower()
            
            # Check for password patterns
            for pattern in self.password_patterns:
                if re.search(pattern, text_lower):
                    return True
            
            # Check for sensitive keywords from config
            sensitive_keywords = self.config.get('privacy.sensitive_keywords', [])
            for keyword in sensitive_keywords:
                if keyword.lower() in text_lower:
                    return True
            
            # Check for potential credit card numbers
            if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
                return True
            
            # Check for potential SSN
            if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking sensitive data: {e}")
            return False
    
    def _sanitize_sensitive_data(self, text: str) -> str:
        """Sanitize sensitive data based on configuration."""
        try:
            sanitization_method = self.config.get('privacy.sanitization_method', 'hash')
            
            if sanitization_method == 'hash':
                # Hash the sensitive data
                if self.keylogger.encryption:
                    return f"[SENSITIVE_DATA_HASH:{self.keylogger.encryption.hash_data(text)[:16]}]"
                else:
                    return "[SENSITIVE_DATA_DETECTED]"
            elif sanitization_method == 'mask':
                # Mask with asterisks
                return '*' * min(len(text), 20)
            elif sanitization_method == 'remove':
                # Remove completely
                return "[SENSITIVE_DATA_REMOVED]"
            else:
                return "[SENSITIVE_DATA_DETECTED]"
            
        except Exception as e:
            logger.error(f"Error sanitizing sensitive data: {e}")
            return "[SENSITIVE_DATA_ERROR]"
    
    def _manage_key_sequence(self, key) -> None:
        """Manage key sequence for analysis."""
        try:
            self.key_sequence.append({
                'key': self._key_to_string(key),
                'timestamp': time.time(),
                'modifiers': list(self.modifier_keys)
            })
            
            # Limit sequence length
            if len(self.key_sequence) > self.max_sequence_length:
                self.key_sequence = self.key_sequence[-self.max_sequence_length:]
            
        except Exception as e:
            logger.error(f"Error managing key sequence: {e}")
    
    def _key_to_string(self, key) -> str:
        """Convert key to string representation."""
        try:
            if isinstance(key, Key):
                return key.name
            elif isinstance(key, KeyCode):
                if hasattr(key, 'char') and key.char:
                    return key.char
                else:
                    return f"KeyCode({key.vk})"
            else:
                return str(key)
        except Exception:
            return "Unknown"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get listener statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'current_text_length': len(self.current_text),
            'sequence_length': len(self.key_sequence),
            'active_modifiers': len(self.modifier_keys)
        }