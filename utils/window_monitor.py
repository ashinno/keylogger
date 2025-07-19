"""Window monitoring utility with performance optimizations."""

import time
import threading
import logging
from typing import Optional, Dict, Any, List
import psutil

try:
    import win32gui
    import win32process
    import win32api
    import win32con
except ImportError:
    win32gui = win32process = win32api = win32con = None
    logging.warning("Windows-specific modules not available")

logger = logging.getLogger(__name__)


class WindowMonitor:
    """Monitor active windows and application usage."""
    
    def __init__(self, keylogger_core):
        self.keylogger = keylogger_core
        self.config = keylogger_core.config
        self.is_running = False
        
        # Window tracking
        self.current_window = None
        self.window_start_time = None
        self.window_history: List[Dict[str, Any]] = []
        self.application_usage: Dict[str, float] = {}
        
        # Performance settings
        self.check_interval = self.config.get('performance.window_check_interval', 1.0)
        self.max_history_size = self.config.get('performance.window_history_size', 1000)
        
        # Privacy settings
        self.log_window_titles = self.config.get('privacy.log_window_titles', True)
        self.sanitize_titles = self.config.get('privacy.sanitize_window_titles', True)
        
        # Statistics
        self.stats = {
            'window_changes': 0,
            'applications_tracked': 0,
            'total_monitoring_time': 0,
            'errors': 0
        }
        
        # Title sanitization patterns
        self.sensitive_patterns = [
            r'password', r'login', r'signin', r'auth',
            r'private', r'confidential', r'secret',
            r'\b\d{4}\b',  # Potential years or codes
            r'@\w+\.\w+',  # Email domains
        ]
    
    def run(self) -> None:
        """Main monitoring loop."""
        if not self._check_platform_support():
            logger.error("Window monitoring not supported on this platform")
            return
        
        self.is_running = True
        start_time = time.time()
        
        logger.info("Window monitor started")
        
        try:
            while self.is_running and not self.keylogger.stop_event.is_set():
                try:
                    self._check_active_window()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Error in window monitoring loop: {e}")
                    self.stats['errors'] += 1
                    time.sleep(self.check_interval * 2)
            
            # Final window time logging
            self._log_final_window_time()
            
        except Exception as e:
            logger.error(f"Critical error in window monitor: {e}")
        finally:
            self.stats['total_monitoring_time'] = time.time() - start_time
            logger.info(f"Window monitor stopped. Stats: {self.stats}")
    
    def _check_platform_support(self) -> bool:
        """Check if platform supports window monitoring."""
        if win32gui is None:
            return False
        return True
    
    def _check_active_window(self) -> None:
        """Check and handle active window changes."""
        try:
            window_info = self._get_active_window_info()
            
            if window_info and window_info != self.current_window:
                self._handle_window_change(window_info)
            
        except Exception as e:
            logger.error(f"Error checking active window: {e}")
            self.stats['errors'] += 1
    
    def _get_active_window_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active window."""
        try:
            if not win32gui:
                return None
            
            # Get active window handle
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            
            # Get window title
            window_title = win32gui.GetWindowText(hwnd)
            if not window_title:
                window_title = "Unknown Window"
            
            # Get process information
            try:
                _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                process = psutil.Process(process_id)
                process_name = process.name()
                executable_path = process.exe()
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                process_name = "Unknown Process"
                executable_path = "Unknown"
            
            # Sanitize window title if needed
            if self.sanitize_titles:
                window_title = self._sanitize_window_title(window_title)
            
            window_info = {
                'hwnd': hwnd,
                'title': window_title,
                'process_name': process_name,
                'executable_path': executable_path,
                'timestamp': time.time()
            }
            
            return window_info
            
        except Exception as e:
            logger.error(f"Error getting active window info: {e}")
            return None
    
    def _sanitize_window_title(self, title: str) -> str:
        """Sanitize window title to remove sensitive information."""
        try:
            import re
            
            sanitized_title = title
            
            # Apply sanitization patterns
            for pattern in self.sensitive_patterns:
                sanitized_title = re.sub(pattern, '[REDACTED]', sanitized_title, flags=re.IGNORECASE)
            
            # Remove potential file paths
            sanitized_title = re.sub(r'[A-Za-z]:\\[^\n]*', '[FILE_PATH]', sanitized_title)
            
            # Remove potential URLs
            sanitized_title = re.sub(r'https?://[^\s]+', '[URL]', sanitized_title)
            
            return sanitized_title
            
        except Exception as e:
            logger.error(f"Error sanitizing window title: {e}")
            return "[SANITIZATION_ERROR]"
    
    def _handle_window_change(self, new_window_info: Dict[str, Any]) -> None:
        """Handle window change event."""
        try:
            current_time = time.time()
            
            # Log time spent on previous window
            if self.current_window and self.window_start_time:
                time_spent = current_time - self.window_start_time
                self._log_window_time(self.current_window, time_spent)
                
                # Update application usage statistics
                app_name = self.current_window.get('process_name', 'Unknown')
                if app_name not in self.application_usage:
                    self.application_usage[app_name] = 0
                self.application_usage[app_name] += time_spent
            
            # Update current window
            self.current_window = new_window_info
            self.window_start_time = current_time
            self.stats['window_changes'] += 1
            
            # Add to history
            self._add_to_history(new_window_info)
            
            # Log window change
            self._log_window_change(new_window_info)
            
            # Update keylogger's active window
            window_name = self._format_window_name(new_window_info)
            self.keylogger.update_active_window(window_name)
            
        except Exception as e:
            logger.error(f"Error handling window change: {e}")
            self.stats['errors'] += 1
    
    def _log_window_time(self, window_info: Dict[str, Any], time_spent: float) -> None:
        """Log time spent on a window."""
        try:
            window_name = self._format_window_name(window_info)
            
            # Skip if application is excluded
            if self.config.is_excluded_application(window_name):
                return
            
            self.keylogger.log_event(
                "Time Spent",
                f"{time_spent:.2f} seconds",
                window_name,
                metadata={
                    'process_name': window_info.get('process_name'),
                    'executable_path': window_info.get('executable_path'),
                    'time_spent': time_spent
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging window time: {e}")
    
    def _log_window_change(self, window_info: Dict[str, Any]) -> None:
        """Log window change event."""
        try:
            window_name = self._format_window_name(window_info)
            
            # Skip if application is excluded
            if self.config.is_excluded_application(window_name):
                return
            
            details = f"Switched to {window_name}"
            if window_info.get('process_name'):
                details += f" ({window_info['process_name']})"
            
            self.keylogger.log_event(
                "Window Change",
                details,
                window_name,
                metadata={
                    'process_name': window_info.get('process_name'),
                    'executable_path': window_info.get('executable_path'),
                    'hwnd': window_info.get('hwnd')
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging window change: {e}")
    
    def _add_to_history(self, window_info: Dict[str, Any]) -> None:
        """Add window to history."""
        try:
            history_entry = {
                'title': window_info.get('title'),
                'process_name': window_info.get('process_name'),
                'timestamp': window_info.get('timestamp'),
                'executable_path': window_info.get('executable_path')
            }
            
            self.window_history.append(history_entry)
            
            # Limit history size
            if len(self.window_history) > self.max_history_size:
                self.window_history = self.window_history[-self.max_history_size:]
            
            # Update applications tracked count
            unique_apps = set(entry.get('process_name') for entry in self.window_history)
            self.stats['applications_tracked'] = len(unique_apps)
            
        except Exception as e:
            logger.error(f"Error adding to history: {e}")
    
    def _format_window_name(self, window_info: Dict[str, Any]) -> str:
        """Format window name for logging."""
        try:
            if self.log_window_titles and window_info.get('title'):
                title = window_info['title']
                process_name = window_info.get('process_name', '')
                
                if process_name and process_name.lower() not in title.lower():
                    return f"{title} - {process_name}"
                else:
                    return title
            else:
                return window_info.get('process_name', 'Unknown Application')
            
        except Exception as e:
            logger.error(f"Error formatting window name: {e}")
            return "Unknown Window"
    
    def _log_final_window_time(self) -> None:
        """Log time spent on the final window before stopping."""
        try:
            if self.current_window and self.window_start_time:
                time_spent = time.time() - self.window_start_time
                self._log_window_time(self.current_window, time_spent)
                
                # Update application usage
                app_name = self.current_window.get('process_name', 'Unknown')
                if app_name not in self.application_usage:
                    self.application_usage[app_name] = 0
                self.application_usage[app_name] += time_spent
            
        except Exception as e:
            logger.error(f"Error logging final window time: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'current_window': self.current_window.get('title') if self.current_window else None,
            'history_size': len(self.window_history),
            'application_count': len(self.application_usage)
        }
    
    def get_application_usage(self) -> Dict[str, float]:
        """Get application usage statistics."""
        # Add current window time if still running
        usage = self.application_usage.copy()
        
        if self.current_window and self.window_start_time:
            current_time = time.time() - self.window_start_time
            app_name = self.current_window.get('process_name', 'Unknown')
            if app_name not in usage:
                usage[app_name] = 0
            usage[app_name] += current_time
        
        return usage
    
    def get_window_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get window history."""
        if limit:
            return self.window_history[-limit:]
        return self.window_history.copy()
    
    def get_top_applications(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top applications by usage time."""
        usage = self.get_application_usage()
        
        sorted_apps = sorted(usage.items(), key=lambda x: x[1], reverse=True)
        
        top_apps = []
        for app_name, time_spent in sorted_apps[:limit]:
            top_apps.append({
                'application': app_name,
                'time_spent': time_spent,
                'percentage': (time_spent / sum(usage.values())) * 100 if usage else 0
            })
        
        return top_apps
    
    def export_usage_report(self) -> Dict[str, Any]:
        """Export comprehensive usage report."""
        return {
            'monitoring_stats': self.get_stats(),
            'application_usage': self.get_application_usage(),
            'top_applications': self.get_top_applications(),
            'window_history': self.get_window_history(100),  # Last 100 windows
            'report_timestamp': time.time()
        }