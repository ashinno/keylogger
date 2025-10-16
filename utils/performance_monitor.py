"""Performance monitoring utility for system resources and keylogger metrics."""

import time
import threading
import logging
import psutil
from typing import Dict, Any, List, Optional
from collections import deque
import gc

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance and keylogger resource usage."""
    
    def __init__(self, keylogger_core):
        self.keylogger = keylogger_core
        self.config = keylogger_core.config
        self.is_running = False
        
        # Performance settings
        self.check_interval = self.config.get('performance.monitor_interval', 30.0)
        self.history_size = self.config.get('performance.history_size', 100)
        self.alert_thresholds = {
            'cpu_percent': self.config.get('performance.cpu_alert_threshold', 80.0),
            'memory_percent': self.config.get('performance.memory_alert_threshold', 85.0),
            'disk_percent': self.config.get('performance.disk_alert_threshold', 90.0),
            'keylogger_memory_mb': self.config.get('performance.keylogger_memory_alert_mb', 100)
        }
        
        # Data storage
        self.system_metrics = deque(maxlen=self.history_size)
        self.keylogger_metrics = deque(maxlen=self.history_size)
        self.alerts_history = deque(maxlen=50)
        
        # Process tracking
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # Statistics
        self.stats = {
            'monitoring_duration': 0,
            'total_checks': 0,
            'alerts_triggered': 0,
            'gc_collections': 0,
            'peak_memory_mb': 0,
            'peak_cpu_percent': 0,
            'errors': 0
        }
        
        # Performance optimization
        self.enable_gc_monitoring = self.config.get('performance.enable_gc_monitoring', True)
        self.auto_gc_threshold = self.config.get('performance.auto_gc_threshold_mb', 50)
        self.detailed_monitoring = self.config.get('performance.detailed_monitoring', True)
    
    def run(self) -> None:
        """Main performance monitoring loop."""
        self.is_running = True
        logger.info("Performance monitor started")
        
        try:
            while self.is_running and not self.keylogger.stop_event.is_set():
                try:
                    # Collect metrics
                    self._collect_system_metrics()
                    self._collect_keylogger_metrics()
                    
                    # Check for alerts
                    self._check_performance_alerts()
                    
                    # Perform maintenance
                    self._perform_maintenance()
                    
                    # Update statistics
                    self.stats['total_checks'] += 1
                    self.stats['monitoring_duration'] = time.time() - self.start_time
                    
                    # Wait for next check
                    if self.keylogger.stop_event.wait(self.check_interval):
                        break
                    
                except Exception as e:
                    logger.error(f"Error in performance monitoring loop: {e}")
                    self.stats['errors'] += 1
                    time.sleep(self.check_interval * 2)
            
        except Exception as e:
            logger.error(f"Critical error in performance monitor: {e}")
        finally:
            logger.info(f"Performance monitor stopped. Stats: {self.stats}")
    
    def _collect_system_metrics(self) -> None:
        """Collect system-wide performance metrics."""
        try:
            timestamp = time.time()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # System load
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have load average
                load_avg = (0, 0, 0)
            
            metrics = {
                'timestamp': timestamp,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent,
                    'swap_total_gb': swap.total / (1024**3),
                    'swap_used_gb': swap.used / (1024**3),
                    'swap_percent': swap.percent
                },
                'disk': {
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'percent': (disk_usage.used / disk_usage.total) * 100,
                    'read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                    'write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
                },
                'network': {
                    'bytes_sent_mb': network_io.bytes_sent / (1024**2) if network_io else 0,
                    'bytes_recv_mb': network_io.bytes_recv / (1024**2) if network_io else 0,
                    'packets_sent': network_io.packets_sent if network_io else 0,
                    'packets_recv': network_io.packets_recv if network_io else 0
                },
                'load_avg': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
            }
            metrics['cpu_percent'] = metrics['cpu']['percent']
            metrics['memory_percent'] = metrics['memory']['percent']
            metrics['disk_percent'] = metrics['disk']['percent']
            
            # Add detailed metrics if enabled
            if self.detailed_monitoring:
                metrics.update(self._get_detailed_system_metrics())
            
            self.system_metrics.append(metrics)
            
            # Update peak values
            self.stats['peak_cpu_percent'] = max(self.stats['peak_cpu_percent'], cpu_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            self.stats['errors'] += 1
    
    def _collect_keylogger_metrics(self) -> None:
        """Collect keylogger-specific performance metrics."""
        try:
            timestamp = time.time()
            
            # Process metrics
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            # Thread information
            threads = self.process.threads()
            thread_count = len(threads)
            
            # File descriptors (Unix) or handles (Windows)
            try:
                if hasattr(self.process, 'num_fds'):
                    open_files = self.process.num_fds()
                else:
                    open_files = self.process.num_handles()
            except (AttributeError, psutil.AccessDenied):
                open_files = 0
            
            # Keylogger-specific metrics
            keylogger_stats = self.keylogger.get_stats()
            
            metrics = {
                'timestamp': timestamp,
                'process': {
                    'memory_rss_mb': memory_info.rss / (1024**2),
                    'memory_vms_mb': memory_info.vms / (1024**2),
                    'cpu_percent': cpu_percent,
                    'thread_count': thread_count,
                    'open_files': open_files,
                    'uptime_hours': (timestamp - self.start_time) / 3600
                },
                'keylogger': {
                    'events_logged': keylogger_stats.get('total_events', 0),
                    'buffer_size': keylogger_stats.get('buffer_size', 0),
                    'log_file_size_mb': keylogger_stats.get('log_file_size_mb', 0),
                    'active_listeners': keylogger_stats.get('active_listeners', 0),
                    'errors': keylogger_stats.get('errors', 0)
                }
            }
            metrics['memory_mb'] = metrics['process']['memory_rss_mb']
            metrics['cpu_percent'] = cpu_percent
            metrics['thread_count'] = thread_count
            metrics['open_files'] = open_files
            
            # Add component-specific metrics
            if hasattr(self.keylogger, 'keyboard_listener'):
                kb_stats = getattr(self.keylogger.keyboard_listener, 'stats', {})
                metrics['keyboard'] = {
                    'keys_pressed': kb_stats.get('keys_pressed', 0),
                    'words_typed': kb_stats.get('words_typed', 0),
                    'sensitive_filtered': kb_stats.get('sensitive_data_filtered', 0)
                }
            
            if hasattr(self.keylogger, 'mouse_listener'):
                mouse_stats = getattr(self.keylogger.mouse_listener, 'stats', {})
                metrics['mouse'] = {
                    'clicks': mouse_stats.get('total_clicks', 0),
                    'movements': mouse_stats.get('total_movements', 0),
                    'distance_pixels': mouse_stats.get('total_distance', 0)
                }
            
            if hasattr(self.keylogger, 'clipboard_listener'):
                clip_stats = getattr(self.keylogger.clipboard_listener, 'stats', {})
                metrics['clipboard'] = {
                    'changes': clip_stats.get('changes_detected', 0),
                    'sensitive_filtered': clip_stats.get('sensitive_data_filtered', 0)
                }
            
            self.keylogger_metrics.append(metrics)
            
            # Update peak memory
            memory_mb = memory_info.rss / (1024**2)
            self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], memory_mb)
            
        except Exception as e:
            logger.error(f"Error collecting keylogger metrics: {e}")
            self.stats['errors'] += 1
    
    def _get_detailed_system_metrics(self) -> Dict[str, Any]:
        """Get additional detailed system metrics."""
        try:
            detailed = {}
            
            # Process count
            detailed['process_count'] = len(psutil.pids())
            
            # Boot time
            detailed['boot_time'] = psutil.boot_time()
            detailed['uptime_hours'] = (time.time() - psutil.boot_time()) / 3600
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    detailed['temperatures'] = {
                        sensor: [temp.current for temp in temps[sensor]]
                        for sensor in temps
                    }
            except (AttributeError, OSError):
                pass
            
            # Battery (if available)
            try:
                battery = psutil.sensors_battery()
                if battery:
                    detailed['battery'] = {
                        'percent': battery.percent,
                        'plugged': battery.power_plugged,
                        'time_left_hours': battery.secsleft / 3600 if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                    }
            except (AttributeError, OSError):
                pass
            
            return detailed
            
        except Exception as e:
            logger.debug(f"Error getting detailed metrics: {e}")
            return {}
    
    def _check_performance_alerts(self) -> None:
        """Check for performance issues and trigger alerts."""
        try:
            if not self.system_metrics or not self.keylogger_metrics:
                return
            
            latest_system = self.system_metrics[-1]
            latest_keylogger = self.keylogger_metrics[-1]
            
            alerts = []
            
            # CPU usage alert
            cpu_percent = latest_system.get('cpu_percent', latest_system.get('cpu', {}).get('percent', 0))
            if cpu_percent > self.alert_thresholds['cpu_percent']:
                alerts.append({
                    'metric': 'cpu_percent',
                    'value': cpu_percent,
                    'threshold': self.alert_thresholds['cpu_percent'],
                    'severity': 'high',
                    'message': f"High CPU usage: {cpu_percent:.1f}%"
                })
            
            # Memory usage alert
            memory_percent = latest_system.get('memory_percent', latest_system.get('memory', {}).get('percent', 0))
            if memory_percent > self.alert_thresholds['memory_percent']:
                alerts.append({
                    'metric': 'memory_percent',
                    'value': memory_percent,
                    'threshold': self.alert_thresholds['memory_percent'],
                    'severity': 'medium',
                    'message': f"High memory usage: {memory_percent:.1f}%"
                })
            
            # Disk usage alert
            disk_percent = latest_system.get('disk_percent', latest_system.get('disk', {}).get('percent', 0))
            if disk_percent > self.alert_thresholds['disk_percent']:
                alerts.append({
                    'metric': 'disk_percent',
                    'value': disk_percent,
                    'threshold': self.alert_thresholds['disk_percent'],
                    'severity': 'medium',
                    'message': f"High disk usage: {disk_percent:.1f}%"
                })
            
            # Keylogger memory alert
            keylogger_memory = latest_keylogger.get('memory_mb', latest_keylogger.get('process', {}).get('memory_rss_mb', 0))
            if keylogger_memory > self.alert_thresholds['keylogger_memory_mb']:
                alerts.append({
                    'metric': 'keylogger_memory_mb',
                    'value': keylogger_memory,
                    'threshold': self.alert_thresholds['keylogger_memory_mb'],
                    'severity': 'high',
                    'message': f"High keylogger memory usage: {keylogger_memory:.1f}MB"
                })
            
            # Process alerts
            for alert in alerts:
                self._trigger_alert(alert)
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
            self.stats['errors'] += 1
    
    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        """Trigger a performance alert."""
        try:
            alert['timestamp'] = time.time()
            self.alerts_history.append(alert)
            self.stats['alerts_triggered'] += 1
            
            # Log the alert
            logger.warning(f"PERFORMANCE ALERT: {alert['message']}")
            
            # Log to keylogger
            self.keylogger.log_event(
                "Performance Alert",
                alert['message'],
                "System",
                metadata={
                    'metric': alert.get('metric'),
                    'value': alert.get('value'),
                    'threshold': alert.get('threshold'),
                    'severity': alert.get('severity', 'medium')
                }
            )
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _perform_maintenance(self) -> None:
        """Perform maintenance tasks to optimize performance."""
        try:
            # Check if garbage collection is needed
            if self.enable_gc_monitoring:
                current_memory = self.process.memory_info().rss / (1024**2)
                
                if current_memory > self.auto_gc_threshold:
                    collected = gc.collect()
                    if collected > 0:
                        self.stats['gc_collections'] += 1
                        logger.debug(f"Garbage collection freed {collected} objects")
            
            # Clean up old data if memory is getting high
            if len(self.system_metrics) > self.history_size * 0.9:
                # Remove oldest 10% of data
                remove_count = int(self.history_size * 0.1)
                for _ in range(remove_count):
                    if self.system_metrics:
                        self.system_metrics.popleft()
                    if self.keylogger_metrics:
                        self.keylogger_metrics.popleft()
            
        except Exception as e:
            logger.error(f"Error performing maintenance: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'system_metrics_count': len(self.system_metrics),
            'keylogger_metrics_count': len(self.keylogger_metrics),
            'alerts_count': len(self.alerts_history),
            'current_memory_mb': self.process.memory_info().rss / (1024**2) if self.process else 0
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent performance metrics."""
        try:
            return {
                'system': self.system_metrics[-1] if self.system_metrics else {},
                'keylogger': self.keylogger_metrics[-1] if self.keylogger_metrics else {},
                'timestamp': time.time()
            }
        except Exception:
            return {'system': {}, 'keylogger': {}, 'timestamp': time.time()}
    
    def get_metrics_history(self, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical performance metrics."""
        try:
            system_history = list(self.system_metrics)
            keylogger_history = list(self.keylogger_metrics)
            
            if limit:
                system_history = system_history[-limit:]
                keylogger_history = keylogger_history[-limit:]
            
            return {
                'system': system_history,
                'keylogger': keylogger_history
            }
        except Exception:
            return {'system': [], 'keylogger': []}
    
    def get_alerts_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get performance alerts history."""
        alerts = list(self.alerts_history)
        if limit:
            alerts = alerts[-limit:]
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        try:
            if not self.system_metrics or not self.keylogger_metrics:
                return {}
            
            system_samples = list(self.system_metrics)
            keylogger_samples = list(self.keylogger_metrics)
            
            avg_cpu = sum(m.get('cpu_percent', 0) for m in system_samples) / len(system_samples)
            avg_memory = sum(m.get('memory_percent', 0) for m in system_samples) / len(system_samples)
            avg_disk = sum(m.get('disk_percent', 0) for m in system_samples) / len(system_samples)
            avg_keylogger_memory = sum(m.get('memory_mb', 0) for m in keylogger_samples) / len(keylogger_samples)
            avg_keylogger_cpu = sum(m.get('cpu_percent', 0) for m in keylogger_samples) / len(keylogger_samples)
            avg_thread_count = sum(m.get('thread_count', 0) for m in keylogger_samples) / len(keylogger_samples)
            
            summary = {
                'system_metrics': {
                    'avg_cpu_percent': avg_cpu,
                    'avg_memory_percent': avg_memory,
                    'avg_disk_percent': avg_disk,
                    'peak_cpu_percent': self.stats['peak_cpu_percent'],
                    'peak_memory_mb': self.stats['peak_memory_mb']
                },
                'keylogger_metrics': {
                    'avg_memory_mb': avg_keylogger_memory,
                    'avg_cpu_percent': avg_keylogger_cpu,
                    'avg_thread_count': avg_thread_count
                },
                'alerts_summary': {
                    'total_alerts': self.stats['alerts_triggered'],
                    'recent_alerts': list(self.alerts_history)[-5:]
                },
                'monitoring': {
                    'duration_hours': self.stats['monitoring_duration'] / 3600 if self.stats['monitoring_duration'] else 0,
                    'total_checks': self.stats['total_checks'],
                    'errors': self.stats['errors']
                }
            }
            summary['recommendations'] = self._generate_recommendations(summary)
            return summary
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate human-readable recommendations based on current metrics."""
        recommendations: List[str] = []
        system_metrics = summary.get('system_metrics', {})
        keylogger_metrics = summary.get('keylogger_metrics', {})
        avg_cpu = system_metrics.get('avg_cpu_percent', 0)
        avg_memory = system_metrics.get('avg_memory_percent', 0)
        avg_disk = system_metrics.get('avg_disk_percent', 0)
        avg_keylogger_memory = keylogger_metrics.get('avg_memory_mb', 0)
        
        if avg_cpu > self.alert_thresholds['cpu_percent'] * 0.8:
            recommendations.append('Investigate CPU-intensive processes or reduce system workload.')
        if avg_memory > self.alert_thresholds['memory_percent'] * 0.8:
            recommendations.append('Consider closing unused applications to free system memory.')
        if avg_disk > self.alert_thresholds['disk_percent'] * 0.8:
            recommendations.append('Free up disk space or plan for additional storage capacity.')
        if avg_keylogger_memory > self.alert_thresholds['keylogger_memory_mb'] * 0.8:
            recommendations.append('Evaluate keylogger modules for potential memory optimizations.')
        if not recommendations:
            recommendations.append('System performance metrics are within expected parameters.')
        return recommendations
    
    def export_performance_data(self) -> Dict[str, Any]:
        """Export all performance data for analysis."""
        summary = self.get_performance_summary()
        return {
            'statistics': self.get_stats(),
            'system_metrics': list(self.system_metrics),
            'keylogger_metrics': list(self.keylogger_metrics),
            'alerts_history': list(self.alerts_history),
            'performance_summary': summary,
            'recommendations': summary.get('recommendations', []) if summary else [],
            'export_timestamp': time.time()
        }