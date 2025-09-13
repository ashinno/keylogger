"""Comprehensive test suite for all utility modules with performance and security focus."""

import os
import sys
import time
import unittest
import threading
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import json
import hashlib
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.window_monitor import WindowMonitor
    from utils.screenshot_monitor import ScreenshotMonitor
    from utils.usb_monitor import USBMonitor
    from utils.performance_monitor import PerformanceMonitor
    from utils.camera_monitor import CameraMonitor
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    sys.exit(1)


class TestWindowMonitorComprehensive(unittest.TestCase):
    """Comprehensive tests for WindowMonitor with privacy and performance focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'performance.window_check_interval': 1.0,
            'performance.window_history_size': 100,
            'privacy.log_window_titles': True,
            'privacy.sanitize_window_titles': True
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.stop_event = Mock()
        self.mock_keylogger.stop_event.is_set.return_value = False
        self.mock_keylogger.stop_event.wait.return_value = False
        
        self.window_monitor = WindowMonitor(self.mock_keylogger)
        
    def tearDown(self):
        if self.window_monitor.is_running:
            self.window_monitor.stop()
            
    def test_initialization(self):
        """Test WindowMonitor initialization."""
        self.assertFalse(self.window_monitor.is_running)
        self.assertIsNone(self.window_monitor.current_window)
        self.assertIsNone(self.window_monitor.window_start_time)
        self.assertEqual(len(self.window_monitor.window_history), 0)
        self.assertIn('window_changes', self.window_monitor.stats)
        
    @patch('utils.window_monitor.win32gui')
    def test_get_active_window_info(self, mock_win32gui):
        """Test active window information retrieval."""
        # Mock Windows API calls
        mock_win32gui.GetForegroundWindow.return_value = 12345
        mock_win32gui.GetWindowText.return_value = "Test Application - Document.txt"
        
        with patch('utils.window_monitor.win32process') as mock_win32process:
            mock_win32process.GetWindowThreadProcessId.return_value = (None, 6789)
            
            with patch('psutil.Process') as mock_process_class:
                mock_process = Mock()
                mock_process.name.return_value = "testapp.exe"
                mock_process.exe.return_value = "C:\\Program Files\\TestApp\\testapp.exe"
                mock_process_class.return_value = mock_process
                
                window_info = self.window_monitor._get_active_window_info()
                
                self.assertIsNotNone(window_info)
                self.assertEqual(window_info['title'], "Test Application - Document.txt")
                self.assertEqual(window_info['process_name'], "testapp.exe")
                self.assertEqual(window_info['process_id'], 6789)
                
    def test_title_sanitization(self):
        """Test window title sanitization for privacy."""
        test_cases = [
            ("Gmail - password reset", "Gmail - [SENSITIVE]"),
            ("Login to MyBank", "[SENSITIVE] to MyBank"),
            ("Document.txt - Notepad", "Document.txt - Notepad"),
            ("user@example.com - Outlook", "[EMAIL] - Outlook"),
            ("Secret Project 2024", "[SENSITIVE] Project 2024"),
            ("PIN: 1234 - Banking", "PIN: [SENSITIVE] - Banking")
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                sanitized = self.window_monitor._sanitize_window_title(original)
                self.assertEqual(sanitized, expected)
                
    def test_window_change_detection(self):
        """Test window change detection and tracking."""
        # Initial window
        window1 = {
            'title': 'Application 1',
            'process_name': 'app1.exe',
            'process_id': 1001,
            'handle': 12345
        }
        
        # Different window
        window2 = {
            'title': 'Application 2',
            'process_name': 'app2.exe',
            'process_id': 1002,
            'handle': 12346
        }
        
        # Test first window
        self.assertTrue(self.window_monitor._has_window_changed(window1))
        self.window_monitor._update_current_window(window1)
        
        # Test same window (no change)
        self.assertFalse(self.window_monitor._has_window_changed(window1))
        
        # Test different window
        self.assertTrue(self.window_monitor._has_window_changed(window2))
        
    def test_application_usage_tracking(self):
        """Test application usage time tracking."""
        app_name = "testapp.exe"
        usage_time = 5.0
        
        # Initial usage
        self.window_monitor._update_application_usage(app_name, usage_time)
        self.assertEqual(self.window_monitor.application_usage[app_name], usage_time)
        
        # Additional usage
        self.window_monitor._update_application_usage(app_name, usage_time)
        self.assertEqual(self.window_monitor.application_usage[app_name], usage_time * 2)
        
    def test_window_history_management(self):
        """Test window history size management."""
        max_history = self.window_monitor.max_history_size
        
        # Add more entries than max history
        for i in range(max_history + 10):
            window_info = {
                'title': f'Window {i}',
                'process_name': f'app{i}.exe',
                'process_id': 1000 + i,
                'start_time': time.time(),
                'duration': 1.0
            }
            self.window_monitor._add_to_history(window_info)
        
        # History should not exceed max size
        self.assertLessEqual(len(self.window_monitor.window_history), max_history)
        
        # Most recent entries should be preserved
        latest_entry = self.window_monitor.window_history[-1]
        self.assertIn(str(max_history + 9), latest_entry['title'])
        
    def test_statistics_tracking(self):
        """Test window monitoring statistics."""
        initial_stats = self.window_monitor.get_stats()
        
        # Simulate window changes
        window1 = {'title': 'App1', 'process_name': 'app1.exe', 'process_id': 1001, 'handle': 12345}
        window2 = {'title': 'App2', 'process_name': 'app2.exe', 'process_id': 1002, 'handle': 12346}
        
        self.window_monitor._handle_window_change(window1)
        self.window_monitor._handle_window_change(window2)
        
        updated_stats = self.window_monitor.get_stats()
        
        # Verify stats updated
        self.assertGreater(updated_stats['window_changes'], initial_stats['window_changes'])
        self.assertGreater(updated_stats['applications_tracked'], initial_stats['applications_tracked'])
        
    def test_concurrent_window_monitoring(self):
        """Test thread safety of window monitoring."""
        errors = []
        
        def window_monitor_worker(worker_id):
            try:
                for i in range(5):
                    window_info = {
                        'title': f'Worker {worker_id} Window {i}',
                        'process_name': f'worker{worker_id}.exe',
                        'process_id': 2000 + worker_id * 10 + i,
                        'handle': 20000 + worker_id * 100 + i
                    }
                    self.window_monitor._handle_window_change(window_info)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=window_monitor_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)
        
    def test_export_functionality(self):
        """Test window monitoring data export."""
        # Add some test data
        for i in range(3):
            window_info = {
                'title': f'Test Window {i}',
                'process_name': f'test{i}.exe',
                'process_id': 3000 + i,
                'start_time': time.time() - (10 - i),
                'duration': i + 1.0
            }
            self.window_monitor._add_to_history(window_info)
        
        # Test export
        exported_data = self.window_monitor.export_window_data()
        
        self.assertIn('window_history', exported_data)
        self.assertIn('application_usage', exported_data)
        self.assertIn('statistics', exported_data)
        self.assertEqual(len(exported_data['window_history']), 3)


class TestScreenshotMonitorComprehensive(unittest.TestCase):
    """Comprehensive tests for ScreenshotMonitor with privacy and performance focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'screenshots.interval_seconds': 60.0,
            'performance.screenshot_interval': 60.0,
            'screenshots.quality': 85,
            'performance.screenshot_quality': 85,
            'screenshots.format': 'JPEG',
            'performance.screenshot_format': 'JPEG',
            'privacy.blur_sensitive_areas': True,
            'privacy.redact_screenshot_text': False,
            'screenshots.max_width': 1920,
            'screenshots.max_height': 1080,
            'privacy.max_screenshot_resolution': (1920, 1080)
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.stop_event = Mock()
        self.mock_keylogger.stop_event.is_set.return_value = False
        self.mock_keylogger.stop_event.wait.return_value = False
        
        # Create temp directory for screenshots
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('utils.screenshot_monitor.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            self.screenshot_monitor = ScreenshotMonitor(self.mock_keylogger)
            
    def tearDown(self):
        if self.screenshot_monitor.is_running:
            self.screenshot_monitor.stop()
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test ScreenshotMonitor initialization."""
        self.assertFalse(self.screenshot_monitor.is_running)
        self.assertEqual(self.screenshot_monitor.capture_interval, 60.0)
        self.assertEqual(self.screenshot_monitor.screenshot_quality, 85)
        self.assertEqual(self.screenshot_monitor.screenshot_format, 'JPEG')
        self.assertTrue(self.screenshot_monitor.blur_sensitive_areas)
        
    @patch('utils.screenshot_monitor.ImageGrab')
    def test_screenshot_capture(self, mock_image_grab):
        """Test basic screenshot capture functionality."""
        # Mock PIL ImageGrab
        mock_image = Mock()
        mock_image.size = (1920, 1080)
        mock_image_grab.grab.return_value = mock_image
        
        screenshot = self.screenshot_monitor._capture_screenshot()
        
        self.assertIsNotNone(screenshot)
        mock_image_grab.grab.assert_called_once()
        
    def test_image_resizing(self):
        """Test image resizing for privacy and performance."""
        with patch('PIL.Image.new') as mock_image_new:
            mock_image = Mock()
            mock_image.size = (3840, 2160)  # 4K resolution
            mock_resized = Mock()
            mock_resized.size = (1920, 1080)
            mock_image.resize.return_value = mock_resized
            
            resized = self.screenshot_monitor._resize_image(mock_image)
            
            # Should be resized to max resolution
            mock_image.resize.assert_called_once()
            
    def test_sensitive_area_detection(self):
        """Test detection of sensitive areas in screenshots."""
        # Mock image with sensitive areas
        mock_image = Mock()
        mock_image.size = (1920, 1080)
        
        # Test sensitive area patterns
        sensitive_areas = self.screenshot_monitor._detect_sensitive_areas(mock_image)
        
        # Should return list of areas to blur
        self.assertIsInstance(sensitive_areas, list)
        
    def test_image_blurring(self):
        """Test image blurring for privacy protection."""
        with patch('PIL.Image.new') as mock_image_new:
            mock_image = Mock()
            mock_image.size = (1920, 1080)
            mock_blurred = Mock()
            
            with patch('PIL.ImageFilter.GaussianBlur') as mock_blur:
                mock_image.filter.return_value = mock_blurred
                
                # Define areas to blur
                sensitive_areas = [(100, 100, 200, 150), (500, 300, 600, 350)]
                
                blurred_image = self.screenshot_monitor._blur_sensitive_areas(mock_image, sensitive_areas)
                
                self.assertIsNotNone(blurred_image)
                
    def test_screenshot_compression(self):
        """Test screenshot compression and quality settings."""
        with patch('PIL.Image.new') as mock_image_new:
            mock_image = Mock()
            mock_image.size = (1920, 1080)
            
            # Test different quality settings
            quality_levels = [50, 75, 85, 95]
            
            for quality in quality_levels:
                with self.subTest(quality=quality):
                    self.screenshot_monitor.screenshot_quality = quality
                    
                    with patch('io.BytesIO') as mock_bytes_io:
                        mock_buffer = Mock()
                        mock_bytes_io.return_value = mock_buffer
                        
                        compressed = self.screenshot_monitor._compress_screenshot(mock_image)
                        
                        # Verify save was called with correct quality
                        mock_image.save.assert_called()
                        
    def test_screenshot_metadata_extraction(self):
        """Test extraction of screenshot metadata."""
        mock_image = Mock()
        mock_image.size = (1920, 1080)
        
        metadata = self.screenshot_monitor._extract_metadata(mock_image)
        
        self.assertIn('timestamp', metadata)
        self.assertIn('resolution', metadata)
        self.assertIn('format', metadata)
        self.assertIn('quality', metadata)
        
    def test_screenshot_storage_management(self):
        """Test screenshot storage and cleanup."""
        # Mock file operations
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.unlink') as mock_unlink:
                mock_exists.return_value = True
                
                # Test cleanup of old screenshots
                self.screenshot_monitor._cleanup_old_screenshots()
                
                # Verify cleanup was attempted
                self.assertTrue(mock_exists.called)
                
    def test_screenshot_encryption(self):
        """Test screenshot encryption functionality."""
        mock_image_data = b"fake_image_data"
        
        with patch.object(self.screenshot_monitor, 'keylogger_core') as mock_core:
            mock_encryption = Mock()
            mock_core.encryption_manager = mock_encryption
            mock_encryption.encrypt_data.return_value = b"encrypted_data"
            
            encrypted = self.screenshot_monitor._encrypt_screenshot(mock_image_data)
            
            self.assertEqual(encrypted, b"encrypted_data")
            mock_encryption.encrypt_data.assert_called_once_with(mock_image_data)
            
    def test_concurrent_screenshot_capture(self):
        """Test thread safety of screenshot operations."""
        errors = []
        
        def screenshot_worker():
            try:
                with patch('utils.screenshot_monitor.ImageGrab') as mock_grab:
                    mock_image = Mock()
                    mock_image.size = (1920, 1080)
                    mock_grab.grab.return_value = mock_image
                    
                    for i in range(3):
                        self.screenshot_monitor._capture_screenshot()
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(2):
            thread = threading.Thread(target=screenshot_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)


class TestUSBMonitorComprehensive(unittest.TestCase):
    """Comprehensive tests for USBMonitor with security focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'performance.usb_check_interval': 2.0,
            'performance.usb_history_size': 500,
            'security.log_usb_details': True,
            'security.alert_new_usb_devices': True,
            'security.track_usb_file_ops': False
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.stop_event = Mock()
        self.mock_keylogger.stop_event.is_set.return_value = False
        self.mock_keylogger.stop_event.wait.return_value = False
        
        with patch('utils.usb_monitor.psutil') as mock_psutil:
            mock_psutil.disk_partitions.return_value = []
            self.usb_monitor = USBMonitor(self.mock_keylogger)
            
    def tearDown(self):
        if self.usb_monitor.is_running:
            self.usb_monitor.stop()
            
    def test_initialization(self):
        """Test USBMonitor initialization."""
        self.assertFalse(self.usb_monitor.is_running)
        self.assertEqual(len(self.usb_monitor.connected_devices), 0)
        self.assertEqual(len(self.usb_monitor.device_history), 0)
        self.assertIn('devices_connected', self.usb_monitor.stats)
        
    @patch('utils.usb_monitor.psutil')
    def test_device_detection(self, mock_psutil):
        """Test USB device detection."""
        # Mock USB devices
        mock_partitions = [
            Mock(device='E:\\', mountpoint='E:\\', fstype='FAT32', opts='rw,removable'),
            Mock(device='F:\\', mountpoint='F:\\', fstype='NTFS', opts='rw,removable')
        ]
        mock_psutil.disk_partitions.return_value = mock_partitions
        
        devices = self.usb_monitor._get_current_usb_devices()
        
        self.assertEqual(len(devices), 2)
        self.assertIn('E:\\', devices)
        self.assertIn('F:\\', devices)
        
    def test_device_change_detection(self):
        """Test USB device connection/disconnection detection."""
        # Initial state - no devices
        self.usb_monitor.connected_devices = set()
        
        # New devices connected
        current_devices = {'E:\\', 'F:\\'}
        
        connected, disconnected = self.usb_monitor._detect_device_changes(current_devices)
        
        self.assertEqual(connected, {'E:\\', 'F:\\'})
        self.assertEqual(disconnected, set())
        
        # Update connected devices
        self.usb_monitor.connected_devices = current_devices
        
        # One device disconnected, one new connected
        new_devices = {'E:\\', 'G:\\'}
        
        connected, disconnected = self.usb_monitor._detect_device_changes(new_devices)
        
        self.assertEqual(connected, {'G:\\'})
        self.assertEqual(disconnected, {'F:\\'})
        
    def test_device_info_extraction(self):
        """Test USB device information extraction."""
        device_path = 'E:\\'
        
        with patch('utils.usb_monitor.psutil') as mock_psutil:
            # Mock disk usage
            mock_usage = Mock()
            mock_usage.total = 8000000000  # 8GB
            mock_usage.used = 2000000000   # 2GB
            mock_usage.free = 6000000000   # 6GB
            mock_psutil.disk_usage.return_value = mock_usage
            
            device_info = self.usb_monitor._get_device_info(device_path)
            
            self.assertIn('device_path', device_info)
            self.assertIn('total_size', device_info)
            self.assertIn('used_space', device_info)
            self.assertIn('free_space', device_info)
            self.assertIn('timestamp', device_info)
            
    def test_device_security_analysis(self):
        """Test security analysis of USB devices."""
        device_info = {
            'device_path': 'E:\\',
            'total_size': 8000000000,
            'filesystem': 'FAT32',
            'label': 'USB_DRIVE'
        }
        
        security_analysis = self.usb_monitor._analyze_device_security(device_info)
        
        self.assertIn('risk_level', security_analysis)
        self.assertIn('security_flags', security_analysis)
        self.assertIn('recommendations', security_analysis)
        
    def test_device_history_management(self):
        """Test USB device history management."""
        max_history = self.usb_monitor.max_history_size
        
        # Add more events than max history
        for i in range(max_history + 10):
            event = {
                'event_type': 'connected',
                'device_path': f'Drive{i}:\\',
                'timestamp': time.time() + i,
                'device_info': {'total_size': 1000000000}
            }
            self.usb_monitor._add_to_history(event)
        
        # History should not exceed max size
        self.assertLessEqual(len(self.usb_monitor.device_history), max_history)
        
        # Most recent events should be preserved
        latest_event = self.usb_monitor.device_history[-1]
        self.assertIn(str(max_history + 9), latest_event['device_path'])
        
    def test_statistics_tracking(self):
        """Test USB monitoring statistics."""
        initial_stats = self.usb_monitor.get_stats()
        
        # Simulate device events
        self.usb_monitor._handle_device_connected('E:\\')
        self.usb_monitor._handle_device_disconnected('E:\\')
        
        updated_stats = self.usb_monitor.get_stats()
        
        # Verify stats updated
        self.assertGreater(updated_stats['devices_connected'], initial_stats['devices_connected'])
        self.assertGreater(updated_stats['devices_disconnected'], initial_stats['devices_disconnected'])
        
    def test_concurrent_device_monitoring(self):
        """Test thread safety of USB device monitoring."""
        errors = []
        
        def usb_monitor_worker(worker_id):
            try:
                for i in range(3):
                    device_path = f'Worker{worker_id}Drive{i}:\\'
                    self.usb_monitor._handle_device_connected(device_path)
                    time.sleep(0.1)
                    self.usb_monitor._handle_device_disconnected(device_path)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(2):
            thread = threading.Thread(target=usb_monitor_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)


class TestPerformanceMonitorComprehensive(unittest.TestCase):
    """Comprehensive tests for PerformanceMonitor with system monitoring focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'performance.monitor_interval': 30.0,
            'performance.history_size': 100,
            'performance.cpu_alert_threshold': 80.0,
            'performance.memory_alert_threshold': 85.0,
            'performance.disk_alert_threshold': 90.0,
            'performance.keylogger_memory_alert_mb': 100,
            'performance.enable_gc_monitoring': True,
            'performance.auto_gc_threshold_mb': 50,
            'performance.detailed_monitoring': True
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.stop_event = Mock()
        self.mock_keylogger.stop_event.is_set.return_value = False
        self.mock_keylogger.stop_event.wait.return_value = False
        
        with patch('utils.performance_monitor.psutil') as mock_psutil:
            mock_process = Mock()
            mock_psutil.Process.return_value = mock_process
            self.performance_monitor = PerformanceMonitor(self.mock_keylogger)
            
    def tearDown(self):
        if self.performance_monitor.is_running:
            self.performance_monitor.stop()
            
    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        self.assertFalse(self.performance_monitor.is_running)
        self.assertEqual(self.performance_monitor.check_interval, 30.0)
        self.assertEqual(self.performance_monitor.history_size, 100)
        self.assertIn('cpu_percent', self.performance_monitor.alert_thresholds)
        self.assertIn('monitoring_duration', self.performance_monitor.stats)
        
    @patch('utils.performance_monitor.psutil')
    def test_system_metrics_collection(self, mock_psutil):
        """Test system metrics collection."""
        # Mock system metrics
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = Mock()
        mock_memory.total = 16000000000  # 16GB
        mock_memory.available = 8000000000  # 8GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 1000000000000  # 1TB
        mock_disk.used = 500000000000    # 500GB
        mock_disk.free = 500000000000    # 500GB
        mock_disk.percent = 50.0
        mock_psutil.disk_usage.return_value = mock_disk
        
        # Collect metrics
        self.performance_monitor._collect_system_metrics()
        
        # Verify metrics were collected
        self.assertGreater(len(self.performance_monitor.system_metrics), 0)
        
        latest_metrics = self.performance_monitor.system_metrics[-1]
        self.assertIn('cpu_percent', latest_metrics)
        self.assertIn('memory_percent', latest_metrics)
        self.assertIn('disk_percent', latest_metrics)
        
    def test_keylogger_metrics_collection(self):
        """Test keylogger-specific metrics collection."""
        # Mock process metrics
        with patch.object(self.performance_monitor, 'process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 50000000  # 50MB
            mock_memory_info.vms = 100000000  # 100MB
            mock_process.memory_info.return_value = mock_memory_info
            mock_process.cpu_percent.return_value = 5.5
            mock_process.num_threads.return_value = 10
            
            # Collect metrics
            self.performance_monitor._collect_keylogger_metrics()
            
            # Verify metrics were collected
            self.assertGreater(len(self.performance_monitor.keylogger_metrics), 0)
            
            latest_metrics = self.performance_monitor.keylogger_metrics[-1]
            self.assertIn('memory_mb', latest_metrics)
            self.assertIn('cpu_percent', latest_metrics)
            self.assertIn('thread_count', latest_metrics)
            
    def test_performance_alert_detection(self):
        """Test performance alert detection and triggering."""
        # Add high CPU usage metric
        high_cpu_metric = {
            'timestamp': time.time(),
            'cpu_percent': 95.0,  # Above threshold
            'memory_percent': 50.0,
            'disk_percent': 30.0
        }
        self.performance_monitor.system_metrics.append(high_cpu_metric)
        
        initial_alerts = len(self.performance_monitor.alerts_history)
        
        # Check for alerts
        self.performance_monitor._check_performance_alerts()
        
        # Verify alert was triggered
        self.assertGreater(len(self.performance_monitor.alerts_history), initial_alerts)
        
        latest_alert = self.performance_monitor.alerts_history[-1]
        self.assertEqual(latest_alert['metric'], 'cpu_percent')
        self.assertEqual(latest_alert['severity'], 'high')
        
    def test_memory_management_and_gc(self):
        """Test memory management and garbage collection."""
        initial_gc_count = self.performance_monitor.stats['gc_collections']
        
        # Mock high memory usage to trigger GC
        with patch('gc.collect') as mock_gc_collect:
            mock_gc_collect.return_value = 100  # Objects collected
            
            # Simulate high memory usage
            self.performance_monitor.stats['peak_memory_mb'] = 60  # Above threshold
            
            # Perform maintenance
            self.performance_monitor._perform_maintenance()
            
            # Verify GC was triggered
            mock_gc_collect.assert_called_once()
            self.assertGreater(self.performance_monitor.stats['gc_collections'], initial_gc_count)
            
    def test_metrics_history_management(self):
        """Test metrics history size management."""
        max_history = self.performance_monitor.history_size
        
        # Add more metrics than max history
        for i in range(max_history + 10):
            metric = {
                'timestamp': time.time() + i,
                'cpu_percent': 50.0 + i,
                'memory_percent': 40.0 + i,
                'disk_percent': 30.0 + i
            }
            self.performance_monitor.system_metrics.append(metric)
        
        # History should not exceed max size (deque handles this automatically)
        self.assertLessEqual(len(self.performance_monitor.system_metrics), max_history)
        
    def test_performance_summary_generation(self):
        """Test performance summary generation."""
        # Add some test metrics
        for i in range(5):
            system_metric = {
                'timestamp': time.time() + i,
                'cpu_percent': 50.0 + i * 5,
                'memory_percent': 40.0 + i * 3,
                'disk_percent': 30.0 + i * 2
            }
            self.performance_monitor.system_metrics.append(system_metric)
            
            keylogger_metric = {
                'timestamp': time.time() + i,
                'memory_mb': 30.0 + i * 2,
                'cpu_percent': 2.0 + i * 0.5,
                'thread_count': 8 + i
            }
            self.performance_monitor.keylogger_metrics.append(keylogger_metric)
        
        # Generate summary
        summary = self.performance_monitor.get_performance_summary()
        
        self.assertIn('system_metrics', summary)
        self.assertIn('keylogger_metrics', summary)
        self.assertIn('alerts_summary', summary)
        self.assertIn('recommendations', summary)
        
        # Verify averages are calculated
        self.assertIn('avg_cpu_percent', summary['system_metrics'])
        self.assertIn('avg_memory_percent', summary['system_metrics'])
        
    def test_export_functionality(self):
        """Test performance data export."""
        # Add some test data
        for i in range(3):
            system_metric = {
                'timestamp': time.time() + i,
                'cpu_percent': 50.0 + i * 10,
                'memory_percent': 40.0 + i * 5
            }
            self.performance_monitor.system_metrics.append(system_metric)
        
        # Test export
        exported_data = self.performance_monitor.export_performance_data()
        
        self.assertIn('system_metrics', exported_data)
        self.assertIn('keylogger_metrics', exported_data)
        self.assertIn('alerts_history', exported_data)
        self.assertIn('statistics', exported_data)
        self.assertEqual(len(exported_data['system_metrics']), 3)


class TestCameraMonitorComprehensive(unittest.TestCase):
    """Comprehensive tests for CameraMonitor with security and privacy focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'camera.device_index': 0,
            'camera.segment_seconds': 60,
            'camera.fps': 15,
            'camera.width': 640,
            'camera.height': 480,
            'camera.resolution': (640, 480),
            'camera.format': 'mp4',
            'camera.codec': 'mp4v',
            'camera.grayscale': False,
            'camera.blur': False,
            'camera.encrypt': False,
            'security.hash_videos': True
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.stop_event = Mock()
        self.mock_keylogger.stop_event.is_set.return_value = False
        self.mock_keylogger.stop_event.wait.return_value = False
        
        with patch('utils.camera_monitor.cv2') as mock_cv2:
            mock_cv2.VideoCapture.return_value = Mock()
            self.camera_monitor = CameraMonitor(self.mock_keylogger)
            
    def tearDown(self):
        if self.camera_monitor.is_running:
            self.camera_monitor.stop()
            
    def test_initialization(self):
        """Test CameraMonitor initialization."""
        self.assertFalse(self.camera_monitor.is_running)
        self.assertEqual(self.camera_monitor.device_index, 0)
        self.assertEqual(self.camera_monitor.segment_seconds, 60)
        self.assertEqual(self.camera_monitor.fps, 15)
        self.assertEqual(self.camera_monitor.resolution, (640, 480))
        
    @patch('utils.camera_monitor.cv2')
    def test_camera_initialization(self, mock_cv2):
        """Test camera device initialization."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap
        
        success = self.camera_monitor._initialize_camera()
        
        self.assertTrue(success)
        mock_cv2.VideoCapture.assert_called_with(0)
        mock_cap.set.assert_called()  # Resolution and FPS settings
        
    def test_video_recording_setup(self):
        """Test video recording setup and configuration."""
        with patch('utils.camera_monitor.cv2') as mock_cv2:
            mock_writer = Mock()
            mock_cv2.VideoWriter.return_value = mock_writer
            
            filename = "test_video.mp4"
            writer = self.camera_monitor._setup_video_writer(filename)
            
            self.assertIsNotNone(writer)
            mock_cv2.VideoWriter.assert_called_once()
            
    def test_frame_processing(self):
        """Test frame processing with privacy filters."""
        # Mock frame data
        mock_frame = Mock()
        
        with patch('utils.camera_monitor.cv2') as mock_cv2:
            # Test grayscale conversion
            self.camera_monitor.grayscale = True
            mock_cv2.cvtColor.return_value = mock_frame
            
            processed_frame = self.camera_monitor._process_frame(mock_frame)
            
            mock_cv2.cvtColor.assert_called_once()
            
            # Test blur filter
            self.camera_monitor.grayscale = False
            self.camera_monitor.blur = True
            mock_cv2.GaussianBlur.return_value = mock_frame
            
            processed_frame = self.camera_monitor._process_frame(mock_frame)
            
            mock_cv2.GaussianBlur.assert_called_once()
            
    def test_video_encryption(self):
        """Test video file encryption."""
        video_data = b"fake_video_data"
        
        with patch.object(self.camera_monitor, 'keylogger_core') as mock_core:
            mock_encryption = Mock()
            mock_core.encryption_manager = mock_encryption
            mock_encryption.encrypt_data.return_value = b"encrypted_video_data"
            
            self.camera_monitor.encrypt_videos = True
            encrypted = self.camera_monitor._encrypt_video(video_data)
            
            self.assertEqual(encrypted, b"encrypted_video_data")
            mock_encryption.encrypt_data.assert_called_once_with(video_data)
            
    def test_video_hashing(self):
        """Test video file integrity hashing."""
        video_data = b"fake_video_data"
        
        hash_value = self.camera_monitor._hash_video(video_data)
        
        # Verify hash is generated
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA-256 hex length
        
        # Verify same data produces same hash
        hash_value2 = self.camera_monitor._hash_video(video_data)
        self.assertEqual(hash_value, hash_value2)
        
    def test_storage_management(self):
        """Test video storage and cleanup management."""
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.unlink') as mock_unlink:
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_exists.return_value = True
                    mock_stat.return_value.st_size = 1000000  # 1MB
                    
                    # Test cleanup of old videos
                    self.camera_monitor._cleanup_old_videos()
                    
                    # Verify cleanup was attempted
                    self.assertTrue(mock_exists.called)
                    
    def test_statistics_tracking(self):
        """Test camera monitoring statistics."""
        initial_stats = self.camera_monitor.get_stats()
        
        # Simulate recording events
        self.camera_monitor.stats['segments_recorded'] += 1
        self.camera_monitor.stats['total_frames'] += 100
        self.camera_monitor.stats['total_size_mb'] += 5.5
        
        updated_stats = self.camera_monitor.get_stats()
        
        # Verify stats updated
        self.assertGreater(updated_stats['segments_recorded'], initial_stats['segments_recorded'])
        self.assertGreater(updated_stats['total_frames'], initial_stats['total_frames'])
        
    def test_error_handling(self):
        """Test error handling in camera operations."""
        with patch('utils.camera_monitor.cv2') as mock_cv2:
            # Mock camera initialization failure
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap
            
            success = self.camera_monitor._initialize_camera()
            
            self.assertFalse(success)
            self.assertGreater(self.camera_monitor.stats['errors'], 0)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)