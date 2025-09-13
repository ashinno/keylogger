"""Comprehensive test suite for all listener components with security and performance focus."""

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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from listeners.keyboard_listener import KeyboardListener
    from listeners.mouse_listener import MouseListener
    from listeners.clipboard_listener import ClipboardListener
    from pynput.keyboard import Key, KeyCode
    from pynput.mouse import Button
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class TestKeyboardListenerComprehensive(unittest.TestCase):
    """Comprehensive tests for KeyboardListener with security focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'performance.text_flush_interval': 1.0,
            'performance.max_key_sequence': 50,
            'performance.typing_timeout': 1.0,
            'privacy.sanitize_passwords': True,
            'privacy.excluded_applications': ['test.exe']
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.session_stats = {'active_window': 'Test Window'}
        
        self.keyboard_listener = KeyboardListener(self.mock_keylogger)
        
    def tearDown(self):
        if self.keyboard_listener.is_running:
            self.keyboard_listener.stop()
            
    def test_initialization(self):
        """Test KeyboardListener initialization."""
        self.assertFalse(self.keyboard_listener.is_running)
        self.assertEqual(self.keyboard_listener.current_text, "")
        self.assertEqual(len(self.keyboard_listener.modifier_keys), 0)
        self.assertIn('keys_pressed', self.keyboard_listener.stats)
        
    @patch('listeners.keyboard_listener.keyboard.Listener')
    def test_start_stop_cycle(self, mock_listener_class):
        """Test keyboard listener start/stop functionality."""
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener
        
        # Test start
        self.keyboard_listener.start()
        self.assertTrue(self.keyboard_listener.is_running)
        mock_listener.start.assert_called_once()
        
        # Test stop
        self.keyboard_listener.stop()
        self.assertFalse(self.keyboard_listener.is_running)
        mock_listener.stop.assert_called_once()
        
    def test_modifier_key_detection(self):
        """Test modifier key detection and handling."""
        modifier_keys = [Key.ctrl, Key.alt, Key.shift, Key.cmd]
        
        for key in modifier_keys:
            with self.subTest(key=key):
                self.assertTrue(self.keyboard_listener._is_modifier_key(key))
                
        # Test non-modifier keys
        non_modifier_keys = [Key.space, Key.enter, KeyCode.from_char('a')]
        for key in non_modifier_keys:
            with self.subTest(key=key):
                self.assertFalse(self.keyboard_listener._is_modifier_key(key))
                
    def test_shortcut_detection(self):
        """Test keyboard shortcut detection."""
        # Add modifier keys
        self.keyboard_listener.modifier_keys.add(Key.ctrl)
        
        # Test shortcut keys
        shortcut_keys = [KeyCode.from_char('c'), KeyCode.from_char('v'), KeyCode.from_char('x')]
        for key in shortcut_keys:
            with self.subTest(key=key):
                self.assertTrue(self.keyboard_listener._is_shortcut(key))
                
        # Test without modifiers
        self.keyboard_listener.modifier_keys.clear()
        self.assertFalse(self.keyboard_listener._is_shortcut(KeyCode.from_char('c')))
        
    def test_sensitive_data_detection(self):
        """Test sensitive data pattern detection."""
        sensitive_texts = [
            "password123",
            "my secret token",
            "enter your pin",
            "auth key here",
            "passwd: test123"
        ]
        
        for text in sensitive_texts:
            with self.subTest(text=text):
                self.assertTrue(self.keyboard_listener._is_sensitive_data(text))
                
        # Test non-sensitive text
        normal_texts = ["hello world", "regular typing", "document content"]
        for text in normal_texts:
            with self.subTest(text=text):
                self.assertFalse(self.keyboard_listener._is_sensitive_data(text))
                
    def test_sensitive_data_sanitization(self):
        """Test sensitive data sanitization."""
        test_cases = [
            ("password123", "[SENSITIVE_DATA_FILTERED]"),
            ("hello password world", "hello [SENSITIVE_DATA_FILTERED] world"),
            ("normal text", "normal text"),
            ("my secret is hidden", "my [SENSITIVE_DATA_FILTERED] is hidden")
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.keyboard_listener._sanitize_sensitive_data(input_text)
                self.assertEqual(result, expected)
                
    def test_key_sequence_management(self):
        """Test key sequence length management."""
        # Fill sequence to max length
        max_length = self.keyboard_listener.max_sequence_length
        
        for i in range(max_length + 10):
            key = KeyCode.from_char(chr(ord('a') + (i % 26)))
            self.keyboard_listener._manage_key_sequence(key)
            
        # Sequence should not exceed max length
        self.assertLessEqual(len(self.keyboard_listener.key_sequence), max_length)
        
    def test_special_key_handling(self):
        """Test handling of special keys."""
        special_keys = {
            Key.enter: "\n",
            Key.tab: "\t",
            Key.space: " ",
            Key.backspace: "[BACKSPACE]"
        }
        
        for key, expected_text in special_keys.items():
            with self.subTest(key=key):
                self.keyboard_listener.current_text = "test"
                self.keyboard_listener._handle_special_key(key)
                
                if key == Key.backspace:
                    # Backspace should remove last character
                    self.assertEqual(self.keyboard_listener.current_text, "tes")
                else:
                    self.assertIn(expected_text, self.keyboard_listener.current_text)
                    
    def test_character_key_handling(self):
        """Test handling of character keys."""
        test_chars = ['a', 'Z', '1', '!', '@']
        
        for char in test_chars:
            with self.subTest(char=char):
                self.keyboard_listener.current_text = ""
                key = KeyCode.from_char(char)
                self.keyboard_listener._handle_character_key(key)
                self.assertEqual(self.keyboard_listener.current_text, char)
                
    def test_text_flushing(self):
        """Test text flushing functionality."""
        self.keyboard_listener.current_text = "test text to flush"
        
        # Mock window name check
        self.mock_config.is_application_excluded = Mock(return_value=False)
        
        self.keyboard_listener._flush_current_text()
        
        # Verify event was logged
        self.mock_keylogger.log_event.assert_called_once()
        
        # Text should be cleared after flush
        self.assertEqual(self.keyboard_listener.current_text, "")
        
    def test_statistics_tracking(self):
        """Test statistics tracking accuracy."""
        initial_stats = self.keyboard_listener.get_stats()
        
        # Simulate key presses
        self.keyboard_listener._on_key_press(KeyCode.from_char('a'))
        self.keyboard_listener._on_key_press(Key.ctrl)
        self.keyboard_listener._on_key_press(KeyCode.from_char('c'))
        
        updated_stats = self.keyboard_listener.get_stats()
        
        # Verify stats updated
        self.assertGreater(updated_stats['keys_pressed'], initial_stats['keys_pressed'])
        
    def test_concurrent_key_processing(self):
        """Test thread safety of key processing."""
        errors = []
        
        def key_press_worker(char):
            try:
                for i in range(10):
                    key = KeyCode.from_char(char)
                    self.keyboard_listener._on_key_press(key)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for char in 'abcde':
            thread = threading.Thread(target=key_press_worker, args=(char,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)
        
    def test_key_to_string_conversion(self):
        """Test key to string conversion."""
        test_cases = [
            (KeyCode.from_char('a'), 'a'),
            (Key.enter, 'Enter'),
            (Key.space, 'Space'),
            (Key.ctrl, 'Ctrl')
        ]
        
        for key, expected in test_cases:
            with self.subTest(key=key):
                result = self.keyboard_listener._key_to_string(key)
                self.assertEqual(result, expected)


class TestMouseListenerComprehensive(unittest.TestCase):
    """Comprehensive tests for MouseListener with performance focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'performance.mouse_move_threshold': 50,
            'performance.mouse_move_interval': 1.0,
            'performance.max_click_sequence': 50,
            'performance.double_click_threshold': 0.5,
            'performance.mouse_batch_size': 10,
            'performance.mouse_batch_timeout': 2.0,
            'privacy.log_mouse_coordinates': True,
            'privacy.coordinate_precision': 10
        }.get(key, default)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.session_stats = {'active_window': 'Test Window'}
        
        self.mouse_listener = MouseListener(self.mock_keylogger)
        
    def tearDown(self):
        if self.mouse_listener.is_running:
            self.mouse_listener.stop()
            
    def test_initialization(self):
        """Test MouseListener initialization."""
        self.assertFalse(self.mouse_listener.is_running)
        self.assertEqual(self.mouse_listener.last_position, (0, 0))
        self.assertFalse(self.mouse_listener.is_dragging)
        self.assertIn('clicks', self.mouse_listener.stats)
        
    @patch('listeners.mouse_listener.Listener')
    def test_start_stop_cycle(self, mock_listener_class):
        """Test mouse listener start/stop functionality."""
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener
        
        # Test start
        self.mouse_listener.start()
        self.assertTrue(self.mouse_listener.is_running)
        mock_listener.start.assert_called_once()
        
        # Test stop
        self.mouse_listener.stop()
        self.assertFalse(self.mouse_listener.is_running)
        mock_listener.stop.assert_called_once()
        
    def test_mouse_movement_threshold(self):
        """Test mouse movement threshold filtering."""
        current_time = time.time()
        
        # Small movement (below threshold)
        small_distance = 10
        should_log = self.mouse_listener._should_log_move(100, 100, current_time, small_distance)
        self.assertFalse(should_log)
        
        # Large movement (above threshold)
        large_distance = 100
        should_log = self.mouse_listener._should_log_move(200, 200, current_time, large_distance)
        self.assertTrue(should_log)
        
    def test_double_click_detection(self):
        """Test double-click detection logic."""
        current_time = time.time()
        
        # Add first click
        self.mouse_listener.click_sequence.append({
            'timestamp': current_time - 0.2,  # Within threshold
            'button': Button.left
        })
        
        # Test double-click detection
        is_double = self.mouse_listener._is_double_click(current_time)
        self.assertTrue(is_double)
        
        # Test with time outside threshold
        is_double = self.mouse_listener._is_double_click(current_time + 1.0)
        self.assertFalse(is_double)
        
    def test_drag_operation_detection(self):
        """Test drag operation detection and tracking."""
        # Simulate mouse press (start drag)
        self.mouse_listener._handle_mouse_press(100, 100, Button.left, time.time())
        
        # Should not be dragging yet
        self.assertFalse(self.mouse_listener.is_dragging)
        
        # Simulate movement while pressed (should start drag)
        self.mouse_listener._on_mouse_move(200, 200)
        
        # Should now be dragging
        self.assertTrue(self.mouse_listener.is_dragging)
        self.assertIsNotNone(self.mouse_listener.drag_start_pos)
        
    def test_coordinate_sanitization(self):
        """Test coordinate privacy sanitization."""
        # Test with precision setting
        precision = self.mouse_listener.coordinate_precision
        
        x, y = 123, 456
        sanitized_x, sanitized_y = self.mouse_listener._sanitize_coordinates(x, y)
        
        # Coordinates should be rounded to precision
        expected_x = (x // precision) * precision
        expected_y = (y // precision) * precision
        
        self.assertEqual(sanitized_x, expected_x)
        self.assertEqual(sanitized_y, expected_y)
        
    def test_distance_calculation(self):
        """Test distance calculation between points."""
        pos1 = (0, 0)
        pos2 = (3, 4)  # 3-4-5 triangle
        
        distance = self.mouse_listener._calculate_distance(pos1, pos2)
        self.assertEqual(distance, 5.0)
        
        # Test same position
        distance = self.mouse_listener._calculate_distance(pos1, pos1)
        self.assertEqual(distance, 0.0)
        
    def test_event_batching(self):
        """Test mouse event batching functionality."""
        # Add events to batch
        for i in range(5):
            event = {
                'type': 'move',
                'x': i * 10,
                'y': i * 10,
                'timestamp': time.time()
            }
            self.mouse_listener._add_to_batch(event)
        
        # Verify events in batch
        self.assertEqual(len(self.mouse_listener.event_batch), 5)
        
        # Flush batch
        self.mouse_listener._flush_event_batch()
        
        # Batch should be empty after flush
        self.assertEqual(len(self.mouse_listener.event_batch), 0)
        
    def test_click_sequence_management(self):
        """Test click sequence length management."""
        max_sequence = self.mouse_listener.max_click_sequence
        
        # Add more clicks than max sequence
        for i in range(max_sequence + 10):
            click_info = {
                'timestamp': time.time() + i * 0.1,
                'button': Button.left,
                'x': i,
                'y': i
            }
            self.mouse_listener._add_to_click_sequence(click_info)
        
        # Sequence should not exceed max length
        self.assertLessEqual(len(self.mouse_listener.click_sequence), max_sequence)
        
    def test_button_statistics_tracking(self):
        """Test button-specific statistics tracking."""
        initial_stats = self.mouse_listener.get_stats()
        
        # Simulate different button clicks
        self.mouse_listener._handle_mouse_press(100, 100, Button.left, time.time())
        self.mouse_listener._handle_mouse_press(100, 100, Button.right, time.time())
        self.mouse_listener._handle_mouse_press(100, 100, Button.middle, time.time())
        
        updated_stats = self.mouse_listener.get_stats()
        
        # Verify stats updated
        self.assertGreater(updated_stats['clicks'], initial_stats['clicks'])
        
    def test_scroll_event_handling(self):
        """Test mouse scroll event handling."""
        # Simulate scroll events
        self.mouse_listener._on_mouse_scroll(100, 100, 0, 1)  # Scroll up
        self.mouse_listener._on_mouse_scroll(100, 100, 0, -1)  # Scroll down
        
        # Verify scroll events were processed
        self.assertGreater(self.mouse_listener.stats['scrolls'], 0)
        
    def test_concurrent_mouse_events(self):
        """Test thread safety of mouse event processing."""
        errors = []
        
        def mouse_event_worker(worker_id):
            try:
                for i in range(10):
                    x, y = worker_id * 100 + i, worker_id * 100 + i
                    self.mouse_listener._on_mouse_move(x, y)
                    self.mouse_listener._on_mouse_click(x, y, Button.left, True)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=mouse_event_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)


class TestClipboardListenerComprehensive(unittest.TestCase):
    """Comprehensive tests for ClipboardListener with security focus."""
    
    def setUp(self):
        # Mock keylogger core
        self.mock_keylogger = Mock()
        self.mock_config = Mock()
        
        # Default config values
        self.mock_config.get.side_effect = lambda key, default=None: {
            'performance.clipboard_check_interval': 0.1,
            'performance.clipboard_history_size': 10,
            'privacy.clipboard_max_length': 1000,
            'privacy.log_clipboard_sensitive': False,
            'privacy.hash_clipboard_sensitive': True,
            'privacy.clipboard_length_only': False
        }.get(key, default)
        
        self.mock_config.is_application_excluded = Mock(return_value=False)
        
        self.mock_keylogger.config = self.mock_config
        self.mock_keylogger.log_event = Mock()
        self.mock_keylogger.session_stats = {'active_window': 'Test Window'}
        
        self.clipboard_listener = ClipboardListener(self.mock_keylogger)
        
    def tearDown(self):
        if self.clipboard_listener.is_running:
            self.clipboard_listener.stop()
            
    def test_initialization(self):
        """Test ClipboardListener initialization."""
        self.assertFalse(self.clipboard_listener.is_running)
        self.assertEqual(self.clipboard_listener.last_content, "")
        self.assertEqual(self.clipboard_listener.change_count, 0)
        self.assertIn('changes_detected', self.clipboard_listener.stats)
        
    @patch('listeners.clipboard_listener.pyperclip')
    def test_start_stop_cycle(self, mock_pyperclip):
        """Test clipboard listener start/stop functionality."""
        mock_pyperclip.paste.return_value = "initial content"
        
        # Test start
        self.clipboard_listener.start()
        self.assertTrue(self.clipboard_listener.is_running)
        
        # Give time for thread to start
        time.sleep(0.2)
        
        # Test stop
        self.clipboard_listener.stop()
        self.assertFalse(self.clipboard_listener.is_running)
        
    def test_content_change_detection(self):
        """Test clipboard content change detection."""
        # Test with same content
        self.clipboard_listener.last_content = "test content"
        self.clipboard_listener.last_content_hash = self.clipboard_listener._hash_content("test content")
        
        has_changed = self.clipboard_listener._has_content_changed("test content")
        self.assertFalse(has_changed)
        
        # Test with different content
        has_changed = self.clipboard_listener._has_content_changed("different content")
        self.assertTrue(has_changed)
        
    def test_content_type_detection(self):
        """Test content type detection patterns."""
        test_cases = [
            ("user@example.com", "email"),
            ("https://example.com", "url"),
            ("123-45-6789", "ssn"),
            ("4111-1111-1111-1111", "credit_card"),
            ("555-123-4567", "phone"),
            ("C:\\Users\\test\\file.txt", "file_path"),
            ("function test() { return true; }", "code"),
            ("regular text content", "text")
        ]
        
        for content, expected_type in test_cases:
            with self.subTest(content=content):
                detected_type = self.clipboard_listener._detect_content_type(content)
                self.assertEqual(detected_type, expected_type)
                
    def test_sensitive_content_detection(self):
        """Test sensitive content pattern detection."""
        sensitive_contents = [
            "password123",
            "secret token here",
            "sk-1234567890abcdef",  # API key pattern
            "user@example.com",
            "4111-1111-1111-1111"  # Credit card
        ]
        
        for content in sensitive_contents:
            with self.subTest(content=content):
                is_sensitive = self.clipboard_listener._is_sensitive_content(content)
                self.assertTrue(is_sensitive)
                
        # Test non-sensitive content
        normal_content = "This is just regular text content"
        self.assertFalse(self.clipboard_listener._is_sensitive_content(normal_content))
        
    def test_content_hashing(self):
        """Test content hashing for change detection."""
        content1 = "test content"
        content2 = "different content"
        
        hash1 = self.clipboard_listener._hash_content(content1)
        hash2 = self.clipboard_listener._hash_content(content2)
        
        # Hashes should be different for different content
        self.assertNotEqual(hash1, hash2)
        
        # Same content should produce same hash
        hash1_repeat = self.clipboard_listener._hash_content(content1)
        self.assertEqual(hash1, hash1_repeat)
        
    def test_history_management(self):
        """Test clipboard history management."""
        max_history = self.clipboard_listener.max_history_size
        
        # Add more items than max history
        for i in range(max_history + 5):
            content = f"content {i}"
            self.clipboard_listener._add_to_history(content)
        
        # History should not exceed max size
        self.assertLessEqual(len(self.clipboard_listener.content_history), max_history)
        
        # Most recent items should be preserved
        latest_content = self.clipboard_listener.content_history[-1]['content']
        self.assertIn(str(max_history + 4), latest_content)
        
    def test_log_details_preparation(self):
        """Test log details preparation based on privacy settings."""
        content = "test content"
        content_type = "text"
        
        # Test with sensitive content and different settings
        test_cases = [
            (False, True, False),   # Don't log sensitive, hash sensitive, not length only
            (True, False, False),   # Log sensitive, don't hash, not length only
            (False, False, True),   # Don't log sensitive, don't hash, length only
        ]
        
        for log_sensitive, hash_sensitive, length_only in test_cases:
            with self.subTest(log_sensitive=log_sensitive, hash_sensitive=hash_sensitive, length_only=length_only):
                self.clipboard_listener.log_sensitive_data = log_sensitive
                self.clipboard_listener.hash_sensitive_data = hash_sensitive
                self.clipboard_listener.content_length_only = length_only
                
                result = self.clipboard_listener._prepare_log_details(content, content_type, False)
                
                if length_only:
                    self.assertIn("Length:", result)
                else:
                    self.assertIsNotNone(result)
                    
    def test_statistics_tracking(self):
        """Test clipboard statistics tracking."""
        initial_stats = self.clipboard_listener.get_stats()
        
        # Simulate content changes
        self.clipboard_listener._handle_clipboard_change("test content 1")
        self.clipboard_listener._handle_clipboard_change("test content 2")
        
        updated_stats = self.clipboard_listener.get_stats()
        
        # Verify stats updated
        self.assertGreater(updated_stats['changes_detected'], initial_stats['changes_detected'])
        self.assertGreater(updated_stats['total_characters'], initial_stats['total_characters'])
        
    def test_content_analysis_export(self):
        """Test content analysis and export functionality."""
        # Add some test content to history
        test_contents = [
            "regular text",
            "user@example.com",
            "https://example.com",
            "password123"
        ]
        
        for content in test_contents:
            self.clipboard_listener._add_to_history(content)
        
        # Get content analysis
        analysis = self.clipboard_listener.get_content_analysis()
        
        # Verify analysis structure
        self.assertIn('total_entries', analysis)
        self.assertIn('content_types', analysis)
        self.assertIn('sensitive_entries', analysis)
        
        # Test export functionality
        exported = self.clipboard_listener.export_history(include_sensitive=False)
        self.assertIsInstance(exported, list)
        
        # Export with sensitive data
        exported_with_sensitive = self.clipboard_listener.export_history(include_sensitive=True)
        self.assertGreaterEqual(len(exported_with_sensitive), len(exported))
        
    @patch('listeners.clipboard_listener.pyperclip')
    def test_clipboard_monitoring_loop(self, mock_pyperclip):
        """Test clipboard monitoring loop functionality."""
        # Setup mock to return different content on successive calls
        mock_pyperclip.paste.side_effect = ["content1", "content1", "content2", "content2"]
        
        self.clipboard_listener.start()
        
        # Let it run briefly
        time.sleep(0.3)
        
        self.clipboard_listener.stop()
        
        # Verify monitoring occurred
        self.assertGreater(mock_pyperclip.paste.call_count, 0)
        
    def test_error_handling_in_monitoring(self):
        """Test error handling during clipboard monitoring."""
        with patch('listeners.clipboard_listener.pyperclip') as mock_pyperclip:
            # Setup mock to raise exception
            mock_pyperclip.paste.side_effect = Exception("Clipboard access error")
            
            self.clipboard_listener.start()
            
            # Let it run briefly
            time.sleep(0.2)
            
            self.clipboard_listener.stop()
            
            # Verify error was handled (stats should show errors)
            stats = self.clipboard_listener.get_stats()
            self.assertGreater(stats['errors'], 0)
            
    def test_clear_history_functionality(self):
        """Test clipboard history clearing."""
        # Add some content to history
        for i in range(5):
            self.clipboard_listener._add_to_history(f"content {i}")
        
        # Verify history has content
        self.assertGreater(len(self.clipboard_listener.content_history), 0)
        
        # Clear history
        self.clipboard_listener.clear_history()
        
        # Verify history is empty
        self.assertEqual(len(self.clipboard_listener.content_history), 0)
        self.assertEqual(self.clipboard_listener.change_count, 0)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)