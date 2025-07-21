"""Mouse listener with performance optimizations and detailed event tracking."""

import time
import threading
import logging
from typing import Optional, Dict, Any, Tuple
from pynput import mouse
from pynput.mouse import Button, Listener
import json

logger = logging.getLogger(__name__)


class MouseListener:
    """Enhanced mouse listener with performance and security features."""
    
    def __init__(self, keylogger_core):
        self.keylogger = keylogger_core
        self.config = keylogger_core.config
        self.listener: Optional[Listener] = None
        self.is_running = False
        
        # Event tracking
        self.last_position = (0, 0)
        self.last_move_time = 0
        self.click_sequence = []
        self.drag_start_pos: Optional[Tuple[int, int]] = None
        self.is_dragging = False
        
        # Performance settings
        self.move_threshold = self.config.get('performance.mouse_move_threshold', 50)
        self.move_interval = self.config.get('performance.mouse_move_interval', 1.0)
        self.max_click_sequence = self.config.get('performance.max_click_sequence', 50)
        self.double_click_threshold = self.config.get('performance.double_click_threshold', 0.5)
        
        # Privacy settings
        self.log_coordinates = self.config.get('privacy.log_mouse_coordinates', False)
        self.coordinate_precision = self.config.get('privacy.coordinate_precision', 10)
        
        # Statistics
        self.stats = {
            'clicks': 0,
            'double_clicks': 0,
            'right_clicks': 0,
            'middle_clicks': 0,
            'scrolls': 0,
            'moves_logged': 0,
            'drags': 0,
            'total_distance': 0.0
        }
        
        # Event batching for performance
        self.event_batch = []
        self.batch_size = self.config.get('performance.mouse_batch_size', 10)
        self.batch_timeout = self.config.get('performance.mouse_batch_timeout', 2.0)
        self.last_batch_time = time.time()
        
        # Setup batch processing
        self._setup_batch_processing()
    
    def _setup_batch_processing(self) -> None:
        """Setup batch processing for mouse events."""
        def batch_processor():
            while self.is_running:
                time.sleep(self.batch_timeout)
                if self.event_batch:
                    self._flush_event_batch()
        
        self.batch_thread = threading.Thread(target=batch_processor, daemon=True)
    
    def start(self) -> None:
        """Start the mouse listener."""
        if self.is_running:
            logger.warning("Mouse listener is already running")
            return
        
        try:
            self.is_running = True
            self.listener = Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            self.listener.start()
            self.batch_thread.start()
            logger.info("Mouse listener started")
            
        except Exception as e:
            logger.error(f"Failed to start mouse listener: {e}")
            self.is_running = False
            raise
    
    def stop(self) -> None:
        """Stop the mouse listener."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            if self.listener:
                self.listener.stop()
            
            # Flush any remaining events
            if self.event_batch:
                self._flush_event_batch()
            
            logger.info("Mouse listener stopped")
            logger.info(f"Mouse stats: {self.stats}")
            
        except Exception as e:
            logger.error(f"Error stopping mouse listener: {e}")
    
    def _on_mouse_move(self, x: int, y: int) -> None:
        """Handle mouse move events."""
        try:
            if not self.is_running:
                return
            
            current_time = time.time()
            
            # Calculate distance moved
            distance = self._calculate_distance((x, y), self.last_position)
            self.stats['total_distance'] += distance
            
            # Check if we should log this move
            if self._should_log_move(x, y, current_time, distance):
                self._add_move_event(x, y, current_time, distance)
                self.last_move_time = current_time
                self.stats['moves_logged'] += 1
            
            self.last_position = (x, y)
            
            # Handle dragging
            if self.is_dragging and self.drag_start_pos:
                drag_distance = self._calculate_distance((x, y), self.drag_start_pos)
                if drag_distance > 5:  # Minimum drag distance
                    self._add_drag_event(x, y, drag_distance)
            
        except Exception as e:
            logger.error(f"Error in mouse move handler: {e}")
    
    def _on_mouse_click(self, x: int, y: int, button: Button, pressed: bool) -> None:
        """Handle mouse click events."""
        try:
            if not self.is_running:
                return
            
            current_time = time.time()
            
            if pressed:
                self._handle_mouse_press(x, y, button, current_time)
            else:
                self._handle_mouse_release(x, y, button, current_time)
            
        except Exception as e:
            logger.error(f"Error in mouse click handler: {e}")
    
    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """Handle mouse scroll events."""
        try:
            if not self.is_running:
                return
            
            self.stats['scrolls'] += 1
            
            scroll_direction = "up" if dy > 0 else "down" if dy < 0 else "horizontal"
            scroll_amount = abs(dy) if dy != 0 else abs(dx)
            
            self._add_scroll_event(x, y, scroll_direction, scroll_amount)
            
        except Exception as e:
            logger.error(f"Error in mouse scroll handler: {e}")
    
    def _should_log_move(self, x: int, y: int, current_time: float, distance: float) -> bool:
        """Determine if mouse move should be logged."""
        # Check time interval
        if current_time - self.last_move_time < self.move_interval:
            return False
        
        # Check distance threshold
        if distance < self.move_threshold:
            return False
        
        return True
    
    def _handle_mouse_press(self, x: int, y: int, button: Button, timestamp: float) -> None:
        """Handle mouse button press."""
        # Start drag detection
        if button == Button.left:
            self.drag_start_pos = (x, y)
            self.is_dragging = True
        
        # Add to click sequence for pattern detection
        click_info = {
            'button': button.name,
            'position': self._sanitize_coordinates(x, y),
            'timestamp': timestamp,
            'action': 'press'
        }
        
        self._add_to_click_sequence(click_info)
    
    def _handle_mouse_release(self, x: int, y: int, button: Button, timestamp: float) -> None:
        """Handle mouse button release."""
        # End drag detection
        if button == Button.left and self.is_dragging:
            if self.drag_start_pos:
                drag_distance = self._calculate_distance((x, y), self.drag_start_pos)
                if drag_distance > 5:
                    self.stats['drags'] += 1
            self.is_dragging = False
            self.drag_start_pos = None
        
        # Update click statistics
        if button == Button.left:
            self.stats['clicks'] += 1
            # Check for double click
            if self._is_double_click(timestamp):
                self.stats['double_clicks'] += 1
        elif button == Button.right:
            self.stats['right_clicks'] += 1
        elif button == Button.middle:
            self.stats['middle_clicks'] += 1
        
        # Add click event
        self._add_click_event(x, y, button, timestamp)
        
        # Add to click sequence
        click_info = {
            'button': button.name,
            'position': self._sanitize_coordinates(x, y),
            'timestamp': timestamp,
            'action': 'release'
        }
        
        self._add_to_click_sequence(click_info)
    
    def _is_double_click(self, timestamp: float) -> bool:
        """Check if this is a double click."""
        if len(self.click_sequence) < 2:
            return False
        
        last_click = self.click_sequence[-1]
        if (timestamp - last_click['timestamp'] < self.double_click_threshold and
            last_click['button'] == 'left' and
            last_click['action'] == 'release'):
            return True
        
        return False
    
    def _add_to_click_sequence(self, click_info: Dict[str, Any]) -> None:
        """Add click to sequence for pattern analysis."""
        self.click_sequence.append(click_info)
        
        # Limit sequence length
        if len(self.click_sequence) > self.max_click_sequence:
            self.click_sequence = self.click_sequence[-self.max_click_sequence:]
    
    def _add_move_event(self, x: int, y: int, timestamp: float, distance: float) -> None:
        """Add mouse move event to batch."""
        event = {
            'type': 'move',
            'position': self._sanitize_coordinates(x, y),
            'timestamp': timestamp,
            'distance': round(distance, 2)
        }
        self._add_to_batch(event)
    
    def _add_click_event(self, x: int, y: int, button: Button, timestamp: float) -> None:
        """Add mouse click event to batch."""
        event = {
            'type': 'click',
            'button': button.name,
            'position': self._sanitize_coordinates(x, y),
            'timestamp': timestamp
        }
        self._add_to_batch(event)
    
    def _add_scroll_event(self, x: int, y: int, direction: str, amount: int) -> None:
        """Add mouse scroll event to batch."""
        event = {
            'type': 'scroll',
            'direction': direction,
            'amount': amount,
            'position': self._sanitize_coordinates(x, y),
            'timestamp': time.time()
        }
        self._add_to_batch(event)
    
    def _add_drag_event(self, x: int, y: int, distance: float) -> None:
        """Add mouse drag event to batch."""
        event = {
            'type': 'drag',
            'position': self._sanitize_coordinates(x, y),
            'distance': round(distance, 2),
            'timestamp': time.time()
        }
        self._add_to_batch(event)
    
    def _add_to_batch(self, event: Dict[str, Any]) -> None:
        """Add event to batch for processing."""
        self.event_batch.append(event)
        
        # Check if batch should be flushed
        if (len(self.event_batch) >= self.batch_size or
            time.time() - self.last_batch_time >= self.batch_timeout):
            self._flush_event_batch()
    
    def _flush_event_batch(self) -> None:
        """Flush event batch to log."""
        if not self.event_batch:
            return
        
        try:
            window_name = self.keylogger.session_stats.get('active_window', 'Unknown')
            
            # Skip if application is excluded
            if self.config.is_application_excluded(window_name):
                self.event_batch.clear()
                return
            
            # Group events by type for efficient logging
            event_groups = {}
            for event in self.event_batch:
                event_type = event['type']
                if event_type not in event_groups:
                    event_groups[event_type] = []
                event_groups[event_type].append(event)
            
            # Log each group
            for event_type, events in event_groups.items():
                if len(events) == 1:
                    # Single event
                    event = events[0]
                    details = self._format_event_details(event)
                else:
                    # Multiple events - summarize
                    details = self._format_batch_details(event_type, events)
                
                self.keylogger.log_event(
                    f"Mouse {event_type.title()}",
                    details,
                    window_name,
                    metadata={
                        'event_count': len(events),
                        'batch_timestamp': time.time()
                    }
                )
            
            self.event_batch.clear()
            self.last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing event batch: {e}")
            self.event_batch.clear()
    
    def _format_event_details(self, event: Dict[str, Any]) -> str:
        """Format single event details."""
        event_type = event['type']
        
        if event_type == 'move':
            if self.log_coordinates:
                return f"Moved to {event['position']}, distance: {event['distance']}px"
            else:
                return f"Mouse moved {event['distance']}px"
        
        elif event_type == 'click':
            if self.log_coordinates:
                return f"{event['button'].title()} click at {event['position']}"
            else:
                return f"{event['button'].title()} click"
        
        elif event_type == 'scroll':
            if self.log_coordinates:
                return f"Scrolled {event['direction']} ({event['amount']}) at {event['position']}"
            else:
                return f"Scrolled {event['direction']} ({event['amount']})"
        
        elif event_type == 'drag':
            if self.log_coordinates:
                return f"Dragged to {event['position']}, distance: {event['distance']}px"
            else:
                return f"Dragged {event['distance']}px"
        
        return str(event)
    
    def _format_batch_details(self, event_type: str, events: list) -> str:
        """Format batch event details."""
        count = len(events)
        
        if event_type == 'move':
            total_distance = sum(e.get('distance', 0) for e in events)
            return f"{count} mouse moves, total distance: {total_distance:.1f}px"
        
        elif event_type == 'click':
            button_counts = {}
            for event in events:
                button = event.get('button', 'unknown')
                button_counts[button] = button_counts.get(button, 0) + 1
            
            details = []
            for button, count in button_counts.items():
                details.append(f"{count} {button}")
            
            return f"{count} clicks: {', '.join(details)}"
        
        elif event_type == 'scroll':
            return f"{count} scroll events"
        
        elif event_type == 'drag':
            total_distance = sum(e.get('distance', 0) for e in events)
            return f"{count} drag events, total distance: {total_distance:.1f}px"
        
        return f"{count} {event_type} events"
    
    def _sanitize_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Sanitize coordinates based on privacy settings."""
        if not self.log_coordinates:
            return (0, 0)
        
        # Round coordinates to reduce precision
        precision = self.coordinate_precision
        x_rounded = (x // precision) * precision
        y_rounded = (y // precision) * precision
        
        return (x_rounded, y_rounded)
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """Get listener statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'batch_size': len(self.event_batch),
            'click_sequence_length': len(self.click_sequence),
            'is_dragging': self.is_dragging,
            'last_position': self.last_position
        }
    
    def get_click_patterns(self) -> Dict[str, Any]:
        """Analyze click patterns for insights."""
        if not self.click_sequence:
            return {}
        
        # Analyze click timing patterns
        intervals = []
        for i in range(1, len(self.click_sequence)):
            interval = self.click_sequence[i]['timestamp'] - self.click_sequence[i-1]['timestamp']
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
        else:
            avg_interval = min_interval = max_interval = 0
        
        return {
            'total_clicks': len(self.click_sequence),
            'average_interval': round(avg_interval, 3),
            'min_interval': round(min_interval, 3),
            'max_interval': round(max_interval, 3),
            'double_click_rate': self.stats['double_clicks'] / max(self.stats['clicks'], 1)
        }