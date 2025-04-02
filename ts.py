import re
from typing import List, Optional

class LogParser:
    def __init__(self):
        self.current_text = []
        self.active_window = None

    def parse_log_entry(self, log_entry: str) -> Optional[str]:
        """Parse a single log entry and return a human-readable sentence."""
        # Remove timestamp if present
        log_entry = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}: ', '', log_entry)

        # Handle different types of log entries
        if 'Text input:' in log_entry:
            match = re.search(r'Text input: (.*?) in', log_entry)
            if match:
                return f"Typed: {match.group(1)}"

        elif 'Key pressed:' in log_entry:
            # Extract the key that was pressed
            match = re.search(r"Key pressed: ['\"](.*?)['\"]", log_entry)
            if match:
                key = match.group(1)
                if key.isalnum() and len(key) == 1:  # Single alphanumeric character
                    self.current_text.append(key)
                    return None  # Don't output individual keypresses

        elif 'Screenshot captured:' in log_entry:
            match = re.search(r'Screenshot captured: (.*?)$', log_entry)
            if match:
                return f"Screenshot taken: {match.group(1)}"

        elif 'Active window:' in log_entry:
            match = re.search(r'Active window: (.*?)$', log_entry)
            if match:
                new_window = match.group(1)
                if new_window != self.active_window:
                    self.active_window = new_window
                    return f"Switched to {new_window}"

        elif 'Mouse pressed' in log_entry:
            match = re.search(r'Mouse pressed at \((\d+), (\d+)\)', log_entry)
            if match:
                x, y = match.group(1), match.group(2)
                return f"Clicked at position ({x}, {y})"

        return None

    def process_log(self, log_entries: List[str]) -> List[str]:
        """Process multiple log entries and return human-readable sentences."""
        readable_entries = []
        current_sentence = ""

        for entry in log_entries:
            parsed = self.parse_log_entry(entry)
            if parsed:
                readable_entries.append(parsed)
            if 'Key pressed: Key.space' in entry or 'Key pressed: Key.enter' in entry:
                if self.current_text:
                    current_sentence = ''.join(self.current_text)
                    if current_sentence.strip():
                        readable_entries.append(f"Typed: {current_sentence}")
                    self.current_text = []

        return readable_entries

def main():
    parser = LogParser()
    try:
        with open('keylog.txt', 'r', encoding='utf-8', errors='ignore') as f:
            log_entries = f.readlines()
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        return

    # Process logs in real-time
    readable_entries = parser.process_log(log_entries)
    for entry in readable_entries:
        print(entry)

if __name__ == "__main__":
    main()