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
            match = re.search(r"Key pressed: ['\"](.*?)['\"]", log_entry)
            if match:
                key = match.group(1)
                if key.isalnum() and len(key) == 1:
                    self.current_text.append(key)
                    return None # Accumulate text
                elif key == 'Key.backspace':
                    return 'Pressed Backspace'
                elif key == 'Key.space':
                    self.current_text.append(' ')
                    return None # Accumulate space
                elif key == 'Key.enter':
                    return 'Pressed Enter'
                elif key.startswith('Key.'):
                    # Handle other special keys if needed, e.g., Shift, Ctrl, Alt, Tab, Delete
                    # For now, just return a generic message or None
                    # return f"Pressed {key.split('.')[-1]}"
                    return None
                # Potentially handle other non-alphanumeric keys if necessary
                return None

        elif 'Screenshot captured:' in log_entry:
            match = re.search(r'Screenshot captured: (.*?)$', log_entry)
            if match:
                return f"Screenshot taken: {match.group(1)}"

        elif 'Active window:' in log_entry:
            match = re.search(r'Active window: (.*?)$', log_entry)
            if match:
                new_window = match.group(1)
                if new_window != self.active_window:
                    # Flush any pending text before switching window context
                    flushed_text = self._flush_current_text()
                    self.active_window = new_window
                    if flushed_text:
                        return f"{flushed_text}\nSwitched to window: {new_window}"
                    else:
                        return f"Switched to window: {new_window}"

        elif 'Mouse pressed' in log_entry:
            match = re.search(r'Mouse pressed at \((\d+), (\d+)\)', log_entry)
            if match:
                x, y = match.group(1), match.group(2)
                return f"Mouse clicked at ({x}, {y})"

        elif 'Mouse scrolled' in log_entry:
            match = re.search(r'Mouse scrolled at \((\d+), (\d+)\) with delta \((.*?)\) in', log_entry)
            if match:
                x, y, delta = match.group(1), match.group(2), match.group(3)
                direction = "up" if "-1" in delta else "down" # Basic direction detection
                return f"Mouse scrolled {direction} at ({x}, {y})"

        return None

    def _flush_current_text(self) -> Optional[str]:
        """Helper function to format and clear the current text buffer."""
        if self.current_text:
            sentence = ''.join(self.current_text).strip()
            self.current_text = []
            if sentence:
                return f"Typed: '{sentence}'"
        return None

    def process_log(self, log_entries: List[str]) -> List[str]:
        """Process multiple log entries and return human-readable sentences."""
        readable_entries = []

        for entry in log_entries:
            parsed = self.parse_log_entry(entry)
            # Handle backspace in the accumulated text
            if parsed == 'Pressed Backspace':
                if self.current_text:
                    self.current_text.pop()
                # Optionally add the 'Pressed Backspace' message itself
                # readable_entries.append(parsed)
            elif parsed == 'Pressed Enter':
                flushed = self._flush_current_text()
                if flushed:
                    readable_entries.append(flushed)
                readable_entries.append("Pressed Enter") # Add Enter press explicitly
            elif parsed:
                # Handle multi-line output from window switch
                if "\n" in parsed:
                    readable_entries.extend(parsed.split('\n'))
                else:
                    readable_entries.append(parsed)

        # Flush any remaining text at the end
        final_text = self._flush_current_text()
        if final_text:
            readable_entries.append(final_text)

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