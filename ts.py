import re
from typing import List, Optional

class LogParser:
    def __init__(self):
        self.current_text = []
        self.active_window = None

    def parse_log_entry(self, log_entry: str) -> Optional[str]:
        """Parse a single log entry. Return action strings or None if accumulating text."""
        # Remove timestamp if present
        log_entry = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}: ', '', log_entry)

        # Handle key presses for text reconstruction
        if 'Key pressed:' in log_entry:
            key_match = re.search(r"Key pressed: (.*?)(?: in |$)", log_entry)
            if key_match:
                key_info = key_match.group(1).strip()

                char_match = re.search(r"^'(.)'", key_info)
                if char_match:
                    char = char_match.group(1)
                    # Handle specific escaped characters if necessary
                    if char == '\\':
                        self.current_text.append('\\') # Append literal backslash
                    else:
                        self.current_text.append(char)
                    return None # Accumulate character

                elif key_info == 'Key.space':
                    self.current_text.append(' ')
                    return None # Accumulate space
                elif key_info == 'Key.backspace':
                    return 'Action: Backspace'
                elif key_info == 'Key.enter':
                    return 'Action: Enter'
                elif key_info.startswith('Key.') or key_info.startswith("'\\x"):
                    return None # Ignore other special keys and control chars
            return None # Ignore if no relevant key press info found

        # Ignore other event types for text report
        elif any(kw in log_entry for kw in ['Mouse pressed', 'Mouse released', 'Mouse scrolled', 'Screenshot captured', 'Text input:', 'Key released:', 'Shortcut used:', 'Active window:']):
             # Active window changes are handled separately in process_log
             return None # Ignore these events

        # Return None for any other unhandled lines
        return None

    def _flush_current_text(self) -> Optional[str]:
        """Helper function to format and clear the current text buffer."""
        if self.current_text:
            sentence = ''.join(self.current_text).strip()
            self.current_text = []
            if sentence:
                # Return only the sentence, context added in process_log
                return f"Typed: '{sentence}'"
        return None

    def process_log(self, log_entries: List[str]) -> List[str]:
        """Process multiple log entries and return human-readable sentences focused on typed text."""
        readable_output = []
        current_window = None
        self.current_text = [] # Ensure text buffer is clear at start

        for entry in log_entries:
            # Check for window change first
            window_match = re.search(r'Active window: (.*?)$', entry)
            if window_match:
                # Remove timestamp if present before extracting window name
                clean_entry = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}: ', '', entry)
                window_match = re.search(r'Active window: (.*?)$', clean_entry)
                if window_match:
                    new_window = window_match.group(1).strip()
                    if new_window != current_window:
                        flushed = self._flush_current_text() # Flush text on window change
                        if flushed:
                            readable_output.append(f"[{current_window or 'Unknown Window'}] {flushed}")
                        current_window = new_window
                        self.active_window = new_window # Update parser's window context
                        # Optionally log window switch:
                        # readable_output.append(f"--- Switched to window: {current_window} ---")
                continue # Don't process the window change line further

            # Process the entry for key presses or actions
            parsed_action = self.parse_log_entry(entry)

            if parsed_action == 'Action: Backspace':
                if self.current_text:
                    self.current_text.pop()
            elif parsed_action == 'Action: Enter':
                flushed = self._flush_current_text()
                if flushed:
                    readable_output.append(f"[{current_window or 'Unknown Window'}] {flushed}")
                # Optionally add an explicit Enter marker:
                readable_output.append(f"[{current_window or 'Unknown Window'}] [Enter]")
                self.current_text = [] # Clear buffer after Enter

            # Note: parse_log_entry now only returns actions or None for accumulation
            # Text is accumulated internally and flushed on window change, Enter, or end of log.

        # Flush any remaining text at the very end
        final_text = self._flush_current_text()
        if final_text:
            readable_output.append(f"[{current_window or 'Unknown Window'}] {final_text}")

        # Filter out potential empty strings if any None results were added inadvertently
        return [line for line in readable_output if line and line.strip()]

def main():
    parser = LogParser()
    log_file_path = 'keylog.txt'
    output_file_path = 'readable_log_report.txt'
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_entries = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return
    except Exception as e:
        print(f"Error reading file {log_file_path}: {e}")
        return

    readable_entries = parser.process_log(log_entries)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for entry in readable_entries:
                f.write(entry + '\n')
        print(f"Readable log report saved to: {output_file_path}")
    except Exception as e:
        print(f"Error writing report file {output_file_path}: {e}")

if __name__ == "__main__":
    main()