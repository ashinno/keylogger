# Keylogger Project

## Overview

This project is a keylogger implemented in Python. It captures and logs various user activities such as keystrokes, mouse clicks, mouse scrolls, active window changes, clipboard changes, screenshots, and USB device connections. The logged data is encrypted for security.

## Features

- **Keystroke Logging**: Captures key press and release events.
- **Mouse Activity Logging**: Captures mouse clicks and scroll events.
- **Active Window Logging**: Logs the active window and the time spent on each window.
- **Clipboard Monitoring**: Monitors and logs clipboard changes.
- **Screenshot Capture**: Periodically captures screenshots.
- **USB Device Monitoring**: Monitors USB device connections and disconnections.
- **Network Activity Logging**: Logs network activity such as visited URLs.
- **Encrypted Log File**: Encrypts the log file for security.

## Requirements

- Python 3.x
- Required Python packages:
  - `psutil`
  - `cryptography`
  - `pynput`
  - `Pillow`
  - `requests`
  - `pyperclip` (optional, for clipboard monitoring)
  - `pywin32` (for Windows-specific functionality)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ashinno/keylogger.git
   cd keylogger
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the keylogger script:
```sh
python keylogger.py
```

The keylogger will start capturing and logging user activities. The log data will be saved in `keylog.txt` and encrypted as `keylog.enc`.

## Configuration

- **Logging**: The logging configuration is set to log messages to `keylog.txt`.
- **Encryption**: A key is generated for encryption using the `cryptography` library. Ensure to store the key securely in a production environment.
- **Screenshot Interval**: The interval for capturing screenshots can be adjusted in the `periodic_screenshot_capture` function.

## Security and Privacy

This keylogger is intended for educational purposes only. Ensure you have proper authorization before using it on any system. Unauthorized use of keyloggers is illegal and unethical.

## Acknowledgements

- `pynput` for keyboard and mouse event capturing.
- `cryptography` for encryption.
- `Pillow` for screenshot capturing.
- `psutil` for process and system utilities.
- `pyperclip` for clipboard monitoring.
- `pywin32` for Windows-specific functionality.