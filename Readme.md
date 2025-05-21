# Keylogger Project

## Overview

This project is a keylogger implemented in Python. It captures and logs various user activities such as keystrokes, mouse clicks, mouse scrolls, active window changes, clipboard changes, screenshots, and USB device connections. The logged data is encrypted for security. The project also includes tools to parse these logs into human-readable formats, offering different perspectives on the captured activity. The primary keylogger script itself generates a readable report upon exit, and separate parser scripts are provided for more detailed or alternative analyses.

## Features

- **Keystroke Logging**: Captures key press and release events.
- **Mouse Activity Logging**: Captures mouse clicks and scroll events.
- **Active Window Logging**: Logs the active window and the time spent on each window.
- **Application Usage Statistics**: Tracks and displays time spent on different applications.
- **Clipboard Monitoring**: Monitors and logs clipboard changes.
- **Screenshot Capture**: Periodically captures screenshots.
- **USB Device Monitoring**: Monitors USB device connections and disconnections.
- **Network Activity Logging**: Logs network activity such as visited URLs.
- **Text Input Reconstruction**: Attempts to reconstruct typed sentences and text entries from raw key presses.
- **Encrypted Log File**: Encrypts the log file for security.

## Project Structure

The project consists of the following main Python scripts:

-   **`keylogger.py`**: This is the core keylogging application. It captures various user activities, logs them to `keylog.txt`, encrypts the log to `keylog.enc`, and can also generate a `readable_log_report.txt` upon termination. It also includes features like screenshot capture, clipboard monitoring, and USB device detection.
-   **`log_parser.py`**: This script provides a comprehensive analysis of the `keylog.txt` file. It generates a `readable_log_report.txt` that includes a summary of application usage (time spent, first/last seen) and reconstructs typed text per application, along with other notable events.
-   **`ts.py`**: (Assuming 'ts' stands for 'text script' or similar) This script focuses primarily on reconstructing and displaying the typed text from `keylog.txt`, saving its output to `readable_log_report.txt`. It offers a more streamlined view of text input compared to the other parsers.

## Requirements

- Python 3.x
- Required Python packages (can be installed via `pip install -r requirements.txt` if a `requirements.txt` file is provided, or individually):
  - `psutil`: For fetching system and process information.
  - `cryptography`: For encrypting the log files.
  - `pynput`: For capturing keyboard and mouse events.
  - `Pillow`: For taking screenshots (a fork of PIL, imported as `PIL`).
  - `requests`: For making HTTP requests (used in remote logging features, if enabled).
  - `pyperclip` (optional): For clipboard monitoring. If not installed, clipboard logging will be disabled.
  - `pywin32` (Windows only): For Windows-specific features like USB monitoring and detailed active window information. Not required on other operating systems.

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

### Running the Keylogger

To start capturing activities:
```sh
python keylogger.py
```
The keylogger will run in the background, logging activities to `keylog.txt` (plain text) and `keylog.enc` (encrypted). Screenshots will be saved in the `screenshots/` directory.
Upon stopping the keylogger (e.g., by pressing the 'Esc' key, if configured, or by terminating the process), it will also generate a summary report named `readable_log_report.txt`.

### Generating Reports

While the main keylogger (`keylogger.py`) produces a readable report upon exit, you can also use the dedicated parser scripts to analyze `keylog.txt` at any time:

1.  **Using `log_parser.py` for a comprehensive report:**
    This script provides a detailed breakdown of application usage, typed text per application, and other events.
    ```sh
    python log_parser.py
    ```
    This will read `keylog.txt` and generate/overwrite `readable_log_report.txt` with its analysis.

2.  **Using `ts.py` for a text-focused report:**
    This script primarily focuses on reconstructing typed text.
    ```sh
    python ts.py
    ```
    This will also read `keylog.txt` and generate/overwrite `readable_log_report.txt` with its text reconstruction.

**Note:** Both `log_parser.py` and `ts.py` will overwrite the `readable_log_report.txt` if it already exists. If you want to keep specific reports, rename them before running a different parser.

## Output Files

The keylogger and associated scripts generate the following files:

-   **`keylog.txt`**: This is the raw, unencrypted log file containing all captured events (keystrokes, mouse activity, window changes, etc.) in a chronological order.
-   **`keylog.enc`**: This is the encrypted version of `keylog.txt` for secure storage of the logged data. The encryption key is generated by `keylogger.py` (and currently logged to the console for development purposes; in a real scenario, this key needs to be managed securely).
-   **`readable_log_report.txt`**: This file contains a human-readable version of the logs. It can be generated by:
    -   `keylogger.py`: Upon termination, it creates a summary report.
    -   `log_parser.py`: Generates a comprehensive report with application usage statistics and typed text.
    -   `ts.py`: Generates a report focused on reconstructed typed text.
    Each script will overwrite this file if it exists.
-   **`screenshots/`**: This directory stores screenshots captured by `keylogger.py`. Each screenshot is named with a timestamp (e.g., `screenshot_YYYYMMDD-HHMMSS.png`).
-   **`config.json`** (Optional): If you create this file, `keylogger.py` will load its configuration from here, allowing you to customize features without modifying the script directly. (More details in the "Configuration" section).

## Configuration

The keylogger (`keylogger.py`) can be configured by creating a `config.json` file in the same directory. If this file is present, its settings will override the defaults in the script.

Here's an example of `config.json` with default settings:

```json
{
    "remote_logging": false,
    "remote_server": "http://example.com/log",
    "stealth_mode": false,
    "email_alerts": false,
    "email_recipient": "",
    "features": {
        "key_logging": true,
        "mouse_logging": true,
        "clipboard_logging": true,
        "screenshots": true,
        "usb_monitoring": true,
        "network_logging": true
    }
}
```

**Explanation of Configuration Options:**

-   **`remote_logging`** (boolean): If `true`, attempts to send log data to a remote server (currently a placeholder feature). Defaults to `false`.
-   **`remote_server`** (string): The URL of the remote server for log submission if `remote_logging` is `true`. Defaults to `"http://example.com/log"`.
-   **`stealth_mode`** (boolean): Placeholder for enabling stealth operations (e.g., hiding console windows). Functionality might vary. Defaults to `false`.
-   **`email_alerts`** (boolean): Placeholder for sending email alerts. Defaults to `false`.
-   **`email_recipient`** (string): The email address to send alerts to if `email_alerts` is `true`.
-   **`features`** (object): Allows toggling specific logging features on or off.
    -   **`key_logging`** (boolean): Enable/disable keyboard logging. Defaults to `true`.
    -   **`mouse_logging`** (boolean): Enable/disable mouse event logging. Defaults to `true`.
    -   **`clipboard_logging`** (boolean): Enable/disable clipboard monitoring. Defaults to `true`.
    -   **`screenshots`** (boolean): Enable/disable periodic screenshot capture. Defaults to `true`.
    -   **`usb_monitoring`** (boolean): Enable/disable USB device connection/disconnection logging (Windows-specific). Defaults to `true`.
    -   **`network_logging`** (boolean): Enable/disable logging of visited URLs (basic implementation). Defaults to `true`.

**Other Configuration Notes:**

-   **Encryption Key**: The encryption key for `keylog.enc` is generated dynamically by `keylogger.py` each time it runs. For persistent decryption or in a real-world scenario, this key would need to be managed externally and securely. The current script prints it to the console, which is suitable only for development/testing.
-   **Screenshot Interval**: The interval for capturing screenshots is hardcoded in `keylogger.py` within the `periodic_screenshot_capture` function (default is 10 seconds). To change this, you would need to modify the script directly.
-   **Log File Names**: The names of the log files (`keylog.txt`, `keylog.enc`) are hardcoded in `keylogger.py`.

## Ethical Use and Privacy Warning

**Important: This project is intended strictly for educational and research purposes.**

Using keyloggers or any surveillance software on systems without explicit, prior, and informed consent from all parties is illegal and unethical in most jurisdictions.

-   **Authorization Required**: Always ensure you have explicit written authorization from the system owner and all users before deploying or using this software on any computer.
-   **Legal Compliance**: You are solely responsible for understanding and complying with all applicable local, state, national, and international laws regarding the use of such software.
-   **No Endorsement of Misuse**: The creators and contributors of this project do not endorse or condone any illegal or unethical use of this software.

**By downloading, copying, or using this software, you acknowledge and agree to these terms and take full responsibility for your actions.**

## Acknowledgements

- `pynput` for keyboard and mouse event capturing.
- `cryptography` for encryption.
- `Pillow` for screenshot capturing.
- `psutil` for process and system utilities.
- `pyperclip` for clipboard monitoring.
- `pywin32` for Windows-specific functionality.