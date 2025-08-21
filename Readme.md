# Enhanced Keylogger Project

A comprehensive, modular keylogger application with advanced security, privacy, and monitoring features.

## Table of Contents
1. Project Overview
2. Technical Specifications
3. Installation Guide
4. Usage Instructions
5. Feature Documentation
6. Contribution Guidelines
7. Troubleshooting
8. License Information

## 1. Project Overview
Purpose: The Enhanced Keylogger helps you monitor and analyze keyboard, mouse, clipboard, window focus, screenshots, USB events, and performance data on authorized systems, with strong security and privacy controls. It is designed for legitimate use cases such as usability testing, parental control, endpoint monitoring within your organization, and academic/security research where you have explicit permission.

Objectives:
- Provide comprehensive, configurable input and activity monitoring
- Preserve privacy via sanitization, hashing, and application exclusions
- Ensure security with encryption and access control
- Offer a web interface and tools for real-time monitoring and post-hoc analysis
- Maintain reliable performance with buffering, rotation, and resource tracking

Target audience and use cases:
- Security analysts and researchers evaluating user behavior or system usage patterns
- IT administrators monitoring corporate endpoints with consent and policy
- Parents or guardians monitoring family devices with consent
- QA/usability teams capturing interaction data during tests

## 2. Technical Specifications
System architecture and components:
- Core services orchestrate configuration, encryption, logging, and the main keylogger runtime.
- Listeners capture inputs/events: keyboard, mouse, and clipboard under listeners/.
- Utilities monitor environment: window focus/usage, screenshots, USB device changes, and performance under utils/.
- Web interface built with Flask for dashboard, logs, configuration, export, and performance views under web/ with templates/ and static/ assets.
- Parsers provide log parsing and export for analysis under parsers/.

Key modules and responsibilities (non-exhaustive):
- core.config_manager: load, validate, and provide configuration values
- core.encryption_manager: manage encryption keys and AES-256-GCM operations
- core.logging_manager: buffered, encrypted log writing with rotation
- core.keylogger: orchestrates listeners, lifecycle, and exports
- listeners.keyboard_listener: keystroke capture, sensitive data filtering, shortcut detection
- listeners.mouse_listener: click, movement, scroll tracking
- listeners.clipboard_listener: clipboard polling with size and privacy limits
- utils.window_monitor: active window tracking and usage timing (Windows-focused)
- utils.screenshot_monitor: periodic screenshots with blur/redaction and compression
- utils.usb_monitor: USB connect/disconnect detection and device details
- utils.performance_monitor: CPU/memory/disk/network plus keylogger metrics
- web.app: Flask routes for login, dashboard, logs, config, export, performance
- parsers.log_parser: parsing, statistics, and export to text/CSV/JSON

Dependencies and requirements:
- Python 3.8+ with pip
- See “Requirements” for the full dependency list and versions
- Windows-specific functionality (active window and some USB details) requires pywin32

Technical constraints and considerations:
- Platform differences: active window and some USB metadata collection are Windows-oriented; on other OSes, these features may be reduced or disabled
- Permissions: some environments require elevated privileges to capture inputs or access system APIs
- Privacy and compliance: configure exclusions, sanitization, and encryption according to your policies and local laws
- Performance: screenshots and high-frequency polling can be resource intensive; adjust intervals and buffer sizes via config.json

## 🚀 Features

### Core Functionality
- **Keyboard Monitoring**: Advanced keystroke capture with modifier key support
- **Mouse Tracking**: Comprehensive mouse event logging (clicks, movements, scrolls)
- **Clipboard Monitoring**: Real-time clipboard content tracking
- **Window Tracking**: Active window and application usage monitoring
- **Screenshot Capture**: Automated screenshot taking with privacy controls
- **USB Device Monitoring**: Track USB device connections/disconnections
- **Performance Monitoring**: System resource usage tracking

### Security & Privacy
- **Data Encryption**: AES-256-GCM encryption for all logged data
- **Privacy Filters**: Automatic detection and sanitization of sensitive data
- **Stealth Mode**: Hidden operation with configurable visibility
- **Access Control**: Password protection and user authentication
- **Secure Key Management**: Encrypted key storage with rotation support

### Performance & Reliability
- **Buffered Logging**: Efficient event buffering with automatic flushing
- **Resource Optimization**: Memory and CPU usage monitoring
- **Error Handling**: Comprehensive error recovery and logging
- **Modular Architecture**: Clean separation of concerns
- **Cross-Platform Support**: Windows, macOS, and Linux compatibility

### Advanced Features
- **Web Interface**: Real-time monitoring dashboard
- **Remote Logging**: Send logs to remote servers
- **Email Alerts**: Automated notifications for important events
- **Log Analysis**: Advanced parsing and reporting tools
- **Configuration Management**: Dynamic configuration updates
- **Export Capabilities**: Multiple output formats (JSON, CSV, TXT)

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- Minimum 4GB RAM
- 1GB free disk space

### Python Dependencies
```
psutil>=5.9.0
cryptography>=3.4.8
pynput>=1.7.6
Pillow>=9.0.0
requests>=2.28.0
pyperclip>=1.8.2
pywin32>=304 (Windows only)
flask>=2.2.0
watchdog>=2.1.9
click>=8.1.0
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
```

## 🛠️ Installation

### Quick Setup (recommended)
```bash
python setup.py
```
This will:
- Check Python version
- Create required directories (keys, logs, screenshots, core, listeners, utils, parsers, web, tests)
- Install dependencies from requirements.txt
- Generate a default config.json
- Generate an encryption key at keys/encryption.key
- Run basic tests and optionally configure autostart

If you prefer manual setup, follow the steps below.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd keylogger
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initial Configuration
- If config.json does not exist, run the setup script above (python setup.py) to create one with sensible defaults.
- Update credentials and secrets:
  - web.admin_username and web.admin_password for the web interface login
  - web.secret_key to a strong random value
- Review privacy, performance, and feature toggles according to your needs.

### 4. Generate Encryption Key
```bash
python -c "from core.encryption_manager import EncryptionManager; EncryptionManager('keys/encryption.key')"
```
This will initialize and persist a secure key at keys/encryption.key if it does not already exist.

## 🚀 Usage

### Basic Usage
```bash
# Start keylogger in interactive mode
python main.py

# Start with web interface
python main.py --web-only

# Run as daemon (background)
python main.py --daemon

# Parse existing log files
python main.py --parse path/to/logfile.txt
```

### CLI Reference
- -c, --config PATH: Configuration file path (default: config.json)
- -w, --web-only: Run web interface only
- -d, --daemon: Run as daemon (background)
- --parse FILE: Parse an existing log file instead of running the keylogger
- -o, --output FILE: Output file when parsing (default varies by format)
- -f, --format {text,csv,json}: Output format when parsing (default: text)
- --version: Show version and exit

### Web Interface
1. Start with web interface: `python main.py --web-only`
2. Open browser to `http://localhost:5000`
3. Login with configured credentials
4. Monitor real-time activity and manage settings

Note: The web interface credentials are read from config.json at web.admin_username and web.admin_password. Older configs using web.username and web.password are automatically migrated to the new keys on the next run; you can also rename them manually if desired.

### Interactive Commands
When running in interactive mode, use these commands:
- `status` - Show current keylogger status
- `stats` - Display statistics
- `export <format> <file>` - Export logs (json/csv/txt)
- `config` - Reload configuration
- `stop` - Stop keylogger
- `quit` - Quit application
- `help` - Show available commands

## 5. Feature Documentation

### Keyboard Listener (listeners/keyboard_listener.py)
- Functionality: Captures key presses/releases, detects shortcuts, buffers text with flush behavior, and filters sensitive input.
- Privacy/Security: Sensitive data detection and sanitization; configurable keywords via privacy.sensitive_keywords and toggles via privacy.sanitize_passwords and privacy.hash_sensitive_data.
- Performance: Adjustable flush thresholds and typing timeouts to minimize overhead.
- Enable/Disable: features.keyboard (true/false).
- Known limitations: OS-level permissions may restrict capture in elevated or protected apps.

### Window Monitor (utils/window_monitor.py)
- Functionality: Tracks active window changes and time spent per application; sanitizes window titles for privacy.
- Config: features.window_tracking, performance.window_check_interval.
- Platform: Primarily Windows-focused; functionality may be reduced on macOS/Linux.
- Known limitations: Some system or UWP apps may not expose details; permissions can limit access.

### Screenshot Monitor (utils/screenshot_monitor.py)
- Functionality: Periodic screenshots with optional blur/redaction, compression, resolution limits; can hash/encrypt outputs.
- Config: features.screenshots (enable/disable), performance.screenshot_interval, privacy exclusions/filters.
- Security: Supports encryption and hashing for tamper detection and confidentiality.
- Performance considerations: Screenshots can be resource-intensive; tune interval and compression.
- Known limitations: Some environments block screenshot capture; redaction rules depend on configuration.

### USB Monitor (utils/usb_monitor.py)
- Functionality: Detects USB device connect/disconnect events and logs details; uses psutil cross-platform and win32 APIs on Windows for richer metadata.
- Config: features.usb_monitoring; polling/check intervals.
- Platform: Works cross-platform with reduced metadata outside Windows.
- Known limitations: Vendor/product details may be unavailable on certain OSes; timing granularity limited by polling interval.

### Performance Monitor (utils/performance_monitor.py)
- Functionality: Tracks CPU, memory, disk, network, and keylogger-specific metrics (process memory/CPU, threads, events logged).
- Config: performance thresholds (e.g., max_memory_usage_mb), history size, and sampling intervals.
- Use cases: Alerting on resource spikes, tuning intervals and buffer sizes based on observed load.

### Web Interface (web/app.py)
- Routes: /login, / (dashboard), /logs (+ /api/logs), /config (+ /api/config), /export (+ /api/export), /performance (+ /api/performance), /test.
- Auth: Basic session auth using credentials from web.admin_username and web.admin_password; secret set via web.secret_key; web.debug controls debug mode.
- Notes: Protect access, change defaults, and bind host/port via web.host and web.port.

### Log Parser (parsers/log_parser.py)
- Functionality: Parses keylogger output for reporting and export to text/CSV/JSON.
- CLI: See Usage > CLI Reference and Log Analysis section for examples.

## 🐛 Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Run with administrator privileges (Windows)
runas /user:Administrator python main.py

# Use sudo on Linux/macOS
sudo python main.py
```

#### Missing Dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

#### Configuration Errors
```bash
# Validate configuration (example check)
python -c "from core.config_manager import ConfigManager; cm = ConfigManager('config.json'); errs = cm.validate_config(); print('Valid' if not errs else f'Invalid: {errs}')"
```

#### Performance Issues
- Reduce buffer sizes in configuration
- Disable unnecessary features
- Increase flush intervals
- Monitor system resources

### Debug Mode
- Set logging.level to "DEBUG" in config.json and restart the application.
- Optionally set web.debug to true for verbose Flask output.
- View runtime logs:
  - Windows (PowerShell): `Get-Content .\keylogger_main.log -Wait`
  - Linux/macOS: `tail -f keylogger_main.log`

## 📝 Changelog

### Version 2.0.0 (Current)
- Complete architectural refactor
- Enhanced security and encryption
- Web interface implementation
- Advanced privacy controls
- Performance optimizations
- Comprehensive testing suite

### Version 1.0.0 (Legacy)
- Basic keystroke logging
- Simple file output
- Minimal configuration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python tests/test_keylogger.py
```

## 📄 License

The Enhanced Keylogger is released under the MIT License. The full text is provided below for convenience.

MIT License

Copyright (c) 2025 Enhanced Keylogger Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third-party licenses: This project depends on open-source libraries (e.g., pynput, cryptography, psutil, Flask, Pillow, requests, pyperclip, pywin32). Each remains under its own license; please review those projects for details.


## ⚠️ Legal Notice

**IMPORTANT**: This software is intended for educational and legitimate monitoring purposes only. Users are responsible for:

- Obtaining proper consent before monitoring
- Complying with local privacy laws
- Using the software ethically and legally
- Securing collected data appropriately

The developers are not responsible for misuse of this software.

## 📞 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Report security issues privately

## 🙏 Acknowledgments

- [pynput](https://github.com/moses-palmer/pynput) for input monitoring
- [cryptography](https://github.com/pyca/cryptography) for encryption
- [psutil](https://github.com/giampaolo/psutil) for system monitoring
- [Flask](https://github.com/pallets/flask) for web interface

---

**Remember**: Always use this software responsibly and in compliance with applicable laws and regulations.