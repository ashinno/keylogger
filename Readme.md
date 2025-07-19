# Enhanced Keylogger Project

A comprehensive, modular keylogger application with advanced security, privacy, and monitoring features.

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
```bash
# Copy default configuration
cp config.json.example config.json

# Edit configuration as needed
notepad config.json  # Windows
nano config.json     # Linux/macOS
```

### 4. Generate Encryption Key
```bash
python -c "from core.encryption_manager import EncryptionManager; from core.config_manager import ConfigManager; em = EncryptionManager(ConfigManager('config.json')); em.save_key(em.generate_random_key())"
```

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
python main.py --parse-logs path/to/logfile.txt
```

### Web Interface
1. Start with web interface: `python main.py --web-only`
2. Open browser to `http://localhost:5000`
3. Login with configured credentials
4. Monitor real-time activity and manage settings

### Interactive Commands
When running in interactive mode, use these commands:
- `status` - Show current keylogger status
- `stats` - Display statistics
- `export <format> <file>` - Export logs (json/csv/txt)
- `reload` - Reload configuration
- `stop` - Stop keylogger
- `help` - Show available commands

## ⚙️ Configuration

The keylogger is configured through `config.json`. Key sections include:

### Logging Configuration
```json
{
  "logging": {
    "level": "INFO",
    "file_path": "logs/keylog.txt",
    "max_size_mb": 100,
    "buffer_size": 100,
    "flush_interval": 5.0,
    "enable_rotation": true,
    "enable_encryption": true
  }
}
```

### Feature Toggles
```json
{
  "features": {
    "keyboard": true,
    "mouse": true,
    "clipboard": true,
    "screenshots": false,
    "usb_monitoring": true,
    "network_monitoring": false,
    "window_tracking": true
  }
}
```

### Privacy Settings
```json
{
  "privacy": {
    "sanitize_passwords": true,
    "hash_sensitive_data": true,
    "clipboard_max_length": 1000,
    "excluded_applications": ["password_manager.exe"],
    "sensitive_keywords": ["password", "ssn", "credit"]
  }
}
```

### Security Configuration
```json
{
  "encryption": {
    "enabled": true,
    "key_file": "keys/encryption.key",
    "algorithm": "AES-256-GCM"
  },
  "stealth": {
    "hide_console": true,
    "process_name": "system_service",
    "startup_delay": 30
  }
}
```

## 📊 Log Analysis

### Using the Log Parser
```bash
# Parse and analyze logs
python -m parsers.log_parser logs/keylog.txt

# Generate readable report
python -m parsers.log_parser logs/keylog.txt --report output_report.txt

# Export to different formats
python -m parsers.log_parser logs/keylog.txt --export-csv output.csv
python -m parsers.log_parser logs/keylog.txt --export-json output.json
```

### Programmatic Analysis
```python
from parsers.log_parser import LogParser
from core.config_manager import ConfigManager

# Initialize parser
config = ConfigManager('config.json')
parser = LogParser(config)

# Parse log file
events = parser.parse_log_file('logs/keylog.txt')

# Get statistics
stats = parser.get_statistics()
print(f"Total events: {stats['total_events']}")
print(f"Total keystrokes: {stats['total_keystrokes']}")

# Generate report
report = parser.generate_readable_report(events, 'analysis_report.txt')
```

## 🧪 Testing

### Run All Tests
```bash
# Run complete test suite
python tests/test_keylogger.py

# Run with pytest for detailed output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Resource usage and speed
- **Security Tests**: Encryption and privacy features

## 🔧 Development

### Project Structure
```
keylogger/
├── core/                   # Core functionality
│   ├── config_manager.py   # Configuration management
│   ├── encryption_manager.py # Encryption/decryption
│   ├── logging_manager.py  # Event logging
│   └── keylogger.py       # Main orchestrator
├── listeners/             # Input listeners
│   ├── keyboard_listener.py
│   ├── mouse_listener.py
│   └── clipboard_listener.py
├── utils/                 # Utility modules
│   ├── window_monitor.py
│   ├── screenshot_monitor.py
│   ├── usb_monitor.py
│   └── performance_monitor.py
├── parsers/              # Log analysis
│   └── log_parser.py
├── web/                  # Web interface
│   └── app.py
├── tests/                # Test suite
│   └── test_keylogger.py
├── config.json           # Configuration file
├── requirements.txt      # Dependencies
├── main.py              # Entry point
└── README.md            # Documentation
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Comprehensive docstrings
- Error handling for all operations
- Security-first approach

### Adding New Features
1. Create feature branch
2. Implement with tests
3. Update configuration schema
4. Add documentation
5. Submit pull request

## 🔒 Security Considerations

### Data Protection
- All sensitive data is encrypted at rest
- Memory is cleared after use
- Secure key derivation and storage
- Regular security audits

### Privacy Compliance
- Configurable data retention policies
- Automatic sensitive data detection
- User consent mechanisms
- Data anonymization options

### Access Control
- Multi-factor authentication support
- Role-based permissions
- Audit logging
- Session management

## 📈 Performance Optimization

### Memory Management
- Efficient buffering strategies
- Automatic garbage collection
- Memory usage monitoring
- Resource leak detection

### CPU Optimization
- Event batching
- Asynchronous processing
- Thread pool management
- Performance profiling

### Storage Efficiency
- Log compression
- Automatic cleanup
- Size-based rotation
- Efficient file formats

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
# Validate configuration
python -c "from core.config_manager import ConfigManager; cm = ConfigManager('config.json'); print('Valid' if cm.validate_config() else 'Invalid')"
```

#### Performance Issues
- Reduce buffer sizes in configuration
- Disable unnecessary features
- Increase flush intervals
- Monitor system resources

### Debug Mode
```bash
# Enable debug logging
python main.py --debug

# Verbose output
python main.py --verbose
```

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

This project is licensed under the MIT License - see the LICENSE file for details.

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