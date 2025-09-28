#!/usr/bin/env python3
"""Minimal Flask test to isolate web interface issues."""

from flask import Flask, render_template
import os

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

app.secret_key = 'test-key'

@app.route('/')
def home():
    """Simple home page."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Keylogger Web Interface Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { background: #e8f5e8; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 Keylogger Web Interface Test</h1>
            <div class="status">
                <h2>✅ Web Interface is Working!</h2>
                <p>The Flask server is running correctly.</p>
                <p><strong>Next steps:</strong></p>
                <ul>
                    <li>Test login functionality</li>
                    <li>Check template rendering</li>
                    <li>Verify static assets</li>
                </ul>
            </div>
            <p><a href="/login">Go to Login Page</a></p>
        </div>
    </body>
    </html>
    '''

@app.route('/login')
def login():
    """Test login page."""
    try:
        return render_template('login.html')
    except Exception as e:
        return f'''
        <h1>Template Test</h1>
        <p>Error loading login.html template: {e}</p>
        <p>Template folder: {app.template_folder}</p>
        <p>Current directory: {os.getcwd()}</p>
        <p><a href="/">Back to Home</a></p>
        '''

@app.route('/test')
def test():
    """Test endpoint."""
    return {
        'status': 'ok',
        'message': 'Web interface is working',
        'template_folder': app.template_folder,
        'static_folder': app.static_folder,
        'current_dir': os.getcwd()
    }

if __name__ == '__main__':
    print("Starting minimal Flask test server...")
    print("Access at: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)