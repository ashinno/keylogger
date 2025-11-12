#!/usr/bin/env python3
"""Minimal Flask test to isolate web interface issues."""

from flask import Flask, render_template, request
import os

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

app.secret_key = 'test-key'

# --- Simple runtime state for dashboard preview ---
RUNNING_STATE = False
SESSION_STATS = {
    'keystrokes': 0,
    'mouse_clicks': 0,
    'clipboard_events': 0,
    'window_changes': 0,
}

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

@app.route('/dashboard')
def dashboard():
    """Stub dashboard route for preview."""
    try:
        # Provide minimal context expected by the template to avoid Jinja errors
        status = {'running': RUNNING_STATE}
        stats = {
            'events_logged': SESSION_STATS.get('keystrokes', 0) + SESSION_STATS.get('mouse_clicks', 0),
            'uptime': '0:00:00',
            'memory_usage': '0 MB'
        }
        features = {
            'keyboard': True,
            'mouse': True,
            'clipboard': True,
            'window_tracking': True,
            'camera': False
        }
        system_info = {
            'platform': 'Windows',
            'python_version': '3.x',
            'cpu_usage': 'N/A',
            'available_memory': 'N/A',
            'disk_space': 'N/A'
        }
        recent_events = []
        return render_template('dashboard.html',
                               status=status,
                               stats=stats,
                               features=features,
                               system_info=system_info,
                               recent_events=recent_events)
    except Exception as e:
        return f'<h1>Dashboard</h1><p>Error rendering template: {e}</p>'

@app.route('/logs')
def logs():
    """Stub logs route for preview."""
    try:
        return render_template('logs.html')
    except Exception:
        return '<h1>Logs</h1><p>Stub route.</p>'

@app.route('/camera')
def camera():
    """Stub camera route for preview."""
    try:
        return render_template('camera.html')
    except Exception:
        return '<h1>Camera</h1><p>Stub route.</p>'

@app.route('/config')
def config():
    """Stub config route for preview."""
    try:
        return render_template('config.html')
    except Exception:
        return '<h1>Config</h1><p>Stub route.</p>'

@app.route('/performance')
def performance():
    """Stub performance route for preview."""
    try:
        return render_template('performance.html')
    except Exception:
        return '<h1>Performance</h1><p>Stub route.</p>'

@app.route('/ml-dashboard')
def ml_dashboard():
    """Stub ML dashboard route for preview."""
    try:
        return render_template('ml_dashboard.html')
    except Exception:
        return '<h1>ML Dashboard</h1><p>Stub route.</p>'

@app.route('/export')
def export():
    """Stub export route for preview."""
    try:
        return render_template('export.html')
    except Exception:
        return '<h1>Export</h1><p>Stub route.</p>'

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

@app.route('/interpretability-dashboard')
def interpretability_dashboard():
    """Render interpretability dashboard for preview."""
    try:
        return render_template('interpretability_dashboard.html')
    except Exception as e:
        return f'<h1>Error</h1><p>{e}</p>'

# --- Preview-only stub APIs to avoid JS errors on the dashboard ---
@app.route('/api/ml/interpretability/latest')
def interpretability_latest_stub():
    return {
        'status': 'success',
        'timestamp': '2024-01-15T10:30:00',
        'prediction': 0,
        'confidence': 0.82,
        'confidence_level': 'high',
        'reliability_score': 0.78,
        'confidence_metrics': {'margin': 0.35},
        'explanations': {
            'uncertainty': {'entropy': 1.2},
            'shap': {
                'feature_importance': [
                    {'feature': 'dwell_time', 'shap_value': 0.12, 'abs_importance': 0.12, 'rank': 1},
                    {'feature': 'flight_time', 'shap_value': -0.10, 'abs_importance': 0.10, 'rank': 2},
                    {'feature': 'is_weekend', 'shap_value': 0.06, 'abs_importance': 0.06, 'rank': 3}
                ],
                'base_value': 0.5,
                'total_impact': 0.28
            },
            'lime': {
                'feature_importance': [
                    {'feature': 'dwell_time', 'lime_value': 0.15, 'abs_importance': 0.15, 'rank': 1},
                    {'feature': 'is_weekend', 'lime_value': -0.07, 'abs_importance': -0.07, 'rank': 2}
                ]
            },
            'feature_importance': {
                'built_in': [
                    {'feature': 'dwell_time', 'importance': 0.18, 'rank': 1},
                    {'feature': 'flight_time', 'importance': 0.12, 'rank': 2}
                ]
            }
        },
        'behavioral_context': {
            'event_type': 'keyboard',
            'timestamp': '2024-01-15T10:30:00',
            'anomaly_score': 0.12,
            'ensemble_agreement': 0.85
        }
    }

@app.route('/api/ml/interpretability/statistics')
def interpretability_statistics_stub():
    return {
        'status': 'success',
        'statistics': {
            'explanations_generated': 42,
            'shap_explanations': 20,
            'lime_explanations': 15,
            'feature_importance_calculated': 30,
            'decision_paths_extracted': 10,
            'total_predictions': 120,
            'high_confidence_predictions': 80,
            'low_confidence_predictions': 12,
            'average_confidence': 0.76,
            'confidence_distribution': {'high': 60, 'medium': 40, 'low': 20}
        },
        'calibrated_models': 2,
        'timestamp': '2024-01-15T10:31:00'
    }

@app.route('/api/ml/interpretability/historical')
def interpretability_historical_stub():
    # Basic mocked trend for preview
    base_ts = '2024-01-15T10:30:00'
    timestamps = [base_ts]
    confidence_scores = [0.78]
    uncertainty_scores = [0.22]
    return {
        'status': 'success',
        'historical_data': {
            'timestamps': timestamps,
            'confidence_scores': confidence_scores,
            'uncertainty_scores': uncertainty_scores
        },
        'timestamp': '2024-01-15T10:31:00'
    }

# --- ML Dashboard stub APIs ---
@app.route('/api/ml/status')
def ml_status_stub():
    return {
        'success': True,
        'ml_components': {
            'behavioral_analytics': {'events_processed': 120, 'anomalies_detected': 2},
            'keystroke_dynamics': {'enrolled_users': 1, 'sessions': 1},
            'insider_threat': {'alerts': 0},
            'risk_scoring': {'current_risk': 0.12}
        },
        'timestamp': '2024-01-15T10:32:00'
    }

@app.route('/api/ml/behavioral/baseline')
def behavioral_baseline_stub():
    return {
        'success': True,
        'baseline': {
            'status': 'established',
            'samples': 200,
            'features': 25,
            'feature_stats': {'dwell_time': {'mean': 0.12, 'std': 0.03}},
            'last_update': '2024-01-15T10:30:00'
        },
        'timestamp': '2024-01-15T10:32:00'
    }

@app.route('/api/ml/threat/summary')
def threat_summary_stub():
    return {
        'success': True,
        'threat_summary': {
            'categories': {'exfiltration': 0, 'privilege_abuse': 0, 'policy_violation': 0},
            'recent_events': []
        },
        'timestamp': '2024-01-15T10:32:00'
    }

@app.route('/api/ml/risk/current')
def risk_current_stub():
    return {
        'success': True,
        'risk_status': {
            'current_score': 0.18,
            'risk_level': 'low',
            'risk_factors': {'behavioral': 0.12, 'keystroke': 0.05},
            'trend': 'stable',
            'recent_alerts': 0,
            'system_status': 'ok'
        },
        'timestamp': '2024-01-15T10:32:00'
    }

@app.route('/api/performance')
def performance_api_stub():
    return {
        'system': {
            'cpu_percent': 22.5,
            'memory_percent': 58.2,
            'disk_percent': 71.4,
            'load_average': 0.0
        },
        'keylogger': {
            'uptime': '1h 12m',
            'events_per_hour': 45.3,
            'buffer_size': 12,
            'log_file_size_mb': 3.27,
            'events_logged': 200,
            'errors': 0
        },
        'components': {
            'keyboard': {'events': 120, 'last_activity': 'just now'},
            'mouse': {'events': 60, 'last_activity': '2m ago'},
            'clipboard': {'events': 20, 'last_activity': '15m ago'},
            'window': {'events': 80, 'last_activity': '5m ago'}
        }
    }

# --- Additional stubs for Dashboard, Logs, Camera, Config, Export ---
@app.route('/api/status')
def api_status_stub():
    from datetime import datetime
    return {
        'success': True,
        'status': 'Running' if RUNNING_STATE else 'Stopped',
        'running': RUNNING_STATE,
        'session_stats': SESSION_STATS,
        'timestamp': datetime.now().isoformat()
    }

@app.route('/api/start', methods=['POST'])
def api_start_stub():
    global RUNNING_STATE, SESSION_STATS
    RUNNING_STATE = True
    # Simulate some activity
    SESSION_STATS['keystrokes'] += 5
    SESSION_STATS['mouse_clicks'] += 2
    return {'success': True, 'message': 'Keylogger started', 'running': RUNNING_STATE, 'stats': SESSION_STATS}

@app.route('/api/stop', methods=['POST'])
def api_stop_stub():
    global RUNNING_STATE
    RUNNING_STATE = False
    return {'success': True, 'message': 'Keylogger stopped', 'running': RUNNING_STATE, 'stats': SESSION_STATS}

@app.route('/api/logs')
def api_logs_stub():
    # Provide minimal JSON structure used by logs page
    return {
        'logs': [
            {'timestamp': '2024-01-15T10:32:00', 'type': 'keyboard', 'content': 'a'},
            {'timestamp': '2024-01-15T10:32:05', 'type': 'mouse', 'content': 'click'},
            {'timestamp': '2024-01-15T10:32:10', 'type': 'clipboard', 'content': 'copied text'}
        ],
        'pagination': {'page': 1, 'per_page': 50, 'total': 3, 'pages': 1}
    }

@app.route('/api/camera/stats')
def api_camera_stats_stub():
    return {
        'success': True,
        'stats': {
            'enabled': True,
            'is_running': False,
            'videos_recorded': 3,
            'videos_encrypted': 1,
            'errors': 0,
            'average_size_bytes': 5_200_000,
            'video_directory': 'data/camera'
        }
    }

@app.route('/api/camera/videos')
def api_camera_videos_stub():
    limit = int(request.args.get('limit', '10')) if request.args.get('limit') else 10
    base_ts = 1736927400  # example epoch seconds
    videos = []
    for i in range(min(limit, 3)):
        videos.append({
            'filename': f'video_{i}.mp4',
            'metadata': {
                'timestamp': base_ts + i * 60,
                'resolution': [1280, 720],
                'file_size': 5_200_000,
                'encrypted': (i % 2 == 0),
                'duration_seconds': 30 + i * 5
            }
        })
    return {'success': True, 'videos': videos}

@app.route('/api/config', methods=['GET', 'POST'])
def api_config_stub():
    if request.method == 'POST':
        return {'success': True, 'message': 'Configuration updated'}
    return {
        'success': True,
        'config': {
            'web': {'enabled': True},
            'ml': {'interpretability': {'enabled': True}},
            'logging': {'level': 'INFO'}
        }
    }

@app.route('/api/export/preview', methods=['POST'])
def api_export_preview_stub():
    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = {}
    return {
        'success': True,
        'preview': {
            'format': payload.get('format', 'json'),
            'estimated_size_mb': 1.2,
            'items': [{'type': 'log', 'count': 200}, {'type': 'video', 'count': 3}]
        }
    }

if __name__ == '__main__':
    print("Starting minimal Flask test server...")
    print("Access at: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
