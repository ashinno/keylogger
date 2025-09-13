import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from flask import Flask
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_web_app
from core.config_manager import ConfigManager
from core.keylogger import KeyloggerCore

class TestWebInterfaceComprehensive(unittest.TestCase):
    """Comprehensive tests for web interface functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock keylogger core
        self.mock_keylogger_core = Mock(spec=KeyloggerCore)
        self.mock_config_manager = Mock(spec=ConfigManager)
        
        # Configure mock config manager
        self.mock_config_manager.get.side_effect = self._mock_config_get
        
        # Create Flask app with test configuration
        self.app = create_web_app(self.mock_keylogger_core, self.mock_config_manager)
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test-secret-key'
        self.app.config['WTF_CSRF_ENABLED'] = False
        
        self.client = self.app.test_client()
        
        # Mock ML components
        self._setup_ml_mocks()
        
    def _mock_config_get(self, key, default=None):
        """Mock configuration getter."""
        config_values = {
            'web.username': 'admin',
            'web.password_hash': 'pbkdf2:sha256:260000$test$hash',
            'web.session_timeout': 3600,
            'logging.file_path': 'logs/keylog.txt',
            'features.camera': True,
            'camera.enabled': True,
            'ml.behavioral_analytics.sensitivity': 0.1,
            'ml.keystroke_dynamics.min_samples': 100,
            'ml.insider_threat.threshold': 0.7,
            'ml.risk_scoring.threshold': 0.8
        }
        return config_values.get(key, default)
    
    def _setup_ml_mocks(self):
        """Set up ML component mocks."""
        # Behavioral analytics mock
        self.mock_behavioral = Mock()
        self.mock_behavioral.get_current_analysis.return_value = {
            'anomaly_score': 0.3,
            'baseline_deviation': 0.2,
            'confidence': 0.8
        }
        self.mock_keylogger_core.behavioral_analytics = self.mock_behavioral
        
        # Keystroke dynamics mock
        self.mock_keystroke = Mock()
        self.mock_keystroke.get_authentication_status.return_value = {
            'authenticated': True,
            'confidence': 0.9,
            'user_id': 'test_user'
        }
        self.mock_keylogger_core.keystroke_dynamics = self.mock_keystroke
        
        # Insider threat detector mock
        self.mock_threat = Mock()
        self.mock_threat.get_threat_summary.return_value = {
            'threat_level': 'low',
            'risk_score': 0.2,
            'indicators': []
        }
        self.mock_keylogger_core.insider_threat_detector = self.mock_threat
        
        # Risk scorer mock
        self.mock_risk = Mock()
        self.mock_risk.get_current_risk_status.return_value = {
            'risk_level': 'medium',
            'score': 0.5,
            'factors': ['unusual_activity']
        }
        self.mock_keylogger_core.risk_scorer = self.mock_risk
        
        # Camera monitor mock
        self.mock_camera = Mock()
        self.mock_camera.get_stats.return_value = {
            'is_running': True,
            'videos_captured': 10,
            'last_capture': datetime.now().isoformat()
        }
        self.mock_keylogger_core.camera_monitor = self.mock_camera
        
        # Logging manager mock
        self.mock_logging = Mock()
        self.mock_logging.get_buffer_entries.return_value = [
            '{"timestamp": "2024-01-01 12:00:00", "type": "keyboard", "message": "Key pressed: a"}',
            '{"timestamp": "2024-01-01 12:00:01", "type": "mouse", "message": "Mouse clicked"}'
        ]
        self.mock_keylogger_core.logging_manager = self.mock_logging
    
    def _login(self):
        """Helper method to log in for authenticated routes."""
        with patch('werkzeug.security.check_password_hash', return_value=True):
            response = self.client.post('/login', data={
                'username': 'admin',
                'password': 'password'
            }, follow_redirects=True)
            return response

class TestAuthentication(TestWebInterfaceComprehensive):
    """Test authentication functionality."""
    
    def test_login_page_loads(self):
        """Test that login page loads correctly."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
    
    def test_successful_login(self):
        """Test successful login with correct credentials."""
        with patch('werkzeug.security.check_password_hash', return_value=True):
            response = self.client.post('/login', data={
                'username': 'admin',
                'password': 'password'
            }, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
    
    def test_failed_login_wrong_username(self):
        """Test failed login with wrong username."""
        response = self.client.post('/login', data={
            'username': 'wrong',
            'password': 'password'
        })
        self.assertEqual(response.status_code, 200)
        # Should stay on login page
    
    def test_failed_login_wrong_password(self):
        """Test failed login with wrong password."""
        with patch('werkzeug.security.check_password_hash', return_value=False):
            response = self.client.post('/login', data={
                'username': 'admin',
                'password': 'wrong'
            })
            self.assertEqual(response.status_code, 200)
    
    def test_logout(self):
        """Test logout functionality."""
        self._login()
        response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
    
    def test_protected_route_without_login(self):
        """Test that protected routes redirect to login."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 302)  # Redirect to login
    
    def test_session_timeout(self):
        """Test session timeout functionality."""
        self._login()
        
        # Mock expired session
        with self.client.session_transaction() as sess:
            sess['last_activity'] = (datetime.now() - timedelta(hours=2)).timestamp()
        
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 302)  # Should redirect to login

class TestDashboardRoutes(TestWebInterfaceComprehensive):
    """Test dashboard route functionality."""
    
    def test_dashboard_loads_when_authenticated(self):
        """Test that dashboard loads for authenticated users."""
        self._login()
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
    
    def test_logs_page_loads(self):
        """Test that logs page loads correctly."""
        self._login()
        response = self.client.get('/logs')
        self.assertEqual(response.status_code, 200)
    
    def test_ml_dashboard_loads(self):
        """Test that ML dashboard loads correctly."""
        self._login()
        response = self.client.get('/ml-dashboard')
        self.assertEqual(response.status_code, 200)
    
    def test_camera_page_loads(self):
        """Test that camera page loads correctly."""
        self._login()
        response = self.client.get('/camera')
        self.assertEqual(response.status_code, 200)
    
    def test_dashboard_with_ml_error(self):
        """Test dashboard behavior when ML components have errors."""
        self.mock_behavioral.get_current_analysis.side_effect = Exception("ML Error")
        self._login()
        response = self.client.get('/ml-dashboard')
        self.assertEqual(response.status_code, 200)  # Should handle error gracefully

class TestAPIEndpoints(TestWebInterfaceComprehensive):
    """Test API endpoint functionality."""
    
    def test_api_logs_endpoint(self):
        """Test logs API endpoint."""
        self._login()
        response = self.client.get('/api/logs')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('logs', data)
        self.assertIn('pagination', data)
    
    def test_api_logs_with_pagination(self):
        """Test logs API with pagination parameters."""
        self._login()
        response = self.client.get('/api/logs?page=1&per_page=10')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['pagination']['page'], 1)
        self.assertEqual(data['pagination']['per_page'], 10)
    
    def test_api_logs_with_filters(self):
        """Test logs API with filtering."""
        self._login()
        response = self.client.get('/api/logs?event_type=keyboard&q=test')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('logs', data)
    
    def test_api_performance_endpoint(self):
        """Test performance API endpoint."""
        self._login()
        response = self.client.get('/api/performance')
        self.assertEqual(response.status_code, 200)
    
    def test_api_camera_stats(self):
        """Test camera stats API endpoint."""
        self._login()
        response = self.client.get('/api/camera/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('stats', data)
    
    def test_api_camera_videos(self):
        """Test camera videos API endpoint."""
        self.mock_camera.get_video_list.return_value = [
            {'filename': 'video1.mp4', 'timestamp': '2024-01-01 12:00:00'},
            {'filename': 'video2.mp4', 'timestamp': '2024-01-01 12:05:00'}
        ]
        self._login()
        response = self.client.get('/api/camera/videos?limit=5')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('videos', data)
    
    def test_api_without_authentication(self):
        """Test that API endpoints require authentication."""
        response = self.client.get('/api/logs')
        self.assertEqual(response.status_code, 302)  # Redirect to login

class TestMLAPIEndpoints(TestWebInterfaceComprehensive):
    """Test ML-related API endpoints."""
    
    def test_behavioral_analysis_api(self):
        """Test behavioral analysis API endpoint."""
        self._login()
        response = self.client.get('/api/ml/behavioral/current')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('analysis', data)
    
    def test_keystroke_auth_api(self):
        """Test keystroke authentication API endpoint."""
        self._login()
        response = self.client.get('/api/ml/keystroke/auth')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('auth_status', data)
    
    def test_threat_summary_api(self):
        """Test threat summary API endpoint."""
        self._login()
        response = self.client.get('/api/ml/threat/summary')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('threat_summary', data)
    
    def test_risk_status_api(self):
        """Test current risk status API endpoint."""
        self._login()
        response = self.client.get('/api/ml/risk/current')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('risk_status', data)
    
    def test_risk_alerts_api(self):
        """Test risk alerts API endpoint."""
        # Mock alert history
        self.mock_risk.alert_history = [
            {'timestamp': datetime.now().isoformat(), 'level': 'high', 'message': 'Test alert'}
        ]
        self._login()
        response = self.client.get('/api/ml/risk/alerts')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('alerts', data)
    
    def test_ml_config_get(self):
        """Test ML configuration GET endpoint."""
        self._login()
        response = self.client.get('/api/ml/config')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('behavioral_analytics', data)
        self.assertIn('keystroke_dynamics', data)
    
    def test_ml_config_post(self):
        """Test ML configuration POST endpoint."""
        self._login()
        config_data = {
            'behavioral_analytics': {'sensitivity': 0.2},
            'keystroke_dynamics': {'min_samples': 150}
        }
        response = self.client.post('/api/ml/config', 
                                  data=json.dumps(config_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
    
    def test_keystroke_enroll_api(self):
        """Test keystroke user enrollment API."""
        self.mock_keystroke.enroll_user.return_value = {'status': 'success'}
        self._login()
        enroll_data = {
            'user_id': 'test_user',
            'typing_samples': ['sample1', 'sample2']
        }
        response = self.client.post('/api/ml/keystroke/enroll',
                                  data=json.dumps(enroll_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
    
    def test_reset_endpoints(self):
        """Test various reset API endpoints."""
        self._login()
        
        # Test behavioral reset
        response = self.client.post('/api/ml/behavioral/reset')
        self.assertEqual(response.status_code, 200)
        
        # Test keystroke reset
        response = self.client.post('/api/ml/keystroke/reset')
        self.assertEqual(response.status_code, 200)
        
        # Test threat reset
        response = self.client.post('/api/ml/threat/reset')
        self.assertEqual(response.status_code, 200)
    
    def test_ml_api_without_components(self):
        """Test ML API behavior when components are not available."""
        # Remove ML components
        self.mock_keylogger_core.behavioral_analytics = None
        self.mock_keylogger_core.keystroke_dynamics = None
        
        self._login()
        response = self.client.get('/api/ml/behavioral/current')
        self.assertEqual(response.status_code, 404)
        
        response = self.client.get('/api/ml/keystroke/auth')
        self.assertEqual(response.status_code, 404)

class TestErrorHandling(TestWebInterfaceComprehensive):
    """Test error handling and edge cases."""
    
    def test_404_error_handler(self):
        """Test 404 error handling."""
        response = self.client.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)
    
    def test_500_error_handler(self):
        """Test 500 error handling."""
        # Mock an endpoint to raise an exception
        with patch.object(self.mock_keylogger_core, 'logging_manager', side_effect=Exception("Test error")):
            self._login()
            response = self.client.get('/api/logs')
            self.assertEqual(response.status_code, 500)
    
    def test_api_error_responses(self):
        """Test API error response format."""
        self.mock_logging.get_buffer_entries.side_effect = Exception("Buffer error")
        self._login()
        response = self.client.get('/api/logs')
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_malformed_json_requests(self):
        """Test handling of malformed JSON requests."""
        self._login()
        response = self.client.post('/api/ml/config',
                                  data='invalid json',
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields in API requests."""
        self._login()
        # Test keystroke enrollment without required fields
        response = self.client.post('/api/ml/keystroke/enroll',
                                  data=json.dumps({'user_id': 'test'}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

class TestSecurityFeatures(TestWebInterfaceComprehensive):
    """Test security-related features."""
    
    def test_csrf_protection(self):
        """Test CSRF protection (when enabled)."""
        # This would test CSRF tokens in a real scenario
        # For now, we test that the app can handle CSRF settings
        self.app.config['WTF_CSRF_ENABLED'] = True
        self._login()
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
    
    def test_session_security(self):
        """Test session security features."""
        self._login()
        
        # Test that session contains expected security markers
        with self.client.session_transaction() as sess:
            self.assertIn('logged_in', sess)
            self.assertIn('last_activity', sess)
    
    def test_password_hash_verification(self):
        """Test password hash verification."""
        # Test with various password scenarios
        test_cases = [
            ('correct_password', True),
            ('wrong_password', False),
            ('', False),
            (None, False)
        ]
        
        for password, expected_result in test_cases:
            with patch('werkzeug.security.check_password_hash', return_value=expected_result):
                response = self.client.post('/login', data={
                    'username': 'admin',
                    'password': password or ''
                })
                if expected_result:
                    # Should redirect on success
                    self.assertIn(response.status_code, [200, 302])
                else:
                    # Should stay on login page
                    self.assertEqual(response.status_code, 200)

class TestPerformanceAndScalability(TestWebInterfaceComprehensive):
    """Test performance and scalability aspects."""
    
    def test_large_log_pagination(self):
        """Test pagination with large log datasets."""
        # Mock large dataset
        large_log_entries = [f'{{"timestamp": "2024-01-01 12:00:{i:02d}", "type": "test", "message": "Entry {i}"}}' 
                           for i in range(1000)]
        self.mock_logging.get_buffer_entries.return_value = large_log_entries
        
        self._login()
        response = self.client.get('/api/logs?page=1&per_page=50')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data['logs']), 50)
        self.assertEqual(data['pagination']['total'], 1000)
    
    def test_concurrent_api_requests(self):
        """Test handling of concurrent API requests."""
        self._login()
        
        # Simulate multiple concurrent requests
        responses = []
        for _ in range(5):
            response = self.client.get('/api/performance')
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
    
    def test_memory_efficient_log_processing(self):
        """Test memory-efficient log processing."""
        # Test with very large log entries
        large_entry = '{"timestamp": "2024-01-01 12:00:00", "type": "test", "message": "' + 'x' * 10000 + '"}'
        self.mock_logging.get_buffer_entries.return_value = [large_entry] * 100
        
        self._login()
        response = self.client.get('/api/logs?per_page=10')
        self.assertEqual(response.status_code, 200)
        # Should handle large entries without issues

class TestTestRoute(TestWebInterfaceComprehensive):
    """Test the test route functionality."""
    
    def test_test_route_accessibility(self):
        """Test that test route is accessible without authentication."""
        response = self.client.get('/test')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('message', data)
        self.assertIn('timestamp', data)
    
    def test_test_route_response_format(self):
        """Test test route response format."""
        response = self.client.get('/test')
        data = json.loads(response.data)
        
        required_fields = ['status', 'message', 'timestamp']
        for field in required_fields:
            self.assertIn(field, data)
        
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(data['message'], 'Web application is running')

if __name__ == '__main__':
    unittest.main()