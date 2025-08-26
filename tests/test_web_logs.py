"""Tests for /api/logs filtering and pagination using Flask test client."""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web.app import create_web_app  # type: ignore


class StubLoggingManager:
    def __init__(self, entries):
        self._entries = entries

    def get_buffer_entries(self):
        return list(self._entries)


class StubConfig:
    def __init__(self, overrides=None):
        self._data = {
            'web.secret_key': 'test-secret',
            'web.debug': False,
            'web.admin_username': 'admin',
            'web.admin_password': 'admin123',
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key, default=None):
        return self._data.get(key, default)


class StubCore:
    def __init__(self, logging_manager):
        self.logging_manager = logging_manager
        self.config = StubConfig()

    def get_session_stats(self):
        # Minimal stats to satisfy any usage in routes if accidentally hit
        return {
            'uptime': 0,
            'total_events': 0,
            'running': True,
        }


def _make_log(timestamp: datetime, event_type: str, message: str) -> str:
    # Format expected by _get_recent_logs: "timestamp: event_type: message"
    return f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {event_type}: {message}"


def build_seed_entries():
    now = datetime.now()
    entries = []
    # Create a mix of events over the last 60 seconds
    types_counts = {
        'Keyboard': 15,
        'Mouse': 10,
        'Clipboard': 8,
        'Window': 5,
        'System': 4,
    }
    counter = 0
    for etype, count in types_counts.items():
        for i in range(count):
            ts = now - timedelta(seconds=counter)
            entries.append(_make_log(ts, etype, f"message {etype} {i}"))
            counter += 1
    # Entries order doesn't matter; the API sorts by timestamp string desc
    return entries


class TestWebLogsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_entries = build_seed_entries()
        cls.total_entries = len(seed_entries)
        cls.counts = {
            'Keyboard': 15,
            'Mouse': 10,
            'Clipboard': 8,
            'Window': 5,
            'System': 4,
        }
        logging_manager = StubLoggingManager(seed_entries)
        cls.core = StubCore(logging_manager)
        cls.config = StubConfig()
        cls.app = create_web_app(cls.core, cls.config)
        cls.app.testing = True
        cls.client = cls.app.test_client()

        # Login once for all tests
        cls.client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)

    def test_fetch_all_logs(self):
        resp = self.client.get('/api/logs')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('logs', data)
        self.assertIn('pagination', data)
        self.assertEqual(data['pagination']['total'], self.total_entries)
        self.assertGreater(len(data['logs']), 0)

    def test_filter_keyboard(self):
        resp = self.client.get('/api/logs?type=Keyboard')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['pagination']['total'], self.counts['Keyboard'])
        # Ensure all returned are of the requested type
        for item in data['logs']:
            self.assertIn('Keyboard', item['type'])

    def test_filter_mouse_pagination(self):
        per_page = 5
        page = 2
        resp = self.client.get(f'/api/logs?type=Mouse&per_page={per_page}&page={page}')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        # Total should match all Mouse entries
        self.assertEqual(data['pagination']['total'], self.counts['Mouse'])
        # Page metadata
        pages = (self.counts['Mouse'] + per_page - 1) // per_page
        self.assertEqual(data['pagination']['pages'], pages)
        self.assertEqual(data['pagination']['page'], page)
        # Page 2 should have per_page items unless last page smaller
        expected_len = per_page if self.counts['Mouse'] > per_page else self.counts['Mouse']
        # If last page and remainder smaller, compute accordingly
        if page == pages and (self.counts['Mouse'] % per_page) != 0:
            expected_len = self.counts['Mouse'] % per_page
        self.assertEqual(len(data['logs']), expected_len)
        # Ensure all returned are Mouse
        for item in data['logs']:
            self.assertIn('Mouse', item['type'])

    def test_unknown_filter_returns_empty(self):
        resp = self.client.get('/api/logs?type=DoesNotExist')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['pagination']['total'], 0)
        self.assertEqual(len(data['logs']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)