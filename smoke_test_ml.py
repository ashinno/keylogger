import requests
import json

BASE = 'http://127.0.0.1:5000'
s = requests.Session()


def log(msg: str):
    print(msg, flush=True)


def preview_json(data):
    try:
        text = json.dumps(data, indent=2, default=str)
        return text if len(text) <= 600 else text[:600] + '...'
    except Exception:
        text = str(data)
        return text if len(text) <= 600 else text[:600] + '...'


def test(method: str, path: str, payload=None):
    url = BASE + path
    try:
        if method.upper() == 'GET':
            r = s.get(url, timeout=10)
        else:
            r = s.post(url, json=payload, timeout=10)
        ct = r.headers.get('content-type', '')
        try:
            data = r.json() if 'application/json' in ct else r.text
        except Exception:
            data = r.text
        log(f"[{method}] {path} -> HTTP {r.status_code}")
        if isinstance(data, (dict, list)):
            log(preview_json(data))
        else:
            body = data
            if isinstance(body, bytes):
                body = body.decode('utf-8', errors='ignore')
            body = body if len(body) <= 600 else body[:600] + '...'
            log(body)
    except Exception as e:
        log(f"[{method}] {path} -> ERROR {e}")


# Login
try:
    r = s.post(BASE + '/login', data={'username': 'admin', 'password': 'admin123'}, allow_redirects=False, timeout=10)
    log(f"Login status: HTTP {r.status_code}; session_cookie={bool(s.cookies.get_dict().get('session'))}")
except Exception as e:
    log(f"Login failed: {e}")

# Smoke tests
endpoints = [
    ('GET', '/api/ml/status', None),
    ('GET', '/api/ml/behavioral/baseline', None),
    ('POST', '/api/ml/keystroke/enroll', {'user_id': 'test_user', 'typing_samples': [{'durations': [100, 120, 110], 'keys': 'test'}]}),
    ('GET', '/api/ml/threat/summary', None),
    ('GET', '/api/ml/risk/current', None),
    ('GET', '/api/ml/risk/alerts', None),
    ('POST', '/api/ml/analytics/events', {'events': [{'id': 'e1', 'timestamp': '2025-09-13T14:16:00Z', 'type': 'keyboard', 'key': 'A'}]}),
    ('GET', '/api/ml/models/status', None),
    ('POST', '/api/ml/export/data', {'type': 'all'}),
]

for method, path, payload in endpoints:
    test(method, path, payload)