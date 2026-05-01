import json
import urllib.request
import time

BASE = 'http://127.0.0.1:8000'
TESTS = [
    ('hello', {'message': 'Hello', 'mode': 'auto', 'max_steps': 10}),
    ('find_projects', {'message': '/task Find my github directory and list projects in it', 'mode': 'auto', 'max_steps': 10}),
    ('create_file', {'message': '/task Create a file named smoke_test.txt in the SlothBrain project directory with the text hello from slothbrain', 'mode': 'auto', 'max_steps': 10}),
    ('read_file', {'message': '/task Read the file smoke_test.txt in the SlothBrain project directory', 'mode': 'auto', 'max_steps': 10}),
    ('research', {'message': '/task Research sun hydraulics llc and provide sources', 'mode': 'auto', 'max_steps': 10}),
    ('sub_agent', {'message': '/task Use a sub agent to summarize what this repository is for', 'mode': 'auto', 'max_steps': 10}),
]

def post(payload):
    req = urllib.request.Request(
        BASE + '/api/chat',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode('utf-8'))

for name, payload in TESTS:
    started = time.time()
    try:
        result = post(payload)
        print('###', name)
        print(json.dumps({'agent': result.get('agent'), 'response': result.get('response'), 'duration': round(time.time()-started, 2)}, ensure_ascii=False))
    except Exception as exc:
        print('###', name)
        print(json.dumps({'error': str(exc), 'duration': round(time.time()-started, 2)}, ensure_ascii=False))
