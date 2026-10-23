"""Real loopback HTTP validation; stores responses and stops only its child server."""
import base64
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
KEY = 'local-validation-only-20261021'
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else None
PHASE = 'derainer' if CHECKPOINT else 'base'
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def multipart(fields=(), files=()):
    boundary = 'lingrui-validation-boundary-20261021'
    chunks = []
    for key, value in fields:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for name, filename, content, mime in files:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + content + b'\r\n')
    chunks.append(f'--{boundary}--\r\n'.encode())
    return b''.join(chunks), f'multipart/form-data; boundary={boundary}'


def png(width, height):
    b = io.BytesIO()
    Image.new('RGB', (width, height), (87, 123, 175)).save(b, 'PNG')
    return b.getvalue()


def call(name, method, path, expected, data=None, content_type=None, key=KEY, expected_json=None):
    headers = {'X-API-Key': key} if key is not None else {}
    if content_type:
        headers['Content-Type'] = content_type
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    started = time.monotonic()
    try:
        response = OPENER.open(request, timeout=120)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        status = response.status
        response_headers = dict(response.headers)
    (OUT / f'{PHASE}_{name}.response.body').write_bytes(raw)
    try:
        decoded = json.loads(raw)
    except ValueError:
        decoded = None
    passed = status == expected and response_headers.get('Content-Type', '').startswith('application/json')
    if expected_json is not None:
        passed = passed and decoded == expected_json
    if status >= 400:
        passed = passed and isinstance(decoded, dict) and isinstance(decoded.get('error'), str)
    entry = {'name': name, 'method': method, 'path': path, 'request_content_type': content_type,
             'request_bytes': len(data or b''), 'request_key': 'omitted' if key is None else ('valid' if key == KEY else 'wrong'),
             'expected_status': expected, 'status': status, 'response_headers': response_headers,
             'response': decoded, 'elapsed_seconds': round(time.monotonic() - started, 4), 'passed': passed}
    (OUT / f'{PHASE}_{name}.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding='utf-8')
    RESULTS.append(entry)
    print(f'{name}: HTTP {status}, expected {expected}, passed={passed}', flush=True)
    return decoded


with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
BASE = f'http://127.0.0.1:{port}'
code = 'from app import create_app; '
if CHECKPOINT:
    code += f'from model_service import Derainer; predictor=Derainer({CHECKPOINT!r}, "cpu"); '
    code += f'create_app(predictor=predictor, api_key={KEY!r})'
else:
    code += f'create_app(api_key={KEY!r})'
code += f'.run(host="127.0.0.1", port={port}, debug=False, use_reloader=False)'
summary = {'phase': PHASE, 'base_url': BASE, 'checkpoint': CHECKPOINT, 'python': sys.executable,
           'model': 'derainer on CPU' if CHECKPOINT else 'none', 'framework': 'Flask development server; local validation only'}
log = (OUT / f'{PHASE}_server.log').open('w', encoding='utf-8')
process = subprocess.Popen([sys.executable, '-u', '-c', code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
summary['pid'] = process.pid
try:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Server exited {process.returncode}; see server log')
        try:
            with OPENER.open(BASE + '/health', timeout=1) as r:
                if r.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    else:
        raise RuntimeError('Server readiness timed out')
    call('health', 'GET', '/health', 200, key=None, expected_json={'status': 'ok', 'model_ready': bool(CHECKPOINT)})
    if CHECKPOINT:
        input_image = png(32, 40)
        (OUT / 'derainer_input.png').write_bytes(input_image)
        data, mime = multipart(files=[('file', 'image.png', input_image, 'image/png')])
        output = call('predict', 'POST', '/predict', 200, data, mime)
        if output and 'image_base64' in output:
            result = base64.b64decode(output['image_base64'], validate=True)
            (OUT / 'derainer_http_output.png').write_bytes(result)
            with Image.open(io.BytesIO(result)) as image:
                assert image.size == (32, 40) and image.format == 'PNG'
            assert output['device'] == 'cpu' and output['width'] == 32 and output['height'] == 40
            summary['decoded_output_verified'] = True
    else:
        call('index', 'GET', '/', 200, key=None)
        for op, result in [('sum', 10), ('subtract', 6), ('multiply', 16), ('divide', 4)]:
            fields = [('Operation', op), ('Data_A', '8'), ('Data_B', '2')]
            data, mime = multipart(fields=fields)
            call(op + '_multipart', 'POST', '/calculate', 200, data, mime, expected_json={'Result': result})
            call(op + '_json', 'POST', '/calculate', 200, json.dumps(dict(fields)).encode(), 'application/json', expected_json={'Result': result})
        form = urllib.parse.urlencode({'Operation': 'sum', 'Data_A': 3, 'Data_B': 4}).encode()
        call('root_urlencoded', 'POST', '/', 200, form, 'application/x-www-form-urlencoded', expected_json={'Result': 7})
        call('missing_key', 'POST', '/calculate', 401, form, 'application/x-www-form-urlencoded', key=None)
        call('wrong_key', 'POST', '/calculate', 401, form, 'application/x-www-form-urlencoded', key='wrong')
        for name, body in [
            ('missing_field', {'Operation': 'sum', 'Data_A': 1}),
            ('divide_zero', {'Operation': 'divide', 'Data_A': 1, 'Data_B': 0}),
            ('bad_operation', {'Operation': 'power', 'Data_A': 1, 'Data_B': 2}),
            ('nan', {'Operation': 'sum', 'Data_A': 'NaN', 'Data_B': 2}),
            ('bool', {'Operation': 'sum', 'Data_A': True, 'Data_B': 2}),
            ('overflow', {'Operation': 'multiply', 'Data_A': '1e308', 'Data_B': 2}),
            ('json_array', []),
        ]:
            call(name, 'POST', '/calculate', 400, json.dumps(body).encode(), 'application/json')
        call('malformed_json', 'POST', '/calculate', 400, b'{', 'application/json')
        call('wrong_calculate_type', 'POST', '/calculate', 415, b'hello', 'text/plain')
        data, mime = multipart(fields=[('Operation', 'sum'), ('Data_A', 1), ('Data_A', 2), ('Data_B', 3)])
        call('duplicate_field', 'POST', '/calculate', 400, data, mime)
        call('wrong_upload_type', 'POST', '/predict', 415, b'{}', 'application/json')
        data, mime = multipart(fields=[('note', 'missing file')])
        call('missing_file', 'POST', '/predict', 400, data, mime)
        for name, content, expected in [('invalid_image', b'not image', 400), ('empty_file', b'', 400), ('valid_image_no_model', png(32, 40), 503), ('oversize_pixels', png(2001, 2000), 400), ('oversize_body', b'x' * (8 * 1024 * 1024 + 1), 413)]:
            data, mime = multipart(files=[('file', name + '.png', content, 'image/png')])
            call(name, 'POST', '/predict', expected, data, mime)
        call('not_found', 'GET', '/absent', 404)
        call('wrong_method', 'GET', '/calculate', 405)
    summary['checks'] = len(RESULTS)
    summary['passed'] = sum(r['passed'] for r in RESULTS)
    summary['failed'] = sum(not r['passed'] for r in RESULTS)
finally:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    log.close()
    summary['server_stopped'] = process.poll() is not None
    summary['server_exit_code'] = process.returncode
    (OUT / f'{PHASE}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary.get('failed') or not summary.get('checks'):
    raise SystemExit(1)
