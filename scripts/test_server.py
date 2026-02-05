import sys
from pathlib import Path
# Добавляем корень репозитория в sys.path чтобы импорт src работал при запуске скрипта напрямую
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.backend.create_app import create_app
import io

app = create_app()
client = app.test_client()

print('GET /')
r = client.get('/')
print(r.status_code, r.content_type)
print(r.data[:200].decode('utf-8', errors='ignore'))

print('\nGET /static/js/base.js')
r = client.get('/static/js/base.js')
print(r.status_code, r.content_type, 'len=', len(r.data))
print(r.data[:200].decode('utf-8', errors='ignore'))

print('\nGET /static/static.css')
r = client.get('/static/static.css')
print(r.status_code, r.content_type, 'len=', len(r.data))
print(r.data[:200].decode('utf-8', errors='ignore'))

print('\nPOST /api/v1/auth/register (multipart)')
data = {
    'name': 'Test Student',
    'group': 'T-1',
    'photo': (io.BytesIO(b'testimagebytes'), 'photo.jpg')
}
# When using test_client, do not set content_type so it constructs multipart form data
r = client.post('/api/v1/auth/register', data=data, content_type=None)
print('status:', r.status_code)
print('response:', r.get_data(as_text=True))
