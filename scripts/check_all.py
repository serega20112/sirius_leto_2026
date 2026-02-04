#!/usr/bin/env python3
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.backend.create_app import create_app
import io

app = create_app()
client = app.test_client()

print("\n" + "="*70)
print("ФИНАЛЬНАЯ ПРОВЕРКА: СТАТИЧЕСКИЕ ФАЙЛЫ И РЕГИСТРАЦИЯ")
print("="*70 + "\n")

# Проверка статических файлов
tests = [
    ('GET / (главная)', '/'),
    ('GET /js/main.js', '/js/main.js'),
    ('GET /static/static.css', '/static/static.css'),
]

print("📦 Проверка статических ресурсов:\n")
for desc, url in tests:
    r = client.get(url)
    status_emoji = "✅" if r.status_code == 200 else "❌"
    content_type = r.content_type if r.status_code == 200 else f"ERROR {r.status_code}"
    print(f"  {status_emoji} {desc:<30} → {content_type}")

# Проверка регистрации
print("\n📝 Проверка регистрации студента:\n")
data = {
    'name': 'Test Student',
    'group': 'T-1',
    'photo': (io.BytesIO(b'test_image_bytes'), 'photo.jpg')
}
r = client.post('/api/v1/auth/register', data=data)
print(f"  POST /api/v1/auth/register")
print(f"  Status: {r.status_code}")
if r.status_code in [200, 201]:
    print(f"  Response: {r.get_data(as_text=True)[:150]}")
    print(f"  ✅ Регистрация работает!")
else:
    print(f"  ❌ Ошибка: {r.get_data(as_text=True)}")

print("\n" + "="*70)
print("✨ Все проверки завершены!")
print("="*70 + "\n")

