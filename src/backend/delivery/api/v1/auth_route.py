from flask import Blueprint, request, jsonify
from src.backend.dependencies.container import container

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    print("--- [AUTH] Начат процесс регистрации ---")

    # Диагностика: покажем content-type и ключи (полезно при пустых request.files)
    print(f"[AUTH] Content-Type: {request.content_type}")
    print(f"[AUTH] form.keys: {list(request.form.keys())}")
    print(f"[AUTH] files.keys: {list(request.files.keys())}")

    # 1. Получаем файл: ожидаем 'photo', но если его нет — возьмём первый файл в request.files (толерантность)
    file = None
    if "photo" in request.files:
        file = request.files["photo"]
    elif len(request.files) == 1:
        # если файл пришёл под другим именем (например браузер/форма), берем первый
        first_key = next(iter(request.files.keys()))
        file = request.files[first_key]
        print(f"[AUTH] Файл найден под ключом '{first_key}', используем его")
    else:
        print("[AUTH] Ошибка: Нет файла фото")
        return jsonify({"error": "No photo uploaded", "debug_files": list(request.files.keys())}), 400

    name = request.form.get("name") or request.form.get("regName")
    group = request.form.get("group") or request.form.get("regGroup")

    print(f"[AUTH] Данные: Имя={name}, Группа={group}, Файл={getattr(file, 'filename', None)}")

    if not name or not group:
        print("[AUTH] Ошибка: Не заполнены имя или группа")
        return jsonify({"error": "Missing name or group", "debug_form": dict(request.form)}), 400

    try:
        # Читаем файл в память
        photo_bytes = file.read()
        print(f"[AUTH] Файл прочитан, размер: {len(photo_bytes)} байт")

        # 2. Вызов логики
        print("[AUTH] Вызываю Use Case...")
        result_student = container.register_use_case.execute(name, group, photo_bytes)

        print(f"[AUTH] УСПЕХ! Студент создан с ID: {result_student.id}")

        return (
            jsonify(
                {"message": "Student registered successfully", "id": result_student.id}
            ),
            201,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[AUTH] КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        return jsonify({"error": str(e)}), 500
