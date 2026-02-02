import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_overlays(frame, tracking_result):
    """Рисует рамки и русский текст на кадре."""
    students = tracking_result.get("students", [])

    for student in students:
        bbox = student["bbox"]
        name = student["name"]
        engagement = student["engagement"]

        x1, y1, x2, y2 = map(int, bbox)

        color = (0, 255, 0)
        if engagement == "low":
            color = (0, 0, 255)
        elif name == "Unknown":
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} | {engagement.upper()}"
        frame = draw_russian_text(frame, label, (x1, y1 - 30), color)

    return frame


def draw_russian_text(img, text, position, color):
    """Вспомогательная функция для отрисовки кириллицы."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
