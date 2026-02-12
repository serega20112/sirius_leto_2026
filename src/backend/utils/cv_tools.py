import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_overlays(frame, tracking_result):
    students = tracking_result.get("students", [])
    h, w = frame.shape[:2]

    for student in students:
        bbox = student.get("bbox")
        name = student.get("name", "Unknown")
        engagement = student.get("engagement", "unknown")

        # безопасная проверка bbox
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            continue

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        if name == "Unknown" or name is None:
            color = (0, 0, 255)  # red for unknown
        else:
            if engagement == "high":
                color = (0, 255, 0)  # green
            elif engagement == "medium":
                color = (0, 255, 255)  # yellow
            elif engagement == "low":
                color = (0, 0, 255)  # red
            else:
                color = (255, 255, 255)  # white for unknown engagement

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} | {str(engagement).upper()}"
        frame = draw_russian_text(frame, label, (x1, max(0, y1 - 30)), color)

    return frame


def draw_russian_text(img, text, position, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except Exception:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
