import cv2


def draw_overlays(frame, tracking_result):
    """
    Рисует рамки и текст на кадре.
    tracking_result: Результат работы TrackAttendanceUseCase.
    """
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
        label = f"{name} ({engagement})"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return frame
