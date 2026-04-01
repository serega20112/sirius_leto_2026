from typing import Any, Protocol


class PersonTracker(Protocol):
    def track_people(self, frame: Any) -> list[dict]:
        ...


class FaceRecognizerContract(Protocol):
    def recognize(self, face_img: Any, track_id: int | None = None) -> str | None:
        ...

    def refresh_db(self) -> None:
        ...

    def detect_faces(self, frame: Any) -> list[dict]:
        ...


class EngagementEstimator(Protocol):
    def estimate_engagement(
        self,
        frame: Any,
        bbox: list[int],
        track_id: int | None = None,
    ) -> str:
        ...

    def forget_track(self, track_id: int) -> None:
        ...
