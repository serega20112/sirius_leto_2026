from typing import Any, Protocol


class PersonTracker(Protocol):
    def track_people(self, frame: Any) -> list[dict]:
        """
        Tracks people.
        
        Args:
            frame: Input value for `frame`.
        
        Returns:
            The computed or transformed result.
        """
        ...


class FaceRecognizerContract(Protocol):
    def recognize(self, face_img: Any, track_id: int | None = None) -> str | None:
        """
        Runs the operation recognize.
        
        Args:
            face_img: Input value for `face_img`.
            track_id: Input value for `track_id`.
        
        Returns:
            The function result.
        """
        ...

    def refresh_db(self) -> None:
        """
        Refreshes db.
        
        Args:
            None.
        
        Returns:
            Does not return a value.
        """
        ...

    def detect_faces(self, frame: Any) -> list[dict]:
        """
        Runs the operation detect faces.
        
        Args:
            frame: Input value for `frame`.
        
        Returns:
            The function result.
        """
        ...


class EngagementEstimator(Protocol):
    def estimate_engagement(
        self,
        frame: Any,
        bbox: list[int],
        track_id: int | None = None,
        face_bbox: list[int] | None = None,
    ) -> str:
        """
        Estimates engagement.
        
        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            track_id: Input value for `track_id`.
            face_bbox: Input value for `face_bbox`.
        
        Returns:
            The computed or transformed result.
        """
        ...

    def forget_track(self, track_id: int) -> None:
        """
        Clears track.
        
        Args:
            track_id: Input value for `track_id`.
        
        Returns:
            Does not return a value.
        """
        ...
