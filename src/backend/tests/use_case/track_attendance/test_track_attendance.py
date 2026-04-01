from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.backend.domain.attendance.entity import EngagementStatus
from src.backend.infrastructure.ai.config import AttendanceTrackingConfig
from src.backend.use_case.track_attendance import TrackAttendanceUseCase


def _build_use_case(
    detector=None,
    recognizer=None,
    pose_estimator=None,
    student_repo=None,
    attendance_repo=None,
    config=None,
):
    return TrackAttendanceUseCase(
        person_detector=detector or MagicMock(),
        face_recognizer=recognizer or MagicMock(),
        pose_estimator=pose_estimator or MagicMock(),
        student_repo=student_repo or MagicMock(),
        attendance_repo=attendance_repo or MagicMock(),
        config=config or AttendanceTrackingConfig(),
    )


def test_execute_logs_known_student_after_presence_confirmation():
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 7}]

    recognizer = MagicMock()
    recognizer.recognize.return_value = "student-1"

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "high"

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = SimpleNamespace(
        id="student-1",
        name="Alice",
    )

    attendance_repo = MagicMock()
    config = AttendanceTrackingConfig(presence_confirmation_seconds=0.0)
    use_case = _build_use_case(
        detector=detector,
        recognizer=recognizer,
        pose_estimator=pose_estimator,
        student_repo=student_repo,
        attendance_repo=attendance_repo,
        config=config,
    )

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    result = use_case.execute(frame)

    assert result["students"][0]["name"] == "Alice"
    assert result["students"][0]["engagement"] == "high"
    attendance_repo.add_log.assert_called_once()

    saved_log = attendance_repo.add_log.call_args.args[0]
    assert saved_log.student_id == "student-1"
    assert saved_log.engagement_score is EngagementStatus.HIGH


def test_execute_retries_unknown_identity_on_next_frame():
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 3}]

    recognizer = MagicMock()
    recognizer.recognize.side_effect = [None, "student-1"]

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "medium"

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = SimpleNamespace(
        id="student-1",
        name="Alice",
    )

    use_case = _build_use_case(
        detector=detector,
        recognizer=recognizer,
        pose_estimator=pose_estimator,
        student_repo=student_repo,
        config=AttendanceTrackingConfig(presence_confirmation_seconds=999.0),
    )

    frame = np.zeros((240, 240, 3), dtype=np.uint8)

    first_result = use_case.execute(frame)
    second_result = use_case.execute(frame)

    assert first_result["students"][0]["name"] == "Unknown"
    assert second_result["students"][0]["name"] == "Alice"
    assert recognizer.recognize.call_count == 2


def test_execute_forgets_stale_tracks():
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 5}]

    recognizer = MagicMock()
    recognizer.recognize.return_value = None

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "low"

    config = AttendanceTrackingConfig(stale_track_ttl_seconds=1.0)
    use_case = _build_use_case(
        detector=detector,
        recognizer=recognizer,
        pose_estimator=pose_estimator,
        config=config,
    )

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    use_case.execute(frame)

    use_case.track_last_seen[5] = datetime.now() - timedelta(seconds=5)
    detector.track_people.return_value = []

    use_case.execute(frame)

    recognizer.forget_track.assert_called_once_with(5)
    pose_estimator.forget_track.assert_called_once_with(5)
