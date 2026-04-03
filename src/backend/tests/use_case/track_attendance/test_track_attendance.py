from datetime import datetime, time, timedelta
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
    """
    Verifies scenario build use case.
    
    Args:
        detector: Input value for `detector`.
        recognizer: Input value for `recognizer`.
        pose_estimator: Input value for `pose_estimator`.
        student_repo: Input value for `student_repo`.
        attendance_repo: Input value for `attendance_repo`.
        config: Input value for `config`.
    
    Returns:
        The function result.
    """
    return TrackAttendanceUseCase(
        person_detector=detector or MagicMock(),
        face_recognizer=recognizer or MagicMock(),
        pose_estimator=pose_estimator or MagicMock(),
        student_repo=student_repo or MagicMock(),
        attendance_repo=attendance_repo or MagicMock(),
        config=config or AttendanceTrackingConfig(),
    )


def test_execute_logs_known_student_after_presence_confirmation():
    """
    Verifies scenario execute logs known student after presence confirmation.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 7}]

    recognizer = MagicMock()
    recognizer.detect_faces.return_value = []
    recognizer.recognize.return_value = "student-1"

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "high"

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = SimpleNamespace(
        id="student-1",
        name="Alice",
    )

    attendance_repo = MagicMock()
    attendance_repo.get_logs_by_student.return_value = []
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
    """
    Verifies scenario execute retries unknown identity on next frame.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 3}]

    recognizer = MagicMock()
    recognizer.detect_faces.return_value = []
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
    """
    Verifies scenario execute forgets stale tracks.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 5}]

    recognizer = MagicMock()
    recognizer.detect_faces.return_value = []
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


def test_execute_passes_matched_face_bbox_to_engagement_estimator():
    """
    Verifies scenario execute passes matched face bbox to engagement estimator.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 9}]

    recognizer = MagicMock()
    recognizer.detect_faces.return_value = [
        {
            "bbox": [20, 20, 90, 120],
            "crop": np.zeros((100, 70, 3), dtype=np.uint8),
            "confidence": 1.0,
        }
    ]
    recognizer.recognize.return_value = "student-9"

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "medium"

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = SimpleNamespace(
        id="student-9",
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
    use_case.execute(frame)

    _, kwargs = pose_estimator.estimate_engagement.call_args
    assert kwargs["face_bbox"] == [20, 20, 90, 120]


def test_execute_logs_student_again_on_new_day():
    """
    Verifies scenario execute logs student again on new day.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [10, 10, 110, 210], "track_id": 7}]

    recognizer = MagicMock()
    recognizer.detect_faces.return_value = []
    recognizer.recognize.return_value = "student-1"

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "high"

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = SimpleNamespace(
        id="student-1",
        name="Alice",
    )

    attendance_repo = MagicMock()
    attendance_repo.get_logs_by_student.return_value = []
    use_case = _build_use_case(
        detector=detector,
        recognizer=recognizer,
        pose_estimator=pose_estimator,
        student_repo=student_repo,
        attendance_repo=attendance_repo,
        config=AttendanceTrackingConfig(presence_confirmation_seconds=0.0),
    )
    use_case.marked_students["student-1"] = datetime.now().date() - timedelta(days=1)

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    use_case.execute(frame)

    attendance_repo.add_log.assert_called_once()


def test_execute_builds_larger_display_bbox_from_face():
    """
    Verifies scenario execute builds larger display bbox from face.

    Args:
        None.

    Returns:
        Does not return a value.
    """
    detector = MagicMock()
    detector.track_people.return_value = [{"bbox": [60, 40, 160, 200], "track_id": 4}]

    recognizer = MagicMock()
    recognizer.detect_faces.return_value = [
        {
            "bbox": [80, 50, 140, 120],
            "crop": np.zeros((70, 60, 3), dtype=np.uint8),
            "confidence": 1.0,
        }
    ]
    recognizer.recognize.return_value = "student-4"

    pose_estimator = MagicMock()
    pose_estimator.estimate_engagement.return_value = "medium"

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = SimpleNamespace(
        id="student-4",
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
    result = use_case.execute(frame)

    display_bbox = result["students"][0]["display_bbox"]
    assert display_bbox[0] < 60
    assert display_bbox[1] < 40
    assert display_bbox[2] > 160
    assert display_bbox[3] > 200


def test_log_visit_uses_lesson_start_time_for_late_mark():
    """
    Verifies scenario log visit uses lesson start time for late mark.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    attendance_repo = MagicMock()
    attendance_repo.get_logs_by_student.return_value = []
    use_case = _build_use_case(
        attendance_repo=attendance_repo,
        config=AttendanceTrackingConfig(
            lesson_start_time=time(9, 0),
            late_after_seconds=300.0,
        ),
    )

    logged = use_case._log_visit(
        "student-1",
        "high",
        datetime(2026, 4, 1, 9, 10, 0),
    )

    assert logged is True
    saved_log = attendance_repo.add_log.call_args.args[0]
    assert saved_log.is_late is True


def test_log_visit_skips_duplicate_log_for_same_day():
    """
    Verifies scenario log visit skips duplicate log for same day.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    attendance_repo = MagicMock()
    attendance_repo.get_logs_by_student.return_value = [
        SimpleNamespace(timestamp=datetime(2026, 4, 1, 9, 5, 0))
    ]
    use_case = _build_use_case(attendance_repo=attendance_repo)

    logged = use_case._log_visit(
        "student-1",
        "medium",
        datetime(2026, 4, 1, 9, 15, 0),
    )

    assert logged is True
    attendance_repo.add_log.assert_not_called()
