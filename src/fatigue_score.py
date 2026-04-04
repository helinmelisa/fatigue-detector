from src.ear import EAR_THRESHOLD
from src.mar import MAR_THRESHOLD

# Weights for combining signals
EAR_WEIGHT = 0.6
MAR_WEIGHT = 0.4

# Score thresholds
ALERT_THRESHOLD = 50
CRITICAL_THRESHOLD = 75


def compute_fatigue_score(ear, mar, closed_frames, yawn_frames):
    """
    Returns a fatigue score from 0 to 100.
    - Eye closure contributes up to 60 points
    - Yawning contributes up to 40 points
    """
    eye_score = 0
    if ear < EAR_THRESHOLD:
        eye_score = min(closed_frames * 2, 60)

    mouth_score = 0
    if mar > MAR_THRESHOLD:
        mouth_score = min(yawn_frames * 2, 40)

    return int(eye_score * EAR_WEIGHT / 0.6 + mouth_score * MAR_WEIGHT / 0.4)


def fatigue_level(score):
    if score >= CRITICAL_THRESHOLD:
        return "CRITICAL", (0, 0, 255)
    elif score >= ALERT_THRESHOLD:
        return "WARNING", (0, 165, 255)
    else:
        return "OK", (0, 255, 0)
