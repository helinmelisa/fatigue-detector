import numpy as np
from scipy.spatial.distance import euclidean

# Eyes are considered closed when EAR drops below this threshold
EAR_THRESHOLD = 0.25

# Number of consecutive frames eyes must be below threshold to trigger drowsiness
EAR_CONSEC_FRAMES = 20


def eye_aspect_ratio(eye_coords):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    eye_coords: array of 6 (x, y) points in order:
        p1 (left corner), p2 (top-left), p3 (top-right),
        p4 (right corner), p5 (bottom-right), p6 (bottom-left)
    """
    p1, p2, p3, p4, p5, p6 = eye_coords
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
