import numpy as np
from scipy.spatial.distance import euclidean

# Mouth is considered open (yawn) when MAR exceeds this threshold
MAR_THRESHOLD = 0.6

# Number of consecutive frames mouth must be open to trigger yawn
MAR_CONSEC_FRAMES = 15


def mouth_aspect_ratio(mouth_coords):
    """
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)

    mouth_coords: array of 8 (x, y) points:
        p1 (left corner), p2 (top-left), p3 (top-center-left), p4 (top-center-right),
        p5 (right corner), p6 (bottom-right), p7 (bottom-center), p8 (bottom-left)
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_coords
    vertical_1 = euclidean(p2, p8)
    vertical_2 = euclidean(p3, p7)
    vertical_3 = euclidean(p4, p6)
    horizontal = euclidean(p1, p5)
    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
