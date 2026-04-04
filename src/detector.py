import cv2
import mediapipe as mp
import numpy as np
from src.landmarks import LEFT_EYE, RIGHT_EYE, MOUTH


class FatigueDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return results

    def get_landmark_coords(self, face_landmarks, indices, frame_shape):
        h, w = frame_shape[:2]
        return np.array([
            (face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h)
            for i in indices
        ], dtype=np.float32)

    def draw_eye_mouth_landmarks(self, frame, results):
        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:
            for indices, color in [
                (LEFT_EYE, (0, 255, 0)),
                (RIGHT_EYE, (0, 255, 0)),
                (MOUTH, (0, 165, 255)),
            ]:
                coords = self.get_landmark_coords(face_landmarks, indices, frame.shape)
                for x, y in coords.astype(int):
                    cv2.circle(frame, (x, y), 2, color, -1)

        return frame

    def release(self):
        self.face_mesh.close()
