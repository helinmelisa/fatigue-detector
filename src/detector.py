import cv2
import numpy as np
from src.landmarks import LEFT_EYE, RIGHT_EYE, MOUTH
from src.ear import eye_aspect_ratio, EAR_THRESHOLD, EAR_CONSEC_FRAMES
import mediapipe as mp


class FatigueDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.closed_frames = 0
        self.drowsy = False

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

    def update_ear(self, face_landmarks, frame_shape):
        left = self.get_landmark_coords(face_landmarks, LEFT_EYE, frame_shape)
        right = self.get_landmark_coords(face_landmarks, RIGHT_EYE, frame_shape)
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

        if ear < EAR_THRESHOLD:
            self.closed_frames += 1
            if self.closed_frames >= EAR_CONSEC_FRAMES:
                self.drowsy = True
        else:
            self.closed_frames = 0
            self.drowsy = False

        return ear

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

    def draw_ear_info(self, frame, ear):
        color = (0, 0, 255) if self.drowsy else (0, 255, 0)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.drowsy:
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    def release(self):
        self.face_mesh.close()
