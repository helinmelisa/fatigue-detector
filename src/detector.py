import cv2
import numpy as np
from src.landmarks import LEFT_EYE, RIGHT_EYE, MOUTH
from src.ear import eye_aspect_ratio, EAR_THRESHOLD, EAR_CONSEC_FRAMES
from src.mar import mouth_aspect_ratio, MAR_THRESHOLD, MAR_CONSEC_FRAMES
from src.fatigue_score import compute_fatigue_score, fatigue_level
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
        self.yawn_frames = 0
        self.yawning = False

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

    def update_mar(self, face_landmarks, frame_shape):
        mouth = self.get_landmark_coords(face_landmarks, MOUTH, frame_shape)
        mar = mouth_aspect_ratio(mouth)

        if mar > MAR_THRESHOLD:
            self.yawn_frames += 1
            if self.yawn_frames >= MAR_CONSEC_FRAMES:
                self.yawning = True
        else:
            self.yawn_frames = 0
            self.yawning = False

        return mar

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

    def draw_info(self, frame, ear, mar):
        score = compute_fatigue_score(ear, mar, self.closed_frames, self.yawn_frames)
        level, level_color = fatigue_level(score)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Score: {score}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, level_color, 2)
        cv2.putText(frame, level, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, level_color, 2)

        # Score bar
        bar_x, bar_y, bar_w, bar_h = 10, 125, 200, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled = int(bar_w * score / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), level_color, -1)

        return frame

    def release(self):
        self.face_mesh.close()
