import cv2
from src.detector import FatigueDetector


def main():
    cap = cv2.VideoCapture(1)
    detector = FatigueDetector()

    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        frame = detector.draw_eye_mouth_landmarks(frame, results)

        if results.multi_face_landmarks:
            ear = detector.update_ear(results.multi_face_landmarks[0], frame.shape)
            mar = detector.update_mar(results.multi_face_landmarks[0], frame.shape)
            frame = detector.draw_info(frame, ear, mar)

        cv2.imshow("Fatigue Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
