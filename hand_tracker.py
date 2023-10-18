import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img_h, img_w = frame.shape[:2]

        results = holistic.process(frame)

        #Right hand landmarks
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(240, 15, 0), thickness=2, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                              )

        #Left hand landmarks
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(15, 255, 0), thickness=2, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                               )

        cv2.imshow('Hand Tracking', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()