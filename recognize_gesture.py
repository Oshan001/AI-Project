import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# ===============================
# LOAD MODELS (SAFE)
# ===============================
single_model = single_scaler = None
both_model = both_scaler = None

if os.path.exists("gesture_model_single.pkl"):
    with open("gesture_model_single.pkl", "rb") as f:
        single_model, single_scaler = pickle.load(f)

if os.path.exists("gesture_model_both.pkl"):
    with open("gesture_model_both.pkl", "rb") as f:
        both_model, both_scaler = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
DIST_THRESHOLD = 0.5

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        text = "No Hand Detected"

        # ===============================
        # ONE HAND
        # ===============================
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            if single_model is None:
                text = "No Gesture Recognized"
            else:
                data = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    data.extend([lm.x, lm.y])

                data_np = np.array(data).reshape(1, -1)
                data_scaled = single_scaler.transform(data_np)

                try:
                    dist, _ = single_model.kneighbors(data_scaled)
                    if dist[0][0] > DIST_THRESHOLD:
                        text = "No Gesture Recognized"
                    else:
                        text = single_model.predict(data_scaled)[0]
                except:
                    text = "No Gesture Recognized"

                mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS
                )

        # ===============================
        # BOTH HANDS
        # ===============================
        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            if both_model is None:
                text = "No Gesture Recognized"
            else:
                data = []
                for hand in results.multi_hand_landmarks:
                    for lm in hand.landmark:
                        data.extend([lm.x, lm.y])
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                data_np = np.array(data).reshape(1, -1)
                data_scaled = both_scaler.transform(data_np)

                try:
                    dist, _ = both_model.kneighbors(data_scaled)
                    if dist[0][0] > DIST_THRESHOLD:
                        text = "No Gesture Recognized"
                    else:
                        text = both_model.predict(data_scaled)[0]
                except:
                    text = "No Gesture Recognized"

        # ===============================
        # DISPLAY
        # ===============================
        cv2.putText(
            frame,
            text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if text != "No Hand Detected" else (0, 0, 255),
            2
        )

        cv2.imshow("Sign Language Recognition", frame)

        if (cv2.waitKey(1) & 0xFF == ord('q') or
            cv2.getWindowProperty(
                "Sign Language Recognition", cv2.WND_PROP_VISIBLE) < 1):
            break

cap.release()

cv2.destroyAllWindows()
