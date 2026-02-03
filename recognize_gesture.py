# ===============================
# RECOGNIZE GESTURE
# ===============================
import cv2
import mediapipe as mp
import pickle
import numpy as np

# ===============================
# LOAD MODEL
# ===============================
with open("gesture_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
DIST_THRESHOLD = 0.5

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

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

        display_texts = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label
                data = []

                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])

                data_np = np.array(data).reshape(1, -1)
                data_scaled = scaler.transform(data_np)

                try:
                    distances, _ = model.kneighbors(data_scaled)
                    if distances[0][0] > DIST_THRESHOLD:
                        prediction = f"{hand_label} hand: Not recognized"
                    else:
                        prediction = f"{hand_label} hand: {model.predict(data_scaled)[0]}"
                except:
                    prediction = f"{hand_label} hand: Not recognized"

                display_texts.append(prediction)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for idx, text in enumerate(display_texts):
            cv2.putText(frame, text, (10, 50 + idx * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Recognition", frame)

        # ✅ EXIT on Q or ❌
        if (cv2.waitKey(1) & 0xFF == ord('q') or
            cv2.getWindowProperty("Sign Language Recognition", cv2.WND_PROP_VISIBLE) < 1):
            break

cap.release()
cv2.destroyAllWindows()
