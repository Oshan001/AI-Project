# ===============================
# GESTURE RECOGNITION - REAL TIME (EXIT ON WINDOW CLOSE)
# ===============================
import cv2
import mediapipe as mp
import pickle
import os
import numpy as np

# ===============================
# LOAD MODELS IF AVAILABLE
# ===============================
models = {}
for hand_type in ["left", "right"]:
    model_path = f"gesture_model_{hand_type}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            models[hand_type] = pickle.load(f)  # (model, scaler)
        print(f"✅ {hand_type.upper()} hand model loaded")
    else:
        print(f"⚠️  {hand_type.upper()} hand model NOT found")

if not models:
    print("❌ No models available. Train at least one hand first.")
    exit()

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Camera not opened")
    exit()
print("✅ Camera opened successfully\n")

window_name = "Gesture Recognition"
cv2.namedWindow(window_name)

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    print("🎥 Gesture Recognition Started - Press 'Q' or close window to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()  # left or right
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Only predict if model exists for this hand
                if hand_label in models:
                    model, scaler = models[hand_label]

                    # Convert landmarks to feature vector
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                    X = scaler.transform([row])
                    pred = model.predict(X)[0]

                    cv2.putText(frame, f"{hand_label.upper()} HAND: {pred}", (10, 50 if hand_label=="left" else 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, f"{hand_label.upper()} HAND: NO MODEL", (10, 50 if hand_label=="left" else 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        # ===============================
        # EXIT CONDITIONS
        # ===============================
        # 1. Press Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Exited by user pressing Q")
            break

        # 2. Click X (close window)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("❌ Window closed by user")
            break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera released, program exited cleanly")
