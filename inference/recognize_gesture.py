# import cv2
# import mediapipe as mp
# import pickle
# import os
# import numpy as np
#
# # ===============================
# # LOAD MODELS
# # ===============================
# models = {}
#
# for hand_type in ["left", "right"]:
#     model_path = f"gesture_model_{hand_type}.pkl"
#     if os.path.exists(model_path):
#         with open(model_path, "rb") as f:
#             models[hand_type] = pickle.load(f)  # (model, scaler)
#
# if not models:
#     print("❌ No models available. Train at least one hand first.")
#     exit()
#
# # ===============================
# # MEDIAPIPE SETUP
# # ===============================
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ ERROR: Camera not opened")
#     exit()
#
# window_name = "Gesture Recognition"
# cv2.namedWindow(window_name)
#
# with mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# ) as hands:
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb)
#
#         h, w, _ = frame.shape
#
#         if results.multi_hand_landmarks and results.multi_handedness:
#             for hand_landmarks, handedness in zip(
#                 results.multi_hand_landmarks,
#                 results.multi_handedness
#             ):
#                 hand_label = handedness.classification[0].label.lower()  # left / right
#
#                 mp_draw.draw_landmarks(
#                     frame,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS
#                 )
#
#                 if hand_label in models:
#                     model, scaler = models[hand_label]
#
#                     row = []
#                     for lm in hand_landmarks.landmark:
#                         row.extend([lm.x, lm.y])
#
#                     X = scaler.transform([row])
#                     pred = model.predict(X)[0]
#
#                     # ===============================
#                     # UI POSITION BASED ON HAND
#                     # ===============================
#                     if hand_label == "left":
#                         x_pos = 15              # LEFT side
#                     else:
#                         x_pos = w - 320         # RIGHT side
#
#                     cv2.putText(
#                         frame,
#                         f"Gesture: {pred}",
#                         (x_pos, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1.3,
#                         (0, 255, 0),
#                         3
#                     )
#
#         cv2.imshow(window_name, frame)
#
#         # EXIT WITH Q
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         # EXIT ON WINDOW CLOSE
#         if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
#             break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import pickle
import os

# ===============================
# LOAD MODELS
# ===============================
models = {}
for hand_type in ["left", "right"]:
    model_path = os.path.join("..", "models", f"gesture_model_{hand_type}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            models[hand_type] = pickle.load(f)

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

window_name = "Gesture Recognition"
cv2.namedWindow(window_name)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if hand_label in models:
                    model, scaler = models[hand_label]
                    row = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
                    X = scaler.transform([row])
                    pred = model.predict(X)[0]

                    # LEFT hand → left corner
                    if hand_label == "left":
                        cv2.putText(frame, f"Gesture : {pred}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                    # RIGHT hand → right corner
                    else:
                        cv2.putText(frame, f"   Gesture: {pred}", (w - 400, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
