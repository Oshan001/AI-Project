# import cv2
# import mediapipe as mp
# import csv
# import os
# import time
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# import pickle
#
# # ===============================
# # SETUP
# # ===============================
# os.makedirs("gestures", exist_ok=True)
#
# gesture_label = input("\nEnter gesture label: ").strip()
# if not gesture_label:
#     print("Gesture label required")
#     exit()
#
# MAX_FRAMES = 150
#
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opened")
#     exit()
#
# # ===============================
# # AUTO DETECT HAND
# # ===============================
# hand_type = None
# # print("Show ONE hand to detect LEFT / RIGHT...")
#
# while hand_type is None:
#     ret, frame = cap.read()
#     if not ret:
#         continue
#
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
#         results = hands.process(rgb)
#         if results.multi_handedness:
#             hand_type = results.multi_handedness[0].classification[0].label
#
#     cv2.imshow("Detecting Hand", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#
# cv2.destroyAllWindows()
# # print(f"Detected hand: {hand_type}")
#
# csv_path = f"gestures/gesture_data_{hand_type.lower()}.csv"
# model_path = f"gesture_model_{hand_type.lower()}.pkl"
#
# # ===============================
# # DATA COLLECTION
# # ===============================
# hand_count = 0
#
# with open(csv_path, "a", newline="") as f:
#     writer = csv.writer(f)
#
#     with mp_hands.Hands(
#         max_num_hands=1,
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.7
#     ) as hands:
#
#         while hand_count < MAX_FRAMES:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame = cv2.flip(frame, 1)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb)
#
#             h, w, _ = frame.shape
#
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     row = []
#                     for lm in hand_landmarks.landmark:
#                         row.extend([lm.x, lm.y])
#                     row.append(gesture_label)
#                     writer.writerow(row)
#                     hand_count += 1
#
#                     mp_draw.draw_landmarks(
#                         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
#                     )
#
#             # ===============================
#             # UI DRAWING
#             # ===============================
#
#             # CENTER TITLE
#             title_text = "TRAINING"
#             (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
#             cv2.putText(
#                 frame,
#                 title_text,
#                 ((w - tw) // 2, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.4,
#                 (255, 255, 255),
#                 3
#             )
#
#             # LEFT INFO
#             cv2.putText(
#                 frame,
#                 f'Gesture : "{gesture_label}"',
#                 (15, 85),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 (255, 255, 255),
#                 2
#             )
#
#             # ===============================
#             # TRANSPARENT SAMPLING BAR
#             # ===============================
#             BAR_WIDTH = 320
#             BAR_HEIGHT = 18
#             x = 15
#             bar_y = 110
#
#             filled = int((hand_count / MAX_FRAMES) * BAR_WIDTH)
#
#             overlay = frame.copy()
#             alpha = 0.4  # transparency
#
#             # background
#             cv2.rectangle(
#                 overlay,
#                 (x, bar_y),
#                 (x + BAR_WIDTH, bar_y + BAR_HEIGHT),
#                 (80, 80, 80),
#                 -1
#             )
#
#             # filled
#             cv2.rectangle(
#                 overlay,
#                 (x, bar_y),
#                 (x + filled, bar_y + BAR_HEIGHT),
#                 (0, 255, 0),
#                 -1
#             )
#
#             # blend
#             cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#
#             # border
#             cv2.rectangle(
#                 frame,
#                 (x, bar_y),
#                 (x + BAR_WIDTH, bar_y + BAR_HEIGHT),
#                 (255, 255, 255),
#                 1
#             )
#
#             # sampling text
#             cv2.putText(
#                 frame,
#                 f"Sampling : ({hand_count} / {MAX_FRAMES})",
#                 (x, bar_y + BAR_HEIGHT + 22),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 255, 255),
#                 2
#             )
#
#             cv2.imshow("Collect Data and Train", frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             time.sleep(0.03)
#
# cap.release()
# cv2.destroyAllWindows()
#
# # ===============================
# # TRAIN MODEL
# # ===============================
# if hand_count > 0:
#     df = pd.read_csv(csv_path, header=None)
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
#
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(X_scaled, y)
#
#     with open(model_path, "wb") as f:
#         pickle.dump((model, scaler), f)
#
#     print("Model trained and saved")
# else:
#     print("No data collected")
#


import cv2
import mediapipe as mp
import csv
import os
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# ===============================
# CREATE FOLDERS
# ===============================
os.makedirs("../datasets", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# ===============================
# FUNCTION TO COUNT GESTURES
# ===============================
def gesture_count(csv_path):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None)
            return len(df.iloc[:, -1].unique())
        except:
            return 0
    return 0

# ===============================
# USER INPUT
# ===============================
gesture_label = input("Enter gesture label : ").strip()
if not gesture_label:
    print("❌ Gesture label cannot be empty")
    exit()

MAX_FRAMES = 150

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Camera not opened")
    exit()

# ===============================
# AUTO DETECT HAND
# ===============================
hand_type = None
print("Show your hand to detect LEFT or RIGHT...")
while hand_type is None:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(rgb)
        if results.multi_handedness:
            hand_type = results.multi_handedness[0].classification[0].label

    cv2.putText(frame, "Show hand to detect LEFT/RIGHT...", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()
print(f"Detected hand: {hand_type.upper()}")

csv_path = os.path.join("..", "datasets", f"gesture_data_{hand_type.lower()}.csv")
model_path = os.path.join("..", "models", f"gesture_model_{hand_type.lower()}.pkl")

# ===============================
# DATA COLLECTION
# ===============================
hand_count = 0

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:

        while hand_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, _ = frame.shape
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
                    row.append(gesture_label)
                    writer.writerow(row)
                    hand_count += 1

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ===============================
            # UI DISPLAY
            # ===============================
            # Center: TRAINING
            cv2.putText(frame, "TRAINING", (w//2 - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

            # Left corner: Gesture label
            cv2.putText(frame, f"Gesture: \"{gesture_label}\"", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

            # Sampling bar
            bar_width = 300
            bar_height = 20
            filled = int((hand_count / MAX_FRAMES) * bar_width)
            cv2.rectangle(frame, (10, 140), (10 + bar_width, 140 + bar_height), (120, 120, 120), 2)
            cv2.rectangle(frame, (10, 140), (10 + filled, 140 + bar_height), (0, 255, 0), -1)
            cv2.putText(frame, f"({hand_count}/{MAX_FRAMES})", (10 + bar_width + 10, 140 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Collect Data and Train", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()

# ===============================
# TRAIN MODEL
# ===============================
if hand_count > 0:
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)

    with open(model_path, "wb") as f:
        pickle.dump((model, scaler), f)

    print(f"✅ {hand_type.upper()} hand model trained & saved!")
else:
    print("❌ No frames collected. Training cancelled.")
