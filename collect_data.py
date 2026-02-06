#
# # ===============================
# # COLLECT DATA + TRAIN MODEL
# # (ONE HAND AT A TIME, AUTO DETECT, COUNT GESTURES)
# # ===============================
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
# # SETUP FOLDERS
# # ===============================
# os.makedirs("gestures", exist_ok=True)
#
# print("\n" + "=" * 60)
# print("🎯 GESTURE RECOGNITION - TRAINING MODE")
# print("=" * 60)
#
# # ===============================
# # FUNCTION TO COUNT GESTURES
# # ===============================
# def gesture_count(csv_path):
#     if os.path.exists(csv_path):
#         try:
#             df = pd.read_csv(csv_path, header=None)
#             return len(df.iloc[:, -1].unique())
#         except:
#             return 0
#     return 0
#
# left_csv = "gestures/gesture_data_left.csv"
# right_csv = "gestures/gesture_data_right.csv"
#
# left_count = gesture_count(left_csv)
# right_count = gesture_count(right_csv)
#
# print("\n📊 Number of gestures collected so far:")
# print(f"   LEFT hand:  {left_count} gesture(s)")
# print(f"   RIGHT hand: {right_count} gesture(s)")
#
# # ===============================
# # USER INPUT
# # ===============================
# gesture_label = input("\nEnter gesture label (e.g., A, HELLO, THUMBSUP): ").strip()
# if not gesture_label:
#     print("❌ Gesture label cannot be empty")
#     exit()
#
# MAX_FRAMES = 150
# print(f"\n📊 Target frames: {MAX_FRAMES}")
#
# # ===============================
# # MEDIAPIPE SETUP
# # ===============================
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# # ===============================
# # OPEN CAMERA
# # ===============================
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ ERROR: Camera not opened")
#     exit()
#
# print("✅ Camera opened successfully\n")
#
# # ===============================
# # AUTO-DETECT HAND TYPE
# # ===============================
# print("🤚 Show your hand to the camera to auto-detect LEFT or RIGHT hand...")
# hand_type = None
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
#         if results.multi_hand_landmarks and results.multi_handedness:
#             hand_type = results.multi_handedness[0].classification[0].label
#
#     cv2.putText(frame, "Show your hand to detect LEFT/RIGHT...", (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.imshow("Hand Detection", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#
# cv2.destroyAllWindows()
# print(f"\n✅ Detected hand: {hand_type.upper()} HAND")
#
# csv_path = f"gestures/gesture_data_{hand_type.lower()}.csv"
# model_path = f"gesture_model_{hand_type.lower()}.pkl"
#
# print(f"📁 Saving data to: {csv_path}")
# print(f"📁 Model will be saved to: {model_path}")
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
#         print(f"\n🎥 Collecting gesture '{gesture_label}' for {hand_type} hand...")
#         print(f"📊 Target: {MAX_FRAMES} frames")
#         print("⌨️  Press Q to exit\n")
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
#             # CAMERA UI (NO LEFT/RIGHT DETECTED TEXT)
#             # ===============================
#             cv2.putText(frame, f"TRAINING MODE", (10, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
#
#             cv2.putText(frame, f"Gesture: '{gesture_label}'", (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             cv2.putText(frame, f"Progress: {hand_count}/{MAX_FRAMES}", (10, 140),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             # Progress bar
#             bar_width = 300
#             filled_width = int((hand_count / MAX_FRAMES) * bar_width)
#             cv2.rectangle(frame, (10, h - 50), (10 + bar_width, h - 30), (100, 100, 100), 2)
#             cv2.rectangle(frame, (10, h - 50), (10 + filled_width, h - 30), (0, 255, 0), -1)
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
# print(f"\n✅ Data collection complete ({hand_count} frames collected)")
#
# # ===============================
# # TRAIN MODEL
# # ===============================
# if hand_count > 0:
#     print(f"\n🧠 Training {hand_type.upper()} hand model...")
#
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
#     print(f"\n✅ {hand_type.upper()} model trained & saved!")
#
# else:
#     print("❌ No frames collected. Training cancelled.")





#
#
#
# # ===============================
# # COLLECT DATA + TRAIN MODEL
# # (ONE HAND AT A TIME, AUTO DETECT)
# # ===============================
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
# os.makedirs("gestures", exist_ok=True)
#
# def gesture_count(csv_path):
#     if os.path.exists(csv_path):
#         try:
#             df = pd.read_csv(csv_path, header=None)
#             return len(df.iloc[:, -1].unique())
#         except:
#             return 0
#     return 0
#
# gesture_label = input("\nEnter gesture label: ").strip()
# if not gesture_label:
#     exit()
#
# MAX_FRAMES = 150
#
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     exit()
#
# # ===============================
# # AUTO DETECT HAND
# # ===============================
# hand_type = None
# while hand_type is None:
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
#         results = hands.process(rgb)
#         if results.multi_handedness:
#             hand_type = results.multi_handedness[0].classification[0].label
#
#     cv2.imshow("Hand Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#
# cv2.destroyAllWindows()
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
#             # SAMPLING UI (CLEAN & COMPACT)
#             # ===============================
#             bar_width = 300
#             bar_height = 18
#             filled_width = int((hand_count / MAX_FRAMES) * bar_width)
#
#             y_bar = h - 40
#
#             # Label
#             cv2.putText(frame, "Sampling :", (10, y_bar - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#
#             # Bar background
#             cv2.rectangle(frame, (120, y_bar - bar_height),
#                           (120 + bar_width, y_bar), (120, 120, 120), 2)
#
#             # Bar fill
#             cv2.rectangle(frame, (120, y_bar - bar_height),
#                           (120 + filled_width, y_bar), (0, 255, 0), -1)
#
#             # Count text
#             cv2.putText(frame, f"{hand_count} / {MAX_FRAMES}",
#                         (120 + bar_width + 10, y_bar - 3),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
# # ===============================
# # COLLECT DATA + TRAIN MODEL
# # (ONE HAND AT A TIME, AUTO DETECT)
# # ===============================
#
# import cv2
# import mediapipe as mp
# # import csv
# import os
# import time
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# import pickle
#
# os.makedirs("gestures", exist_ok=True)
#
# gesture_label = input("\nEnter gesture label: ").strip()
# if not gesture_label:
#     exit()
#
# MAX_FRAMES = 150
#
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     exit()
#
# # ===============================
# # AUTO DETECT HAND
# # ===============================
# hand_type = None
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
#     cv2.imshow("Hand Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#
# cv2.destroyAllWindows()
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
#             # ======== TRAINING UI =========
#             # ===============================
#
#             # ---- CENTER TITLE ----
#             title = "TRAINING"
#             (tw, th), _ = cv2.getTextSize(
#                 title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
#             )
#             cv2.putText(
#                 frame,
#                 title,
#                 ((w - tw) // 2, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.0,
#                 (0, 255, 255),
#                 2
#             )
#
#             # ---- LEFT CORNER UI ----
#             x = 10
#             y = 80
#
#             cv2.putText(
#                 frame,
#                 f'Gesture : "{gesture_label}"',
#                 (x, y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 255, 255),
#                 2
#             )
#
#             # Sampling bar
#             BAR_WIDTH = 220
#             BAR_HEIGHT = 16
#             filled = int((hand_count / MAX_FRAMES) * BAR_WIDTH)
#
#             bar_y = y + 20
#
#             # Bar background
#             cv2.rectangle(
#                 frame,
#                 (x, bar_y),
#                 (x + BAR_WIDTH, bar_y + BAR_HEIGHT),
#                 (80, 80, 80),
#                 -1
#             )
#
#             # Bar fill
#             cv2.rectangle(
#                 frame,
#                 (x, bar_y),
#                 (x + filled, bar_y + BAR_HEIGHT),
#                 (0, 255, 0),
#                 -1
#             )
#
#             # Bar border
#             cv2.rectangle(
#                 frame,
#                 (x, bar_y),
#                 (x + BAR_WIDTH, bar_y + BAR_HEIGHT),
#                 (255, 255, 255),
#                 1
#             )
#
#             # Sampling text (same line)
#             cv2.putText(
#                 frame,
#                 f"Sampling : ({hand_count} / {MAX_FRAMES})",
#                 (x + BAR_WIDTH + 10, bar_y + BAR_HEIGHT - 2),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 255),
#                 1
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




# ===============================
# COLLECT DATA + TRAIN MODEL
# (ONE HAND AT A TIME, AUTO DETECT)
# CLEAN UI + TRANSPARENT SAMPLING BAR
# ===============================

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
# SETUP
# ===============================
os.makedirs("gestures", exist_ok=True)

gesture_label = input("\nEnter gesture label: ").strip()
if not gesture_label:
    print("Gesture label required")
    exit()

MAX_FRAMES = 150

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opened")
    exit()

# ===============================
# AUTO DETECT HAND
# ===============================
hand_type = None
print("Show ONE hand to detect LEFT / RIGHT...")

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

    cv2.imshow("Detecting Hand", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()
print(f"Detected hand: {hand_type}")

csv_path = f"gestures/gesture_data_{hand_type.lower()}.csv"
model_path = f"gesture_model_{hand_type.lower()}.pkl"

# ===============================
# DATA COLLECTION
# ===============================
hand_count = 0

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

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
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                    row.append(gesture_label)
                    writer.writerow(row)
                    hand_count += 1

                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # ===============================
            # UI DRAWING
            # ===============================

            # CENTER TITLE
            title_text = "TRAINING"
            (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.putText(
                frame,
                title_text,
                ((w - tw) // 2, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                3
            )

            # LEFT INFO
            cv2.putText(
                frame,
                f'Gesture : "{gesture_label}"',
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            # ===============================
            # TRANSPARENT SAMPLING BAR
            # ===============================
            BAR_WIDTH = 320
            BAR_HEIGHT = 18
            x = 15
            bar_y = 110

            filled = int((hand_count / MAX_FRAMES) * BAR_WIDTH)

            overlay = frame.copy()
            alpha = 0.4  # transparency

            # background
            cv2.rectangle(
                overlay,
                (x, bar_y),
                (x + BAR_WIDTH, bar_y + BAR_HEIGHT),
                (80, 80, 80),
                -1
            )

            # filled
            cv2.rectangle(
                overlay,
                (x, bar_y),
                (x + filled, bar_y + BAR_HEIGHT),
                (0, 255, 0),
                -1
            )

            # blend
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # border
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + BAR_WIDTH, bar_y + BAR_HEIGHT),
                (255, 255, 255),
                1
            )

            # sampling text
            cv2.putText(
                frame,
                f"Sampling : ({hand_count} / {MAX_FRAMES})",
                (x, bar_y + BAR_HEIGHT + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

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

    print("Model trained and saved")
else:
    print("No data collected")

