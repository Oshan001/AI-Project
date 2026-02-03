# ===============================
# COLLECT DATA + TRAIN MODEL
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
# USER INPUT
# ===============================
label = input("Enter gesture label (e.g., A, HELLO): ")
MAX_FRAMES = 150

# ===============================
# CSV SETUP
# ===============================
os.makedirs("gestures", exist_ok=True)
csv_path = "gestures/gesture_data.csv"

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ===============================
# CAMERA OPEN
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Camera not opened")
    exit()

print("✅ Camera opened successfully")

# ===============================
# DATA COLLECTION
# ===============================
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        count = 0
        print("🎥 Collecting data... Press Q or ❌ to exit")

        while count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_side = results.multi_handedness[i].classification[0].label

                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])

                    row.append(f"{label}_{hand_side}")
                    writer.writerow(row)
                    count += 1

                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(
                frame,
                f"Collecting {label}: {count}/{MAX_FRAMES}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collect & Train", frame)

            # ✅ EXIT on Q or ❌
            if (cv2.waitKey(1) & 0xFF == ord('q') or
                cv2.getWindowProperty("Collect & Train", cv2.WND_PROP_VISIBLE) < 1):
                break

            time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()

print("✅ Data collection complete")

# ===============================
# TRAIN MODEL
# ===============================
print("🧠 Training model...")

df = pd.read_csv(csv_path, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

with open("gesture_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model trained & saved")
print("📌 Classes:", model.classes_)
