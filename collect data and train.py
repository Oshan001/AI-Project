# ===============================
# COLLECT DATA + TRAIN MODEL (ONE HAND AT A TIME, AUTO DETECT, COUNT GESTURES)
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
# SETUP FOLDERS
# ===============================
os.makedirs("gestures", exist_ok=True)

print("\n" + "="*60)
print("🎯 GESTURE RECOGNITION - TRAINING MODE")
print("="*60)

# ===============================
# FUNCTION TO COUNT GESTURES IN CSV
# ===============================
def gesture_count(csv_path):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None)
            return len(df.iloc[:, -1].unique())  # Last column = gesture labels
        except:
            return 0
    return 0

left_csv = "gestures/gesture_data_left.csv"
right_csv = "gestures/gesture_data_right.csv"

# Count gestures for display
left_count = gesture_count(left_csv)
right_count = gesture_count(right_csv)

print("\n📊 Number of gestures collected so far:")
print(f"   LEFT hand:  {left_count} gesture(s)")
print(f"   RIGHT hand: {right_count} gesture(s)")

# ===============================
# USER INPUT FOR GESTURE LABEL
# ===============================
gesture_label = input("\nEnter gesture label (e.g., A, HELLO, THUMBSUP): ").strip()
if not gesture_label:
    print("❌ Gesture label cannot be empty")
    exit()

MAX_FRAMES = 150
print(f"\n📊 Target frames: {MAX_FRAMES}")

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ===============================
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Camera not opened")
    exit()
print("✅ Camera opened successfully\n")

# ===============================
# DETERMINE HAND TO TRAIN (FIRST DETECTED HAND)
# ===============================
print("🤚 Show your hand to the camera to auto-detect LEFT or RIGHT hand...")
hand_type = None
while hand_type is None:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(rgb)
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_type = results.multi_handedness[0].classification[0].label  # "Left" or "Right"

    cv2.putText(frame, "Show your hand to detect LEFT/RIGHT...", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Detection cancelled by user")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()
print(f"\n✅ Detected hand: {hand_type.upper()} HAND")

csv_path = f"gestures/gesture_data_{hand_type.lower()}.csv"
model_path = f"gesture_model_{hand_type.lower()}.pkl"
print(f"📁 Saving data to: {csv_path}")
print(f"📁 Model will be saved to: {model_path}")

# ===============================
# DATA COLLECTION
# ===============================
hand_count = 0

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        print(f"\n🎥 Collecting gesture '{gesture_label}' for {hand_type} hand...")
        print(f"📊 Target: {MAX_FRAMES} frames")
        print("⌨️  Press Q to exit\n")

        while hand_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, c = frame.shape
            hand_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                    row.append(gesture_label)
                    writer.writerow(row)
                    hand_count += 1

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # DISPLAY INFO
            cv2.putText(frame, f"TRAINING: {hand_type.upper()} HAND", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            cv2.putText(frame, f"Gesture: '{gesture_label}'", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {hand_count}/{MAX_FRAMES}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if hand_detected:
                cv2.putText(frame, f"✅ {hand_type.upper()} HAND DETECTED", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"❌ Show {hand_type.upper()} HAND to camera", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Progress bar
            bar_width = 300
            bar_height = 20
            filled_width = int((hand_count / MAX_FRAMES) * bar_width)
            cv2.rectangle(frame, (10, h - 50), (10 + bar_width, h - 30), (100, 100, 100), 2)
            cv2.rectangle(frame, (10, h - 50), (10 + filled_width, h - 30), (0, 255, 0), -1)

            cv2.imshow("Collect Data and Train", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Data collection complete ({hand_count} frames collected)")

# ===============================
# TRAIN MODEL
# ===============================
if hand_count > 0:
    print(f"\n🧠 Training {hand_type.upper()} hand model...")

    try:
        df = pd.read_csv(csv_path, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_scaled, y)

        with open(model_path, "wb") as f:
            pickle.dump((model, scaler), f)

        print(f"\n✅ {hand_type.upper()} model trained & saved!")

        # SHOW UPDATED GESTURE COUNTS
        left_count = gesture_count(left_csv)
        right_count = gesture_count(right_csv)
        print("\n📊 Updated number of gestures:")
        print(f"   LEFT hand:  {left_count} gesture(s)")
        print(f"   RIGHT hand: {right_count} gesture(s)")

    except Exception as e:
        print(f"❌ Error during training: {e}")
else:
    print("❌ No frames collected. Training cancelled.")
