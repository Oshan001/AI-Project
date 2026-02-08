# Hand Gesture Recognition System (Left & Right Hand)

A real-time hand gesture recognition system using **OpenCV**, **MediaPipe**, and **Machine Learning**.  
The system supports **separate training and recognition for left and right hands**, allowing different gestures for each hand.

---

## 📌 Features

- Real-time hand tracking using MediaPipe
- Separate gesture models for **Left Hand** and **Right Hand**
- One-hand-at-a-time gesture training
- Automatic hand detection (Left / Right)
- Live gesture recognition with camera UI
- Clean and minimal user interface
- Exit safely using **Q key** or window close (X)

---

## 🗂️ Project Structure
GestureRecognition/
│
├── training/
│   └── collect_and_train.py         # Data collection & model training
│
├── inference/
│   └── recognize_gesture.py        # Real-time gesture recognition
│
├── models/
│   ├── gesture_model_left.pkl
│   └── gesture_model_right.pkl
│
└── datasets/
    ├── gesture_data_left.csv
    └── gesture_data_right.csv
