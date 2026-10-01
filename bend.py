import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# (base joint, middle joint, tip) landmark numbers for each finger
FINGERS = {
    "Thumb": (2, 3, 4),
    "Index": (5, 6, 8),
    "Middle": (9, 10, 12),
    "Ring": (13, 14, 16),
    "Pinky": (17, 18, 20),
}

def angle(a, b, c):
    """Angle at point b, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def bend_percent(ang, straight=170, curled=70):
    """Turn an angle into 0-100% bent."""
    pct = (straight - ang) / (straight - curled) * 100
    return int(np.clip(pct, 0, 100))

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            pts = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

            y = 30
            for name, (a, b, c) in FINGERS.items():
                pct = bend_percent(angle(pts[a], pts[b], pts[c]))
                cv2.putText(frame, f"{name}: {pct}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 35

        cv2.imshow("Hand", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()