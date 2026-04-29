import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((480, 640, 3), np.uint8)

# Drawing color (Purple)
drawColor = (255, 0, 255)

# Webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if len(lmList) != 0:
                # Index finger tip
                x1, y1 = lmList[8]

                # Middle finger tip
                x2, y2 = lmList[12]

                # DRAW MODE (only index finger up)
                if abs(y2 - y1) > 40:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1, y1

                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), drawColor, 5)
                    prev_x, prev_y = x1, y1

                else:
                    prev_x, prev_y = 0, 0

                # ERASE MODE (fingers close)
                if abs(x2 - x1) < 30 and abs(y2 - y1) < 30:
                    cv2.circle(canvas, (x1, y1), 30, (0, 0, 0), -1)

            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Merge canvas with webcam
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    # Title
    cv2.putText(img, "AI Virtual Painter", (140, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("AI Virtual Painter", img)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()