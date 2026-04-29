import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="AI Virtual Painter", layout="centered")

st.title("🎨 AI Virtual Painter")
st.markdown("Draw in air using your finger ✋")

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Sidebar controls
color_option = st.sidebar.selectbox("Choose Color", ["Purple", "Red", "Green", "Blue"])

color_map = {
    "Purple": (255, 0, 255),
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0)
}

drawColor = color_map[color_option]

class Painter(VideoTransformerBase):
    def __init__(self):
        self.canvas = np.zeros((480, 640, 3), np.uint8)
        self.prev_x, self.prev_y = 0, 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

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
                    x1, y1 = lmList[8]   # index finger
                    x2, y2 = lmList[12]  # middle finger

                    # DRAW MODE
                    if abs(y2 - y1) > 40:
                        if self.prev_x == 0 and self.prev_y == 0:
                            self.prev_x, self.prev_y = x1, y1

                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), drawColor, 5)
                        self.prev_x, self.prev_y = x1, y1
                    else:
                        self.prev_x, self.prev_y = 0, 0

                    # ERASE MODE
                    if abs(x2 - x1) < 30 and abs(y2 - y1) < 30:
                        cv2.circle(self.canvas, (x1, y1), 30, (0, 0, 0), -1)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # Merge canvas
        imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.canvas)

        return img

webrtc_streamer(
    key="painter",
    video_transformer_factory=Painter,
    media_stream_constraints={"video": True, "audio": False},
)