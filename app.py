import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="AI Virtual Painter", layout="centered")

st.title("🎨 AI Virtual Painter (Web Demo)")
st.markdown("Draw using mouse (Deployed Version)")

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Drawing state
if "drawing" not in st.session_state:
    st.session_state.drawing = False

# Canvas display
frame_placeholder = st.empty()

# Mouse drawing simulation
color = st.sidebar.color_picker("Pick a color", "#ff00ff")

# Convert hex to BGR
color_bgr = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))

# Simple drawing area
img = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Draw demo shapes
cv2.putText(img, "Demo Mode (Cloud)", (120, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

frame_placeholder.image(img, channels="BGR")

st.info("⚠️ Full AI hand-tracking version runs locally (main.py)")