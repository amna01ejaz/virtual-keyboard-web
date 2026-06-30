import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av
import os
import tempfile
import urllib.request
import numpy as np

st.set_page_config(page_title="Global Virtual Keyboard", layout="wide")
st.title("⌨️ Real-Time Virtual AI Keyboard")
st.write("Type in the air using hand tracking! Access this page from any device globally.")

# Writable temporary path for downloading the hand model asset
TEMP_DIR = tempfile.gettempdir()
MODEL_PATH = os.path.join(TEMP_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading hand tracking model... Please wait."):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

# Session state to hold the typed text globally across stream updates
if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""

# Define simple keyboard keys layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

class KeyboardProcessor(VideoProcessorBase):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.cooldown = 0  # Frame cooldown to prevent duplicate typing animations

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Draw the virtual keyboard keys overlay onto the frame
        key_w, key_h = int(w / 11), 50
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                x = col_idx * key_w + 20
                y = row_idx * key_h + 50
                cv2.rectangle(img, (x, y), (x + key_w - 5, y + key_h - 5), (255, 255, 255), 2)
                cv2.putText(img, key, (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Handle frame conversion and hand tracking
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = self.detector.detect(mp_image)
        
        if self.cooldown > 0:
            self.cooldown -= 1

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Landmark 8 is Index Finger Tip, Landmark 12 is Middle Finger Tip
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                
                # Draw tracker on index finger tip
                cv2.circle(img, (ix, iy), 8, (0, 165, 255), -1)
                
                # Check distance between index and middle finger tips (Click action)
                distance = np.hypot(ix - mx, iy - my)
                
                if distance < 30 and self.cooldown == 0:
                    # Check which key boundary the index finger falls into
                    for row_idx, row in enumerate(keys):
                        for col_idx, key in enumerate(row):
                            x = col_idx * key_w + 20
                            y = row_idx * key_h + 50
                            if x < ix < x + key_w and y < iy < y + key_h:
                                cv2.rectangle(img, (x, y), (x + key_w - 5, y + key_h - 5), (0, 255, 0), -1)
                                # Global text reference update safely
                                st.session_state.typed_text += key
                                self.cooldown = 15  # Cooldown delay

        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Create two columns layout: left for streaming, right for viewing your live output
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="virtual-keyboard",
        video_processor_factory=KeyboardProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📝 Live Typed Output")
    st.write("---")
    st.title(f"`{st.session_state.typed_text}`")
    if st.button("Clear Text"):
        st.session_state.typed_text = ""
        st.rerun()