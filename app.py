import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av

st.set_page_config(page_title="Cloud-Safe Virtual Keyboard", layout="wide")
st.title("⌨️ Pure Vision Virtual Keyboard")
st.write("Hold up a bright colored object (like a blue pen cap or marker) to point and type!")

if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""

# Define simple keyboard keys layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

class ColorKeyboardProcessor(VideoProcessorBase):
    def __init__(self):
        self.cooldown = 0
        # Default HSV Range for a bright Blue object (adjust if you use Green/Red)
        self.lower_color = np.array([100, 100, 100])
        self.upper_color = np.array([140, 255, 255])

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Draw the virtual keys
        key_w, key_h = int(w / 11), 50
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                x = col_idx * key_w + 20
                y = row_idx * key_h + 50
                cv2.rectangle(img, (x, y), (x + key_w - 5, y + key_h - 5), (255, 255, 255), 2)
                cv2.putText(img, key, (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Track the colored object using HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.cooldown > 0:
            self.cooldown -= 1

        if len(contours) > 0:
            # Find the largest colored object cluster
            c = max(contours, key=cv2.contourArea)
            ((x_coord, y_coord), radius) = cv2.minEnclosingCircle(c)
            
            if radius > 10:
                cx, cy = int(x_coord), int(y_coord)
                # Draw a pointer target circle on the screen
                cv2.circle(img, (cx, cy), int(radius), (0, 255, 255), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                
                # If pointer hovers over a key, consider it pressed (auto-click style)
                if self.cooldown == 0:
                    for row_idx, row in enumerate(keys):
                        for col_idx, key in enumerate(row):
                            kx = col_idx * key_w + 20
                            ky = row_idx * key_h + 50
                            if kx < cx < kx + key_w and ky < cy < ky + key_h:
                                cv2.rectangle(img, (kx, ky), (kx + key_w - 5, ky + key_h - 5), (0, 255, 0), -1)
                                st.session_state.typed_text += key
                                self.cooldown = 12  # Cooldown frames

        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

col1, col2 = st.columns([2, 1])
with col1:
    webrtc_streamer(
        key="color-keyboard",
        video_processor_factory=ColorKeyboardProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📝 Live Typed Output")
    st.title(f"`{st.session_state.typed_text}`")
    if st.button("Clear Text"):
        st.session_state.typed_text = ""
        st.rerun()