import streamlit as st
import av
import cv2
import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Gesture Virtual Keyboard", layout="wide")

st.title("🖐 Gesture Virtual Keyboard")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class HandProcessor(VideoProcessorBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.8,
            max_num_hands=2
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    img,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="virtual-keyboard",
    video_processor_factory=HandProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)