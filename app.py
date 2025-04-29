import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import av
import aubio
import google.generativeai as genai

# ✅ Gemini API Key (Hardcoded)
API_KEY = "AIzaSyAW_b4mee9l8eP931cqd9xqErHV34f7OEw"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# ✅ Note-to-Prompt Mapping
note_prompts = {
    "E": "Write a Python function to reverse a string.",
    "A": "Write a Python for-loop that prints numbers from 1 to 10.",
    "D": "Write a Python function to check if a number is prime.",
    "G": "Generate a Python class for a basic calculator.",
    "B": "Write a Python script to read a text file and print its contents.",
    "C": "Write a Python program that uses recursion to calculate factorial."
}

# ✅ Frequency to Note Conversion
def freq_to_note(freq):
    if freq == 0:
        return None
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4 = 440.0
    semitones = int(round(12 * np.log2(freq / A4)))
    note_index = (semitones + 9) % 12
    return note_names[note_index]

# ✅ Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.pitch_o = aubio.pitch("default", 1024, 512, 44100)
        self.pitch_o.set_unit("Hz")
        self.pitch_o.set_silence(-40)
        self.last_note = None

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        samples = frame.to_ndarray().astype(np.float32).flatten()
        pitch = self.pitch_o(samples)[0]
        note = freq_to_note(pitch)

        if note and note != self.last_note:
            self.last_note = note
            st.session_state["note"] = note
            st.session_state["frequency"] = pitch

            if note in note_prompts:
                prompt = note_prompts[note]
                st.session_state["prompt"] = prompt
                try:
                    response = model.generate_content(prompt)
                    st.session_state["response"] = response.text.strip()
                except Exception as e:
                    st.session_state["response"] = f"❌ Gemini API Error: {e}"
        return frame

# ✅ Streamlit UI
st.set_page_config(page_title="🎸 Guitar-to-Gemini", layout="centered")
st.title("🎸 Live Guitar-to-Gemini Code Generator")
st.markdown("Play a single guitar string into your microphone. This app detects the note and sends a prompt to Gemini Pro to generate Python code.")

# ✅ Streamlit output placeholders
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎵 Detected Note")
    st.info(st.session_state.get("note", "Waiting..."))
    st.write(f"**Frequency:** {st.session_state.get('frequency', 0):.2f} Hz")

with col2:
    st.subheader("🧠 Gemini Prompt")
    st.success(st.session_state.get("prompt", "Waiting for note to map..."))

st.subheader("💻 Python Code Output")
st.code(st.session_state.get("response", "# Play a note on your guitar to get started..."), language="python")

# ✅ Start WebRTC Streamer
webrtc_streamer(
    key="guitar-code",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor
)
