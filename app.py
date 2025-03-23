import streamlit as st

# Check for required packages
try:
    import cv2
    import numpy as np
    from deepface import DeepFace
except ImportError as e:
    st.error(f"""
        Error: {e}
        Please install the required packages using:
        ```
        pip install opencv-python deepface streamlit numpy
        ```
    """)
    st.stop()

# Streamlit UI
st.title("🎭 Real-Time Emotion Detection App")
st.write("Detects emotions using your webcam.")

frame_placeholder = st.empty()
stop_button = st.button("Stop Emotion Detection", key="stop")

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Ensures compatibility with Windows DirectShow

if st.button("Start Emotion Detection"):
    if not cap.isOpened():
        st.error("🚨 Could not access the webcam. Make sure it's properly connected.")
        st.stop()

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to capture image.")
            break

        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

            # Extract dominant emotion
            if isinstance(result, list) and len(result) > 0:
                emotion = result[0]["dominant_emotion"]
            else:
                emotion = "No face detected"

            # Display results on frame
            cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert frame for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

st.write("Press 'Start Emotion Detection' to begin.")
