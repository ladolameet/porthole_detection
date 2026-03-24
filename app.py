import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detection", layout="wide")

st.title("🚧 Pothole & Road Anomaly Detection System")

# Load model
MODEL_PATH = "best.pt"  # Put your trained model here
model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    
    st.video(uploaded_file)

    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Output file
    output_path = "output.mp4"

    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.write("⏳ Processing video...")

    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        frame_count += 1
        progress.progress(frame_count / total_frames)

    cap.release()
    out.release()

    st.success("✅ Processing Complete!")

    # Show output video
    st.video(output_path)

    # Download button
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Download Processed Video",
            data=f,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )