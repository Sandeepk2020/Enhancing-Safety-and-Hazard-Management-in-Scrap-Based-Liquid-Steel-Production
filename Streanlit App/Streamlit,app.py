import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the trained YOLOv8 model
model_path = os.path.join(r"C:\Users\sande\Downloads\skbest.pt")
model = YOLO(model_path)

# Streamlit app title
st.title("Enhancing Safety and Hazard Management in Scrap-Based Liquid Steel Production")

# File uploader for video files
uploaded_file = st.file_uploader("Upload a Video or Image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        # Handle image file
        image = Image.open(uploaded_file)
        image = np.array(image)  # Convert image to numpy array

        # Perform detection on the image
        results = model(image)[0]  # Get the first result

        # Draw bounding boxes and labels on the image
        for result in results.boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            conf = result.conf[0]  # Confidence score
            cls = int(result.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image
        st.image(image, channels="BGR", use_column_width=True)

    elif uploaded_file.type.startswith("video"):
        # Handle video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_filename = tfile.name

        # Load the video with OpenCV
        video = cv2.VideoCapture(temp_filename)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        stframe = st.empty()  # Placeholder to display frames

        # Process frames and display results
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Perform detection on the frame
            results = model(frame)[0]  # Get the first result

            # Draw bounding boxes and labels on the frame
            for result in results.boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
                conf = result.conf[0]  # Confidence score
                cls = int(result.cls[0])  # Class ID
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            stframe.image(frame, channels="BGR", use_column_width=True)

        video.release()
        os.remove(temp_filename)  # Clean up temporary file

st.write("Upload a video or image file to see hazardous material detection in action.")