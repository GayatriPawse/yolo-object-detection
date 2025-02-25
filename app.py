# import cv2
# import numpy as np  # Fixed import (numpy as numpy → numpy as np)

# # Load YOLO
# net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")  # Ensure correct file names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Load COCO class labels
# with open("coco-labels-2014_2017.txt", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Initialize video capture (use 0 for webcam or "video.mp4" for a file)
# cap = cv2.VideoCapture("peoplenyc.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Stop if video ends

#     height, width, channels = frame.shape

#     # Prepare Image for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)  # Fixed 1/225 → 1/255
#     net.setInput(blob)

#     # Perform Forward Pass (Object Detection)
#     detections = net.forward(output_layers)

#     # Process YOLO Output
#     boxes, confidences, class_ids = [], [], []  # Ensure lists are initialized before use
#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]  # Fixed 'score' typo
#             if confidence > 0.5:  # Filter weak detections
#                 center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, int(w), int(h)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Non-Maximum Suppression (NMS) to remove redundant boxes
#     if len(boxes) > 0:  # Ensure there are detections before applying NMS
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Show Video Feed
#     cv2.imshow("YOLO Real-Time Object Detection", frame)

#     # Press 'q' to exit the video stream
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os

# Load YOLO Model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco-labels-2014_2017.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Streamlit UI
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🚀 YOLO Object Detection with Streamlit")

# Sidebar UI
st.sidebar.header("🔍 Upload a Video for Detection")
uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
stop_button = st.sidebar.button("⛔ Stop Detection")

st.sidebar.markdown("---")
st.sidebar.info("This app runs real-time object detection on videos using the YOLO model. Upload a video to start.")

def process_frame(frame):
    """Runs YOLO detection on a single frame."""
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:  # Ensure indices is not empty before flattening
        indices = np.array(indices).flatten()
        for i in indices:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()  # ✅ Close the temp file to release it

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    fps_display = st.sidebar.empty()  # Display FPS

    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        if stop_button:
            st.warning("⛔ Detection Stopped by User")
            break

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        stframe.image(processed_frame, channels="BGR", use_container_width=True)


        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fps_display.write(f"🎯 FPS: {fps:.2f}")

    cap.release()  # ✅ Fully release the video file
    cv2.destroyAllWindows()  # ✅ Close OpenCV windows

    try:
        os.remove(tfile.name)  # ✅ Safe deletion after release
    except PermissionError:
        st.error(f"⚠️ Could not delete temp file: {tfile.name}. Please delete it manually.")

    st.success("✅ Video processing completed!")
