import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import tempfile

# --- Page Config ---
st.set_page_config(page_title="Motorcycle ANPR System", page_icon="🏍️", layout="wide")

st.title("🏍️ ANPR: Automatic Number Plate Recognition")
st.markdown("Upload an image or video to detect motorcycle license plates and extract text.")

# --- Load Model & OCR (Cached) ---
@st.cache_resource
def load_model():
    # Load the YOLO model
    model = YOLO("best.pt")
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=False for free CPU hosting
    return model, reader

try:
    with st.spinner("Loading Model... Please wait."):
        model, reader = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Make sure 'best.pt' is in the same folder. {e}")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
source_type = st.sidebar.radio("Select Source", ["Image", "Video"])

# --- Main Logic ---
if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        # Inference button
        if st.button("Detect License Plate"):
            with st.spinner("Processing..."):
                results = model.predict(image, conf=confidence_threshold)
                
                # Draw boxes and extract text
                for result in results:
                    for box in result.boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Crop plate for OCR
                        plate_crop = image[y1:y2, x1:x2]
                        
                        # OCR
                        try:
                            ocr_result = reader.readtext(plate_crop)
                            text = " ".join([res[1] for res in ocr_result]).upper()
                        except:
                            text = "Unknown"
                            
                        # Draw on image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        st.info(f"Detected Text: **{text}**")

                with col2:
                    st.image(image, caption="Processed Image", use_container_width=True)

elif source_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        st.video(uploaded_video)
        
        if st.button("Process Video (First 100 Frames)"):
            st.warning("Video processing on free CPU tiers is slow. Processing first 100 frames only.")
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            frame_count = 0
            
            while cap.isOpened() and frame_count < 100:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame, conf=confidence_threshold)
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # (Optional) Skip OCR on video frames for speed, or add it here
                
                st_frame.image(frame, caption=f"Frame {frame_count}", use_container_width=True)
                frame_count += 1
            
            cap.release()
            st.success("Processing complete!")