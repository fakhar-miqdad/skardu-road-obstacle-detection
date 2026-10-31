import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Skardu Road AI",
    page_icon="🏔️",
    layout="wide"
)

st.title("🏔️ Skardu Road Obstacle Detection")
st.write("AI-powered obstacle detection for mountain roads.")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a mountain road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("AI is detecting objects..."):
            results = model.predict(
                np.array(image),
                conf=0.25
            )

            result = results[0]
            st.success("Detection completed!")

            st.metric("Total Objects Detected", len(result.boxes))

            if len(result.boxes) > 0:
                st.subheader("Detection Results")

                detections = []
                for box in result.boxes:
                    detections.append({
                        "Object": model.names[int(box.cls[0])],
                        "Confidence": f"{float(box.conf[0]) * 100:.2f}%"
                    })

                st.dataframe(detections)

            else:
                st.info("No supported objects detected.")

            st.subheader("AI Detection Image")
            st.image(
                result.plot()[..., ::-1],
                caption="Detected objects with bounding boxes",
                use_container_width=True
            )