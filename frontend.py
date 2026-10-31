
import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(
    page_title="Skardu Road AI",
    page_icon="🏔️",
    layout="wide"
)

st.title("🏔️ Skardu Road Obstacle Detection")
st.write("AI-powered road object detection system")

API_URL = "http://127.0.0.1:8000/detect"

uploaded_file = st.file_uploader(
    "Upload a mountain road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("Original Road Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("AI is detecting objects..."):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=120
                )

                response.raise_for_status()
                data = response.json()

                st.success("Detection Completed!")

                st.metric(
                    "Total Objects Detected",
                    data["total_objects"]
                )

                if data["detections"]:
                    st.subheader("Detection Results")
                    st.dataframe(data["detections"])

                    image_data = base64.b64decode(
                        data["annotated_image"]
                    )

                    detected_image = Image.open(
                        BytesIO(image_data)
                    )

                    st.subheader("AI Detection Image")
                    st.image(
                        detected_image,
                        caption="Objects detected with bounding boxes",
                        use_container_width=True
                    )
                else:
                    st.info("No supported objects detected.")

            except requests.exceptions.RequestException as error:
                st.error(f"Backend connection error: {error}")
