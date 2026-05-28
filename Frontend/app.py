import streamlit as st
import requests
import cv2
import numpy as np
import base64

st.set_page_config(
    page_title="Number Plate Detection",
    layout="centered"
)

st.title("🚘 Number Plate Detection System for Cars, Buses and Trucks")

# ======================================
# IMAGE COMPRESSION FUNCTION
# ======================================

def compress_image(image_bytes, max_size_kb=200):

    nparr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    quality = 95

    while quality >= 10:

        success, encoded_img = cv2.imencode(
            ".jpg",
            img,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        if not success:
            return image_bytes

        compressed_bytes = encoded_img.tobytes()

        size_kb = len(compressed_bytes) / 1024

        if size_kb <= max_size_kb:
            return compressed_bytes

        quality -= 5

    return compressed_bytes

# ======================================
# OPTION SELECTION
# ======================================

option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Capture From Camera"]
)

image_file = None

# ======================================
# UPLOAD OPTION
# ======================================

if option == "Upload Image":

    image_file = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )

# ======================================
# CAMERA OPTION
# ======================================

elif option == "Capture From Camera":

    image_file = st.camera_input(
        "Capture Vehicle Image"
    )

# ======================================
# PROCESS IMAGE
# ======================================

if image_file is not None:

    st.image(
        image_file,
        caption="Input Image",
        use_container_width=True
    )

    if st.button("Detect Number Plate"):

        # Backend URL
        FASTAPI_URL = "https://subhojit156-number-plate-detection-backend.hf.space/detect"

        # Compress image to <=200KB
        original_image = image_file.getvalue()

        compressed_image = compress_image(
            original_image,
            max_size_kb=200
        )

        st.write(
            f"Compressed Image Size: {len(compressed_image)/1024:.2f} KB"
        )

        files = {
            "file": compressed_image
        }

        with st.spinner("Detecting Number Plate..."):

            response = requests.post(
                FASTAPI_URL,
                files=files
            )

        # ======================================
        # RESPONSE
        # ======================================

        if response.status_code == 200:

            data = response.json()

            st.success("Detection Completed")

            # ======================================
            # SHOW DETECTED TEXT
            # ======================================

            if len(data["detections"]) > 0:

                st.subheader("Detected Number Plates")

                for det in data["detections"]:

                    st.write(
                        f"Plate Number: {det['text']}"
                    )

            else:
                st.warning(
                    "No Number Plate Detected"
                )

            # ======================================
            # SHOW OUTPUT IMAGE
            # ======================================

            image_data = base64.b64decode(
                data["image"]
            )

            nparr = np.frombuffer(
                image_data,
                np.uint8
            )

            img = cv2.imdecode(
                nparr,
                cv2.IMREAD_COLOR
            )

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                img,
                caption="Detection Output",
                use_container_width=True
            )

        else:
            st.error(
                "Backend API Error"
            )
