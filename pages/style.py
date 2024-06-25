import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page configuration


# Function to convert photo to sketch
def photo_to_sketch(image, blur_intensity):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = 255 - gray_image
    blurred = cv2.GaussianBlur(inverted_image, (blur_intensity, blur_intensity), 0)
    inverted_blurred = 255 - blurred
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    return sketch


# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            color: #31333F;
            font-family: 'Source Sans Pro', sans-serif;
        }
        .header {
            color: #31333F;
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 24px;
        }
        .info {
            color: #5C5C5C;
            font-family: 'Source Sans Pro', sans-serif;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# Title and instructions
st.title("Chuyển ảnh thành tranh vẽ", anchor="title")
st.markdown(
    "<p class='info'>Tải ảnh lên và tùy chỉnh để chuyển đổi ảnh thành tranh vẽ.</p>",
    unsafe_allow_html=True,
)

# Sidebar settings
st.sidebar.title("Cài đặt")
blur_intensity = st.sidebar.slider("Độ mờ", 1, 101, 21, step=2)

# About section
with st.expander("Về ứng dụng"):
    st.markdown(
        "<p class='info'>Ứng dụng này giúp bạn chuyển đổi ảnh thành tranh vẽ bằng cách sử dụng kỹ thuật xử lý ảnh. Điều chỉnh độ mờ để có hiệu ứng tốt nhất.</p>",
        unsafe_allow_html=True,
    )

# File uploader
uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Read image file
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Show progress indicator
        with st.spinner("Đang xử lý..."):
            sketch_image = photo_to_sketch(image, blur_intensity)

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 class='header'>Ảnh gốc</h3>", unsafe_allow_html=True)
            st.image(image, caption="Ảnh gốc", use_column_width=True)

        with col2:
            st.markdown(
                "<h3 class='header'>Tranh vẽ từ ảnh gốc</h3>", unsafe_allow_html=True
            )
            st.image(sketch_image, caption="Tranh vẽ từ ảnh gốc", use_column_width=True)

        # Download button
        _, col, _ = st.columns([2, 1, 2])
        with col:
            st.download_button(
                label="Tải tranh vẽ về",
                data=cv2.imencode(".png", sketch_image)[1].tobytes(),
                file_name="sketch.png",
                mime="image/png",
            )
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
else:
    st.info("Vui lòng tải ảnh lên để chuyển thành tranh vẽ.")
