import streamlit as st
from st_pages import Page, show_pages, add_page_title

# Add the title and icon to the current page
# add_page_title()

# Specify the pages to be shown in the sidebar with titles and icons
show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page("pages/face-detection.py", "Nhận diện khuôn mặt", "🤦"),
        Page("pages/fruit-identification.py", "Nhận diện trái cây", "🍎"),
        Page("pages/recognize-handwritten-digits.py", "Nhận dạng chữ số MNIST", "🖍️"),
        Page("pages/digital-image-processing.py", "Xử lý ảnh", "🎞️"),
        Page("pages/sample.py", "X", "🔥"),
    ]
)

# Set the title of the main page
# st.title("Digital Image Processing Project")

# Introduction section
st.write(
    """
# Welcome to the Digital Image Processing Project! 👋

This project demonstrates various applications of digital image processing using Streamlit. You can explore different functionalities through the sidebar.

### Available Features
- **Face Recognition**: Recognize faces of at least 5 people.
- **Object Detection**: Identify 5 types of objects using YOLOv8.
- **Handwritten Digit Recognition**: Recognize handwritten digits from the MNIST dataset.
- **Image Processing**: Apply various image processing techniques.

### Additional Information
If you complete all four tasks perfectly, you can earn up to 8 points. Additional tasks related to image processing can earn you extra points, with a maximum of 2 bonus points.

### Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [Community Forums](https://discuss.streamlit.io)
"""
)

# Enhanced UI/UX for main page
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://www.transparenttextures.com/patterns/scribble-light.png')
    }
    .sidebar .sidebar-content {
        background: url('https://www.transparenttextures.com/patterns/scribble-light.png')
    }
    h1 {
        color: #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.image("/mnt/data/image.png", caption="Project Overview")

# Footer section with additional links
st.markdown(
    """
    ### Learn More
    - Check out the official [Streamlit website](https://streamlit.io)
    - Visit our [documentation](https://docs.streamlit.io)
    - Join the discussion on our [community forums](https://discuss.streamlit.io)
    """
)
