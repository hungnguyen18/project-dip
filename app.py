import streamlit as st
from st_pages import Page, show_pages, add_page_title


# Chỉ định các trang để hiển thị trong thanh bên với tiêu đề và biểu tượng
show_pages(
    [
        Page("app.py", "Trang chủ", "🏠"),
        Page("pages/face-detect.py", "Nhận diện khuôn mặt", "🤦"),
        Page("pages/fruit-identification.py", "Nhận diện trái cây", "🍎"),
        Page("pages/recognize-handwritten-digits.py", "Nhận dạng chữ số từ MNIST", "🖍️"),
        Page("pages/digital-image-processing.py", "Xử lý ảnh", "🎞️"),
        Page("pages/style.py", "Chuyển ảnh thành tranh vẽ", "🎨"),
    ]
)


# Phần giới thiệu
st.write(
    """
# Dự án kết thúc môn DIP! 🔥

Dự án này thể hiện các ứng dụng khác nhau của xử lý ảnh số bằng Streamlit.

### Các Chức Năng Có Sẵn
- **Nhận diện Khuôn mặt**: Nhận diện khuôn mặt của ít nhất 5 người.
- **Nhận diện Đối tượng**: Nhận biết 5 loại đối tượng sử dụng YOLOv8.
- **Nhận dạng Chữ số viết tay**: Nhận dạng chữ số viết tay từ bộ dữ liệu MNIST.
- **Xử lý Ảnh**: Áp dụng các kỹ thuật xử lý ảnh đa dạng.
- **Chuyển ảnh thành tranh vẽ**: Áp dụng các kỹ thuật xử lý ảnh đa dạng.

### Thành viên
- **Nguyễn Kim Hưng**
- **Đào Đức Khải**

### Tài Nguyên
- [Tài liệu Streamlit](https://docs.streamlit.io)
- [Diễn đàn cộng đồng](https://discuss.streamlit.io)
"""
)

# Giao diện người dùng nâng cao cho trang chính
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

# Phần chân trang với các liên kết bổ sung
st.markdown(
    """
    ### Tìm Hiểu Thêm
    - Ghé thăm trang web chính thức của [Streamlit](https://streamlit.io)
    - Xem tài liệu [ở đây](https://docs.streamlit.io)
    """
)
