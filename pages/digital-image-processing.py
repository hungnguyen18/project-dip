import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Hàm xử lý ảnh
def negative_image(image):
    return cv.bitwise_not(image)


def logarithmic_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image


def power_law_transform(image, gamma=1.0):
    image = np.array(255 * (image / 255) ** gamma, dtype="uint8")
    return image


def equalize_histogram(image):
    if len(image.shape) == 2:
        return cv.equalizeHist(image)
    else:
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)


def gaussian_blur(image, ksize):
    return cv.GaussianBlur(image, (ksize, ksize), 0)


def edge_detection(image):
    return cv.Canny(image, 100, 200)


# Ứng dụng Streamlit
# st.set_page_config(page_title="Xử lý ảnh số", page_icon="📷", layout="wide")
st.title("📷 Xử lý ảnh số")

st.sidebar.header("📂 Tải lên ảnh")
uploaded_file = st.sidebar.file_uploader("Tải lên một ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.sidebar.header("⚙️ Tùy chọn xử lý ảnh")
    option = st.sidebar.selectbox(
        "Chọn kiểu xử lý ảnh",
        (
            "Làm âm ảnh",
            "Logarit ảnh",
            "Lũy thừa ảnh",
            "Cân bằng Histogram",
            "Lọc Gaussian",
            "Phát hiện biên cạnh",
        ),
    )

    if option == "Làm âm ảnh":
        st.sidebar.markdown(
            "**Làm âm ảnh**: Đổi tất cả các điểm ảnh thành màu đối lập của chúng."
        )
        result = negative_image(image)
    elif option == "Logarit ảnh":
        st.sidebar.markdown(
            "**Logarit ảnh**: Áp dụng phép biến đổi logarit để tăng độ sáng của các điểm tối."
        )
        result = logarithmic_transform(image)
    elif option == "Lũy thừa ảnh":
        st.sidebar.markdown(
            "**Lũy thừa ảnh**: Áp dụng phép biến đổi gamma để điều chỉnh độ sáng của ảnh."
        )
        gamma = st.sidebar.slider("Chọn giá trị gamma", 0.1, 5.0, 1.0)
        result = power_law_transform(image, gamma)
    elif option == "Cân bằng Histogram":
        st.sidebar.markdown(
            "**Cân bằng Histogram**: Cân bằng lại histogram của ảnh để cải thiện độ tương phản."
        )
        result = equalize_histogram(image)
    elif option == "Lọc Gaussian":
        st.sidebar.markdown("**Lọc Gaussian**: Áp dụng lọc Gaussian để làm mờ ảnh.")
        ksize = st.sidebar.slider("Chọn kích thước kernel", 3, 15, 3, step=2)
        result = gaussian_blur(image, ksize)
    elif option == "Phát hiện biên cạnh":
        st.sidebar.markdown(
            "**Phát hiện biên cạnh**: Sử dụng thuật toán Canny để phát hiện các biên cạnh trong ảnh."
        )
        result = edge_detection(image)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Ảnh gốc")
        st.image(image, use_column_width=True)
    with col2:
        st.header("Ảnh sau khi xử lý")
        st.image(result, use_column_width=True)

    # Histogram visualization
    if st.sidebar.checkbox("Hiển thị Histogram"):
        st.sidebar.markdown(
            "**Hiển thị Histogram**: Hiển thị biểu đồ histogram của ảnh gốc và ảnh sau khi xử lý."
        )
        fig, ax = plt.subplots()
        if len(image.shape) == 2:  # Ảnh xám
            ax.hist(image.ravel(), bins=256, color="black", alpha=0.5, label="Gốc")
            ax.hist(result.ravel(), bins=256, color="red", alpha=0.5, label="Xử lý")
        else:  # Ảnh màu
            for i, color in enumerate(["r", "g", "b"]):
                ax.hist(
                    image[:, :, i].ravel(),
                    bins=256,
                    color=color,
                    alpha=0.5,
                    label=f"Gốc {color}",
                )
                if len(result.shape) == 2:
                    ax.hist(
                        result.ravel(),
                        bins=256,
                        color=color,
                        alpha=0.5,
                        linestyle="dashed",
                        label=f"Xử lý {color}",
                    )
                else:
                    ax.hist(
                        result[:, :, i].ravel(),
                        bins=256,
                        color=color,
                        alpha=0.5,
                        linestyle="dashed",
                        label=f"Xử lý {color}",
                    )
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Vui lòng tải lên một ảnh để bắt đầu xử lý.")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .css-1aumxhk {
            background-color: #f0f2f6;
        }
        .stButton>button {
            color: white;
            background: #4CAF50;
        }
        .stFileUploader>label {
            text-align: center;
        }
    </style>
""",
    unsafe_allow_html=True,
)
