import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os

# Thiết lập ứng dụng Streamlit
st.set_page_config(page_title="Nhận Diện Trái Cây", layout="wide")
st.sidebar.title('Nhận Diện Trái Cây')
st.sidebar.write("""
    ### Hướng dẫn:
    1. Tải lên một hình ảnh của một loại trái cây bằng cách sử dụng nút "Browse files".
    2. Điều chỉnh các thông số "Confidence Threshold" và "Overlap Threshold" để có kết quả dự đoán tốt nhất.
    3. Nhấn nút "Dự đoán" để mô hình nhận diện loại trái cây trong hình ảnh.
    
    ### Thông tin về mô hình:
    - **Confidence Threshold**: Ngưỡng tin cậy để quyết định mức độ chắc chắn của dự đoán.
    - **Overlap Threshold**: Ngưỡng chồng lấp để quyết định mức độ chồng lấp giữa các đối tượng trong hình ảnh.
    
    ### Chú ý:
    - Mô hình có thể nhận diện các loại trái cây phổ biến.
    - Vui lòng tải lên hình ảnh có định dạng bmp, png, jpg, hoặc jpeg.
""")
st.title('🍎 Nhận Diện Trái Cây')
st.write("Tải lên một hình ảnh của một loại trái cây và mô hình sẽ dự đoán loại trái cây đó.")

@st.cache_resource
def load_model():
    # Khởi tạo mô hình Roboflow
    rf = Roboflow(api_key="pdA499ZD1TBm2eMVDAxd")
    project = rf.workspace().project("fruits-97szb")
    model = project.version(3).model
    return model

# Tải mô hình với một spinner
with st.spinner('Đang tải mô hình...'):
    model = load_model()

# Tải lên file và nhập thông số
img_file_buffer = st.file_uploader("Tải lên một hình ảnh", type=["bmp", "png", "jpg", "jpeg"])

if img_file_buffer is not None:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, caption='Hình ảnh đã tải lên.', use_column_width=True)

        # Thêm thanh trượt để điều chỉnh thông số confidence và overlap
        st.write("Điều chỉnh các thông số để dự đoán:")
        confidence = st.slider('Confidence Threshold', min_value=0, max_value=100, value=80)
        overlap = st.slider('Overlap Threshold', min_value=0, max_value=100, value=40)

        if st.button('Dự đoán'):
            with st.spinner('Đang dự đoán...'):
                # Lưu hình ảnh đã tải lên vào một vị trí tạm thời
                img_path = os.path.join('/tmp', img_file_buffer.name)
                image.save(img_path)

                # Thực hiện suy luận trên hình ảnh cục bộ
                prediction = model.predict(img_path, confidence=confidence, overlap=overlap).json()

                # Hiển thị kết quả dự đoán
                st.write("Kết quả dự đoán:")
                st.json(prediction)

                # Lưu và hiển thị hình ảnh dự đoán
                prediction_img_path = os.path.join('/tmp', 'prediction_' + img_file_buffer.name)
                model.predict(img_path, confidence=confidence, overlap=overlap).save(prediction_img_path)
                prediction_img = Image.open(prediction_img_path)
                st.image(prediction_img, caption='Hình ảnh dự đoán.', use_column_width=True)
                
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")

else:
    st.info("Vui lòng tải lên một hình ảnh để bắt đầu.")
