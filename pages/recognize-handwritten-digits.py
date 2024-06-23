import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import random

# Register the Sequential class
get_custom_objects()['Sequential'] = Sequential

def create_random_image():
    image = np.zeros((10*28, 10*28), np.uint8)
    data = np.zeros((100, 28, 28, 1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x*28:(x+1)*28, y*28:(y+1)*28] = sample[:, :, 0]
    return image, data

# Sidebar customization
st.sidebar.title("Ứng dụng nhận dạng chữ số MNIST")
st.sidebar.write("""
    ### Hướng dẫn:
    1. Nhấn "Tạo Ảnh" để tạo một hình ảnh chứa các chữ số ngẫu nhiên.
    2. Nhấn "Nhận diện" để mô hình nhận diện các chữ số trong hình ảnh.
""")

st.title("🖍️ Ứng dụng nhận dạng chữ số MNIST")
st.write("""
### Tạo ảnh ngẫu nhiên từ bộ dữ liệu MNIST và nhận diện các chữ số
Nhấn vào nút "Tạo Ảnh" để tạo một hình ảnh mới từ các chữ số ngẫu nhiên.
Sau đó nhấn "Nhận diện" để nhận diện các chữ số trong ảnh.
""")

if 'is_load' not in st.session_state:
    with st.spinner('Đang tải mô hình và dữ liệu...'):
        # Load model
        model_architecture = './models/digit_config.json'
        model_weights = './models/digit_weight.h5'
        model = model_from_json(open(model_architecture).read())
        model.load_weights(model_weights)

        OPTIMIZER = tf.keras.optimizers.Adam()
        model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
        st.session_state.model = model

        # Load data
        (_, _), (X_test, _) = datasets.mnist.load_data()
        X_test = X_test.reshape((10000, 28, 28, 1))
        st.session_state.X_test = X_test

        st.session_state.is_load = True
        st.success('Đã tải mô hình và dữ liệu thành công!')
else:
    st.info('Mô hình và dữ liệu đã được tải trước đó.')

if st.button('Tạo Ảnh'):
    with st.spinner('Đang tạo ảnh ngẫu nhiên...'):
        image, data = create_random_image()
        st.session_state.image = image
        st.session_state.data = data
        st.success('Đã tạo ảnh thành công!')

if 'image' in st.session_state:
    st.image(st.session_state.image, caption='Ảnh chứa các chữ số ngẫu nhiên', use_column_width=True)

    if st.button('Nhận diện'):
        with st.spinner('Đang nhận diện các chữ số...'):
            data = st.session_state.data
            data = data / 255.0
            data = data.astype('float32')
            results = st.session_state.model.predict(data)
            count = 0
            s = ''
            for x in results:
                s = s + '%d ' % (np.argmax(x))
                count = count + 1
                if (count % 10 == 0) and (count < 100):
                    s = s + '\n'
            st.text(s)
            st.success('Nhận diện hoàn tất!')
