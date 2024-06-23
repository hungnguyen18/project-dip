import streamlit as st
import numpy as np
import cv2 as cv
import os
from datetime import datetime
from PIL import Image
import time
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Helper functions
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_face_images(folder_name, cap):
    count = 0
    progress = st.progress(0)
    saved_images = []
    while count < 150:
        ret, frame = cap.read()
        if not ret:
            st.write("Không thể chụp khung hình")
            break
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            img_name = f"{folder_name}/{str(count)}.jpg"
            cv.imwrite(img_name, face_img)
            saved_images.append(img_name)
            progress.progress(int((count / 150) * 100))
        if count >= 150:
            break
    cap.release()
    return saved_images

class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 

def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            ext = os.path.splitext(f)[1]
            if ext in ['.jpg', '.jpeg', '.bmp']:
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

# Helper functions
def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Streamlit UI for Face Detection
st.title('🤦 Ứng dụng Nhận diện Khuôn mặt')
st.write("Ứng dụng này giúp bạn thu thập, huấn luyện và nhận diện khuôn mặt từ camera theo thời gian thực. Bạn có thể thu thập hình ảnh khuôn mặt để huấn luyện mô hình và sau đó sử dụng mô hình này để nhận diện khuôn mặt.")

st.sidebar.title("Điều hướng")
st.sidebar.write("Ứng dụng này giúp bạn thu thập, huấn luyện và nhận diện khuôn mặt từ camera theo thời gian thực. Bạn có thể thu thập hình ảnh khuôn mặt để huấn luyện mô hình và sau đó sử dụng mô hình này để nhận diện khuôn mặt.")
choice = st.sidebar.selectbox("Chọn hành động", ["Huấn luyện", "Nhận diện khuôn mặt", "Xem Khuôn mặt đã Lưu"])

def train_model(metadata):
    if len(set(m.name for m in metadata)) < 2:
        st.error("Cần ít nhất hai người khác nhau để huấn luyện mô hình.")
        return

    recognizer = cv.FaceRecognizerSF.create(
        "./models/face_recognition_sface_2021dec.onnx", "")

    embedded = np.zeros((metadata.shape[0], 128))
    progress = st.progress(0)

    for i, m in enumerate(metadata):
        img = cv.imread(m.image_path(), cv.IMREAD_COLOR)
        face_feature = recognizer.feature(img)
        embedded[i] = face_feature
        progress.progress(int((i / len(metadata)) * 100))

    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)

    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 5 != 0
    test_idx = np.arange(metadata.shape[0]) % 5 == 0
    X_train = embedded[train_idx]
    X_test = embedded[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    svc = SVC(probability=True)  # Using SVC with probability estimation
    svc.fit(X_train, y_train)
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    st.write(f'Độ chính xác của SVM: {acc_svc:.6f}')
    joblib.dump(svc, './models/svc.pkl')
    joblib.dump(encoder, './models/label_encoder.pkl')
    st.success("Huấn luyện mô hình hoàn tất và đã lưu.")

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar sections
if choice == "Huấn luyện":
    st.sidebar.header("Chế độ Huấn luyện")
    st.header("Chế độ Huấn luyện")
    st.write("""
        ### Hướng dẫn:
        1. Nhấn **Bắt đầu Camera** để bắt đầu chụp hình ảnh.
        2. Khi camera đang chạy, nhấn **Lưu** và nhập tên để lưu hình ảnh khuôn mặt.
        3. Hình ảnh khuôn mặt sẽ được lưu và hiển thị bên dưới.
    """)

    if 'cap' not in st.session_state:
        st.session_state.cap = None

    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        start_button = st.sidebar.button("Bắt đầu Camera")
    else:
        start_button = None
    
    if start_button:
        st.session_state.cap = cv.VideoCapture(0)
        st.session_state.start = True

    if 'start' in st.session_state and st.session_state.start:
        FRAME_WINDOW = st.image([])
        stop_button = st.sidebar.button("Dừng Camera")
        save_button = st.sidebar.button("Lưu")

        if stop_button:
            st.session_state.cap.release()
            st.session_state.start = False

        prev_time = 0

        while st.session_state.start:
            curr_time = time.time()
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.write("Không thể chụp khung hình")
                break
            faces = face_cascade.detectMultiScale(frame, 1.1, 4)
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            FRAME_WINDOW.image(frame, channels="BGR")

            if save_button:
                name = st.text_input("Nhập tên cho khuôn mặt", key='name_input')
                if name:
                    folder_name = f"./data/faces/{name}"
                    create_folder(folder_name)
                    
                    with st.spinner('Đang lưu hình ảnh...'):
                        saved_images = save_face_images(folder_name, st.session_state.cap)
                        st.balloons()  # Show visual feedback for success
                        st.success(f"Đã lưu 150 hình ảnh vào {folder_name}")
                        st.write(f"Đã lưu hình ảnh cho {name}")

                        # Display saved images with timestamp
                        st.header(f"Hình ảnh cho {name}")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        cols = [col1, col2, col3, col4, col5]
                        for idx, img_path in enumerate(saved_images):
                            img = Image.open(img_path)
                            cols[idx % 5].image(img, caption=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if choice == "Nhận diện khuôn mặt":
    st.sidebar.header("Chế độ Nhận diện")
    st.header("Chế độ Nhận diện")
    st.write("""
        ### Hướng dẫn:
        1. Đảm bảo rằng các mô hình đã được huấn luyện và lưu trước đó.
        2. Nhập đường dẫn đến mô hình phát hiện và nhận diện khuôn mặt.
        3. Điều chỉnh các thông số như ngưỡng điểm, ngưỡng NMS, và Top K.
        4. Nhấn **Bắt đầu Nhận diện** để bắt đầu quá trình nhận diện khuôn mặt theo thời gian thực.
    """)

    # Load models and label dictionary
    svc = joblib.load('./models/svc.pkl')
    encoder = joblib.load('./models/label_encoder.pkl')

    face_detection_model_path = st.text_input('Đường dẫn đến mô hình phát hiện khuôn mặt', './models/face_detection_yunet_2023mar.onnx')
    face_recognition_model_path = st.text_input('Đường dẫn đến mô hình nhận diện khuôn mặt', './models/face_recognition_sface_2021dec.onnx')
    score_threshold = st.slider('Ngưỡng Điểm', 0.0, 1.0, 0.9)
    nms_threshold = st.slider('Ngưỡng NMS', 0.0, 1.0, 0.3)
    top_k = st.number_input('Top K', min_value=1, value=5000)
    
    if st.button('Bắt đầu Nhận diện'):
        cap = cv.VideoCapture(0)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        detector = cv.FaceDetectorYN.create(
            face_detection_model_path,
            "",
            (frameWidth, frameHeight),  # Update input size
            score_threshold,
            nms_threshold,
            top_k
        )
        recognizer = cv.FaceRecognizerSF.create(face_recognition_model_path, "")
        tm = cv.TickMeter()

        FRAME_WINDOW = st.image([])
        stop_detection = st.button('Dừng Nhận diện')
        while cap.isOpened():
            hasFrame, frame = cap.read()
            if not hasFrame:
                st.write('Không có khung hình nào!')
                break

            detector.setInputSize((frameWidth, frameHeight))  # Ensure the input size is set correctly
            
            tm.start()
            faces = detector.detect(frame)
            tm.stop()

            if faces[1] is not None:
                for face in faces[1]:
                    face_align = recognizer.alignCrop(frame, face)
                    face_feature = recognizer.feature(face_align)
                    test_predict = svc.predict(face_feature)
                    test_proba = svc.predict_proba(face_feature)
                    if max(test_proba[0]) > 0.6:  # Threshold for recognizing the face
                        result = encoder.inverse_transform(test_predict)[0]
                        coords = face[:-1].astype(np.int32)
                        cv.putText(frame, result, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                    else:
                        coords = face[:-1].astype(np.int32)
                        cv.putText(frame, 'Không rõ', (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 0, 255), 2)

            visualize(frame, faces, tm.getFPS())
            FRAME_WINDOW.image(frame, channels="BGR")

            if stop_detection:
                break

        cap.release()
        cv.destroyAllWindows()

if choice == "Xem Khuôn mặt đã Lưu":
    st.sidebar.header("Xem Khuôn mặt đã Lưu")
    st.header("Xem Khuôn mặt đã Lưu")
    st.write("""
        ### Hướng dẫn:
        1. Chọn tên từ danh sách để xem các hình ảnh đã lưu cho khuôn mặt đó.
        2. Các hình ảnh sẽ được hiển thị bên dưới.
    """)

    if os.path.exists('./data/faces'):
        face_folders = os.listdir('./data/faces')
        selected_face = st.sidebar.selectbox("Chọn khuôn mặt để xem hình ảnh", face_folders)

        if selected_face:
            folder_path = f"./data/faces/{selected_face}"
            st.header(f"Hình ảnh cho {selected_face}")
            col1, col2, col3, col4, col5 = st.columns(5)
            cols = [col1, col2, col3, col4, col5]
            for idx, img in enumerate(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, img)
                img = Image.open(img_path)
                cols[idx % 5].image(img, caption=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        st.write("Chưa có khuôn mặt nào được huấn luyện.")
