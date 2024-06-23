import streamlit as st
import numpy as np
import cv2 as cv
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
from PIL import Image


class IdentityMetadata:
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
            if ext == ".jpg" or ext == ".jpeg" or ext == ".bmp":
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(
                input,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                (0, 255, 0),
                thickness,
            )
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(
        input, f"FPS: {fps:.2f}", (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )


def train_model(metadata):
    if len(set(m.name for m in metadata)) < 2:
        st.error("Need images of at least two different people to train the model.")
        return

    recognizer = cv.FaceRecognizerSF.create(
        "./models/face_recognition_sface_2021dec.onnx", ""
    )

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
    st.write(f"SVM accuracy: {acc_svc:.6f}")
    joblib.dump(svc, "./models/svc.pkl")
    joblib.dump(encoder, "./models/label_encoder.pkl")
    st.success("Model training completed and saved.")
    progress.empty()


def save_face_images(folder_name, cap, detector):
    count = 0
    progress = st.progress(0)
    saved_images = []
    info_message = st.info(
        "Vui lòng giữ khuôn mặt của bạn trong tầm camera. Nên tạo nhiều kiểu gương mặt khác nhau để tăng chất lượng mô hình."
    )
    success = st.success
    with st.spinner("Đang lưu ảnh..."):
        while count < 150:
            ret, frame = cap.read()
            if not ret:
                st.write("Không thể chụp khung hình")
                break
            detector.setInputSize((frame.shape[1], frame.shape[0]))  # Update input size
            faces = detector.detect(frame)
            if faces[1] is not None:
                for idx, face in enumerate(faces[1]):
                    coords = face[:-1].astype(np.int32)
                    x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                    count += 1
                    face_img = frame[y : y + h, x : x + w]
                    img_name = f"{folder_name}/{str(count)}.jpg"
                    cv.imwrite(img_name, face_img)
                    saved_images.append(img_name)
                    progress.progress(int((count / 150) * 100))
                    if count >= 150:
                        break
    cap.release()
    if count >= 150:
        success("Đã lưu 150 ảnh thành công.")
    info_message.empty()
    progress.empty()
    return saved_images


# Set up Streamlit app
st.set_page_config(page_title="Face Detection and Recognition", layout="wide")
st.title("Real-Time Face Detection and Recognition")

# Path to models (ensure these paths are correct)
face_detection_model_path = "./models/face_detection_yunet_2023mar.onnx"
face_recognition_model_path = "./models/face_recognition_sface_2021dec.onnx"

# Initialize models
detector = cv.FaceDetectorYN.create(
    face_detection_model_path, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv.FaceRecognizerSF.create(face_recognition_model_path, "")

# Streamlit state management
if "start" not in st.session_state:
    st.session_state.start = False
if "stop" not in st.session_state:
    st.session_state.stop = False
if "save" not in st.session_state:
    st.session_state.save = False
if "face_name" not in st.session_state:
    st.session_state.face_name = ""
if "saved_images" not in st.session_state:
    st.session_state.saved_images = []


def start_button_callback():
    st.session_state.start = True
    st.session_state.stop = False


def stop_button_callback():
    st.session_state.stop = True
    st.session_state.start = False
    st.session_state.save = False


def save_button_callback():
    st.session_state.save = True


def clear_session_state():
    st.session_state.start = False
    st.session_state.stop = True
    st.session_state.save = False


# Define tabs
tab1, tab2, tab3 = st.tabs(["Training", "Detect Face", "View Saved Images"])

with tab1:
    st.header("Training")
    st.write(
        """
    - **Bước 1**: Nhấn nút `Start Training` để bắt đầu.
    - **Bước 2**: Nhấn nút `Save` và nhập tên khuôn mặt để bắt đầu lưu hình ảnh.
    - **Bước 3**: Camera sẽ chụp 150 tấm hình của khuôn mặt bạn. Vui lòng giữ khuôn mặt trong tầm camera và thay đổi biểu cảm khuôn mặt để cải thiện chất lượng mô hình.
    - **Bước 4**: Sau khi lưu đủ 150 hình, quá trình huấn luyện mô hình sẽ tự động bắt đầu.
    """
    )
    if not st.session_state.start:
        if st.button("Start Training"):
            start_button_callback()
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if not st.session_state.stop:
                if st.button("Stop"):
                    stop_button_callback()
        with col2:
            if not st.session_state.save:
                if st.button("Save"):
                    save_button_callback()

        if st.session_state.save:
            face_name = st.text_input("Nhập tên khuôn mặt:")
            if face_name:
                st.session_state.face_name = face_name
                folder_name = f"./data/faces/{face_name}"
                os.makedirs(folder_name, exist_ok=True)
                cap = cv.VideoCapture(0)

                saved_images = save_face_images(folder_name, cap, detector)

                if saved_images:
                    # # Hide loading and info, success messages before training
                    # info_message.empty()
                    # spinner.empty()
                    # success_message.empty()

                    # Train the model after saving the images
                    # Initialize the training_message
                    training_message = st.empty()
                    training_message.text("Training started...")
                    metadata = load_metadata("./data/faces")
                    train_model(metadata)

                    # Hide training message
                    training_message.empty()

                st.session_state.save = False

        # Capture from camera
        cap = cv.VideoCapture(0)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

        if cap.isOpened():
            tm = cv.TickMeter()
            stframe = st.empty()

            while not st.session_state.stop:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    st.write("No frames grabbed!")
                    break

                # Inference
                tm.start()
                detector.setInputSize(
                    (frame.shape[1], frame.shape[0])
                )  # Update input size
                faces = detector.detect(frame)
                tm.stop()

                # Draw results on the input image
                visualize(frame, faces, tm.getFPS())

                # Display frame
                stframe.image(frame, channels="BGR")

            cap.release()
        else:
            st.write("Failed to open webcam.")

        # Close OpenCV windows
        cv.destroyAllWindows()
        if st.session_state.stop:
            st.session_state.start = False
            st.session_state.stop = False

with tab2:
    st.header("Detect Face")
    st.write(
        "This tab will be used for real-time face detection once the model is trained."
    )
    if st.session_state.start:
        st.write("Please stop training before detecting faces.")
    else:
        # Add your face detection code here
        pass

with tab3:
    st.write(
        """
        ### Hướng dẫn:
        1. Chọn tên từ danh sách để xem các hình ảnh đã lưu cho khuôn mặt đó.
        2. Các hình ảnh sẽ được hiển thị bên dưới.
    """
    )
    if st.session_state.start:
        st.write("Please stop training before viewing saved images.")
    else:
        if os.path.exists("./data/faces"):
            face_folders = os.listdir("./data/faces")
            selected_face = st.selectbox("Chọn khuôn mặt để xem hình ảnh", face_folders)

            if selected_face:
                folder_path = f"./data/faces/{selected_face}"
                st.header(f"Hình ảnh cho {selected_face}")
                col1, col2, col3, col4, col5 = st.columns(5)
                cols = [col1, col2, col3, col4, col5]
                for idx, img in enumerate(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, img)
                    img = Image.open(img_path)
                    cols[idx % 5].image(
                        img, caption=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
        else:
            st.write("Chưa có khuôn mặt nào được train.")

# To run the Streamlit app, save this script and run it using:
# streamlit run your_script_name.py
