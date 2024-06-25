import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os
from datetime import datetime
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


# Class to store metadata for each identity
class IdentityMetadata:
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


# Load metadata for images
def load_metadata(path):
    metadata = []
    for name in sorted(os.listdir(path)):
        for file in sorted(os.listdir(os.path.join(path, name))):
            ext = os.path.splitext(file)[1]
            if ext.lower() in [".jpg", ".jpeg", ".bmp"]:
                metadata.append(IdentityMetadata(path, name, file))
    return np.array(metadata)


# Load an image using OpenCV
def load_image(path):
    img = cv.imread(path, 1)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


# Calculate the distance between two embeddings
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


# Visualize detected faces and FPS on the input image
def visualize(input_img, faces, fps, thickness=2, names=None):
    if faces[1] is not None:
        for i, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(
                input_img,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                (0, 255, 0),
                thickness,
            )
            for j in range(4, 14, 2):
                cv.circle(
                    input_img, (coords[j], coords[j + 1]), 2, (255, 0, 0), thickness
                )
            if names and i < len(names):
                cv.putText(
                    input_img,
                    names[i],
                    (coords[0], coords[1] - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
    cv.putText(
        input_img,
        f"FPS: {fps:.2f}",
        (1, 16),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


# Train the SVM model using face embeddings
def train_model(data_path, recognizer):
    metadata = load_metadata(data_path)
    embedded = np.zeros((metadata.shape[0], 128))

    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_step = 100 / metadata.shape[0]

    for i, m in enumerate(metadata):
        img = cv.imread(m.image_path(), cv.IMREAD_COLOR)
        face_feature = recognizer.feature(img)
        embedded[i] = face_feature

        # Update progress
        progress_bar.progress(int(progress_step * (i + 1)))
        status_text.text(f"Processing image {i + 1} of {metadata.shape[0]}")

    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 5 != 0
    test_idx = np.arange(metadata.shape[0]) % 5 == 0
    X_train, X_test = embedded[train_idx], embedded[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    acc_svc = accuracy_score(y_test, svc.predict(X_test))

    model_path = "./models/svc.pkl"
    joblib.dump(svc, model_path)

    progress_bar.empty()
    status_text.text("Training complete!")

    return acc_svc, model_path


# Get the list of face folders in the dataset
def get_face_folders(data_path):
    return sorted(
        [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    )


# Save face images captured from the webcam
def save_face_images(folder_name, cap, detector):
    count = 0
    progress = st.progress(0)
    saved_images = []
    info_message = st.info(
        "Please keep your face in the camera's view. Creating various expressions can improve the model's quality."
    )
    success = st.success
    with st.spinner("Saving images..."):
        while count < 150:
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to capture frame")
                break
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            faces = detector.detect(frame)
            if faces[1] is not None:
                for face in faces[1]:
                    coords = face[:-1].astype(np.int32)
                    x, y, w, h = coords[:4]
                    count += 1
                    face_img = frame[y : y + h, x : x + w]
                    img_name = f"{folder_name}/{count}.jpg"
                    cv.imwrite(img_name, face_img)
                    saved_images.append(img_name)
                    progress.progress(int((count / 150) * 100))
                    if count >= 150:
                        break
    cap.release()
    if count >= 150:
        success("Successfully saved 150 images.")
    info_message.empty()
    progress.empty()
    return saved_images


# Initialize Streamlit app
st.set_page_config(page_title="Face Detection and Recognition", layout="wide")
st.title("Real-Time Face Detection and Recognition")

# Paths to pre-trained models
face_detection_model_path = "./models/face_detection_yunet_2023mar.onnx"
face_recognition_model_path = "./models/face_recognition_sface_2021dec.onnx"

# Load face detection and recognition models
detector = cv.FaceDetectorYN.create(
    face_detection_model_path, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv.FaceRecognizerSF.create(face_recognition_model_path, "")

# Load trained SVM model and face folders
svc = joblib.load("./models/svc.pkl")
data_path = "./data/faces"
mydict = get_face_folders(data_path)

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


# Callback functions for buttons
def start_button_callback():
    st.session_state.start = True
    st.session_state.stop = False


def stop_button_callback():
    st.session_state.stop = True
    st.session_state.start = False


def save_button_callback():
    st.session_state.save = True


# Reset session state
def clear_session_state():
    st.session_state.start = False
    st.session_state.stop = True
    st.session_state.save = False


# Define Streamlit tabs
tab1, tab2, tab3 = st.tabs(["Training", "Detect Face", "View Saved Images"])

with tab1:
    st.header("Training")
    st.write(
        """
    - **Step 1**: Press the `Start Training` button to begin.
    - **Step 2**: Press the `Save` button and enter the face name to start saving images.
    - **Step 3**: The camera will capture 150 images of your face. Please keep your face in the camera's view and vary expressions to improve model quality.
    - **Step 4**: After saving 150 images, the model training will automatically start.
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
            face_name = st.text_input("Enter face name:")
            if face_name:
                st.session_state.face_name = face_name
                folder_name = f"./data/faces/{face_name}"
                os.makedirs(folder_name, exist_ok=True)
                cap = cv.VideoCapture(0)

                saved_images = save_face_images(folder_name, cap, detector)

                st.session_state.save = False

                if saved_images:
                    st.write("Training the model...")
                    acc_svc, model_path = train_model("./data/faces", recognizer)
                    st.write(f"SVM accuracy: {acc_svc:.6f}")
                    st.write(f"Model saved at: {model_path}")

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
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                faces = detector.detect(frame)
                tm.stop()

                fps = tm.getFPS()
                visualize(frame, faces, fps)
                stframe.image(frame, channels="BGR")

            cap.release()
            cv.destroyAllWindows()

with tab2:
    st.header("Detect Face")
    st.write(
        """
    - Press the `Start` button to begin face detection.
    - Press the `Stop` button to stop detection.
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.start:
            if st.button("Start"):
                start_button_callback()
    with col2:
        if st.session_state.start:
            if st.button("Stop"):
                stop_button_callback()

    if st.session_state.start:
        stframe = st.empty()
        st_fps = st.empty()
        cap = cv.VideoCapture(0)
        tm = cv.TickMeter()
        if cap.isOpened():
            detector = cv.FaceDetectorYN.create(
                face_detection_model_path,
                "",
                (320, 320),
                0.9,
                0.3,
                5000,
            )
            recognizer = cv.FaceRecognizerSF.create(face_recognition_model_path, "")

            frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            detector.setInputSize([frameWidth, frameHeight])

            with st.spinner("Detecting faces..."):
                while not st.session_state.stop:
                    hasFrame, frame = cap.read()
                    if not hasFrame:
                        st.write("No frames grabbed!")
                        break

                    tm.start()
                    faces = detector.detect(frame)
                    tm.stop()

                    names = []
                    if faces[1] is not None:
                        for face in faces[1]:
                            face_align = recognizer.alignCrop(frame, face)
                            face_feature = recognizer.feature(face_align)
                            face_feature = face_feature.reshape(1, -1)
                            test_predict = svc.predict(face_feature)
                            result = mydict[test_predict[0]]
                            names.append(result)

                    fps = tm.getFPS()
                    visualize(frame, faces, fps, names=names)
                    stframe.image(frame, channels="BGR")
                    st_fps.text(f"FPS: {fps:.2f}")

            cap.release()
            cv.destroyAllWindows()
        else:
            st.write("Failed to open webcam.")

with tab3:
    st.write(
        """
        ### Instructions:
        1. Select a name from the list to view saved images for that face.
        2. The images will be displayed below.
    """
    )
    if st.session_state.start:
        st.write("Please stop training before viewing saved images.")
    else:
        if os.path.exists(data_path):
            face_folders = os.listdir(data_path)
            selected_face = st.selectbox("Select face to view images", face_folders)

            if selected_face:
                folder_path = os.path.join(data_path, selected_face)
                st.header(f"Images for {selected_face}")
                col1, col2, col3, col4, col5 = st.columns(5)
                cols = [col1, col2, col3, col4, col5]
                for idx, img in enumerate(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, img)
                    img = Image.open(img_path)
                    cols[idx % 5].image(
                        img, caption=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
        else:
            st.write("No faces trained yet.")
