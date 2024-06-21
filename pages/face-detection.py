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
            st.write("Failed to grab frame")
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
            if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
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
st.title('Face Recognition App')
choice = st.sidebar.selectbox("Choose an action", ["Training", "Detect Face"])

def train_model(metadata):
    if len(set(m.name for m in metadata)) < 2:
        st.error("Need images of at least two different people to train the model.")
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
    st.write(f'SVM accuracy: {acc_svc:.6f}')
    joblib.dump(svc, './models/svc.pkl')
    joblib.dump(encoder, './models/label_encoder.pkl')
    st.success("Model training completed and saved.")

# Streamlit UI
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if choice == "Training":
    st.header("Training Mode")
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        start_button = st.button("Start Camera")
    else:
        start_button = None
    
    if start_button:
        st.session_state.cap = cv.VideoCapture(0)
        st.session_state.start = True

    if 'start' in st.session_state and st.session_state.start:
        FRAME_WINDOW = st.image([])
        stop_button = st.button("Stop Camera")
        save_button = st.button("Save")

        if stop_button:
            st.session_state.cap.release()
            st.session_state.start = False

        prev_time = 0

        while st.session_state.start:
            curr_time = time.time()
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            faces = face_cascade.detectMultiScale(frame, 1.1, 4)
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            FRAME_WINDOW.image(frame, channels="BGR")

            if save_button:
                name = st.text_input("Enter name for the face", key='name_input')
                if name:
                    folder_name = f"./data/faces/{name}"
                    create_folder(folder_name)
                    
                    with st.spinner('Saving images...'):
                        saved_images = save_face_images(folder_name, st.session_state.cap)
                        st.balloons()  # Show visual feedback for success
                        st.success(f"Saved 150 images to {folder_name}")
                        st.write(f"Saved images for {name}")

                        # Display saved images with timestamp
                        st.header(f"Images for {name}")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        cols = [col1, col2, col3, col4, col5]
                        for idx, img_path in enumerate(saved_images):
                            img = Image.open(img_path)
                            cols[idx % 5].image(img, caption=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                        # Train the model after saving the images
                        metadata = load_metadata('./data/faces')
                        train_model(metadata)

                    st.session_state.start = False

elif choice == "Detect Face":
    st.header("Detection Mode")
    # Load models and label dictionary
    svc = joblib.load('./models/svc.pkl')
    encoder = joblib.load('./models/label_encoder.pkl')

    face_detection_model_path = st.text_input('Path to the face detection model', './models/face_detection_yunet_2023mar.onnx')
    face_recognition_model_path = st.text_input('Path to the face recognition model', './models/face_recognition_sface_2021dec.onnx')
    score_threshold = st.slider('Score Threshold', 0.0, 1.0, 0.9)
    nms_threshold = st.slider('NMS Threshold', 0.0, 1.0, 0.3)
    top_k = st.number_input('Top K', min_value=1, value=5000)
    
    if st.button('Start Detection'):
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
        stop_detection = st.button('Stop Detection')
        while cap.isOpened():
            hasFrame, frame = cap.read()
            if not hasFrame:
                st.write('No frames grabbed!')
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
                        cv.putText(frame, 'Unknown', (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 0, 255), 2)

            visualize(frame, faces, tm.getFPS())
            FRAME_WINDOW.image(frame, channels="BGR")

            if stop_detection:
                break

        cap.release()
        cv.destroyAllWindows()

# Display saved images only if not in detection mode
if choice != "Detect Face":
    st.header("View Saved Faces")
    if os.path.exists('./data/faces'):
        face_folders = os.listdir('./data/faces')
        selected_face = st.selectbox("Select a face to view images", face_folders)

        if selected_face:
            folder_path = f"./data/faces/{selected_face}"
            st.header(f"Images for {selected_face}")
            col1, col2, col3, col4, col5 = st.columns(5)
            cols = [col1, col2, col3, col4, col5]
            for idx, img in enumerate(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, img)
                img = Image.open(img_path)
                cols[idx % 5].image(img, caption=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        st.write("No faces have been trained yet.")
