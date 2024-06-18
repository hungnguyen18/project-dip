import streamlit as st
import numpy as np
import cv2 as cv
import os
from datetime import datetime
from PIL import Image
import time
import joblib

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

# Streamlit UI
st.title('Face Recognition App')
choice = st.sidebar.selectbox("Choose an action", ["Training", "Detect Face"])

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if choice == "Training":
    st.header("Training Mode")
    start_button = st.button("Start Training")

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
        fps = 0

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
                    st.session_state.start = False

elif choice == "Detect Face":
    st.header("Detection Mode")
    cap = cv.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    if 'stop' not in st.session_state:
        st.session_state.stop = False

    press = st.button('Stop')
    if press:
        if st.session_state.stop == False:
            st.session_state.stop = True
            cap.release()
        else:
            st.session_state.stop = False

    if 'frame_stop' not in st.session_state:
        frame_stop = cv.imread('./assets/stop.jpg')
        st.session_state.frame_stop = frame_stop

    if st.session_state.stop == True:
        FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
    else:
        if os.path.exists('./data/svc.pkl'):
            svc, le = joblib.load('./data/svc.pkl')

            prev_time = 0
            fps = 0
            for img_name in os.listdir(f"./data/faces/{selected_face}"):
                img_path = os.path.join(f"./data/faces/{selected_face}", img_name)
                face_img = cv.imread(img_path)
                if face_img is None:
                    continue
                faces = face_cascade.detectMultiScale(face_img, 1.1, 4)
                for (x, y, w, h) in faces:
                    face_img_resized = cv.resize(face_img[y:y+h, x:x+w], (128, 128)).flatten().reshape(1, -1)
                    pred = svc.predict(face_img_resized)
                    label = le.inverse_transform(pred)[0]
                    cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv.putText(face_img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                FRAME_WINDOW.image(face_img, channels="BGR")
        else:
            st.write("Model not trained yet. Please train the model first.")

# Display saved images
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
