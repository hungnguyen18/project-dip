import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.title('Fruit Identification')

# Load the model only once
try:
    if st.session_state["LoadModel"] == True:
        print('Model already loaded')
except:
    st.session_state["LoadModel"] = True
    st.session_state["Net"] = cv2.dnn.readNet("./models/fruits.onnx")
    print('Model loaded for the first time')

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process(input_image, outputs):
    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)
    return input_image

img_file_buffer = st.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
    st.image(image)
    if st.button('Predict'):
        classes = ['Buoi', 'Cam', 'Coc', 'Khe', 'Mit']
        detections = pre_process(frame, st.session_state["Net"])
        img = post_process(frame.copy(), detections)
        t, _ = st.session_state["Net"].getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        print(label)
        cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
        st.image(img)
