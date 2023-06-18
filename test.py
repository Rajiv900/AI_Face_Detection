import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime

# Define the attendance log file
attendance_log = "attendance.txt"


def get_class_name(class_no):
    if class_no == 0:
        return "Rajiv"
    elif class_no == 1:
        return "Phushan Thapa Magar"
    elif class_no == 2:
        return "Rabin"
    elif class_no == 3:
        return "Pratik"
    elif class_no == 4:
        return "Deepak"


# Load the trained model
model = keras.models.load_model('keras_model.h5')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    success, img_original = cap.read()
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_original, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)  # Draw rectangle around face
        crop_img = img_original[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_name = get_class_name(class_index)

        cv2.putText(img_original, class_name,
                    (x, y + h + 20), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        # Register attendance in the log file
        with open(attendance_log, "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance_entry = f"{timestamp} - {class_name}\n"
            file.write(attendance_entry)

        # Display attendance registration message
        cv2.putText(img_original, "Attendance Registered",
                    (10, 30), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", img_original)
    cv2.waitKey(1)

    if len(faces) > 0:
        # Wait for 3 seconds (3000 milliseconds) after face detection
        cv2.waitKey(3000)
        break

cap.release()
cv2.destroyAllWindows()
