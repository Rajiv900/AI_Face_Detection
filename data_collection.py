import cv2
import os

# Prompt the user to enter a name
name = input("Enter your name: ")

# Create a directory with the provided name
directory = "images/{}".format(name)
os.makedirs(directory, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Counter for image filenames
count = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Capture", frame)

    # Wait for the 's' key to save the image
    if cv2.waitKey(1) & 0xFF == ord('s') and len(faces) > 0:
        # Save the captured image
        image_path = "{}/{}.jpg".format(directory, count)
        cv2.imwrite(image_path, frame)
        print("Image saved:", image_path)
        count += 1

    # Wait for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
