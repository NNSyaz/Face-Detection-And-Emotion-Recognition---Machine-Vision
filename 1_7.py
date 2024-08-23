import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model = load_model('trained_emotion_detection_model.keras')

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and analyze emotions in the image
def detect_emotions(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi_color = image_rgb[y:y + h, x:x + w]  # Extract face region in color
        
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi_color = cv2.resize(face_roi_color, (48, 48))  # Resize color image
        
        # Convert grayscale face to RGB (3 channels)
        face_roi_color = cv2.cvtColor(face_roi_color, cv2.COLOR_RGB2BGR)
        
        # Preprocess the face to match the model input shape
        face_roi_color = face_roi_color.astype('float') / 255.0
        face_roi_color = np.expand_dims(face_roi_color, axis=0)

        # Predict emotions
        prediction = emotion_model.predict(face_roi_color)
        emotion = np.argmax(prediction)
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotion_text = emotions[emotion]

        # Draw rectangle around detected face and label the emotion
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the image with emotions analyzed
    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the image path
detect_emotions(r'C:\Users\syazw\Desktop\BTI3423 MACHINE VISION\Project\1.5.jpeg')  # Replace '1.5.jpeg' with your image path
