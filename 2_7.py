import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model = load_model('trained_emotion_detection_model.keras')

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and analyze emotions in a video stream
def detect_emotions_in_video():
    cap = cv2.VideoCapture(0)  # Access webcam; change to 'filename.mp4' for video file
    
    # Get video frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi_color = frame[y:y + h, x:x + w]
            
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi_color = cv2.resize(face_roi_color, (48, 48))
            face_roi_color = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
            
            face_roi_color = face_roi_color.astype('float') / 255.0
            face_roi_color = np.expand_dims(face_roi_color, axis=0)
            
            prediction = emotion_model.predict(face_roi_color)
            emotion = np.argmax(prediction)
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emotion_text = emotions[emotion]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        out.write(frame)  # Write frame to output video
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()  # Release VideoWriter
    cv2.destroyAllWindows()

# Call the function to start detecting emotions in the video stream and save the result
detect_emotions_in_video()
