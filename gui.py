import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget, QRadioButton, QButtonGroup, QHBoxLayout, QComboBox, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from tensorflow.keras.models import load_model
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

class EmotionDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Emotion Detection')
        self.setGeometry(100, 100, 800, 600)

        try:
            self.emotion_model = load_model('trained_emotion_model3_2.keras')
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            self.show_error(f"Error loading models: {e}")
            logging.error(f"Error loading models: {e}")
            return

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.mode_layout = QHBoxLayout()
        self.layout.addLayout(self.mode_layout)

        self.video_mode = QRadioButton('Video')
        self.image_mode = QRadioButton('Image')
        self.video_mode.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.video_mode)
        self.mode_group.addButton(self.image_mode)

        self.mode_layout.addWidget(self.video_mode)
        self.mode_layout.addWidget(self.image_mode)

        self.camera_selection = QComboBox()
        self.camera_selection.addItem("Built-in Webcam (0)")
        self.camera_selection.addItem("External Webcam (1)")  # Change or add more options as needed
        self.camera_selection.addItem("Upload Video File")  # Option to upload a video file
        self.layout.addWidget(self.camera_selection)

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.running = False

    def start_detection(self):
        logging.debug("Starting detection")
        if self.running:
            return
        
        self.running = True
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        if self.video_mode.isChecked():
            self.start_video_detection()
        else:
            self.start_image_detection()

    def stop_detection(self):
        logging.debug("Stopping detection")
        self.running = False
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def start_video_detection(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()

            selected_option = self.camera_selection.currentText()

            if selected_option == "Upload Video File":
                video_source = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")[0]
                if not video_source:
                    self.stop_detection()
                    return
                self.cap = cv2.VideoCapture(video_source)
            else:
                webcam_index = self.camera_selection.currentIndex()  # Index of selected webcam
                self.cap = cv2.VideoCapture(webcam_index)

            if not self.cap.isOpened():
                self.show_error("Error opening video source.")
                self.stop_detection()
                return

            self.timer.start(30)
        except Exception as e:
            self.show_error(f"Error starting video detection: {e}")
            logging.error(f"Error starting video detection: {e}")
            self.stop_detection()

    def start_image_detection(self):
        try:
            image_path = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.jpg *.jpeg *.png)")[0]
            if not image_path:
                self.stop_detection()
                return

            self.detect_emotions_in_image(image_path)
        except Exception as e:
            self.show_error(f"Error starting image detection: {e}")
            logging.error(f"Error starting image detection: {e}")
            self.stop_detection()
    
    def detect_emotions_in_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi_color = image_rgb[y:y + h, x:x + w]

                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi_color = cv2.resize(face_roi_color, (48, 48))

                face_roi_color = face_roi_color.astype('float') / 255.0
                face_roi_color = np.expand_dims(face_roi_color, axis=0)

                prediction = self.emotion_model.predict(face_roi_color)
                emotion = np.argmax(prediction)
                emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
                emotion_text = emotions[emotion]

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            self.display_image(image)
        except Exception as e:
            self.show_error(f"Error detecting emotions in image: {e}")
            logging.error(f"Error detecting emotions in image: {e}")

    def update_frame(self):
        try:
            if not self.running or not self.cap:
                return

            ret, frame = self.cap.read()
            if not ret:
                self.stop_detection()
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi_color = frame[y:y + h, x:x + w]

                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi_color = cv2.resize(face_roi_color, (48, 48))

                face_roi_color = face_roi_color.astype('float') / 255.0
                face_roi_color = np.expand_dims(face_roi_color, axis=0)

                prediction = self.emotion_model.predict(face_roi_color)
                emotion = np.argmax(prediction)
                emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
                emotion_text = emotions[emotion]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            self.display_image(frame)
        except Exception as e:
            self.show_error(f"Error updating frame: {e}")
            logging.error(f"Error updating frame: {e}")
            self.stop_detection()

    def display_image(self, image):
        try:
            # Convert color format from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
        except Exception as e:
            self.show_error(f"Error displaying image: {e}")
            logging.error(f"Error displaying image: {e}")

    def show_error(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = EmotionDetectionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
