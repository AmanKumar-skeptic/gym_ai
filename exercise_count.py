import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
import time
import pygame
import threading
import os
from collections import deque

# Constants
EXERCISE_CLASSES = [
    "push-up", "squat", "crunch", "jumping_jack", "lunge", 
    "plank", "pull-up", "burpee", "sit-up", "deadlift"
]
CONFIDENCE_THRESHOLD = 0.7
REP_THRESHOLD = 0.8
SEQUENCE_LENGTH = 16  # Frames to consider for exercise detection
COOLDOWN_FRAMES = 15  # Frames to wait before counting a new rep

class ExerciseCounter:
    def __init__(self):
        # Initialize mediapipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load or create exercise classification model
        self.model = self.load_model()
        
        # Counters and state variables for each exercise
        self.counters = {exercise: 0 for exercise in EXERCISE_CLASSES}
        self.exercise_states = {exercise: False for exercise in EXERCISE_CLASSES}  # True if in "up" position
        self.current_exercise = None
        self.last_exercise = None
        self.exercise_confidence = 0
        self.confidence_buffer = deque(maxlen=10)  # Store last 10 confidence values
        
        # Frame sequence for temporal analysis
        self.pose_sequence = deque(maxlen=SEQUENCE_LENGTH)
        
        # Cooldown to prevent counting reps too quickly
        self.cooldown = 0
        
        # Sound effects
        pygame.mixer.init()
        self.rep_sound = pygame.mixer.Sound('sounds/count.wav') if os.path.exists('sounds/count.wav') else None
        self.exercise_change_sound = pygame.mixer.Sound('sounds/change.wav') if os.path.exists('sounds/change.wav') else None
        
        # UI variables
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.ui_color = (0, 255, 0)  # Green
        self.rep_flash_counter = 0
        
    def load_model(self):
        """Load or create pre-trained exercise classification model"""
        # Check if model exists
        if os.path.exists('model/exercise_classifier.h5'):
            print("Loading existing model...")
            return tf.keras.models.load_model('model/exercise_classifier.h5')
        
        # Create model architecture with MobileNetV2 base
        print("Creating new model with MobileNetV2 base...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze the base model layers
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(len(EXERCISE_CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created. Note: This model will need training with exercise data.")
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess frame for the model"""
        # Resize
        resized = cv2.resize(frame, (224, 224))
        # Convert to array and preprocess
        img_array = img_to_array(resized)
        img_array = preprocess_input(img_array)
        return img_array
    
    def detect_pose(self, frame):
        """Detect human pose in frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        results = self.pose.process(rgb_frame)
        return results
    
    def extract_pose_features(self, results):
        """Extract relevant features from pose landmarks"""
        if not results.pose_landmarks:
            return None
        
        # Extract 3D coordinates of all landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def predict_exercise(self, frame, pose_features):
        """Predict the exercise being performed"""
        if pose_features is None:
            return None, 0.0
        
        # Add the current pose to our sequence
        self.pose_sequence.append(pose_features)
        
        # If we don't have enough frames yet, return None
        if len(self.pose_sequence) < SEQUENCE_LENGTH:
            return None, 0.0
        
        # Preprocess frame for CNN
        preprocessed_frame = self.preprocess_frame(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        
        # Get prediction
        prediction = self.model.predict(preprocessed_frame, verbose=0)[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        # Update confidence buffer
        self.confidence_buffer.append(confidence)
        avg_confidence = sum(self.confidence_buffer) / len(self.confidence_buffer)
        
        # Only return a prediction if confidence is high enough
        if avg_confidence > CONFIDENCE_THRESHOLD:
            return EXERCISE_CLASSES[predicted_class_idx], avg_confidence
        else:
            return None, avg_confidence
    
    def count_rep(self, exercise, pose_features):
        """Count a repetition based on pose sequence analysis"""
        if exercise is None or self.cooldown > 0:
            return
        
        # Exercise-specific detection logic
        if exercise == "push-up":
            # Get shoulder and elbow height
            shoulders = (pose_features[33], pose_features[34])  # y-coordinates of shoulders
            elbows = (pose_features[39], pose_features[40])    # y-coordinates of elbows
            
            # Calculate average heights
            avg_shoulder_height = sum(shoulders) / 2
            avg_elbow_height = sum(elbows) / 2
            
            # Determine if in "down" position (elbows bent)
            is_down = abs(avg_shoulder_height - avg_elbow_height) < 0.05
            
            # If transitioning from down to up, count a rep
            if is_down and self.exercise_states[exercise]:
                self.counters[exercise] += 1
                self.rep_flash_counter = 10
                self.cooldown = COOLDOWN_FRAMES
                if self.rep_sound:
                    threading.Thread(target=self.rep_sound.play).start()
            
            # Update state
            self.exercise_states[exercise] = not is_down
            
        elif exercise == "squat":
            # Get hip and knee height
            hips = (pose_features[69], pose_features[70])  # y-coordinates of hips
            knees = (pose_features[75], pose_features[76])  # y-coordinates of knees
            
            # Calculate average heights
            avg_hip_height = sum(hips) / 2
            avg_knee_height = sum(knees) / 2
            
            # Determine if in "down" position (knees bent)
            is_down = abs(avg_hip_height - avg_knee_height) < 0.1
            
            # If transitioning from down to up, count a rep
            if is_down and self.exercise_states[exercise]:
                self.counters[exercise] += 1
                self.rep_flash_counter = 10
                self.cooldown = COOLDOWN_FRAMES
                if self.rep_sound:
                    threading.Thread(target=self.rep_sound.play).start()
            
            # Update state
            self.exercise_states[exercise] = not is_down
            
        # Similar logic would be implemented for other exercises
        # For the sake of this example, I'm providing a simplified implementation
        # In a full application, each exercise would have specific detection criteria
        
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect human pose
        pose_results = self.detect_pose(frame)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Extract features
        pose_features = self.extract_pose_features(pose_results)
        
        # Predict exercise
        predicted_exercise, confidence = self.predict_exercise(frame, pose_features)
        
        # Update current exercise if we have a consistent prediction
        if predicted_exercise:
            if self.current_exercise != predicted_exercise:
                if self.exercise_change_sound:
                    threading.Thread(target=self.exercise_change_sound.play).start()
                print(f"Exercise changed to: {predicted_exercise}")
            
            self.current_exercise = predicted_exercise
            self.exercise_confidence = confidence
            
            # Count repetition
            if pose_features is not None:
                self.count_rep(self.current_exercise, pose_features)
        
        # Decrease cooldown counter
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # Decrease rep flash counter
        if self.rep_flash_counter > 0:
            self.rep_flash_counter -= 1
        
        return frame
    
    def add_ui(self, frame):
        """Add UI elements to the frame"""
        h, w, _ = frame.shape
        
        # Add background overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add exercise name and confidence
        if self.current_exercise:
            exercise_text = f"Exercise: {self.current_exercise.title()}"
            confidence_text = f"Confidence: {self.exercise_confidence:.2f}"
            
            cv2.putText(frame, exercise_text, (10, 30), self.font, 0.7, self.ui_color, 2)
            cv2.putText(frame, confidence_text, (10, 55), self.font, 0.6, self.ui_color, 1)
        else:
            cv2.putText(frame, "No exercise detected", (10, 30), self.font, 0.7, (0, 0, 255), 2)
        
        # Add rep counter
        if self.current_exercise:
            count = self.counters[self.current_exercise]
            count_color = (0, 255, 255) if self.rep_flash_counter > 0 else (255, 255, 255)
            count_text = f"Count: {count}"
            
            # Calculate text size to center it
            text_size = cv2.getTextSize(count_text, self.font, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            
            cv2.putText(frame, count_text, (text_x, h - 30), self.font, 1.5, count_color, 3)
        
        # Add all exercise counters on the right side
        y_offset = 90
        for exercise, count in self.counters.items():
            if count > 0:
                text = f"{exercise.title()}: {count}"
                cv2.putText(frame, text, (w - 200, y_offset), self.font, 0.6, (255, 255, 255), 1)
                y_offset += 25
        
        return frame
    
    def run(self, src=0):
        """Run the exercise counter on a video source"""
        # Open video capture
        cap = cv2.VideoCapture(src)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        # Set camera properties for higher resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window
        cv2.namedWindow('Exercise Counter', cv2.WINDOW_NORMAL)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Add UI elements
            output_frame = self.add_ui(processed_frame)
            
            # Display the resulting frame
            cv2.imshow('Exercise Counter', output_frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        
        # Release MediaPipe resources
        self.pose.close()

if __name__ == "__main__":
    # Create 'sounds' and 'model' directories if they don't exist
    os.makedirs('sounds', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Start the exercise counter
    counter = ExerciseCounter()
    counter.run()