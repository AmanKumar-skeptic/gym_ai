import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
from collections import Counter

# Configuration
EXERCISE_CLASSES = [
    "push-up", "squat", "crunch", "jumping_jack", "lunge", 
    "plank", "pull-up", "burpee", "sit-up", "deadlift"
]
MODEL_WEIGHTS_PATH = "exercise_recognition_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.5  # Lowered from 0.5 to 0.3 for more predictions
FRAME_SKIP = 3  # Reduced from 5 to 2 for more frequent predictions
BUFFER_SIZE = 4  # Reduced from 8 to 4 for faster processing
INFERENCE_RESOLUTION = (160, 160)  # Lower resolution for faster processing

class ExerciseRecognitionModel(nn.Module):
    """
    The deep learning model for exercise recognition.
    Combines a 3D CNN backbone with pose information if available.
    """
    def __init__(self, num_classes=len(EXERCISE_CLASSES)):
        super(ExerciseRecognitionModel, self).__init__()
        
        # Load pre-trained 3D ResNet model as backbone
        # This is where a pre-trained model from torchvision is loaded
        self.backbone = models.video.r3d_18(pretrained=True)
        print("Loaded pre-trained R3D-18 backbone from torchvision")
        
        # Replace the last fully connected layer for our task
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Pose estimation integration
        self.pose_encoder = nn.Sequential(
            nn.Linear(17 * 2, 256),  # 17 keypoints with x,y coordinates
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Fusion layer to combine video features with pose features
        self.fusion = nn.Sequential(
            nn.Linear(in_features + 128, in_features),
            nn.ReLU()
        )

    def forward(self, x, pose_data=None):
        # Extract features from video frames
        features = self.backbone(x)  # [batch_size, in_features]
        
        # If pose data is available, integrate it
        if pose_data is not None:
            pose_features = self.pose_encoder(pose_data)
            features = self.fusion(torch.cat([features, pose_features], dim=1))
        
        # Apply classification head
        output = self.classifier(features)
        
        return output

class FrameBuffer:
    """
    Maintains a buffer of recent frames for temporal analysis.
    This enables the model to recognize actions across multiple frames.
    """
    def __init__(self, buffer_size=16, frame_width=224, frame_height=224):
        self.buffer_size = buffer_size
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.clear()
        
    def clear(self):
        self.buffer = np.zeros((self.buffer_size, 3, self.frame_height, self.frame_width), dtype=np.float32)
        self.current_index = 0
        self.is_full = False
        
    def add_frame(self, frame):
        # Convert frame to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Normalize pixel values
        frame = frame.transpose(2, 0, 1) / 255.0
        
        # Add to buffer
        self.buffer[self.current_index] = frame
        self.current_index = (self.current_index + 1) % self.buffer_size
        
        if self.current_index == 0:
            self.is_full = True
            
    def get_buffer(self):
        if not self.is_full:
            # If buffer isn't full, return only the filled portion
            return self.buffer[:self.current_index]
        return self.buffer
    
    def get_tensor(self):
        # Convert buffer to tensor of shape [1, 3, frames, height, width]
        buffer_data = self.get_buffer()
        tensor = torch.FloatTensor(buffer_data).unsqueeze(0)
        tensor = tensor.permute(0, 2, 1, 3, 4)  # [1, 3, frames, h, w] -> [1, channels, frames, h, w]
        return tensor

class PoseEstimator:
    """
    Handles human pose estimation using pre-trained models.
    Tries to use MoveNet from TensorFlow Hub first, with OpenPose as fallback.
    """
    def __init__(self):
        # Try to load MoveNet from TensorFlow Hub
        # This is where the pre-trained pose estimation model is loaded
        self.model = None
        self.model_type = None
        
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            print("Loading MoveNet from TensorFlow Hub (pre-trained pose estimation model)...")
            self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
            self.movenet = self.model.signatures['serving_default']  # Get the callable signature
            self.model_type = "movenet"
            print("Successfully loaded MoveNet for pose estimation")
        except Exception as e:
            print(f"Could not load MoveNet: {e}")
            
            # Fallback to OpenPose if available
            try:
                print("Attempting to load OpenPose as fallback...")
                self.model = cv2.dnn.readNetFromCaffe(
                    "pose/pose_deploy_linevec.prototxt",
                    "pose/pose_iter_440000.caffemodel"
                )
                self.model_type = "openpose"
                print("Successfully loaded OpenPose for pose estimation")
            except Exception as e:
                print(f"Could not load OpenPose: {e}")
                print("Warning: No pose estimation model loaded. Using only video features.")
    
    def estimate_pose(self, frame):
        if self.model is None:
            return None
            
        if self.model_type == "movenet":
            try:
                import tensorflow as tf
                # Preprocess for MoveNet
                input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
                input_image = tf.cast(input_image, dtype=tf.int32)
                
                # Run inference using the signature
                results = self.movenet(input_image)  # Use the signature instead of calling model directly
                keypoints = results['output_0'].numpy().squeeze()
                
                # Format keypoints as [x1, y1, x2, y2, ...]
                return keypoints[:, :2].flatten()
            except Exception as e:
                print(f"Error in MoveNet pose estimation: {e}")
                return None
            
        elif self.model_type == "openpose":
            try:
                # Preprocess for OpenPose
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
                
                # Run inference
                self.model.setInput(blob)
                output = self.model.forward()
                
                # Extract keypoints
                keypoints = []
                for i in range(17):  # COCO model has 17 keypoints
                    prob_map = output[0, i, :, :]
                    _, prob, _, point = cv2.minMaxLoc(prob_map)
                    
                    if prob > 0.1:
                        x = int(point[0] * width / output.shape[3])
                        y = int(point[1] * height / output.shape[2])
                        keypoints.extend([x / width, y / height])
                    else:
                        keypoints.extend([0, 0])  # Use zeros for low confidence keypoints
                        
                return np.array(keypoints, dtype=np.float32)
            except Exception as e:
                print(f"Error in OpenPose pose estimation: {e}")
                return None
        
        return None

class ExerciseRecognitionSystem:
    """
    Main system that integrates the model, frame buffer, and pose estimator.
    Handles prediction and visualization.
    """
    def __init__(self):
        print(f"Initializing Exercise Recognition System on {DEVICE}")
        
        # Initialize model
        self.model = ExerciseRecognitionModel()
        
        # This method will be implemented to load pre-trained weights
        self.load_pretrained_models()
        
        self.model.to(DEVICE)
        self.model.eval()
        
        # Initialize frame buffer with smaller buffer size
        self.frame_buffer = FrameBuffer(buffer_size=BUFFER_SIZE, 
                                        frame_width=INFERENCE_RESOLUTION[0], 
                                        frame_height=INFERENCE_RESOLUTION[1])
        
        # Initialize pose estimator with pre-trained model
        self.pose_estimator = PoseEstimator()
        
        # For smoothing predictions
        self.prediction_history = []
        self.history_size = 3  # Reduced from 5 to 3 for faster response
        
        # For FPS calculation
        self.prev_time = 0
        self.fps = 0
        
        # Current prediction
        self.current_prediction = -1
        self.current_confidence = 0.0
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        
        # For frame skipping
        self.frame_count = 0
        
        # Flag to control pose estimation frequency
        self.pose_estimation_frame = 0
        self.pose_estimation_interval = 3  # Only run pose estimation every N frames
        self.last_valid_pose = None
        
        print("System initialized and ready")
    
    def load_pretrained_models(self):
        """
        Load pre-trained models for exercise recognition.
        There are three options:
        1. Custom weights if available
        2. Kinetics pre-trained weights if available
        3. Default ImageNet pre-trained weights (already loaded in __init__)
        """
        # Option 1: Try to load custom pre-trained weights
        try:
            print(f"Attempting to load custom model weights from {MODEL_WEIGHTS_PATH}")
            self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
            print(f"Successfully loaded model weights from {MODEL_WEIGHTS_PATH}")
            return
        except Exception as e:
            print(f"Could not load custom weights: {e}")
        
        # Option 2: Try to load Kinetics pre-trained weights
        try:
            print("Attempting to load Kinetics pre-trained weights...")
            # This requires torchvision 0.13.0 or later
            kinetics_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            # Copy weights from backbone
            self.model.backbone.load_state_dict(kinetics_model.blocks[:5].state_dict(), strict=False)
            print("Successfully loaded Kinetics pre-trained weights")
            return
        except Exception as e:
            print(f"Could not load Kinetics weights: {e}")
            print("Using default ImageNet pre-trained weights from model initialization")
    
    def preprocess_frame(self, frame):
        # Resize and normalize frame
        resized_frame = cv2.resize(frame, INFERENCE_RESOLUTION)
        self.frame_buffer.add_frame(resized_frame)
        
        # Estimate pose if pose estimator is available, but only every few frames
        pose_data = None
        if self.pose_estimator.model is not None:
            self.pose_estimation_frame = (self.pose_estimation_frame + 1) % self.pose_estimation_interval
            
            if self.pose_estimation_frame == 0:
                # Only run pose estimation every N frames
                pose_data = self.pose_estimator.estimate_pose(frame)
                if pose_data is not None:
                    self.last_valid_pose = pose_data
            elif self.last_valid_pose is not None:
                # Use the last valid pose for other frames
                pose_data = self.last_valid_pose
                
            if pose_data is not None:
                pose_data = torch.FloatTensor(pose_data).unsqueeze(0).to(DEVICE)
        
        return pose_data
    
    def predict(self, frame):
        # Preprocess frame
        pose_data = self.preprocess_frame(frame)
        
        # Only predict if buffer has enough frames
        # Reduced the minimum required frames to 2 for immediate predictions
        if self.frame_buffer.current_index >= 2 or self.frame_buffer.is_full:
            # Get tensor from buffer
            input_tensor = self.frame_buffer.get_tensor()
            
            # Apply normalization to each frame individually
            for i in range(input_tensor.shape[2]):  # For each frame
                input_tensor[0, :, i] = self.transform(input_tensor[0, :, i])
            
            input_tensor = input_tensor.to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor, pose_data)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Get prediction
                confidence, prediction = torch.max(probabilities, 0)
                confidence = confidence.item()
                prediction = prediction.item()
                
                # Smooth predictions
                self.prediction_history.append((prediction, confidence))
                self.prediction_history = self.prediction_history[-self.history_size:]
                
                # Get most common prediction from history
                if len(self.prediction_history) >= 1:  # Only need 1 prediction
                    predictions = [p[0] for p in self.prediction_history]
                    confidences = [p[1] for p in self.prediction_history]
                    
                    # Count occurrences of each prediction
                    counter = Counter(predictions)
                    most_common = counter.most_common(1)[0][0]
                    
                    # Get average confidence for the most common prediction
                    avg_confidence = sum([c for p, c in zip(predictions, confidences) if p == most_common]) / counter[most_common]
                    
                    self.current_prediction = most_common
                    self.current_confidence = avg_confidence
                else:
                    self.current_prediction = prediction
                    self.current_confidence = confidence
                
                return self.current_prediction, self.current_confidence
        
        # If buffer doesn't have enough frames yet
        return -1, 0.0
    
    def run_webcam(self):
        """Simplified webcam processing loop for better performance"""
        print("Starting webcam capture...")
        cap = cv2.VideoCapture(0)
        
        # Set lower resolution for webcam capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Webcam opened successfully. Press 'q' to quit.")
        
        # For FPS calculation
        fps_counter = 0
        fps_timer = time.time()
        display_fps = 0
        
        # Clear the frame buffer to start fresh
        self.frame_buffer.clear()
        
        frame_count = 0
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer > 1.0:
                display_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
            
            # Process every Nth frame for prediction
            frame_count = (frame_count + 1) % FRAME_SKIP
            if frame_count == 0:
                prediction, confidence = self.predict(frame)
            else:
                # Still add the frame to buffer
                self.preprocess_frame(frame)
            
            # Display result on frame
            if self.current_prediction >= 0 and self.current_confidence > CONFIDENCE_THRESHOLD:
                exercise_name = EXERCISE_CLASSES[self.current_prediction]
                cv2.putText(frame, f"{exercise_name} ({self.current_confidence:.2f})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display buffer fill status
            buffer_status = f"Buffer: {self.frame_buffer.current_index}/{self.frame_buffer.buffer_size}"
            cv2.putText(frame, buffer_status, (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display frame
            cv2.imshow('Exercise Recognition', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    
    # Move the global declaration to the beginning of the function
    global EXERCISE_CLASSES
    
    parser = argparse.ArgumentParser(description='Exercise Recognition System')
    parser.add_argument('--efficient', action='store_true', 
                        help='Run in efficient mode for less powerful systems')
    parser.add_argument('--classes', type=str, default=','.join(EXERCISE_CLASSES),
                        help='Comma-separated list of exercise classes to recognize')
    
    args = parser.parse_args()
    
    # Update exercise classes if provided
    if args.classes:
        EXERCISE_CLASSES = args.classes.split(',')
        print(f"Using custom exercise classes: {EXERCISE_CLASSES}")
    
    # Create the system
    system = ExerciseRecognitionSystem()
    
    # Run the appropriate mode
    if args.efficient:
        system.run_efficient()
    else:
        system.run_webcam()

if __name__ == "__main__":
    main()