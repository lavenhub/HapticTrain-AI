import cv2
import mediapipe as mp
import numpy as np

class PoseEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        # Math to find angle between three points
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame):
        # Convert BGR to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        error_detected = False
        angle = 0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Example: Track Left Elbow Angle (Shoulder: 11, Elbow: 13, Wrist: 15)
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Haptic Logic: Trigger if elbow is flaring (angle > 160)
            if angle > 160:
                error_detected = True
                
            # Draw landmarks on the original frame
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
        return frame, error_detected, angle