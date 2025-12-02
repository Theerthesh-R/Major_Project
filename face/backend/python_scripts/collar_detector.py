import cv2
import numpy as np
from typing import Tuple, Optional, List
import time

class RoundCollarDetector:
    """
    Advanced round collar detection system for classroom attendance
    Detects round-neck t-shirts to prevent photo spoofing
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize collar detector
        
        Args:
            debug: Enable debugging visualizations
        """
        self.debug = debug
        self.detection_history = []
        self.last_detection_time = 0
        self.confidence_threshold = 0.65
        
        print("✅ Round Collar Detector initialized")
        
    def preprocess_neck_region(self, frame: np.ndarray, face_box: Tuple) -> np.ndarray:
        """
        Extract and preprocess neck region from frame
        
        Args:
            frame: Input BGR image
            face_box: (x1, y1, x2, y2) face bounding box
            
        Returns:
            Preprocessed neck region ROI
        """
        x1, y1, x2, y2 = [int(coord) for coord in face_box]
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate neck region (below face)
        neck_height = int(face_height * 0.8)
        neck_y1 = y2
        neck_y2 = min(frame.shape[0], y2 + neck_height)
        
        # Width slightly wider than face
        neck_x1 = max(0, x1 - int(face_width * 0.2))
        neck_x2 = min(frame.shape[1], x2 + int(face_width * 0.2))
        
        # Extract neck ROI
        neck_roi = frame[neck_y1:neck_y2, neck_x1:neck_x2]
        
        if self.debug:
            print(f"Neck ROI: {neck_roi.shape}")
            
        return neck_roi, (neck_x1, neck_y1, neck_x2, neck_y2)
    
    def detect_collar_edges(self, neck_roi: np.ndarray) -> np.ndarray:
        """
        Detect edges in neck region for collar detection
        
        Args:
            neck_roi: Neck region image
            
        Returns:
            Edge map
        """
        if neck_roi.size == 0:
            return np.zeros((10, 10), dtype=np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better edge detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)
        
        # Canny edge detection with adaptive thresholds
        edges = cv2.Canny(blurred, 30, 100)
        
        # Morphological operations to clean edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_collar_candidates(self, edges: np.ndarray) -> List[Tuple]:
        """
        Find potential collar shapes in edge map
        
        Args:
            edges: Binary edge image
            
        Returns:
            List of candidate circles (x, y, radius, circularity)
        """
        candidates = []
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 100 or area > 5000:  # Skip too small or too large
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Round collar should be circular
            if 0.6 <= circularity <= 1.0:
                # Get minimal enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Calculate additional features
                aspect_ratio = self.calculate_aspect_ratio(contour)
                solidity = self.calculate_solidity(contour)
                
                # Calculate confidence score
                confidence = self.calculate_confidence(circularity, aspect_ratio, solidity)
                
                candidates.append((int(x), int(y), int(radius), confidence))
                
        # Sort by confidence
        candidates.sort(key=lambda x: x[3], reverse=True)
        
        return candidates
    
    def calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate aspect ratio of contour bounding box"""
        _, _, w, h = cv2.boundingRect(contour)
        if h == 0:
            return 0
        return w / h
    
    def calculate_solidity(self, contour: np.ndarray) -> float:
        """Calculate solidity (area / convex hull area)"""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0
        area = cv2.contourArea(contour)
        return area / hull_area
    
    def calculate_confidence(self, circularity: float, aspect_ratio: float, solidity: float) -> float:
        """Calculate detection confidence score"""
        # Weight different features
        circ_weight = 0.5
        aspect_weight = 0.2
        solid_weight = 0.3
        
        # Aspect ratio should be close to 1 for circles
        aspect_score = 1 - min(abs(aspect_ratio - 1), 0.5)
        
        # Circularity is already normalized (0-1)
        # Solidity should be high for solid shapes
        solidity_score = solidity
        
        confidence = (circularity * circ_weight + 
                     aspect_score * aspect_weight + 
                     solidity_score * solid_weight)
        
        return min(max(confidence, 0), 1)
    
    def validate_collar_position(self, candidate: Tuple, neck_roi_shape: Tuple) -> bool:
        """
        Validate if candidate is in proper collar position
        
        Args:
            candidate: (x, y, radius, confidence)
            neck_roi_shape: (height, width) of neck ROI
            
        Returns:
            True if valid collar position
        """
        x, y, radius, confidence = candidate
        roi_height, roi_width = neck_roi_shape[:2]
        
        # Collar should be in upper part of neck region
        if y > roi_height * 0.7:  # Too low
            return False
        
        # Collar should not be too close to edges
        if x < radius or x > roi_width - radius:
            return False
            
        # Collar should not be too small or too large
        if radius < 10 or radius > min(roi_height, roi_width) * 0.4:
            return False
            
        return True
    
    def temporal_filtering(self, detection: bool) -> bool:
        """
        Apply temporal filtering to reduce false positives
        
        Args:
            detection: Current frame detection result
            
        Returns:
            Filtered detection result
        """
        current_time = time.time()
        
        # Add to history
        self.detection_history.append((current_time, detection))
        
        # Keep only last 1 second of history
        one_second_ago = current_time - 1.0
        self.detection_history = [(t, d) for t, d in self.detection_history if t >= one_second_ago]
        
        if len(self.detection_history) < 3:
            return detection
        
        # Require detection in majority of recent frames
        recent_detections = [d for _, d in self.detection_history[-5:]]
        positive_count = sum(recent_detections)
        
        return positive_count >= 3  # At least 3 out of 5 frames
    
    def detect(self, frame: np.ndarray, face_box: Tuple) -> Tuple[bool, Optional[Tuple]]:
        """
        Main detection function - detects round collar in frame
        
        Args:
            frame: Input BGR image
            face_box: (x1, y1, x2, y2) face bounding box
            
        Returns:
            (detected, collar_info)
            collar_info: (x, y, radius, confidence) if detected
        """
        # Step 1: Preprocess neck region
        neck_roi, roi_coords = self.preprocess_neck_region(frame, face_box)
        
        if neck_roi.size == 0:
            return False, None
        
        # Step 2: Detect edges
        edges = self.detect_collar_edges(neck_roi)
        
        # Step 3: Find collar candidates
        candidates = self.find_collar_candidates(edges)
        
        # Step 4: Select best candidate
        best_candidate = None
        for candidate in candidates:
            if self.validate_collar_position(candidate, neck_roi.shape):
                if candidate[3] >= self.confidence_threshold:
                    best_candidate = candidate
                    break
        
        # Step 5: Convert to global coordinates
        detected = best_candidate is not None
        collar_info = None
        
        if detected:
            nx, ny, radius, confidence = best_candidate
            neck_x1, neck_y1, _, _ = roi_coords
            
            # Convert to global frame coordinates
            global_x = neck_x1 + nx
            global_y = neck_y1 + ny
            
            collar_info = (global_x, global_y, radius, confidence)
            
            # Apply temporal filtering
            detected = self.temporal_filtering(True)
            self.last_detection_time = time.time()
        else:
            detected = self.temporal_filtering(False)
        
        # Debug visualization
        if self.debug and detected:
            self.visualize_detection(frame, neck_roi, edges, roi_coords, collar_info)
        
        return detected, collar_info
    
    def visualize_detection(self, frame: np.ndarray, neck_roi: np.ndarray, 
                          edges: np.ndarray, roi_coords: Tuple, 
                          collar_info: Optional[Tuple]):
        """Create debug visualization"""
        neck_x1, neck_y1, neck_x2, neck_y2 = roi_coords
        
        # Draw neck ROI rectangle
        cv2.rectangle(frame, (neck_x1, neck_y1), (neck_x2, neck_y2), (255, 255, 0), 2)
        
        # Draw collar circle
        if collar_info:
            x, y, radius, confidence = collar_info
            cv2.circle(frame, (x, y), radius, (0, 255, 255), 3)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            
            # Add confidence text
            cv2.putText(frame, f"Collar: {confidence:.2f}", (x - 30, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Create debug window
        debug_height = 200
        debug_width = 600
        
        # Resize images for debug display
        neck_resized = cv2.resize(neck_roi, (200, debug_height))
        edges_resized = cv2.resize(edges, (200, debug_height))
        edges_color = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
        
        # Create debug panel
        debug_panel = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
        debug_panel[:, :200] = neck_resized
        debug_panel[:, 200:400] = edges_color
        
        # Add labels
        cv2.putText(debug_panel, "Neck ROI", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(debug_panel, "Edges", (210, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show debug window
        cv2.imshow("Collar Detection Debug", debug_panel)
    
    def get_detection_stats(self) -> dict:
        """Get detection statistics"""
        if len(self.detection_history) == 0:
            return {"total_frames": 0, "detection_rate": 0.0}
        
        total_frames = len(self.detection_history)
        positive_frames = sum([d for _, d in self.detection_history])
        
        return {
            "total_frames": total_frames,
            "positive_frames": positive_frames,
            "detection_rate": positive_frames / total_frames,
            "last_detection": self.last_detection_time
        }


# Factory function for easy integration
def create_collar_detector(debug: bool = False) -> RoundCollarDetector:
    """Create and configure collar detector instance"""
    return RoundCollarDetector(debug=debug)


# Example usage function
def test_collar_detection():
    """Test the collar detector with webcam"""
    detector = RoundCollarDetector(debug=True)
    cap = cv2.VideoCapture(0)
    
    print("Testing collar detection...")
    print("Press 'q' to quit")
    print("Wear a round collar t-shirt for testing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate face detection (you would replace this with actual face detection)
        height, width = frame.shape[:2]
        fake_face_box = (width//2 - 100, height//2 - 150, 
                        width//2 + 100, height//2 + 50)
        
        # Detect collar
        detected, collar_info = detector.detect(frame, fake_face_box)
        
        # Draw face box (simulated)
        x1, y1, x2, y2 = fake_face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display status
        status = "Collar: DETECTED" if detected else "Collar: NOT DETECTED"
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show stats
        stats = detector.get_detection_stats()
        cv2.putText(frame, f"Confidence: {stats['detection_rate']:.2f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Collar Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    # Run test if file is executed directly
    test_collar_detection()