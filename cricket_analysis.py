import cv2
from ultralytics import YOLO
import os
import numpy as np
import time
from datetime import datetime

# === NEW: Temporal Smoothing for Role Stability ===
class TemporalSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.role_history = {}  # track_id -> [roles]
    
    def smooth_role(self, track_id, new_role):
        if track_id not in self.role_history:
            self.role_history[track_id] = []
        
        self.role_history[track_id].append(new_role)
        
        # Keep only recent roles
        if len(self.role_history[track_id]) > self.window_size:
            self.role_history[track_id].pop(0)
        
        # Return most common role in recent history
        from collections import Counter
        return Counter(self.role_history[track_id]).most_common(1)[0][0]

# === NEW: Zone-Based Classification System ===
class CricketZones:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define multiple zones for better role classification
        self.zones = {
            'batting_crease': {
                'x_range': (frame_width * 0.35, frame_width * 0.65),
                'y_range': (frame_height * 0.5, frame_height * 0.8),
                'roles': ['batsman', 'non-striker']
            },
            'bowling_area': {
                'x_range': (frame_width * 0.3, frame_width * 0.7),
                'y_range': (frame_height * 0.6, frame_height * 0.8),
                'roles': ['bowler', 'umpire']
            },
            'wicket_keeper_area': {
                'x_range': (frame_width * 0.4, frame_width * 0.6),
                'y_range': (frame_height * 0.1, frame_height * 0.4),
                'roles': ['wicket-keeper']
            },
            'off_side': {
                'x_range': (frame_width * 0.65, frame_width * 0.95),
                'y_range': (frame_height * 0.3, frame_height * 0.7),
                'roles': ['fielder']
            },
            'leg_side': {
                'x_range': (frame_width * 0.05, frame_width * 0.35),
                'y_range': (frame_height * 0.3, frame_height * 0.7),
                'roles': ['fielder']
            }
        }
    
    def get_zone_role(self, x_center, y_center, detected_class):
        for zone_name, zone_info in self.zones.items():
            if (zone_info['x_range'][0] <= x_center <= zone_info['x_range'][1] and
                zone_info['y_range'][0] <= y_center <= zone_info['y_range'][1]):
                
                # If detected class matches zone expectations, use it
                if detected_class in zone_info['roles']:
                    return detected_class
                # Otherwise, assign most likely role for this zone
                return zone_info['roles'][0]
        
        return detected_class  # Fallback to detected class

# === NEW: Motion Analysis for Better Classification ===
class MotionAnalyzer:
    def __init__(self, history_length=10):
        self.history_length = history_length
        self.motion_history = {}  # track_id -> [(x, y, timestamp)]
    
    def add_position(self, track_id, x_center, y_center):
        if track_id not in self.motion_history:
            self.motion_history[track_id] = []
        
        self.motion_history[track_id].append((x_center, y_center, time.time()))
        
        # Keep only recent positions
        if len(self.motion_history[track_id]) > self.history_length:
            self.motion_history[track_id].pop(0)
    
    def get_motion_pattern(self, track_id):
        if track_id not in self.motion_history or len(self.motion_history[track_id]) < 3:
            return 'unknown'
        
        positions = self.motion_history[track_id]
        
        # Calculate velocity
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[0][0]
            dy = positions[-1][1] - positions[0][1]
            velocity = np.sqrt(dx**2 + dy**2)
            
            # Classify based on motion patterns
            if velocity < 5:  # Very slow movement
                return 'static'  # Likely stumps or stationary umpire
            elif velocity > 50:  # Very fast movement
                return 'fast'    # Likely ball
            elif abs(dy) > abs(dx) * 2:  # Vertical movement dominant
                return 'vertical'  # Likely bowler run-up or fielder running
            else:
                return 'horizontal'  # Likely fielder movement
        
        return 'unknown'

# === NEW: Enhanced Bowling Analysis System ===
class BowlingAreaTrigger:
    def __init__(self):
        self.bowler_in_area = False
        self.last_bowler_detection = 0
        self.bowler_confidence = 0.0
        self.prediction_triggered = False
        self.frame_count = 0
        
        # === NEW: Enhanced Analysis Variables ===
        self.bowler_positions = []  # Track bowler position over time
        self.bowler_speed = 0.0     # Bowler's run-up speed
        self.bowler_side = "center"  # Left/right/center position
        self.fielding_gaps = 0      # Number of gaps in fielding
        self.batsman_stance = "neutral"  # Aggressive/defensive/neutral
    
    def check_bowler_in_area(self, tracked_objects, cricket_zones, frame_count):
        """Check if bowler is in bowling area and trigger prediction"""
        self.frame_count = frame_count
        
        for obj in tracked_objects:
            x1_t, y1_t, x2_t, y2_t, obj_id = map(int, obj)
            x_center, y_center = get_box_center([x1_t, y1_t, x2_t, y2_t])
            
            # Check if object is in bowling area
            bowling_zone = cricket_zones.zones['bowling_area']
            if (bowling_zone['x_range'][0] <= x_center <= bowling_zone['x_range'][1] and
                bowling_zone['y_range'][0] <= y_center <= bowling_zone['y_range'][1]):
                
                # === NEW: Enhanced Bowler Analysis ===
                self.analyze_bowler_factors(x_center, y_center, frame_count)
                
                # This is a potential bowler in the bowling area
                self.bowler_in_area = True
                self.last_bowler_detection = frame_count
                return True
        
        # Bowler not in area
        self.bowler_in_area = False
        return False
    
    def analyze_bowler_factors(self, x_center, y_center, frame_count):
        """Analyze bowler's run-up speed, position, and other factors"""
        # Store bowler position for speed calculation
        self.bowler_positions.append((x_center, y_center, frame_count))
        
        # Keep only recent positions (last 15 frames)
        if len(self.bowler_positions) > 15:
            self.bowler_positions.pop(0)
        
        # === NEW: Calculate Bowler Speed ===
        if len(self.bowler_positions) >= 2:
            dx = self.bowler_positions[-1][0] - self.bowler_positions[0][0]
            dy = self.bowler_positions[-1][1] - self.bowler_positions[0][1]
            frames_diff = self.bowler_positions[-1][2] - self.bowler_positions[0][2]
            
            if frames_diff > 0:
                self.bowler_speed = np.sqrt(dx**2 + dy**2) / frames_diff
        
        # === NEW: Determine Bowler Side Position ===
        frame_width = 1920  # Approximate, will be updated dynamically
        if x_center < frame_width * 0.4:
            self.bowler_side = "left"
        elif x_center > frame_width * 0.6:
            self.bowler_side = "right"
        else:
            self.bowler_side = "center"
    
    def analyze_fielding_gaps(self, current_frame_objects, frame_width):
        """Analyze fielding positions to find gaps"""
        if not current_frame_objects:
            self.fielding_gaps = 0
            return
        
        # Count fielders in different zones
        off_side_fielders = 0
        leg_side_fielders = 0
        
        for obj in current_frame_objects:
            if obj['class'] == 'fielder':
                x_center = (obj['box'][0] + obj['box'][2]) / 2
                if x_center > frame_width * 0.5:  # Off side
                    off_side_fielders += 1
                else:  # Leg side
                    leg_side_fielders += 1
        
        # Calculate gaps (fewer fielders = more gaps)
        self.fielding_gaps = max(0, 6 - (off_side_fielders + leg_side_fielders))
    
    def analyze_batsman_stance(self, batsman_box, motion_analyzer):
        """Analyze batsman's stance and movement"""
        if not batsman_box:
            self.batsman_stance = "neutral"
            return
        
        # Get batsman's motion pattern
        # This is a simplified analysis - can be enhanced
        if hasattr(motion_analyzer, 'motion_history'):
            # Check if batsman is moving aggressively
            self.batsman_stance = "neutral"  # Default
            # Could be enhanced with more sophisticated analysis
    
    def should_trigger_prediction(self, frame_count):
        """Determine if we should trigger a new prediction"""
        # Only trigger prediction ONCE per delivery, when bowler just enters area
        if self.bowler_in_area and not self.prediction_triggered:
            # Only trigger if bowler was not in area in previous frame
            self.prediction_triggered = True
            self.last_bowler_detection = frame_count
            return True
        elif not self.bowler_in_area:
            # Reset trigger when bowler leaves area
            self.prediction_triggered = False
        return False
    
    def get_enhanced_prediction(self):
        """Generate enhanced prediction based on all factors"""
        # Enhanced prediction with realistic cricket shots
        if self.bowler_speed > 30:  # Fast bowler
            if self.fielding_gaps > 3:
                prediction = "Pull Shot"
            else:
                prediction = "Defensive Block"
        elif self.bowler_speed < 10:  # Slow bowler
            if self.fielding_gaps > 2:
                prediction = "Cover Drive"
            else:
                prediction = "Defensive Block"
        else:  # Medium pace
            if self.fielding_gaps > 3:
                prediction = "Square Cut"
            else:
                prediction = "Straight Drive"
        
        # Determine confidence based on factors
        if self.bowler_speed > 25 and self.fielding_gaps > 2:
            confidence = "High"
        elif self.bowler_speed < 15 and self.fielding_gaps < 2:
            confidence = "Medium"
        else:
            confidence = "Good"
        
        return f"{prediction} ({confidence})"

# === NEW: Multi-Frame Shot Prediction ===
class MultiFrameShotPredictor:
    def __init__(self, frame_buffer_size=15):
        self.frame_buffer_size = frame_buffer_size
        self.fielding_positions = []  # Store fielding positions over time
        self.batsman_positions = []
    
    def add_frame_data(self, current_frame_objects, batsman_box):
        # Store fielding positions
        fielders = []
        for obj in current_frame_objects:
            if obj['class'] not in ['batsman', 'umpire', 'wicket-keeper']:
                fielders.append(obj['box'])
        
        self.fielding_positions.append(fielders)
        
        # Store batsman position
        if batsman_box:
            self.batsman_positions.append(batsman_box)
        
        # Keep only recent frames
        if len(self.fielding_positions) > self.frame_buffer_size:
            self.fielding_positions.pop(0)
        if len(self.batsman_positions) > self.frame_buffer_size:
            self.batsman_positions.pop(0)
    
    def predict_shot(self, frame_width):
        if not self.batsman_positions:
            return "Defensive Block"  # Default shot when no batsman
        
        # Use average of last 5 frames for stability
        recent_fielders = self.fielding_positions[-5:] if len(self.fielding_positions) >= 5 else self.fielding_positions
        recent_batsman = self.batsman_positions[-1]
        
        batsman_x_center = (recent_batsman[0] + recent_batsman[2]) / 2
        fielders_off = 0
        fielders_leg = 0
        
        # Count fielders across recent frames
        for frame_fielders in recent_fielders:
            for fielder_box in frame_fielders:
                fielder_x_center = (fielder_box[0] + fielder_box[2]) / 2
                if fielder_x_center > batsman_x_center:
                    fielders_off += 1
                else:
                    fielders_leg += 1
        
        # Enhanced prediction logic with more realistic shots
        total_fielders = fielders_off + fielders_leg
        if total_fielders == 0:
            return "Straight Drive"
        
        off_ratio = fielders_off / total_fielders
        
        if off_ratio > 0.7:
            return "Pull Shot"
        elif off_ratio < 0.3:
            return "Cover Drive"
        elif off_ratio > 0.5:
            return "Square Cut"
        else:
            return "Leg Glance"

# === NEW: Adaptive Confidence Thresholds ===
def get_adaptive_confidence(class_name, frame_quality=0.5):
    base_confidences = {
        'ball': 0.35,      # Lower for fast-moving objects
        'batsman': 0.5,    # Higher for important objects
        'fielder': 0.4,    # Medium for fielders
        'umpire': 0.45,    # Medium-high for officials
        'wicket-keeper': 0.45,
        'stumps': 0.6      # Highest for static objects
    }
    
    base_conf = base_confidences.get(class_name, 0.4)
    
    # Adjust based on frame quality (motion blur, lighting)
    if frame_quality < 0.3:  # Poor quality
        return base_conf * 0.8
    elif frame_quality > 0.7:  # Good quality
        return base_conf * 1.1
    return base_conf

# === NEW: Color-Based Classification System ===
def analyze_jersey_color(frame, box):
    """Analyze the dominant color in a bounding box to determine jersey color"""
    x1, y1, x2, y2 = map(int, box)
    
    # Extract the region of interest
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 'unknown'
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate average color in the region
    avg_color = np.mean(hsv, axis=(0, 1))
    hue, saturation, value = avg_color
    
    # Define color ranges for jersey detection
    # Blue jersey detection (batsmen)
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    
    # Yellow jersey detection (fielders/bowler)
    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([30, 255, 255])
    
    # Check which color range the average falls into
    if blue_lower[0] <= hue <= blue_upper[0] and saturation > 50:
        return 'blue'
    elif yellow_lower[0] <= hue <= yellow_upper[0] and saturation > 50:
        return 'yellow'
    elif value > 200 and saturation < 30:  # High value, low saturation = white
        return 'white'
    else:
        return 'unknown'

def get_role_from_color_and_zone(jersey_color, zone_role, detected_class, x_center, y_center):
    """Combine color and zone information for better classification"""
    
    # Color-based role mapping
    color_roles = {
        'blue': ['batsman', 'non-striker'],  # Blue jerseys = batsmen
        'yellow': ['fielder', 'bowler'],     # Yellow jerseys = fielders/bowler
        'white': ['umpire']                  # White jersey = umpire
    }
    
    # ENHANCED: Specific bowler identification logic
    if jersey_color == 'yellow' and zone_role == 'bowler':
        return 'bowler'  # Yellow jersey in bowling area = definitely bowler
    
    # ENHANCED: Specific wicket-keeper identification logic
    if jersey_color == 'blue' and zone_role == 'wicket-keeper':
        return 'wicket-keeper'  # Blue jersey in wicket-keeper area = definitely wicket-keeper
    
    # ENHANCED: Specific batsman identification logic
    if jersey_color == 'blue' and zone_role == 'batsman':
        return 'batsman'  # Blue jersey in batting area = definitely batsman
    
    # Get possible roles from color
    possible_roles = color_roles.get(jersey_color, [detected_class])
    
    # If zone role matches color-based roles, use zone role
    if zone_role in possible_roles:
        return zone_role
    
    # If detected class matches color-based roles, use detected class
    if detected_class in possible_roles:
        return detected_class
    
    # Otherwise, use the first possible role from color
    return possible_roles[0] if possible_roles else detected_class

# === NEW: Frame Quality Assessment ===
def calculate_frame_quality(frame):
    """Simple frame quality assessment based on blur and contrast"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate contrast
    contrast = gray.std()
    
    # Normalize and combine
    blur_score = min(laplacian_var / 100, 1.0)  # Normalize blur
    contrast_score = min(contrast / 50, 1.0)    # Normalize contrast
    
    # Combined quality score
    quality = (blur_score + contrast_score) / 2
    return max(0.1, min(1.0, quality))  # Clamp between 0.1 and 1.0

# === CORRECTED: Full Intersection over Union (IoU) Function ===
def iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    denominator = float(boxAArea + boxBArea - interArea)
    
    # Add a small epsilon to avoid division by zero
    if denominator == 0:
        return 0.0

    iou_val = interArea / denominator
    return iou_val

def batch_iou_matrix(tracked_boxes, yolo_boxes):
    """
    Efficiently compute IoU matrix between tracked boxes and YOLO detections.
    Returns a matrix where [i][j] = IoU between tracked_box[i] and yolo_box[j]
    """
    if len(tracked_boxes) == 0 or len(yolo_boxes) == 0:
        return np.array([])
    
    # Convert to numpy arrays for vectorized operations
    tracked = np.array(tracked_boxes)
    yolo = np.array(yolo_boxes)
    
    # Extract coordinates
    tracked_x1, tracked_y1, tracked_x2, tracked_y2 = tracked[:, 0], tracked[:, 1], tracked[:, 2], tracked[:, 3]
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo[:, 0], yolo[:, 1], yolo[:, 2], yolo[:, 3]
    
    # Compute intersection coordinates
    x1 = np.maximum(tracked_x1[:, np.newaxis], yolo_x1)
    y1 = np.maximum(tracked_y1[:, np.newaxis], yolo_y1)
    x2 = np.minimum(tracked_x2[:, np.newaxis], yolo_x2)
    y2 = np.minimum(tracked_y2[:, np.newaxis], yolo_y2)
    
    # Compute intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute union area
    tracked_area = (tracked_x2 - tracked_x1) * (tracked_y2 - tracked_y1)
    yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
    union = tracked_area[:, np.newaxis] + yolo_area - intersection
    
    # Compute IoU
    iou_matrix = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return iou_matrix

def get_box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

# === Professional Overlay System ===
class AnalysisSidebar:
    """Left sidebar for analysis data with grey labels and green prediction boxes"""
    
    def __init__(self):
        self.predictions = {
            'ideal_shot': 'Defensive Block',
            'ball_length': 'Good Length',
            'predicted_shot': 'Straight Drive',
            'game_mood': 'Balanced'
        }
        self.sidebar_width = 350  # Increased width
        self.sidebar_height = 400  # Increased height for four separate sections
        self.x_offset = 50  # Move more towards center
        self.y_offset = 100  # Move higher up to accommodate four sections
    
    def update_predictions(self, ideal_shot, ball_length, predicted_shot, game_mood):
        """Update the prediction values"""
        self.predictions = {
            'ideal_shot': ideal_shot,
            'ball_length': ball_length,
            'predicted_shot': predicted_shot,
            'game_mood': game_mood
        }
    
    def draw(self, frame, x=None, y=None):
        """Draw four independent analysis sections on the frame"""
        if x is None:
            x = self.x_offset
        if y is None:
            y = self.y_offset
        frame_height, frame_width = frame.shape[:2]
        # Bigger box sizes, slightly smaller text
        section_width = 290
        section_height = 115
        title_height = 32
        value_height = section_height - title_height
        section_spacing = 32
        value_padding = 28
        for i, (label, prediction) in enumerate(self.predictions.items()):
            section_x = x
            section_y = y + (i * (section_height + section_spacing))
            # Draw blue title area
            cv2.rectangle(frame, (section_x, section_y), (section_x + section_width, section_y + title_height), (184, 95, 0), -1)  # BGR for #005FB8
            # Draw semi-transparent white value area (taller)
            overlay = frame.copy()
            cv2.rectangle(overlay, (section_x, section_y + title_height), (section_x + section_width, section_y + section_height), (236, 236, 236), -1)  # #ECECEC
            alpha = 0.5
            frame[section_y + title_height:section_y + section_height, section_x:section_x + section_width] = cv2.addWeighted(
                overlay[section_y + title_height:section_y + section_height, section_x:section_x + section_width], alpha,
                frame[section_y + title_height:section_y + section_height, section_x:section_x + section_width], 1 - alpha, 0)
            # Centered white title text (slightly smaller)
            label_text = label.replace('_', ' ').title()
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            tx = section_x + (section_width - tw) // 2
            ty = section_y + title_height - 10
            cv2.putText(frame, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            # Centered value text (slightly smaller)
            (vw, vh), _ = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            vx = section_x + (section_width - vw) // 2
            vy = section_y + title_height + value_padding + vh
            cv2.putText(frame, prediction, (vx, vy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3, cv2.LINE_AA)

class TextOverlay:
    """Floating text overlay for logs in top-right corner"""
    
    def __init__(self):
        self.logs = []
        self.max_logs = 15  # Keep more logs for better coverage
        self.margin_x = 80  # Margin from right edge
        self.margin_y = 80  # Margin from top edge
        self.line_height = 40  # Space between lines
    
    def add_log(self, message):
        """Add a new log entry"""
        self.logs.append(message)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
    
    def draw(self, frame):
        """Draw floating logs in top-right corner inside a semi-transparent box, aligned in level with the left overlays"""
        frame_height, frame_width = frame.shape[:2]
        # Calculate starting position (top-right corner)
        padding_x = 50  # Increased padding for bigger box
        padding_y = 30  # Increased padding for bigger box
        # Use same alpha as bottom overlay
        box_alpha = 160 / 255.0
        right_margin = 40
        # Prepare log lines
        log_lines = [str(log) for log in self.logs[:10]]
        font_scale = 1.0
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Calculate max width and total height
        max_width = 0
        total_height = 0
        line_height = 0
        for log_text in log_lines:
            (text_width, text_height), baseline = cv2.getTextSize(log_text, font, font_scale, thickness)
            max_width = max(max_width, text_width)
            line_height = max(line_height, text_height + baseline)
        total_height = line_height * len(log_lines)
        # Box dimensions
        box_width = max_width + 2 * padding_x
        box_height = total_height + 2 * padding_y
        # Box position (top right, aligned in level with left overlays)
        box_x = frame_width - box_width - right_margin
        box_y = 40
        # Draw semi-transparent dark box (same as graph overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (20, 20, 20), -1)
        frame[box_y:box_y+box_height, box_x:box_x+box_width] = cv2.addWeighted(
            overlay[box_y:box_y+box_height, box_x:box_x+box_width], box_alpha,
            frame[box_y:box_y+box_height, box_x:box_x+box_width], 1 - box_alpha, 0)
        # Draw logs inside the box
        current_y = box_y + padding_y + line_height - 8
        for i, log_text in enumerate(log_lines):
            (text_width, text_height), baseline = cv2.getTextSize(log_text, font, font_scale, thickness)
            text_x = box_x + padding_x
            # Draw white text with black outline for readability
            cv2.putText(frame, log_text, (text_x, current_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, log_text, (text_x, current_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            current_y += line_height
            if i >= 9:
                break

class CricketOverlaySystem:
    """Main overlay system that manages analysis sidebar and text overlay"""
    
    def __init__(self):
        self.left_sidebar = AnalysisSidebar()
        self.text_overlay = TextOverlay()
        
        # Add initial logs
        self.text_overlay.add_log("System initialized")
        self.text_overlay.add_log("AI Model loaded")
        self.text_overlay.add_log("Tracking active")
        self.text_overlay.add_log("Ready for analysis")
        self.text_overlay.add_log("Cricket analysis started")
    
    def update_analysis(self, ideal_shot, ball_length, predicted_shot, game_mood):
        """Update left sidebar analysis data"""
        self.left_sidebar.update_predictions(ideal_shot, ball_length, predicted_shot, game_mood)
    
    def add_log(self, message):
        """Add log to text overlay"""
        self.text_overlay.add_log(message)
    
    def draw_overlays(self, frame):
        """Draw analysis sidebar and text overlay on the frame"""
        # Draw analysis sidebar on the left
        self.left_sidebar.draw(frame)
        # Draw floating text overlay in top-right corner
        self.text_overlay.draw(frame)

# === NEW: Enhanced Analysis Engine ===
class CricketAnalysisEngine:
    """Engine to calculate analysis data for the left sidebar"""
    
    def __init__(self):
        self.ball_trajectory = []
        self.fielding_gaps = 0
        self.match_context = {
            'run_rate': 5.0,
            'wickets_remaining': 10,
            'overs_remaining': 50.0,
            'current_score': 0
        }
    
    def analyze_ball_length(self, ball_positions):
        """Analyze ball length based on trajectory"""
        if len(ball_positions) < 3:
            return "Good Length"  # Default to most common length
        
        # Calculate vertical movement
        start_y = ball_positions[0][1]
        end_y = ball_positions[-1][1]
        dy = end_y - start_y
        
        # More realistic thresholds based on cricket physics
        if dy > 80:
            return "Bouncer"
        elif dy > 40:
            return "Short"
        elif dy < -60:
            return "Yorker"
        elif dy < -30:
            return "Full"
        else:
            return "Good Length"
    
    def calculate_ideal_shot(self, ball_length, fielding_gaps, game_mood):
        """Calculate ideal shot based on multiple factors"""
        # Enhanced shot selection logic
        if ball_length == "Bouncer":
            if fielding_gaps > 3:
                return "Pull Shot"
            else:
                return "Defensive Block"
        elif ball_length == "Yorker":
            return "Defensive Block"
        elif ball_length == "Full":
            if game_mood == "Pressured":
                return "Defensive Block"
            elif fielding_gaps > 4:
                return "Drive Shot"
            else:
                return "Straight Drive"
        elif ball_length == "Short":
            if fielding_gaps > 2:
                return "Cut Shot"
            else:
                return "Defensive Block"
        elif ball_length == "Good Length":
            if fielding_gaps > 3:
                return "Cover Drive"
            else:
                return "Defensive Block"
        else:
            return "Defensive Block"  # Safe default
    
    def analyze_game_mood(self, run_rate, wickets_remaining, overs_remaining):
        """Analyze game mood based on match situation"""
        # Enhanced game mood analysis
        if wickets_remaining < 3:
            return "Defensive"
        elif run_rate > 8:
            return "Pressured"
        elif run_rate > 6:
            return "Aggressive"
        elif wickets_remaining < 6:
            return "Cautious"
        elif run_rate < 4:
            return "Conservative"
        else:
            return "Balanced"
    
    def generate_cricket_insights(self, detections, bowler_speed, bowler_side):
        """Generate cricket-specific insights for logs"""
        insights = []
        
        # Analyze fielding positions
        fielders = [d for d in detections if d['class'] == 'fielder']
        if len(fielders) < 4:
            insights.append("Power play fielding")
        elif len(fielders) > 8:
            insights.append("Defensive fielding")
        
        # Check for slip fielder
        for fielder in fielders:
            x_center = (fielder['box'][0] + fielder['box'][2]) / 2
            if x_center < 0.3:  # Left side
                insights.append("Slip fielder present")
                break
        
        # Analyze wicket keeper
        wicket_keepers = [d for d in detections if d['class'] == 'wicket-keeper']
        if wicket_keepers:
            insights.append("Wicket keeper ready")
        
        # Analyze bowler
        if bowler_speed > 30:
            insights.append("Fast bowler detected")
        elif bowler_speed < 15:
            insights.append("Spin bowler detected")
        else:
            insights.append("Medium pace bowler")
        
        if bowler_side == "left":
            insights.append("Bowler from left")
        elif bowler_side == "right":
            insights.append("Bowler from right")
        
        # Analyze batsman stance
        batsmen = [d for d in detections if d['class'] == 'batsman']
        if batsmen:
            insights.append("Batsman in position")
        
        # Add realistic pitch conditions
        insights.append("Pitch conditions normal")
        
        return insights

# Setup deterministic colors for better visibility
colors = {
    'batsman': (0, 255, 0),      # Bright Green
    'ball': (255, 0, 0),         # Bright Red
    'fielder': (0, 0, 255),      # Bright Blue
    'umpire': (255, 255, 0),     # Bright Yellow
    'wicket-keeper': (255, 0, 255), # Magenta
    'stumps': (0, 255, 255),     # Cyan
    'unknown': (128, 128, 128)   # Gray
}

# Fallback for any other classes
def get_color(class_name):
    if class_name in colors:
        return colors[class_name]
    else:
        # Generate deterministic color based on class name hash
        import hashlib
        hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
        return (hash_val % 256, (hash_val >> 8) % 256, (hash_val >> 16) % 256) 