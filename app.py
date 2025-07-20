import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from datetime import datetime
import plotly.express as px
from ultralytics import YOLO
from sort import Sort
import pandas as pd

# Import our existing analysis classes directly from local files
from cricket_analysis import (
    CricketOverlaySystem, 
    CricketAnalysisEngine, 
    BowlingAreaTrigger,
    MultiFrameShotPredictor,
    TemporalSmoother,
    CricketZones,
    MotionAnalyzer,
    get_color,
    get_box_center,
    batch_iou_matrix,
    analyze_jersey_color,
    get_role_from_color_and_zone,
    calculate_frame_quality,
    get_adaptive_confidence
)

# Page configuration
st.set_page_config(
    page_title="Cricket AI Analyzer",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 500px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .analysis-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .prediction-medium {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .prediction-low {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the custom trained YOLO model with caching"""
    try:
        # Priority 1: Custom trained model from runs/detect/train/weights/
        custom_model_path = "runs/detect/train/weights/best.pt"
        if os.path.exists(custom_model_path):
            model = YOLO(custom_model_path)
            st.success("✅ Custom trained cricket model loaded successfully!")
            return model
        
        # Priority 2: Custom model in models directory
        models_dir_path = "models/best.pt"
        if os.path.exists(models_dir_path):
            model = YOLO(models_dir_path)
            st.success("✅ Custom cricket model loaded from models directory!")
            return model
        
        # Priority 3: Default YOLO model as fallback
        default_model_path = "models/yolov8n.pt"
        if os.path.exists(default_model_path):
            model = YOLO(default_model_path)
            st.warning("⚠️ Using default YOLO model (custom model not found)")
            return model
        else:
            model = YOLO("yolov8n.pt")
            st.warning("⚠️ Using default YOLO model (downloading)")
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

class StreamlitCricketAnalyzer:
    def __init__(self):
        self.model = load_model()
        self.tracker = Sort(max_age=3, min_hits=2, iou_threshold=0.3)
        self.overlay_system = CricketOverlaySystem()
        self.analysis_engine = CricketAnalysisEngine()
        self.bowling_trigger = BowlingAreaTrigger()
        self.shot_predictor = MultiFrameShotPredictor()
        self.temporal_smoother = TemporalSmoother()
        self.motion_analyzer = MotionAnalyzer()
        
        # Analysis tracking
        self.frame_analysis = []
        self.detection_stats = {
            'batsman': 0, 'ball': 0, 'fielder': 0, 
            'umpire': 0, 'wicket-keeper': 0, 'stumps': 0
        }
        self.prediction_history = []
        
    def process_frame(self, frame, frame_count):
        """Process a single frame and return analysis results"""
        if self.model is None:
            return frame, {}
        
        # Calculate frame quality
        frame_quality = calculate_frame_quality(frame)
        adaptive_conf = get_adaptive_confidence('batsman', frame_quality)
        high_conf = max(0.75, adaptive_conf)
        
        # YOLO detection
        results = self.model.predict(source=frame, conf=high_conf, iou=0.5, verbose=False, agnostic_nms=True)
        
        yolo_detections = []
        detections_for_tracker_list = []
        
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence > 0.4:
                    yolo_detections.append([int(coord) for coord in box.xyxy[0]] + [int(box.cls[0])])
                    detections_for_tracker_list.append([*(box.xyxy[0]), confidence])
        
        detections_for_tracker = np.array(detections_for_tracker_list)
        
        # Tracking
        try:
            tracked_objects = self.tracker.update(detections_for_tracker) if len(detections_for_tracker) > 0 else np.empty((0, 5))
        except Exception as e:
            st.error(f"Tracking error: {e}")
            tracked_objects = np.empty((0, 5))
        
        # Initialize zones if not done
        if not hasattr(self, 'cricket_zones'):
            frame_height, frame_width = frame.shape[:2]
            self.cricket_zones = CricketZones(frame_width, frame_height)
        
        current_frame_objects = []
        batsman_box = None
        
        # Process tracked objects
        if len(tracked_objects) > 0 and len(yolo_detections) > 0:
            tracked_boxes = [[obj[0], obj[1], obj[2], obj[3]] for obj in tracked_objects]
            yolo_boxes = [det[:4] for det in yolo_detections]
            iou_matrix = batch_iou_matrix(tracked_boxes, yolo_boxes)
            
            for obj_idx, obj in enumerate(tracked_objects):
                x1_t, y1_t, x2_t, y2_t, obj_id = map(int, obj)
                tracked_box = [x1_t, y1_t, x2_t, y2_t]
                x_center, y_center = get_box_center(tracked_box)
                
                # Motion analysis
                self.motion_analyzer.add_position(obj_id, x_center, y_center)
                
                # Classification
                class_name = "unknown"
                if len(iou_matrix) > 0 and obj_idx < len(iou_matrix):
                    best_iou_val = np.max(iou_matrix[obj_idx])
                    best_match_idx = np.argmax(iou_matrix[obj_idx])
                    
                    if best_iou_val > 0.6:
                        best_yolo_det = yolo_detections[best_match_idx]
                        detected_class = self.model.names[best_yolo_det[4]]
                        
                        # Enhanced classification
                        zone_role = self.cricket_zones.get_zone_role(x_center, y_center, detected_class)
                        jersey_color = analyze_jersey_color(frame, tracked_box)
                        class_name = get_role_from_color_and_zone(jersey_color, zone_role, detected_class, x_center, y_center)
                        class_name = self.temporal_smoother.smooth_role(obj_id, class_name)
                
                # Draw bounding box
                color = get_color(class_name)
                cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), color, 3)
                cv2.rectangle(frame, (x1_t+1, y1_t+1), (x2_t-1, y2_t-1), (255, 255, 255), 1)
                
                # Draw label
                label = f"{class_name.upper()}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1_t, y1_t - label_size[1] - 10), (x1_t + label_size[0], y1_t), color, -1)
                cv2.putText(frame, label, (x1_t, y1_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                current_frame_objects.append({'class': class_name, 'box': tracked_box})
                if class_name == 'batsman':
                    batsman_box = tracked_box
                
                # Update detection stats
                if class_name in self.detection_stats:
                    self.detection_stats[class_name] += 1
        
        # Analysis
        self.shot_predictor.add_frame_data(current_frame_objects, batsman_box)
        shot_prediction = self.shot_predictor.predict_shot(frame.shape[1])
        
        # Bowling analysis
        bowler_in_area = self.bowling_trigger.check_bowler_in_area(tracked_objects, self.cricket_zones, frame_count)
        self.bowling_trigger.analyze_fielding_gaps(current_frame_objects, frame.shape[1])
        
        # Ball trajectory analysis
        ball_objects = [obj for obj in current_frame_objects if obj['class'] == 'ball']
        if ball_objects:
            ball_center = get_box_center(ball_objects[0]['box'])
            self.analysis_engine.ball_trajectory.append(ball_center)
            if len(self.analysis_engine.ball_trajectory) > 10:
                self.analysis_engine.ball_trajectory.pop(0)
        
        # Calculate analysis data
        ball_length = self.analysis_engine.analyze_ball_length(self.analysis_engine.ball_trajectory)
        game_mood = self.analysis_engine.analyze_game_mood(5.0, 10, 50.0)
        ideal_shot = self.analysis_engine.calculate_ideal_shot(ball_length, self.bowling_trigger.fielding_gaps, game_mood)
        
        # Update overlay system
        self.overlay_system.update_analysis(ideal_shot, ball_length, shot_prediction, game_mood)
        
        # Add logs
        if frame_count % 30 == 0:
            self.overlay_system.add_log(f"Frame: {frame_count}")
        if frame_count % 50 == 0:
            self.overlay_system.add_log(f"Detected: {len(current_frame_objects)} objects")
        
        # Store analysis data
        analysis_data = {
            'frame': frame_count,
            'shot_prediction': shot_prediction,
            'ball_length': ball_length,
            'ideal_shot': ideal_shot,
            'game_mood': game_mood,
            'objects_detected': len(current_frame_objects),
            'bowler_in_area': bowler_in_area,
            'fielding_gaps': self.bowling_trigger.fielding_gaps
        }
        
        self.frame_analysis.append(analysis_data)
        self.prediction_history.append(shot_prediction)
        
        # Draw overlays
        self.overlay_system.draw_overlays(frame)
        
        return frame, analysis_data

def main():
    # Header
    st.markdown('<h1 class="main-header">🏏 Cricket AI Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Cricket Video Analysis with AI-Powered Shot Prediction")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StreamlitCricketAnalyzer()
    
    # Sidebar controls
    st.sidebar.markdown("## ⚙️ Analysis Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Cricket Video", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a cricket video for analysis"
    )
    
    # Analysis parameters
    st.sidebar.markdown("### 📊 Analysis Settings")
    
    show_overlays = st.sidebar.checkbox("Show Analysis Overlays", value=True)
    show_statistics = st.sidebar.checkbox("Show Real-time Statistics", value=True)
    
    # Main content area
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Video processing
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.markdown("### 📹 Video Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", total_frames)
            with col2:
                st.metric("FPS", fps)
            with col3:
                duration = total_frames / fps if fps > 0 else 0
                st.metric("Duration", f"{duration:.1f}s")
            with col4:
                st.metric("File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
            
            # Processing controls
            st.markdown("### 🎮 Processing Controls")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                start_frame = st.number_input("Start Frame", min_value=0, max_value=total_frames-1, value=0)
            with col2:
                end_frame = st.number_input("End Frame", min_value=start_frame, max_value=total_frames-1, value=min(start_frame+100, total_frames-1))
            with col3:
                frame_step = st.number_input("Frame Step", min_value=1, max_value=10, value=1)
            
            # Process button
            if st.button("🚀 Start Analysis", type="primary"):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Video display placeholder
                video_placeholder = st.empty()
                
                # Statistics placeholders
                if show_statistics:
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        detection_chart = st.empty()
                    with stats_col2:
                        prediction_chart = st.empty()
                
                # Process frames
                processed_frames = []
                frame_count = start_frame
                
                while frame_count <= end_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, analysis_data = st.session_state.analyzer.process_frame(frame, frame_count)
                    
                    if show_overlays:
                        processed_frames.append(processed_frame)
                    
                    # Update progress
                    progress = (frame_count - start_frame) / (end_frame - start_frame)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{end_frame}")
                    
                    # Display current frame
                    if show_overlays and len(processed_frames) > 0:
                        # Convert BGR to RGB for display
                        display_frame = cv2.cvtColor(processed_frames[-1], cv2.COLOR_BGR2RGB)
                        video_placeholder.image(display_frame, caption=f"Frame {frame_count}", use_container_width=True)
                    
                    # Update statistics
                    if show_statistics and frame_count % 10 == 0:
                        # Detection statistics
                        detection_df = pd.DataFrame(list(st.session_state.analyzer.detection_stats.items()), 
                                                   columns=['Object', 'Count'])
                        fig1 = px.bar(detection_df, x='Object', y='Count', 
                                     title="Object Detection Statistics",
                                     color='Count', color_continuous_scale='viridis')
                        detection_chart.plotly_chart(fig1, use_container_width=True)
                        
                        # Prediction history
                        if len(st.session_state.analyzer.prediction_history) > 0:
                            prediction_df = pd.DataFrame({
                                'Frame': range(len(st.session_state.analyzer.prediction_history)),
                                'Prediction': st.session_state.analyzer.prediction_history
                            })
                            fig2 = px.line(prediction_df, x='Frame', y='Prediction', 
                                          title="Shot Prediction History")
                            prediction_chart.plotly_chart(fig2, use_container_width=True)
                    
                    frame_count += frame_step
                    
                    # Add small delay for UI responsiveness
                    time.sleep(0.01)
                
                # Analysis complete
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis Complete!")
                
                # Display final results
                st.markdown("### 📊 Analysis Results")
                
                # Summary statistics
                if len(st.session_state.analyzer.frame_analysis) > 0:
                    analysis_df = pd.DataFrame(st.session_state.analyzer.frame_analysis)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_objects = analysis_df['objects_detected'].mean()
                        st.metric("Avg Objects/Frame", f"{avg_objects:.1f}")
                    with col2:
                        unique_predictions = analysis_df['shot_prediction'].nunique()
                        st.metric("Unique Predictions", unique_predictions)
                    with col3:
                        bowler_detections = analysis_df['bowler_in_area'].sum()
                        st.metric("Bowler Detections", bowler_detections)
                    with col4:
                        avg_gaps = analysis_df['fielding_gaps'].mean()
                        st.metric("Avg Fielding Gaps", f"{avg_gaps:.1f}")
                    
                    # Detailed analysis table
                    st.markdown("#### 📋 Frame-by-Frame Analysis")
                    st.dataframe(analysis_df, use_container_width=True)
                    
                    # Charts
                    st.markdown("#### �� Analysis Charts")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Shot prediction distribution
                        prediction_counts = analysis_df['shot_prediction'].value_counts()
                        fig = px.pie(values=prediction_counts.values, names=prediction_counts.index,
                                   title="Shot Prediction Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Objects detected over time
                        fig = px.line(analysis_df, x='frame', y='objects_detected',
                                    title="Objects Detected Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.markdown("### 💾 Download Results")
                if len(st.session_state.analyzer.frame_analysis) > 0:
                    # Create CSV for download
                    csv = analysis_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis CSV",
                        data=csv,
                        file_name=f"cricket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"❌ Error processing video: {e}")
        
        finally:
            # Cleanup
            cap.release()
            os.unlink(video_path)
    
    else:
        # Welcome message
        st.markdown("""
        ### 🎯 How to Use:
        1. **Upload Video**: Use the sidebar to upload a cricket video file
        2. **Configure Settings**: Adjust analysis parameters as needed
        3. **Start Analysis**: Click the "Start Analysis" button
        4. **View Results**: Watch real-time analysis and view final statistics
        
        ### �� Supported Features:
        - **Object Detection**: Batsman, Ball, Fielders, Umpire, Wicket-keeper, Stumps
        - **Shot Prediction**: AI-powered shot recommendations
        - **Ball Length Analysis**: Yorker, Full, Good Length, Short, Bouncer
        - **Fielding Analysis**: Gap detection and positioning
        - **Real-time Overlays**: Live analysis display on video
        - **Statistics Dashboard**: Comprehensive analysis metrics
        """)
        
        # Sample video upload area
        st.markdown("### 📁 Upload Area")
        st.info("�� Use the sidebar to upload your cricket video and start analysis!")

if __name__ == "__main__":
    main()