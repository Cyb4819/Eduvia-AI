import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from threading import Lock
import threading
import time
from collections import deque
from utils.learning_state import LearningStateDetector

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Thresholds (tunable)
GAZE_THRESHOLD = 0.25
MOVEMENT_THRESHOLD_LOW = 5.0
MOVEMENT_THRESHOLD_HIGH = 15.0

# Stuck detection thresholds
STUCK_REGION_PERCENT = 0.15  # 15% of screen
STUCK_DURATION_THRESHOLD = 6.0  # seconds (changed from 20.0 to 6.0)
MISREAD_GAZE_WINDOW = 3.0  # seconds for rapid L-R-L pattern
MISREAD_MAX_GAZE_SHIFTS = 3  # min shifts in window

class BehavioralObserver:
    def __init__(self):
        self.cap = None
        self.face_mesh = None
        self.pose = None
        self.running = False
        self.current_signals = {}
        self.lock = Lock()
        self.signal_interval = 2.0
        self.last_signal_time = 0
        self.prev_head_pos = None
        self.prev_pose_keypoints = None
        self.learning_state_detector = LearningStateDetector(history_window_seconds=60)
        
        # Frame buffer for frontend display
        self._current_frame = None
        self._frame_lock = Lock()
        
        # Gaze tracking for stuck detection
        self._gaze_history = deque(maxlen=300)  # ~10 seconds at 30fps
        self._gaze_window_start = None
        self._is_stuck = False
        self._stuck_start_time = None
        self._misread_detected = False
        self._reading_region = None
        
    def start(self):
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
    
    def get_signals(self):
        with self.lock:
            return self.current_signals.copy()
    
    def get_learning_state_summary(self):
        """Get detailed learning state analysis."""
        return self.learning_state_detector.get_state_summary()
    
    def set_state_change_callback(self, callback):
        """Set callback for non-focused state changes."""
        self._state_change_callback = callback
    
    def get_current_frame(self):
        """Get the latest processed frame as JPEG-encoded bytes."""
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame
            return None
    
    def _update_frame_buffer(self, frame):
        """Encode frame to JPEG and store in buffer."""
        try:
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self._frame_lock:
                self._current_frame = encoded.tobytes()
        except Exception:
            pass
    
    def _run_loop(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            signals = self._process_frame(frame)
            
            # Update learning state detector
            learning_state, confidence = self.learning_state_detector.update(signals)
            signals['learning_state'] = learning_state
            signals['learning_confidence'] = round(confidence, 2)
            
            # Stuck/misread detection (only when focused)
            stuck_signals = self._detect_stuck_and_misread(frame, signals)
            signals.update(stuck_signals)
            
            # Build full signals dict for callback
            full_signals = signals.copy()
            full_signals['learning_state'] = learning_state
            full_signals['learning_confidence'] = confidence
            full_signals.update(stuck_signals)
            
            # Notify bridge of state changes (for voice trigger)
            if hasattr(self, '_state_change_callback') and self._state_change_callback:
                if learning_state in ['distracted', 'overloaded', 'low_engagement']:
                    try:
                        self._state_change_callback(learning_state, confidence, full_signals)
                    except Exception:
                        pass  # Don't let callback errors break observer
                # Also notify if misread
                elif learning_state == 'focused' and stuck_signals.get('misread_detected'):
                    try:
                        full_signals['misread_confidence'] = stuck_signals.get('misread_confidence', 0.6)
                        self._state_change_callback('misread_detected', stuck_signals.get('misread_confidence', 0.6), full_signals)
                    except Exception:
                        pass
            
            if time.time() - self.last_signal_time >= self.signal_interval:
                with self.lock:
                    self.current_signals = signals
                self.last_signal_time = time.time()
            
            frame_count += 1
        
        self.cap.release()
    
    def _get_gaze_direction(self, landmarks, h, w):
        """
        Detects if the user is looking at the screen.
        Uses scale-invariant relative displacement of the iris within the eye eyelids.
        Also uses nose symmetry to detect if the head is turned away.
        """
        try:
            # 1. Right Eye Iris Displacement
            right_eye_pts = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_coords = np.array([[landmarks[p].x, landmarks[p].y] for p in right_eye_pts])
            right_eye_center = np.mean(right_eye_coords, axis=0)
            right_iris_center = np.array([landmarks[473].x, landmarks[473].y])
            right_width = np.linalg.norm(np.array([landmarks[33].x, landmarks[33].y]) - np.array([landmarks[133].x, landmarks[133].y]))
            right_norm_disp = (right_iris_center - right_eye_center) / max(0.001, right_width)

            # 2. Left Eye Iris Displacement
            left_eye_pts = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            left_eye_coords = np.array([[landmarks[p].x, landmarks[p].y] for p in left_eye_pts])
            left_eye_center = np.mean(left_eye_coords, axis=0)
            left_iris_center = np.array([landmarks[468].x, landmarks[468].y])
            left_width = np.linalg.norm(np.array([landmarks[263].x, landmarks[263].y]) - np.array([landmarks[362].x, landmarks[362].y]))
            left_norm_disp = (left_iris_center - left_eye_center) / max(0.001, left_width)

            # 3. Gaze offset metric (magnitude of pupil displacement relative to eyelids center)
            gaze_offset = (np.linalg.norm(right_norm_disp) + np.linalg.norm(left_norm_disp)) / 2.0

            # 4. Head Turn Ratio (relative displacement of nose tip between cheeks)
            p_nose = np.array([landmarks[1].x, landmarks[1].y])
            p_right_cheek = np.array([landmarks[234].x, landmarks[234].y])
            p_left_cheek = np.array([landmarks[454].x, landmarks[454].y])
            dist_r = np.linalg.norm(p_nose - p_right_cheek)
            dist_l = np.linalg.norm(p_nose - p_left_cheek)
            face_width = max(0.001, np.linalg.norm(p_left_cheek - p_right_cheek))
            head_turn_ratio = abs(dist_r - dist_l) / face_width

            # Define thresholds:
            # - gaze_offset < 0.20 means eyes are looking generally straight at the camera/screen.
            # - head_turn_ratio < 0.25 means the head is facing forward.
            is_looking_at_screen = bool(gaze_offset < 0.20 and head_turn_ratio < 0.25)
            return is_looking_at_screen
        except Exception:
            return True  # Fallback to True to avoid false distraction alerts on mesh issues

    def _get_gaze_position(self, landmarks):
        """
        Get normalized gaze position (x, y) mapped to screen space.
        Maps the iris offset within the eyes to a screen coordinate (0..1, 0..1).
        This makes stuck and misread detection work based on actual gaze movements!
        """
        try:
            # Right Eye
            right_eye_pts = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_coords = np.array([[landmarks[p].x, landmarks[p].y] for p in right_eye_pts])
            right_eye_center = np.mean(right_eye_coords, axis=0)
            right_iris_center = np.array([landmarks[473].x, landmarks[473].y])
            right_width = np.linalg.norm(np.array([landmarks[33].x, landmarks[33].y]) - np.array([landmarks[133].x, landmarks[133].y]))
            right_norm_disp = (right_iris_center - right_eye_center) / max(0.001, right_width)

            # Left Eye
            left_eye_pts = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            left_eye_coords = np.array([[landmarks[p].x, landmarks[p].y] for p in left_eye_pts])
            left_eye_center = np.mean(left_eye_coords, axis=0)
            left_iris_center = np.array([landmarks[468].x, landmarks[468].y])
            left_width = np.linalg.norm(np.array([landmarks[263].x, landmarks[263].y]) - np.array([landmarks[362].x, landmarks[362].y]))
            left_norm_disp = (left_iris_center - left_eye_center) / max(0.001, left_width)

            # Average normalized displacement
            dx = (right_norm_disp[0] + left_norm_disp[0]) / 2.0
            dy = (right_norm_disp[1] + left_norm_disp[1]) / 2.0

            # Scale and offset to map to (0..1, 0..1)
            # Typically displacement range is -0.15 to 0.15. Let's scale it by 3.3 to map to -0.5 to 0.5.
            gaze_x = 0.5 + dx * 3.3
            gaze_y = 0.5 + dy * 3.3

            # Add low-frequency head movement contribution for absolute gaze mapping
            # This makes reading scanlines even more accurate!
            # Use nose tip position relative to center of cheeks
            p_nose = np.array([landmarks[1].x, landmarks[1].y])
            p_right_cheek = np.array([landmarks[234].x, landmarks[234].y])
            p_left_cheek = np.array([landmarks[454].x, landmarks[454].y])
            face_center = (p_right_cheek + p_left_cheek) / 2.0
            head_offset_x = p_nose[0] - face_center[0]
            head_offset_y = p_nose[1] - face_center[1]

            # Incorporate head offset (e.g. if user turns head right, head_offset_x becomes positive)
            gaze_x += head_offset_x * 1.5
            gaze_y += head_offset_y * 1.5

            gaze_x = float(np.clip(gaze_x, 0.0, 1.0))
            gaze_y = float(np.clip(gaze_y, 0.0, 1.0))

            return (gaze_x, gaze_y)
        except Exception:
            # Fallback to nose/eyes blend if mesh refinement fails
            nose_tip = landmarks[1]
            return (nose_tip.x, nose_tip.y)
    
    def _detect_stuck_and_misread(self, frame, signals):
        """
        Detect if user is stuck or misreading while focused.
        
        Stuck: Gaze stays in small region for extended time while focused.
        Misread: Rapid left-right-left gaze pattern indicating confusion.
        """
        h, w, _ = frame.shape
        result = {
            'is_stuck': False,
            'stuck_duration': 0.0,
            'stuck_confidence': 0.0,
            'misread_detected': False,
            'misread_confidence': 0.0,
            'reading_region': None,
            'gaze_position': None
        }
        
        # Only detect stuck/misread when focused and present
        if signals.get('learning_state') != 'focused' or not signals.get('presence', False):
            self._gaze_history.clear()
            self._gaze_window_start = None
            self._is_stuck = False
            self._stuck_start_time = None
            self._misread_detected = False
            return result
        
        # Get current gaze position
        gaze_pos = signals.get('gaze_position')
        if gaze_pos is None:
            return result
        
        result['gaze_position'] = gaze_pos
        
        # Add to history
        current_time = time.time()
        self._gaze_history.append({
            'x': gaze_pos[0],
            'y': gaze_pos[1],
            'time': current_time
        })
        
        # --- Stuck Detection ---
        # Check if gaze has stayed within a small region
        if len(self._gaze_history) >= 30:  # At least ~1 second of data
            recent_gazes = list(self._gaze_history)[-180:]  # Last ~6 seconds
            xs = [g['x'] for g in recent_gazes]
            ys = [g['y'] for g in recent_gazes]
            
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            if x_range < STUCK_REGION_PERCENT and y_range < STUCK_REGION_PERCENT:
                if self._stuck_start_time is None:
                    self._stuck_start_time = recent_gazes[0]['time']
                
                stuck_duration = current_time - self._stuck_start_time
                
                if stuck_duration >= STUCK_DURATION_THRESHOLD:
                    self._is_stuck = True
                    result['is_stuck'] = True
                    result['stuck_duration'] = round(stuck_duration, 1)
                    # Confidence increases with duration
                    result['stuck_confidence'] = min(0.9, 0.5 + (stuck_duration - STUCK_DURATION_THRESHOLD) / 20.0)
                    result['reading_region'] = {
                        'x_min': min(xs), 'x_max': max(xs),
                        'y_min': min(ys), 'y_max': max(ys)
                    }
            else:
                self._stuck_start_time = None
                self._is_stuck = False
        
        # --- Misread Detection ---
        # Detect rapid left-right-left gaze shifts (confusion pattern)
        if len(self._gaze_history) >= 20:
            window_gazes = [g for g in self._gaze_history 
                           if current_time - g['time'] <= MISREAD_GAZE_WINDOW]
            
            if len(window_gazes) >= 15:
                # Count direction changes
                direction_changes = 0
                last_direction = None
                
                for i in range(1, len(window_gazes)):
                    dx = window_gazes[i]['x'] - window_gazes[i-1]['x']
                    if abs(dx) > 0.03:  # Significant horizontal movement
                        direction = 'right' if dx > 0 else 'left'
                        if last_direction is not None and direction != last_direction:
                            direction_changes += 1
                        last_direction = direction
                
                if direction_changes >= MISREAD_MAX_GAZE_SHIFTS:
                    self._misread_detected = True
                    result['misread_detected'] = True
                    result['misread_confidence'] = min(0.85, 0.5 + direction_changes * 0.1)
                else:
                    self._misread_detected = False
        
        return result
    
    def _get_movement_level(self, head_pos, pose_kps):
        if head_pos is None or self.prev_head_pos is None or self.prev_pose_keypoints is None:
            return "low"
        head_delta = np.linalg.norm(np.array(head_pos) - np.array(self.prev_head_pos))
        pose_deltas = [np.linalg.norm(np.array(kps) - np.array(prev_kps)) for kps, prev_kps in zip(pose_kps, self.prev_pose_keypoints)]
        pose_delta = np.mean(pose_deltas) if pose_deltas else 0.0
        movement = (head_delta + pose_delta) / 2
        if movement < MOVEMENT_THRESHOLD_LOW:
            return "low"
        elif movement < MOVEMENT_THRESHOLD_HIGH:
            return "medium"
        return "high"
    
    def _get_attention_state(self, presence, gaze_on_screen, movement):
        if not presence:
            return "low"
        if gaze_on_screen and movement == "low":
            return "focused"
        return "distracted"
    
    def _process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face
        face_results = self.face_mesh.process(rgb_frame)
        presence = face_results.multi_face_landmarks is not None
        
        gaze_on_screen = False
        head_pos = None
        pose_kps = []
        gaze_position = None
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[1]
            head_pos = (nose_tip.x * w, nose_tip.y * h)
            gaze_on_screen = self._get_gaze_direction(face_landmarks.landmark, h, w)
            gaze_position = self._get_gaze_position(face_landmarks.landmark)
            
            # Draw face mesh overlay on frame
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            
            # Draw gaze indicator
            if gaze_position:
                gx = int(gaze_position[0] * w)
                gy = int(gaze_position[1] * h)
                cv2.circle(frame, (gx, gy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (gx, gy), 12, (0, 0, 255), 2)
        
        # Pose
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            kps = [(lm.x * w, lm.y * h) for lm in [pose_landmarks[11], pose_landmarks[12], pose_landmarks[23], pose_landmarks[24]]]
            pose_kps = kps
            
            # Draw pose overlay
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )
        
        movement_level = self._get_movement_level(head_pos, pose_kps)
        
        self.prev_head_pos = head_pos
        self.prev_pose_keypoints = pose_kps
        
        attention_state = self._get_attention_state(presence, gaze_on_screen, movement_level)
        
        # Add status text overlay
        status_text = f"State: {attention_state.upper()}"
        if gaze_on_screen:
            status_text += " | Gaze: ON"
        else:
            status_text += " | Gaze: OFF"
        status_text += f" | Move: {movement_level.upper()}"
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Update frame buffer for frontend
        self._update_frame_buffer(frame)
        
        return {
            "presence": presence,
            "gaze_on_screen": gaze_on_screen,
            "movement_level": movement_level,
            "attention_state": attention_state,
            "gaze_position": gaze_position,
            "timestamp": datetime.now().isoformat()
        }
