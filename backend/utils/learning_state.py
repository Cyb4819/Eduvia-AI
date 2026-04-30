"""
Learning State Detection Module

Converts raw behavioral signals into meaningful learning states:
- Focused: Present, looking at screen, low movement
- Distracted: Looking away for extended periods
- Overloaded: High movement + low focus (signs of cognitive overload)
- Low Engagement: Absent or inactive for extended periods

Uses rule-based logic with temporal tracking for confidence scoring.
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time


class LearningStateDetector:
    """
    Detects learning states from behavioral signals with temporal awareness.
    Tracks signal history to identify patterns indicating learner state.
    """
    
    def __init__(self, history_window_seconds=60):
        """
        Initialize the learning state detector.
        
        Args:
            history_window_seconds: How long to track signal history (default 60s)
        """
        self.history_window = history_window_seconds
        self.signal_history = deque(maxlen=100)  # Store last 100 signals
        self.state_history = deque(maxlen=50)     # Store last 50 state inferences
        
        self.gaze_away_start_time = None
        self.movement_start_time = None
        self.no_presence_start_time = None
        
        self.current_state = "low_engagement"
        self.current_confidence = 0.0
        
    def update(self, signals: Dict) -> Tuple[str, float]:
        """
        Update learning state based on latest behavioral signals.
        
        Args:
            signals: Dict with keys: presence, gaze_on_screen, movement_level, attention_state, timestamp
            
        Returns:
            Tuple of (learning_state, confidence_score)
            learning_state: one of ['focused', 'distracted', 'overloaded', 'low_engagement']
            confidence_score: 0.0 to 1.0
        """
        if not signals or not signals.get('timestamp'):
            return self.current_state, self.current_confidence
        
        # Add to history with timestamp
        signals_with_time = signals.copy()
        signals_with_time['received_time'] = time.time()
        self.signal_history.append(signals_with_time)
        
        # Apply rule-based logic
        state, confidence = self._infer_state(signals)
        
        self.current_state = state
        self.current_confidence = confidence
        
        # Track state history
        self.state_history.append({
            'state': state,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        return state, confidence
    
    def _infer_state(self, current_signal: Dict) -> Tuple[str, float]:
        """
        Apply rule-based logic to infer learning state and confidence.
        
        Rules:
        1. No presence → Low Engagement (high confidence)
        2. Present + on-screen gaze + low movement → Focused (high confidence)
        3. Present + off-screen gaze for 5+ seconds → Distracted (confidence based on duration)
        4. Present + high movement + off-screen gaze → Overloaded (high confidence)
        5. Present + inactivity for 10+ seconds → Low Engagement (medium confidence)
        """
        presence = current_signal.get('presence', False)
        gaze_on_screen = current_signal.get('gaze_on_screen', False)
        movement_level = current_signal.get('movement_level', 'low')
        
        current_time = time.time()
        
        # Rule 1: No presence → Low Engagement
        if not presence:
            self.no_presence_start_time = self.no_presence_start_time or current_time
            time_absent = current_time - self.no_presence_start_time
            
            # Increase confidence with time absent (up to 0.95)
            confidence = min(0.95, 0.5 + (time_absent / 30.0) * 0.45)
            return 'low_engagement', confidence
        else:
            self.no_presence_start_time = None
        
        # Rule 2: Focused (Present + on-screen + low movement)
        if presence and gaze_on_screen and movement_level == 'low':
            self.gaze_away_start_time = None
            self.movement_start_time = None
            return 'focused', 0.9
        
        # Rule 3: Distracted (Present + off-screen gaze for 5+ seconds)
        if presence and not gaze_on_screen:
            self.gaze_away_start_time = self.gaze_away_start_time or current_time
            time_away = current_time - self.gaze_away_start_time
            
            if time_away >= 5.0:
                # Confidence increases with time looking away (up to 0.85)
                confidence = min(0.85, 0.6 + (min(time_away, 15.0) / 15.0) * 0.25)
                
                # But if there's high movement, it might be overloading instead
                if movement_level == 'high':
                    self.movement_start_time = self.movement_start_time or current_time
                    return 'overloaded', confidence * 0.95
                
                return 'distracted', confidence
        else:
            self.gaze_away_start_time = None
        
        # Rule 4: Overloaded (Present + high movement + off-screen gaze)
        if presence and movement_level == 'high' and not gaze_on_screen:
            self.movement_start_time = self.movement_start_time or current_time
            time_moving = current_time - self.movement_start_time
            
            if time_moving >= 3.0:
                confidence = min(0.88, 0.65 + (min(time_moving, 10.0) / 10.0) * 0.23)
                return 'overloaded', confidence
        
        # Rule 5: Low Engagement (Present but inactive for 10+ seconds)
        if presence and movement_level == 'low' and not gaze_on_screen:
            # Check if this has been going on for a while
            if self.gaze_away_start_time:
                time_away = current_time - self.gaze_away_start_time
                if time_away >= 10.0:
                    confidence = min(0.8, 0.5 + (min(time_away, 20.0) / 20.0) * 0.3)
                    return 'low_engagement', confidence
        
        # Default: return previous state if no strong signal
        return self.current_state, self.current_confidence
    
    def get_state_summary(self) -> Dict:
        """
        Get current learning state with detailed information.
        
        Returns:
            Dict with current state, confidence, and trend analysis
        """
        if len(self.state_history) < 2:
            return {
                'current_state': self.current_state,
                'confidence': self.current_confidence,
                'recent_states': [],
                'trend': None,
                'state_duration': 0
            }
        
        # Analyze recent state history (last 30 seconds)
        recent_states = list(self.state_history)[-10:]
        state_counts = {}
        for entry in recent_states:
            state = entry['state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Determine trend
        if len(recent_states) > 1:
            prev_state = recent_states[-2]['state']
            curr_state = recent_states[-1]['state']
            trend = 'stable' if prev_state == curr_state else 'changing'
        else:
            trend = 'stable'
        
        # Calculate how long in current state
        state_duration = sum(1 for s in recent_states if s['state'] == self.current_state)
        
        return {
            'current_state': self.current_state,
            'confidence': round(self.current_confidence, 2),
            'trend': trend,
            'state_duration_samples': state_duration,
            'recent_state_distribution': state_counts,
            'total_observations': len(self.signal_history)
        }
    
    def reset(self):
        """Reset the detector state."""
        self.gaze_away_start_time = None
        self.movement_start_time = None
        self.no_presence_start_time = None
        self.signal_history.clear()
        self.state_history.clear()
        self.current_state = 'low_engagement'
        self.current_confidence = 0.0
