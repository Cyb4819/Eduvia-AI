import time
import threading
from typing import Optional, Callable, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    intervention_states = {"distracted", "overloaded", "low_engagement", "misread_detected"}
    # Cooldown between auto-interventions for distraction states (seconds)
    intervention_cooldown: float = 15.0
    # Cooldown between stuck/misread interventions (shorter, more responsive)
    stuck_cooldown: float = 15.0  # Reduced from 20s to 15s for more responsive help
    # Minimum confidence to trigger
    min_confidence: float = 0.6
    # Enable/disable auto-intervention
    auto_intervention_enabled: bool = False


class VoiceTriggerController:
    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()
        self._last_intervention_time: float = 0
        self._last_stuck_time: float = 0
        self._current_state: str = "unknown"
        self._current_confidence: float = 0.0
        self._current_signals: dict = {}
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_intervention: Optional[Callable] = None
        self._on_voice_input: Optional[Callable] = None
        
        logger.info("🎛️ Voice Trigger Controller initialized")
    
    def update_learning_state(self, state: str, confidence: float = 0.0, signals: dict = None):
        """Update current learning state from observer."""
        with self._lock:
            self._current_state = state
            self._current_confidence = confidence
            self._current_signals = signals or {}
        
        # Check if we should auto-intervene
        if self._should_auto_intervene():
            self._trigger_intervention("state_change", state)
    
    def _should_auto_intervene(self) -> bool:
        """Check if auto-intervention should trigger."""
        if not self.config.auto_intervention_enabled:
            return False
        
        # Only intervene on specific states
        if self._current_state not in self.config.intervention_states:
            return False
        
        # Check confidence threshold
        if self._current_confidence < self.config.min_confidence:
            return False
        
        # Use appropriate cooldown based on state type
        if self._current_state in {"misread_detected"}:
            time_since_last = time.time() - self._last_stuck_time
            cooldown = self.config.stuck_cooldown
        else:
            time_since_last = time.time() - self._last_intervention_time
            cooldown = self.config.intervention_cooldown
        
        if time_since_last < cooldown:
            return False
        
        return True
    
    def _trigger_intervention(self, trigger_type: str, context: str):
        """Trigger an intervention."""
        # Update appropriate cooldown timestamp
        if self._current_state in {"misread_detected"}:
            self._last_stuck_time = time.time()
        else:
            self._last_intervention_time = time.time()
        
        logger.info(f"Auto-intervention triggered: {trigger_type} → {context}")
        
        if self._on_intervention:
            try:
                self._on_intervention({
                    "type": trigger_type,
                    "state": self._current_state,
                    "confidence": self._current_confidence,
                    "context": context,
                    "timestamp": time.time(),
                    "signals": self._current_signals
                })
            except Exception as e:
                logger.error(f"Intervention callback error: {e}")
    
    def handle_voice_input(self, transcript: str) -> bool:
        """
        Handle user voice input.
        
        Voice input ALWAYS triggers the model, regardless of state.
        Returns True if processed successfully.
        """
        logger.info(f"🎤 Voice input received: '{transcript}'")
        
        if self._on_voice_input:
            try:
                self._on_voice_input({
                    "transcript": transcript,
                    "state": self._current_state,
                    "timestamp": time.time()
                })
                return True
            except Exception as e:
                logger.error(f"Voice input callback error: {e}")
                return False
        
        return False
    
    def should_respond_to_state(self, state: str) -> bool:
        """Check if the system should respond to a given learning state."""
        return state in self.config.intervention_states
    
    def get_status(self) -> Dict:
        """Get current trigger controller status."""
        time_since_last = time.time() - self._last_intervention_time
        time_since_stuck = time.time() - self._last_stuck_time
        cooldown_remaining = max(0, self.config.intervention_cooldown - time_since_last)
        stuck_cooldown_remaining = max(0, self.config.stuck_cooldown - time_since_stuck)
        
        return {
            "current_state": self._current_state,
            "current_confidence": self._current_confidence,
            "auto_intervention_enabled": self.config.auto_intervention_enabled,
            "intervention_cooldown": self.config.intervention_cooldown,
            "stuck_cooldown": self.config.stuck_cooldown,
            "cooldown_remaining": round(cooldown_remaining, 1),
            "stuck_cooldown_remaining": round(stuck_cooldown_remaining, 1),
            "can_intervene": self._should_auto_intervene(),
            "last_intervention": self._last_intervention_time,
            "last_stuck_intervention": self._last_stuck_time
        }
    
    def set_callbacks(self, on_intervention: Optional[Callable] = None,
                      on_voice_input: Optional[Callable] = None):
        """Set callbacks for trigger events."""
        self._on_intervention = on_intervention
        self._on_voice_input = on_voice_input
    
    def reset_cooldown(self):
        """Reset intervention cooldown."""
        self._last_intervention_time = 0
        self._last_stuck_time = 0


# Singleton instance
_trigger_controller = None


def get_trigger_controller() -> VoiceTriggerController:
    """Get or create singleton trigger controller."""
    global _trigger_controller
    if _trigger_controller is None:
        _trigger_controller = VoiceTriggerController()
    return _trigger_controller

