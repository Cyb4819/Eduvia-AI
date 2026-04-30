"""
Adaptive Learning Engine
Core Intelligence Loop: observe → infer → adapt → teach → repeat

Automatically adapts teaching without waiting for user input.
"""

import threading
import time
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LearningState(Enum):
    """Learning states from Phase 1 observer."""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    OVERLOADED = "overloaded"
    LOW_ENGAGEMENT = "low_engagement"
    UNKNOWN = "unknown"


@dataclass
class AdaptationConfig:
    """Configuration for adaptation behavior."""
    # Timing
    check_interval_seconds: int = 5
    adaptation_cooldown_seconds: int = 30
    
    # Thresholds
    distraction_duration_threshold: int = 10  # seconds
    overload_movement_threshold: float = 0.8
    low_engagement_duration_threshold: int = 15
    
    # Behavior flags
    auto_speak_enabled: bool = True
    auto_teach_enabled: bool = True
    re_engage_enabled: bool = True


@dataclass
class TeachingContent:
    """Content to teach with metadata."""
    topic: str
    content: str
    difficulty: str = "beginner"
    estimated_duration: int = 60  # seconds
    key_points: List[str] = field(default_factory=list)


class AdaptationStrategy:
    """
    Strategy pattern for different learning states.
    Each state has its own adaptation behavior.
    """
    
    # Tutor phrases for each state
    TUTOR_PHRASES = {
        LearningState.FOCUSED: [
            "Let me explain this in more detail...",
            "Building on what we just covered...",
            "This is an important concept, so let me elaborate...",
        ],
        LearningState.DISTRACTED: [
            "Let me simplify this for you...",
            "Should I slow down a bit?",
            "Let me catch your attention with this...",
        ],
        LearningState.OVERLOADED: [
            "Let's break this into smaller steps...",
            "One thing at a time - let's focus on this first...",
            "Take a deep breath - we'll tackle this piece by piece...",
        ],
        LearningState.LOW_ENGAGEMENT: [
            "Hey! Let's try a different approach...",
            "Interesting question - did you know...?",
            "Let me show you something cool...",
        ]
    }
    
    @staticmethod
    def get_strategy(state: LearningState) -> 'dict':
        """Get adaptation strategy for a learning state."""
        strategies = {
            LearningState.FOCUSED: {
                "teaching_style": "detailed",
                "explanation_depth": "comprehensive",
                "pace": "normal",
                "interactivity": "high",
                "examples": "multiple",
                "voice_rate": 1.0,
                "content_complexity": "full"
            },
            LearningState.DISTRACTED: {
                "teaching_style": "simplified",
                "explanation_depth": "brief",
                "pace": "slower",
                "interactivity": "medium",
                "examples": "single",
                "voice_rate": 0.9,
                "content_complexity": "reduced"
            },
            LearningState.OVERLOADED: {
                "teaching_style": "step_by_step",
                "explanation_depth": "minimal",
                "pace": "slow",
                "interactivity": "low",
                "examples": "none",
                "voice_rate": 0.8,
                "content_complexity": "basic"
            },
            LearningState.LOW_ENGAGEMENT: {
                "teaching_style": "engaging",
                "explanation_depth": "varied",
                "pace": "dynamic",
                "interactivity": "high",
                "examples": "stories",
                "voice_rate": 1.1,
                "content_complexity": "varied"
            }
        }
        return strategies.get(state, strategies[LearningState.FOCUSED])
    
    @staticmethod
    def get_tutor_phrase(state: LearningState) -> str:
        """Get a tutor phrase for the state."""
        phrases = AdaptationStrategy.TUTOR_PHRASES.get(state, [])
        if phrases:
            import random
            return random.choice(phrases)
        return "Let me explain..."


class AdaptiveLearningEngine:
    """
    Main adaptive learning engine.
    
    Loop: observe → infer → adapt → teach → repeat
    """
    
    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()
        
        # State
        self._current_state = LearningState.UNKNOWN
        self._previous_state = LearningState.UNKNOWN
        self._state_history: List[Dict] = []
        self._last_adaptation_time = 0
        self._adaptation_count = 0
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_adapt: Optional[Callable] = None
        self._on_teach: Optional[Callable] = None
        
        # Content queue
        self._content_queue: List[TeachingContent] = []
        self._current_teaching: Optional[TeachingContent] = None
        
        # Control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Learning content (from indexed document)
        self._indexed_content: str = ""
        self._learning_chunks: List[Dict] = []
        
        logger.info("🧠 Adaptive Learning Engine initialized")
    
    def start(self):
        """Start the adaptive learning loop."""
        if self._running:
            logger.warning("Engine already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("▶️ Adaptive Learning Engine started")
    
    def stop(self):
        """Stop the adaptive learning loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("⏹️ Adaptive Learning Engine stopped")
    
    def set_learning_content(self, content: str, chunks: List[Dict] = None):
        """Set the content to teach from."""
        self._indexed_content = content
        self._learning_chunks = chunks or []
        logger.info(f"📚 Set learning content: {len(content)} chars, {len(self._learning_chunks)} chunks")
    
    def update_observer_state(self, signals: Dict):
        """
        Update with latest observer signals.
        Called from Phase 1 observer.
        """
        # Infer learning state from signals
        new_state = self._infer_state(signals)
        
        if new_state != self._current_state:
            self._previous_state = self._current_state
            self._current_state = new_state
            
            # Record state change
            self._state_history.append({
                "timestamp": time.time(),
                "previous": self._previous_state.value,
                "current": self._current_state.value,
                "signals": signals
            })
            
            # Trigger adaptation if state changed significantly
            if self._should_adapt():
                self._adapt_and_teach()
    
    def _infer_state(self, signals: Dict) -> LearningState:
        """
        Infer learning state from observer signals.
        """
        presence = signals.get('presence', False)
        gaze_on_screen = signals.get('gaze_on_screen', True)
        movement_level = signals.get('movement_level', 'low')
        attention_state = signals.get('attention_state', 'unknown')
        
        # State inference logic
        if not presence:
            return LearningState.LOW_ENGAGEMENT
        
        if attention_state == 'distracted':
            return LearningState.DISTRACTED
        
        if attention_state == 'overloaded':
            return LearningState.OVERLOADED
        
        if attention_state == 'focused':
            return LearningState.FOCUSED
        
        # Fallback: check gaze and movement
        if not gaze_on_screen and movement_level == 'low':
            return LearningState.DISTRACTED
        
        if movement_level == 'high' and not gaze_on_screen:
            return LearningState.OVERLOADED
        
        if not presence or attention_state == 'low_engagement':
            return LearningState.LOW_ENGAGEMENT
        
        return LearningState.FOCUSED
    
    def _should_adapt(self) -> bool:
        """Check if adaptation should trigger."""
        # Check cooldown
        time_since_last = time.time() - self._last_adaptation_time
        if time_since_last < self.config.adaptation_cooldown_seconds:
            return False
        
        # Only adapt on significant state changes
        significant_states = {
            LearningState.FOCUSED: 0,
            LearningState.DISTRACTED: 1,
            LearningState.OVERLOADED: 2,
            LearningState.LOW_ENGAGEMENT: 3
        }
        
        current_level = significant_states.get(self._current_state, 0)
        previous_level = significant_states.get(self._previous_state, 0)
        
        # Adapt if state changed significantly
        return abs(current_level - previous_level) >= 1
    
    def _adapt_and_teach(self):
        """Core adaptation + teaching method."""
        self._last_adaptation_time = time.time()
        self._adaptation_count += 1
        
        strategy = AdaptationStrategy.get_strategy(self._current_state)
        tutor_phrase = AdaptationStrategy.get_tutor_phrase(self._current_state)
        
        # Generate adapted teaching content
        teaching_content = self._generate_adapted_content(strategy)
        
        # Build adaptation event
        adaptation_event = {
            "timestamp": time.time(),
            "state": self._current_state.value,
            "strategy": strategy,
            "tutor_phrase": tutor_phrase,
            "content": teaching_content.content[:200] + "...",
            "adaptation_number": self._adaptation_count
        }
        
        logger.info(f"🔄 Adaptation #{self._adaptation_count}: {self._current_state.value} → {strategy['teaching_style']}")
        
        # Trigger callbacks
        if self._on_adapt:
            self._on_adapt(adaptation_event)
        
        if self._on_teach and self.config.auto_teach_enabled:
            self._on_teach({
                "phrase": tutor_phrase,
                "content": teaching_content,
                "state": self._current_state.value,
                "strategy": strategy
            })
    
    def _generate_adapted_content(self, strategy: Dict) -> TeachingContent:
        """Generate teaching content based on strategy."""
        style = strategy.get('teaching_style', 'detailed')
        
        # Select content based on teaching style
        if style == "detailed":
            topic = "Deep Dive"
            content = self._get_detailed_content()
        elif style == "simplified":
            topic = "Quick Overview"
            content = self._get_simplified_content()
        elif style == "step_by_step":
            topic = "Step-by-Step Guide"
            content = self._get_step_content()
        elif style == "engaging":
            topic = "Interesting Fact"
            content = self._get_engaging_content()
        else:
            topic = "Summary"
            content = self._get_brief_content()
        
        return TeachingContent(
            topic=topic,
            content=content,
            difficulty=strategy.get('content_complexity', 'beginner'),
            key_points=self._extract_key_points(content)
        )
    
    def _get_detailed_content(self) -> str:
        """Get detailed explanation content."""
        if self._learning_chunks:
            # Use first chunk with full explanation
            chunk = self._learning_chunks[0] if self._learning_chunks else {}
            return chunk.get('concept', '') + "\n\n" + chunk.get('points', '')
        return "Let me explain this concept in depth. " + (self._indexed_content[:1000] if self._indexed_content else "No content loaded.")
    
    def _get_simplified_content(self) -> str:
        """Get simplified content."""
        if self._indexed_content:
            # Return first few sentences
            sentences = self._indexed_content.split('.')
            return '.'.join(sentences[:3]) + '.'
        return "Here's a simple way to think about this..."
    
    def _get_step_content(self) -> str:
        """Get step-by-step content."""
        steps = [
            "Step 1: Let's start with the basics.",
            "Step 2: Focus on this one idea.",
            "Step 3: We'll build from here.",
            "Step 4: Now you try.",
            "Step 5: Great job!"
        ]
        return "\n".join(steps)
    
    def _get_engaging_content(self) -> str:
        """Get engaging, re-interesting content."""
        hooks = [
            "Did you know something interesting about this?",
            "Here's a cool fact that might surprise you!",
            "Let me share something fascinating...",
            "What if I told you this relates to everyday life?"
        ]
        import random
        hook = random.choice(hooks)
        return hook + " " + (self._indexed_content[:500] if self._indexed_content else "...")
    
    def _get_brief_content(self) -> str:
        """Get brief summary content."""
        if self._indexed_content:
            return self._indexed_content[:300] + "..."
        return "Let me give you a quick overview..."
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Simple extraction - split by sentences
        sentences = content.split('.')
        points = [s.strip() for s in sentences[:3] if s.strip()]
        return points
    
    def _run_loop(self):
        """Main loop - monitors and adapts continuously."""
        while self._running:
            try:
                # Get current state
                state = self.get_current_state()
                
                # Check if we need to adapt
                if self._should_adapt():
                    self._adapt_and_teach()
                
                # Wait before next check
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Adaptive loop error: {e}")
                time.sleep(5)
    
    def get_current_state(self) -> LearningState:
        """Get current learning state."""
        return self._current_state
    
    def get_state_summary(self) -> Dict:
        """Get summary of adaptive engine state."""
        return {
            "current_state": self._current_state.value,
            "previous_state": self._previous_state.value,
            "adaptation_count": self._adaptation_count,
            "last_adaptation": self._last_adaptation_time,
            "strategy": AdaptationStrategy.get_strategy(self._current_state),
            "state_history_count": len(self._state_history)
        }
    
    def set_callbacks(self, on_adapt: Callable = None, on_teach: Callable = None):
        """Set callbacks for adaptation events."""
        self._on_adapt = on_adapt
        self._on_teach = on_teach
    
    def force_adaptation(self, state: LearningState = None):
        """Force an adaptation to a specific state."""
        if state:
            self._previous_state = self._current_state
            self._current_state = state
        self._adapt_and_teach()


# Singleton
_engine_instance = None


def get_adaptive_engine() -> AdaptiveLearningEngine:
    """Get or create singleton adaptive engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AdaptiveLearningEngine()
    return _engine_instance