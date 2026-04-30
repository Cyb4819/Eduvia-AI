import threading
import time
import logging
from typing import Optional, Dict, Callable
from queue import Queue

from utils.voice_trigger_controller import VoiceTriggerController, get_trigger_controller
from utils.tts_engine import get_tts

logger = logging.getLogger(__name__)


class CameraVoiceBridge:
    """
    Bridges camera observations to voice responses.
    
    Flow:
    1. Observer detects state change
    2. Trigger controller checks if intervention is needed
    3. If yes, bridge generates tutor phrase via Gemma 4 (in background thread)
    4. TTS speaks the response
    """
    
    def __init__(self, trigger_controller: Optional[VoiceTriggerController] = None):
        self.trigger_controller = trigger_controller or get_trigger_controller()
        self.tts = get_tts()
        
        # State tracking
        self._last_state: str = "unknown"
        self._is_speaking: bool = False
        self._is_generating: bool = False
        self._lock = threading.Lock()
        
        # Message queue for frontend polling
        self.message_queue: Queue = Queue()
        
        # Setup trigger controller callbacks
        self.trigger_controller.set_callbacks(
            on_intervention=self._handle_intervention,
            on_voice_input=self._handle_voice_input
        )
        
        # Callbacks for UI updates
        self._on_speak_start: Optional[Callable] = None
        self._on_speak_end: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_thinking: Optional[Callable] = None
        
        logger.info("🔗 Camera-Voice Bridge initialized")
    
    def update_state(self, state: str, confidence: float = 0.0, signals: dict = None):
        """Update learning state from observer."""
        with self._lock:
            old_state = self._last_state
            self._last_state = state
        
        # Only log state changes for debugging
        if old_state != state:
            logger.info(f"📊 State changed: {old_state} → {state} (confidence: {confidence:.2f})")
        
        # Pass to trigger controller
        self.trigger_controller.update_learning_state(state, confidence, signals)
    
    def _handle_intervention(self, event: Dict):
        """Handle auto-intervention event from trigger controller (non-blocking)."""
        state = event["state"]
        logger.info(f"🎯 Intervention triggered for state: {state}")
        
        # Run generation + speech in background so observer thread stays free
        threading.Thread(
            target=self._run_intervention,
            args=(state,),
            daemon=True
        ).start()
    
    def _run_intervention(self, state: str):
        """Background thread: generate response and speak."""
        try:
            self._set_generating(True)
            
            # Generate contextual response (instant fallback)
            response = self._generate_intervention_response(state)
            
            # Push to queue for frontend polling
            self._push_message({
                "type": "intervention",
                "state": state,
                "response": response,
                "triggered_by": "camera_observer",
                "timestamp": time.time()
            })
            
            # Speak the response
            self._speak(response)
            
        except Exception as e:
            logger.error(f"Intervention run failed: {e}")
            # Ensure something is still delivered even on unexpected error
            fallback = self._get_fallback_response(state)
            self._push_message({
                "type": "intervention",
                "state": state,
                "response": fallback,
                "triggered_by": "camera_observer",
                "timestamp": time.time()
            })
            self._speak(fallback)
        finally:
            self._set_generating(False)
    
    def _handle_voice_input(self, event: Dict):
        """Handle voice input from user (non-blocking)."""
        transcript = event["transcript"]
        state = event["state"]
        logger.info(f"🎤 Voice input received: '{transcript}' (state: {state})")
        
        # Run generation + speech in background
        threading.Thread(
            target=self._run_voice_response,
            args=(transcript, state),
            daemon=True
        ).start()
    
    def _run_voice_response(self, transcript: str, state: str):
        """Background thread: generate tutor answer and speak."""
        try:
            self._set_generating(True)
            
            # Get RAG-based answer
            answer = self._get_tutor_answer(transcript, state)
            
            # Push to queue for frontend polling
            self._push_message({
                "type": "voice_response",
                "question": transcript,
                "response": answer,
                "state": state,
                "triggered_by": "voice_input",
                "timestamp": time.time()
            })
            
            # Speak the answer
            self._speak(answer)
            
        finally:
            self._set_generating(False)
    
    def _set_generating(self, value: bool):
        """Thread-safe set is_generating flag and notify UI."""
        with self._lock:
            self._is_generating = value
        if value and self._on_thinking:
            try:
                self._on_thinking(True)
            except Exception:
                pass
        elif not value and self._on_thinking:
            try:
                self._on_thinking(False)
            except Exception:
                pass
    
    def _push_message(self, msg: Dict):
        """Push a message to the frontend polling queue."""
        self.message_queue.put(msg)
        # Also notify via callback if available
        if self._on_response:
            try:
                self._on_response(msg)
            except Exception as e:
                logger.warning(f"Response callback error: {e}")
    
    def pop_messages(self) -> list:
        """Pop all pending messages for frontend polling."""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except Exception:
                break
        return messages
    
    def _generate_intervention_response(self, state: str) -> str:
        """
        Generate a contextual tutor response for the given learning state.
        Uses fallback responses directly for fast, reliable auto-intervention.
        """
        return self._get_fallback_response(state)
    
    def _generate_stuck_explanation(self) -> str:
        """
        When user is stuck while focused, try to explain the content.
        Uses RAG if available, otherwise uses model's general knowledge.
        """
        try:
            from utils.content_processor import get_content_processor
            import os
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
            processor = get_content_processor(model_path)
            
            # Get RAG pipeline and check if document is indexed
            rag = processor._get_rag_pipeline()
            
            if rag.indexed_docs and len(rag.indexed_docs) > 0:
                # RAG available - query the document
                stuck_query = (
                    "The learner has been stuck on a specific part of the material for 6+ seconds. "
                    "Please find the most important or complex concept from the current section "
                    "and explain it clearly and simply. Identify what might be confusing about this concept."
                )
                
                result = processor.tutor_interaction(stuck_query, "stuck_while_focused", "default")
                answer = result.get("answer", "")
                
                if not answer or len(answer) < 10:
                    return "I notice you've been focused on this part for a while. Let me clarify — this concept explains that..."
                
                return f"I've been watching you focus on this part. Let me help clarify: {answer}"
            else:
                # RAG not available - use general knowledge
                return self._get_general_knowledge_response(
                    "I've been stuck on this concept - can you explain what this might be about?",
                    "stuck_while_focused"
                )
            
        except Exception as e:
            logger.error(f"Error generating stuck explanation: {e}")
            return "I notice you've been on this part for a while. Would you like me to explain it differently?"
    
    def _get_tutor_answer(self, question: str, learning_state: str, is_intervention: bool = False) -> str:
        """Get tutor answer via RAG + Gemma 4. Falls back to model's general knowledge if RAG fails."""
        try:
            from utils.content_processor import get_content_processor
            import os
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
            processor = get_content_processor(model_path)
            
            # Ensure model is loaded
            if not processor._is_loaded:
                logger.info("🔄 Loading model before generating response...")
                processor.load_model()
            
            # Check if pipeline is available
            if processor.pipeline is None:
                logger.error("❌ Pipeline is None - model may not have loaded properly")
                return self._get_fallback_response(learning_state)
            
            # Check if document is indexed for RAG
            rag = processor._get_rag_pipeline()
            
            # Check if RAG has documents indexed
            if rag.indexed_docs and len(rag.indexed_docs) > 0:
                # RAG is available - use it
                if is_intervention:
                    enhanced_question = (
                        f"[AUTO-INTERVENTION - State: {learning_state}]\n\n"
                        f"{question}\n\n"
                        f"Keep your response concise (2-3 sentences max) and natural. "
                        f"Speak directly to the learner as a supportive tutor."
                    )
                    result = processor.tutor_interaction(enhanced_question, learning_state, "default")
                else:
                    result = processor.tutor_interaction(question, learning_state, "default")
                
                answer = result.get("answer", "")
                
                # Check for error or empty response
                if not answer or len(answer.strip()) < 10:
                    logger.warning(f"⚠️ Low quality answer, using fallback")
                    return self._get_fallback_response(learning_state)
                
                return answer
            else:
                # RAG not available - use model's general knowledge
                logger.info("📚 RAG not indexed, using model's general knowledge")
                return self._get_general_knowledge_response(question, learning_state)
            
        except Exception as e:
            logger.error(f"Tutor answer failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response(learning_state)
    
    def _get_general_knowledge_response(self, question: str, learning_state: str) -> str:
        """
        Use the model's general knowledge when RAG is not available.
        The model should still respond even if the question is not from the PDF.
        """
        try:
            from utils.content_processor import get_content_processor
            import os
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
            processor = get_content_processor(model_path)
            
            # Ensure model is loaded before using pipeline
            if not processor._is_loaded:
                logger.info("🔄 Loading model for general knowledge response...")
                processor.load_model()
            
            # Check if pipeline is available
            if processor.pipeline is None:
                logger.error("❌ Pipeline not available")
                return self._get_fallback_response(learning_state)
            
            # Build prompt for general knowledge response
            state_instructions = {
                "focused": "Provide detailed, comprehensive explanation.",
                "distracted": "Keep explanation brief and engaging. Use simple terms.",
                "overloaded": "Break into small steps. One concept at a time.",
                "low_engagement": "Use an engaging, question-based approach.",
                "stuck_while_focused": "Be patient and explain clearly.",
                "misread_detected": "Clarify the concept in a different way."
            }
            instruction = state_instructions.get(learning_state, state_instructions["focused"])
            
            prompt = f"""<start_of_turn>user
You are an expert tutor helping a student.

LEARNER STATE: {learning_state.upper()}
{instruction}

The student asked: "{question}"

Note: This question may not be directly from our loaded document, but you should still help using your general knowledge.

Respond in a helpful, tutor-like manner:
1. If you know the answer: Provide a clear explanation
2. If uncertain: "I don't have specific information on this, but based on what I know..."
3. Be encouraging and offer to help further

Keep response concise (2-3 sentences).
<end_of_turn>
<start_of_turn>model
"""
            
            output = processor.pipeline(prompt, max_new_tokens=512)[0]['generated_text']
            
            if '<start_of_turn>model' in output:
                answer = output.split('<start_of_turn>model')[-1].strip()
            else:
                answer = output.strip()
            
            if not answer or len(answer.strip()) < 10:
                return self._get_fallback_response(learning_state)
            
            return answer
            
        except Exception as e:
            logger.error(f"General knowledge response failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response(learning_state)
    
    def _get_fallback_response(self, state: str) -> str:
        """Get a contextual fallback response when RAG fails."""
        fallback_responses = {
            "distracted": "Hey! Let's try something different. Are you following along?",
            "overloaded": "Let's take it one step at a time. What's the part that's confusing you?",
            "low_engagement": "Hey there! Want to try a different approach to this?",
            "focused": "Great focus! Let me explain this a bit more clearly.",
            "stuck_while_focused": "I notice you've been on this part for a while. Let me clarify — this concept is about understanding the key ideas. Would you like me to explain it differently?",
            "misread_detected": "Let me re-explain that part - it can be tricky!"
        }
        response = fallback_responses.get(state, "I'm here to help! Let me know what you need.")
        
        # Add more context for stuck state
        if state == "stuck_while_focused":
            response = "I notice you've been focused on this part for a while. Let me help clarify — this concept explains the main idea. Would you like me to break it down step by step?"
        
        return response
    
    def _speak(self, text: str):
        """Speak text via TTS."""
        if not text:
            return
        
        with self._lock:
            if self._is_speaking:
                logger.info("🛑 Stopping previous speech")
                self.tts.stop()
            self._is_speaking = True
        
        # Notify start
        if self._on_speak_start:
            try:
                self._on_speak_start(text)
            except Exception:
                pass
        
        try:
            logger.info(f"🔊 Speaking: '{text[:100]}...'")
            self.tts.speak_async(text, word_by_word=False)
            
            # Wait for speech to complete
            while self.tts.is_speaking():
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            with self._lock:
                self._is_speaking = False
            
            # Notify end
            if self._on_speak_end:
                try:
                    self._on_speak_end()
                except Exception:
                    pass
    
    def stop_speaking(self):
        """Stop current speech."""
        self.tts.stop()
        with self._lock:
            self._is_speaking = False
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._lock:
            return self._is_speaking
    
    def is_generating(self) -> bool:
        """Check if currently generating a response."""
        with self._lock:
            return self._is_generating
    
    def set_callbacks(self, on_speak_start: Optional[Callable] = None,
                      on_speak_end: Optional[Callable] = None,
                      on_response: Optional[Callable] = None,
                      on_thinking: Optional[Callable] = None):
        """Set callbacks for UI updates."""
        self._on_speak_start = on_speak_start
        self._on_speak_end = on_speak_end
        self._on_response = on_response
        self._on_thinking = on_thinking
    
    def get_status(self) -> Dict:
        """Get bridge status."""
        return {
            "last_state": self._last_state,
            "is_speaking": self.is_speaking(),
            "is_generating": self.is_generating(),
            "trigger_status": self.trigger_controller.get_status()
        }


# Singleton instance
_bridge_instance = None


def get_camera_voice_bridge() -> CameraVoiceBridge:
    """Get or create singleton bridge."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = CameraVoiceBridge()
    return _bridge_instance

