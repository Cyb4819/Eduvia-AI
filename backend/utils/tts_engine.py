"""
Real-Time Text-to-Speech Module
Speaks text incrementally word-by-word for live tutor feel.
"""

import threading
import queue
import time
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class RealTimeTTS:
    """
    Real-time TTS that speaks as text is generated.
    Uses pyttsx3 for local TTS with word-level streaming.
    """
    
    def __init__(self, rate: int = 150, volume: float = 1.0, voice_id: int = 0):
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.engine = None
        self._is_speaking = False
        self._should_stop = False
        self._current_thread = None
        self._word_callback: Optional[Callable] = None
    
    def initialize(self):
        """Initialize the TTS engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_id:
                self.engine.setProperty('voice', voices[self.voice_id].id)
            
            logger.info("✅ TTS engine initialized")
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            raise
    
    def speak(self, text: str, word_by_word: bool = True):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            word_by_word: If True, speaks word-by-word for real-time feel
        """
        if self.engine is None:
            self.initialize()
        
        self._should_stop = False
        
        if word_by_word:
            self._speak_word_by_word(text)
        else:
            self._speak_full(text)
    
    def _speak_word_by_word(self, text: str):
        """Speak text word by word with callbacks for each word."""
        self._is_speaking = True
        words = text.split()
        
        for i, word in enumerate(words):
            if self._should_stop:
                break
            
            # Notify callback (for highlighting in UI)
            if self._word_callback:
                self._word_callback(word, i, len(words))
            
            # Speak single word
            try:
                self.engine.say(word)
                self.engine.runAndWait()
            except Exception as e:
                logger.warning(f"Word speak error: {e}")
            
            # Small pause between words for natural feel
            time.sleep(0.02)
        
        self._is_speaking = False
    
    def _speak_full(self, text: str):
        """Speak entire text at once."""
        self._is_speaking = True
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS speak error: {e}")
        finally:
            self._is_speaking = False
    
    def speak_async(self, text: str, word_by_word: bool = True):
        """Speak text in a background thread."""
        if self._current_thread and self._current_thread.is_alive():
            self.stop()
        
        self._current_thread = threading.Thread(
            target=self.speak, 
            args=(text, word_by_word),
            daemon=True
        )
        self._current_thread.start()
    
    def stop(self):
        """Stop current speech."""
        self._should_stop = True
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        self._is_speaking = False
    
    def set_word_callback(self, callback: Callable):
        """Set callback for word-level events."""
        self._word_callback = callback
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking


class WebSpeechTTS:
    """
    Browser-based TTS using Web Speech API.
    Works in modern browsers without server-side TTS.
    """
    
    def __init__(self):
        self._is_speaking = False
    
    def get_js_init(self) -> str:
        """Get JavaScript code to initialize browser TTS."""
        return """
        // Web Speech API TTS
        let speechSynthesis = window.speechSynthesis;
        let currentUtterance = null;
        let wordIndex = 0;
        
        function speakText(text, wordByWord = true) {
            stopSpeaking();
            
            if (wordByWord) {
                // Word by word mode
                const words = text.split(' ');
                let index = 0;
                
                function speakNextWord() {
                    if (index >= words.length) {
                        window.onTTSComplete && window.onTTSComplete();
                        return;
                    }
                    
                    const word = words[index];
                    const utterance = new SpeechSynthesisUtterance(word);
                    utterance.rate = 1.1;
                    utterance.pitch = 1.0;
                    
                    utterance.onend = () => {
                        index++;
                        window.onWordSpeak && window.onWordSpeak(word, index, words.length);
                        setTimeout(speakNextWord, 50);
                    };
                    
                    speechSynthesis.speak(utterance);
                }
                
                speakNextWord();
            } else {
                // Full text mode
                currentUtterance = new SpeechSynthesisUtterance(text);
                currentUtterance.rate = 1.0;
                speechSynthesis.speak(currentUtterance);
            }
            
            window.onTTSStart && window.onTTSStart();
        }
        
        function stopSpeaking() {
            speechSynthesis.cancel();
        }
        
        function isSpeaking() {
            return speechSynthesis.speaking;
        }
        
        // Get available voices
        function getVoices() {
            return speechSynthesis.getVoices();
        }
        """


# Singleton instances
_tts_instance = None


def get_tts() -> RealTimeTTS:
    """Get or create TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = RealTimeTTS()
    return _tts_instance