"""
Speech-to-Text Module
Handles voice input for optional user speech recognition.
"""

import threading
import queue
import time
from typing import Optional, Callable, Dict
import logging

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Speech recognition using Web Speech API (browser-based).
    For server-side, would use speech_recognition library with microphone.
    """
    
    def __init__(self):
        self._is_listening = False
        self._transcript_callback: Optional[Callable] = None
    
    def get_js_init(self) -> str:
        """Get JavaScript for browser-based speech recognition."""
        return """
        // Web Speech API Speech Recognition
        let recognition = null;
        let isListening = false;
        
        function initSpeechRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            if (!SpeechRecognition) {
                console.log('Speech recognition not supported');
                return false;
            }
            
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onresult = (event) => {
                const result = event.results[event.results.length - 1];
                const transcript = result[0].transcript;
                const isFinal = result.isFinal;
                
                window.onSpeechResult && window.onSpeechResult(transcript, isFinal);
            };
            
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                window.onSpeechError && window.onSpeechError(event.error);
            };
            
            recognition.onend = () => {
                isListening = false;
                window.onSpeechEnd && window.onSpeechEnd();
            };
            
            return true;
        }
        
        function startListening() {
            if (recognition && !isListening) {
                recognition.start();
                isListening = true;
                window.onSpeechStart && window.onSpeechStart();
            }
        }
        
        function stopListening() {
            if (recognition && isListening) {
                recognition.stop();
                isListening = false;
            }
        }
        
        function getListeningStatus() {
            return isListening;
        }
        """
    
    def get_html_elements(self) -> str:
        """Get HTML elements for voice input UI."""
        return """
        <div id="voice-controls" style="margin: 10px 0;">
            <button id="micBtn" onclick="toggleVoiceInput()" style="background: #e74c3c;">
                🎤 Start Voice Input
            </button>
            <span id="voiceStatus" style="margin-left: 10px; color: #7f8c8d;">Click to speak</span>
        </div>
        <div id="transcript" style="background: #f8f9fa; padding: 10px; border-radius: 5px; display: none;">
            <strong>You said:</strong> <span id="spokenText"></span>
        </div>
        """


class VoiceActivityDetector:
    """
    Detects voice activity from audio for re-engagement.
    """
    
    def __init__(self):
        self._is_active = False
        self._silence_threshold = 0.01
        self._silence_duration = 5.0  # seconds
    
    def get_js_init(self) -> str:
        """Get JavaScript for voice activity detection."""
        return """
        // Voice Activity Detection
        let audioContext = null;
        let analyser = null;
        let microphone = null;
        let vadInterval = null;
        let silenceTime = 0;
        
        async function initVAD() {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                microphone = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                microphone.connect(analyser);
                
                startVAD();
                return true;
            } catch (e) {
                console.error('VAD init failed:', e);
                return false;
            }
        }
        
        function startVAD() {
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            vadInterval = setInterval(() => {
                analyser.getByteFrequencyData(dataArray);
                
                // Calculate average volume
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const average = sum / dataArray.length / 255;
                
                if (average < 0.01) {
                    silenceTime += 0.1;
                    if (silenceTime > 5) {
                        window.onLowEngagement && window.onLowEngagement();
                    }
                } else {
                    silenceTime = 0;
                    window.onVoiceActivity && window.onVoiceActivity(average);
                }
            }, 100);
        }
        
        function stopVAD() {
            if (vadInterval) {
                clearInterval(vadInterval);
            }
        }
        """


# Singleton
_recognizer_instance = None


def get_recognizer() -> SpeechRecognizer:
    """Get or create speech recognizer."""
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = SpeechRecognizer()
    return _recognizer_instance