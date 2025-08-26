import logging
import time
from typing import List

logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Wake word detection using speech recognition
    
    This detector uses STT to recognize wake words from audio data.
    It buffers audio and checks for wake words in the transcribed text.
    """

    def __init__(self, wake_words: List[str]):
        self.wake_words = [w.lower() for w in wake_words]
        self.sensitivity = 0.5
        self.audio_buffer = b""
        self.buffer_max_size = 32000  # ~2 seconds at 16kHz
        self.last_check_time = time.time()
        self.check_interval = 0.5  # Check every 0.5 seconds
        
        # Import STT here to avoid circular imports
        from .stt import SpeechToText
        self.stt = SpeechToText()
        logger.info(f"Wake word detector initialized with words: {self.wake_words}")

    def detect(self, audio_data) -> bool:
        """Detect wake word in audio_data by accumulating audio and using STT"""
        try:
            if not audio_data:
                return False
                
            # Add to buffer
            self.audio_buffer += audio_data
            
            # Limit buffer size (rolling window)
            if len(self.audio_buffer) > self.buffer_max_size:
                # Keep only the most recent audio
                self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
            
            # Only check periodically to avoid excessive processing
            current_time = time.time()
            if current_time - self.last_check_time < self.check_interval:
                return False
                
            self.last_check_time = current_time
            
            # Need sufficient audio to process
            if len(self.audio_buffer) < 8000:  # At least 0.5 seconds
                return False
            
            # Use STT to transcribe recent audio
            transcription = self.stt.transcribe(self.audio_buffer)
            
            if transcription and transcription.strip():
                transcription_lower = transcription.lower()
                logger.debug(f"Wake word check: '{transcription_lower}'")
                
                # Check if any wake word is in the transcription
                for wake_word in self.wake_words:
                    if wake_word in transcription_lower:
                        logger.info(f"Wake word detected: '{wake_word}' in '{transcription_lower}'")
                        self.audio_buffer = b""  # Clear buffer after detection
                        return True
                        
            return False
            
        except Exception as e:
            logger.debug(f"Wake word detection error: {e}")
            return False

    def detect_manual_trigger(self) -> bool:
        """Manual trigger method for testing - triggers every 3 seconds"""
        current_time = time.time()
        if current_time - self.last_check_time > 3.0:
            self.last_check_time = current_time
            logger.info("Manual wake word trigger activated")
            return True
        return False

    def reset_buffer(self):
        """Reset the audio buffer"""
        self.audio_buffer = b""
