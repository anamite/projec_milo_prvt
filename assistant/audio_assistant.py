import json
import threading
import time
import queue
from typing import Dict
import logging

from .wake_word import WakeWordDetector
from .audio_capture import AudioCapture
from .stt import SpeechToText
from .tool_matcher import ToolMatcher
from .tool_executor import ToolExecutor
from .tts import TextToSpeech

logger = logging.getLogger(__name__)

class AudioAssistant:
    """Main orchestrator class for the AI home assistant"""

    def __init__(self, config_file: str = "config.json"):
        self.config = self.load_config(config_file)

        # Initialize components (components are resilient to missing deps)
        self.wake_word_detector = WakeWordDetector(self.config.get("wake_words", ["hey assistant"]))
        self.audio_capture = AudioCapture()
        self.speech_to_text = SpeechToText()
        self.tool_matcher = ToolMatcher(self.config.get("tools", []))
        self.tool_executor = ToolExecutor()
        self.text_to_speech = TextToSpeech()

        # State management
        self.is_listening = False
        self.is_processing = False
        self.audio_queue = queue.Queue()

    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Default configuration with tools"""
        return {
            "wake_words": ["hey assistant", "hello assistant"],
            "tools": []
        }

    def start(self):
        """Start the assistant"""
        logger.info("Starting AI Home Assistant...")

        # Start audio capture in separate thread
        audio_thread = threading.Thread(target=self.audio_capture.start_capture,
                                       args=(self.audio_queue,))
        audio_thread.daemon = True
        audio_thread.start()

        # Main processing loop
        try:
            self.main_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down assistant...")
        finally:
            self.cleanup()

    def main_loop(self):
        """Main processing loop"""
        while True:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()

                if not self.is_listening:
                    # Check for wake word
                    if self.wake_word_detector.detect(audio_data):
                        logger.info("Wake word detected!")
                        self.is_listening = True
                        self.text_to_speech.speak("Yes, how can I help you?")
                        continue

                if self.is_listening and not self.is_processing:
                    # Process speech
                    self.process_speech(audio_data)

            time.sleep(0.1)

    def process_speech(self, audio_data):
        """Process captured speech"""
        self.is_processing = True

        try:
            # Convert speech to text
            text = self.speech_to_text.transcribe(audio_data)

            if text:
                logger.info(f"User said: {text}")

                # Check for silence or sentence completion
                if self.is_sentence_complete(text):
                    # Find matching tool
                    tool_match = self.tool_matcher.find_best_match(text)

                    if tool_match:
                        # Execute tool
                        result = self.tool_executor.execute_tool(tool_match, text)

                        # Speak result
                        if result:
                            self.text_to_speech.speak(result)
                    else:
                        self.text_to_speech.speak("I'm sorry, I didn't understand that.")

                    # Reset listening state
                    self.is_listening = False

        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            self.text_to_speech.speak("Sorry, there was an error processing your request.")

        finally:
            self.is_processing = False

    def is_sentence_complete(self, text: str) -> bool:
        """Check if sentence is complete based on punctuation or silence"""
        # Simple implementation - can be enhanced
        return text and (text.strip().endswith(('.', '!', '?')) or len(text.split()) > 10)

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.audio_capture.stop()
        except Exception:
            pass
        try:
            self.speech_to_text.cleanup()
        except Exception:
            pass
