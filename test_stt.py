import logging
import time
import queue
import threading
import sys
from assistant.wake_word import WakeWordDetector
from assistant.stt import SpeechToText
from assistant.audio_capture import AudioCapture

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class STTTestRunner:
    def __init__(self):
        self.wake_word_detector = WakeWordDetector(["hello", "hey", "assistant"])
        self.stt = SpeechToText()
        self.audio_capture = AudioCapture()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.transcribed_text = ""
        self.capture_thread = None
        
    def test_wake_word_and_stt(self):
        """Test wake word detection followed by STT with 2-second timeout"""
        print("\n" + "="*50)
        print("🎤 STT and Wake Word Test")
        print("="*50)
        print("Say 'hello', 'hey', or 'assistant' to trigger wake word detection...")
        print("Press Ctrl+C to exit")
        
        # Start audio capture in background thread
        self.capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Wait for wake word
        wake_detected = False
        start_time = time.time()
        
        print("🔍 Listening for wake word...")
        
        while not wake_detected:
            # Check if there's audio data to process
            if not self.audio_queue.empty():
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    if audio_chunk and len(audio_chunk) > 0:
                        # Use real wake word detection on audio
                        if self.wake_word_detector.detect(audio_chunk):
                            wake_detected = True
                            print("\n✓ Wake word detected!")
                            break
                except queue.Empty:
                    pass
            
            # Timeout after 30 seconds and offer manual trigger
            if time.time() - start_time > 30:
                print("\n⏰ No wake word detected in 30 seconds.")
                print("Press Enter to manually trigger or Ctrl+C to exit...")
                try:
                    input()
                    wake_detected = True
                    print("✓ Wake word manually triggered!")
                except KeyboardInterrupt:
                    print("\n❌ Test cancelled")
                    return
                break
                
            time.sleep(0.1)
        
        if wake_detected:
            # Clear any remaining audio from queue before starting STT
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._start_listening()
    
    def _audio_capture_loop(self):
        """Continuously capture audio and put in queue"""
        try:
            self.audio_capture.start_capture(self.audio_queue)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
    
    def _start_listening(self):
        """Start listening for speech with 2-second timeout"""
        print("🎤 Listening...")
        self.is_listening = True
        self.transcribed_text = ""
        
        start_time = time.time()
        last_audio_time = start_time
        silence_threshold = 2.0  # 2 seconds of silence
        
        audio_buffer = b""
        last_partial = ""
        
        while self.is_listening:
            current_time = time.time()
            
            # Check for audio in queue
            audio_processed = False
            while not self.audio_queue.empty():
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    if audio_chunk and len(audio_chunk) > 0:
                        audio_buffer += audio_chunk
                        last_audio_time = current_time
                        audio_processed = True
                except queue.Empty:
                    break
            
            # Process accumulated audio every 0.5 seconds or when we have enough data
            if len(audio_buffer) > 8000:  # Roughly 0.5 seconds at 16kHz
                try:
                    result, is_final = self.stt.transcribe_streaming(audio_buffer)
                    if result and result.strip():
                        if is_final:
                            self.transcribed_text = result
                            print(f"📝 Final: {result}")
                        else:
                            if result != last_partial:  # Only show if different from last partial
                                print(f"📝 Partial: {result}")
                                last_partial = result
                except AttributeError:
                    # Fallback to regular transcribe if streaming not available
                    result = self.stt.transcribe(audio_buffer)
                    if result and result.strip():
                        self.transcribed_text = result
                        print(f"📝 Transcribed: {result}")
                
                audio_buffer = b""
            
            # Check for silence timeout
            if current_time - last_audio_time > silence_threshold:
                print("🔇 Silence detected (2 seconds), stopping...")
                self.is_listening = False
                break
                
            # Overall timeout (10 seconds max)
            if current_time - start_time > 10:
                print("⏰ Maximum listening time reached (10 seconds), stopping...")
                self.is_listening = False
                break
                
            time.sleep(0.1)
        
        # Process any remaining audio
        if audio_buffer:
            try:
                result, _ = self.stt.transcribe_streaming(audio_buffer)
                if result and result.strip():
                    self.transcribed_text = result
            except AttributeError:
                result = self.stt.transcribe(audio_buffer)
                if result and result.strip():
                    self.transcribed_text = result
        
        # Output final result
        self._show_final_result()
        
        # Stop audio capture
        self.audio_capture.stop_capture()
    
    def _show_final_result(self):
        """Display the final transcription result"""
        print("\n" + "="*50)
        print("🎯 FINAL RESULT:")
        if self.transcribed_text and self.transcribed_text.strip():
            print(f"   Output: '{self.transcribed_text}'")
        else:
            print("   Output: [No speech detected]")
        print("="*50)
    
    def cleanup(self):
        """Clean up resources"""
        self.is_listening = False
        self.audio_capture.stop_capture()
        if self.stt:
            self.stt.cleanup()

def main():
    """Main test function"""
    test_runner = None
    try:
        test_runner = STTTestRunner()
        test_runner.test_wake_word_and_stt()
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
    finally:
        if test_runner:
            test_runner.cleanup()
        print("👋 Test completed")

if __name__ == "__main__":
    main()
