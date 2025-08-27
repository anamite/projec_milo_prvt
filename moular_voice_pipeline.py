#!/usr/bin/env python3

import argparse
import queue
import sys
import json
import time
import numpy as np
from threading import Thread, Event
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openwakeword.model import Model as WakeWordModel
import pyttsx3
import threading


class AudioManager:
    """Handles all audio input/output operations"""
    
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.audio_queue = queue.Queue()
        self.wake_queue = queue.Queue()
    
    def wake_word_callback(self, indata, frames, time, status):
        """Callback for wake word detection"""
        if status:
            print(status, file=sys.stderr)
        
        # Convert to int16 for openWakeWord
        audio_int16 = np.frombuffer(indata, dtype=np.float32) * 32767
        audio_int16 = audio_int16.astype(np.int16)
        self.wake_queue.put(audio_int16)
    
    def command_callback(self, indata, frames, time, status):
        """Callback for command recording (VOSK)"""
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))


class TTSManager:
    """Handles Text-to-Speech functionality"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.setup_voice()
        self.speaking = threading.Event()
    
    def setup_voice(self):
        """Configure TTS voice settings"""
        voices = self.engine.getProperty('voices')
        
        # Try to set a female voice if available
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        # Set speech rate and volume
        self.engine.setProperty('rate', 180)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
    def speak(self, text, blocking=False):
        """Convert text to speech"""
        if not text.strip():
            return
        
        print(f"🔊 Speaking: {text}")
        
        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # Non-blocking speech in a separate thread
            def speak_thread():
                self.speaking.set()
                self.engine.say(text)
                self.engine.runAndWait()
                self.speaking.clear()
            
            Thread(target=speak_thread, daemon=True).start()
    
    def is_speaking(self):
        """Check if TTS is currently speaking"""
        return self.speaking.is_set()
    
    def stop(self):
        """Stop current speech"""
        self.engine.stop()


class WakeWordDetector:
    """Handles wake word detection"""
    
    def __init__(self, audio_manager, wake_words=None):
        if wake_words is None:
            wake_words = ["alexa", "hey_jarvis"]
        
        self.audio_manager = audio_manager
        self.wake_model = WakeWordModel(
            wakeword_models=wake_words, 
            inference_framework="onnx"
        )
        self.wake_detected = Event()
        self.running = True
        self.confidence_threshold = 0.5
    
    def detect_wake_word(self):
        """Thread for continuous wake word detection"""
        print("Wake word detection started. Say 'alexa' or 'hey jarvis'...")
        
        while self.running:
            try:
                if not self.audio_manager.wake_queue.empty():
                    audio_data = self.audio_manager.wake_queue.get_nowait()
                    
                    # Check for wake word
                    prediction = self.wake_model.predict(audio_data)
                    
                    # Check if any wake word was detected
                    for wake_word, confidence in prediction.items():
                        if confidence > self.confidence_threshold:
                            print(f"\n🎯 Wake word '{wake_word}' detected! (confidence: {confidence:.2f})")
                            self.wake_detected.set()
                            break
            
            except Exception as e:
                print(f"Wake word detection error: {e}")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    
    def start_detection(self):
        """Start wake word detection thread"""
        wake_thread = Thread(target=self.detect_wake_word)
        wake_thread.daemon = True
        wake_thread.start()
        return wake_thread


class SpeechRecognizer:
    """Handles speech-to-text conversion"""
    
    def __init__(self, model_lang="en-us", samplerate=16000):
        self.vosk_model = Model(lang=model_lang)
        self.samplerate = samplerate
        self.listening_for_command = Event()
    
    def listen_for_command(self, audio_manager):
        """Listen for command using VOSK"""
        print("🎤 Listening for command... (speak now)")
        
        # Create recognizer
        rec = KaldiRecognizer(self.vosk_model, self.samplerate)
        
        # Start recording
        with sd.RawInputStream(
            samplerate=self.samplerate, 
            blocksize=8000,
            dtype="int16", 
            channels=1, 
            callback=audio_manager.command_callback
        ):
            command_text = ""
            silence_start = None
            max_silence_duration = 2.0  # Stop after 2 seconds of silence
            
            while self.listening_for_command.is_set():
                try:
                    data = audio_manager.audio_queue.get(timeout=0.1)
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if result['text']:
                            command_text = result['text']
                            print(f"📝 Final: {command_text}")
                            break
                    else:
                        partial = json.loads(rec.PartialResult())
                        if partial['partial']:
                            print(f"📝 Partial: {partial['partial']}", end='\r')
                            silence_start = None  # Reset silence timer on speech
                        else:
                            # No speech detected
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > max_silence_duration:
                                print(f"\n⏰ Silence timeout, processing: '{command_text}'")
                                break
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Command recognition error: {e}")
                    break
            
            return command_text


class CommandProcessor:
    """Processes and matches user commands using embeddings"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define commands
        self.commands = [
            "turn on all lights", "turn on bedroom lights", "turn off bedroom lights",
            "turn off living room lamp", "turn on living room lamp", "turn off all lights",
            "get weather", "get time", "set timer", "play music or song", 
            "call for help", "search for bluetooth speaker"
        ]
        
        # Define songs
        self.songs = [
            "Manathe Chandanakkeeru", "Entammede Jimikki Kammal", "Aaro Viral Meeti",
            "Kizhakku Pookkum", "Mazhaye Thoomazhaye", "Oru Madhurakinavin",
            "Ponveene", "Thumbi Vaa", "Melle Melle", "Nee Himamazhayayi"
        ]
        
        # Pre-compute embeddings for performance
        print("Computing command embeddings...")
        self.command_embeddings = self.sentence_model.encode(self.commands)
        self.song_embeddings = self.sentence_model.encode(self.songs)
    
    def process_command(self, text):
        """Process command using embedding similarity"""
        if not text.strip():
            return None, 0, None, 0
        
        user_embedding = self.sentence_model.encode(text)
        
        # Check commands
        similarities = cosine_similarity([user_embedding], self.command_embeddings)
        best_match_index = similarities.argmax()
        best_command = self.commands[best_match_index]
        command_confidence = similarities[0, best_match_index] * 100
        
        song_match = None
        song_confidence = 0
        
        # If it's a music command, also check songs
        if command_confidence > 20 and best_command == "play music or song":
            song_similarities = cosine_similarity([user_embedding], self.song_embeddings)
            best_song_index = song_similarities.argmax()
            song_match = self.songs[best_song_index]
            song_confidence = song_similarities[0, best_song_index] * 100
        
        if command_confidence < 20:
            best_command = "ask_llm"
        
        return best_command, command_confidence, song_match, song_confidence


class CommandExecutor:
    """Executes matched commands and provides responses"""
    
    def __init__(self, tts_manager):
        self.tts = tts_manager
    
    def execute_command(self, command, song=None, confidence=0):
        """Execute the matched command with TTS response"""
        print(f"\n🚀 Executing: {command}")
        
        response_text = ""
        
        if command == "turn on all lights":
            print("💡 All lights are now ON")
            response_text = "All lights are now on"
            
        elif command == "turn off all lights":
            print("🔌 All lights are now OFF")
            response_text = "All lights are now off"
            
        elif "bedroom lights" in command:
            action = "ON" if "on" in command else "OFF"
            print(f"🛏️ Bedroom lights are now {action}")
            response_text = f"Bedroom lights are now {action.lower()}"
            
        elif "living room lamp" in command:
            action = "ON" if "on" in command else "OFF"
            print(f"🏠 Living room lamp is now {action}")
            response_text = f"Living room lamp is now {action.lower()}"
            
        elif command == "get weather":
            print("🌤️ Today's weather: Sunny, 22°C")
            response_text = "Today's weather is sunny, 22 degrees celsius"
            
        elif command == "get time":
            current_time = time.strftime("%I:%M %p")
            print(f"🕐 Current time: {current_time}")
            response_text = f"The current time is {current_time}"
            
        elif command == "set timer":
            print("⏰ Timer set for 5 minutes")
            response_text = "Timer set for 5 minutes"
            
        elif command == "play music or song":
            if song:
                print(f"🎵 Playing: {song}")
                response_text = f"Playing {song}"
            else:
                print("🎵 Playing music...")
                response_text = "Playing music"
                
        elif command == "call for help":
            print("🚨 Emergency call initiated!")
            response_text = "Emergency call initiated"
            
        elif command == "search for bluetooth speaker":
            print("🔍 Searching for Bluetooth speakers...")
            response_text = "Searching for Bluetooth speakers"
            
        elif command == "ask_llm":
            print("🤖 I'll need to ask an AI assistant for that...")
            response_text = "I'm not sure about that. Let me ask my AI assistant"
            
        else:
            print(f"❓ Unknown command: {command}")
            response_text = "I didn't understand that command"
        
        # Speak the response
        if response_text:
            self.tts.speak(response_text)
        
        return response_text


class VoiceAssistant:
    """Main voice assistant orchestrator"""
    
    def __init__(self, model_lang="en-us", samplerate=16000):
        print("Initializing Voice Assistant...")
        
        # Initialize all modules
        self.audio_manager = AudioManager(samplerate)
        self.tts_manager = TTSManager()
        self.wake_detector = WakeWordDetector(self.audio_manager)
        self.speech_recognizer = SpeechRecognizer(model_lang, samplerate)
        self.command_processor = CommandProcessor()
        self.command_executor = CommandExecutor(self.tts_manager)
        
        self.samplerate = samplerate
        self.running = True
        
        print("Voice Assistant initialized successfully!")
    
    def run(self):
        """Main assistant loop"""
        try:
            # Start wake word detection thread
            self.wake_detector.start_detection()
            
            # Start wake word audio stream
            with sd.RawInputStream(
                samplerate=self.samplerate, 
                blocksize=1024,
                dtype="float32", 
                channels=1, 
                callback=self.audio_manager.wake_word_callback
            ):
                print("\n" + "=" * 60)
                print("🎙️  VOICE ASSISTANT READY")
                print("🔊  Say 'alexa' or 'hey jarvis' to start")
                print("⌨️  Press Ctrl+C to stop")
                print("=" * 60)
                
                # Welcome message
                self.tts_manager.speak("Voice assistant ready. Say alexa or hey jarvis to start.")
                
                while self.running:
                    try:
                        # Wait for wake word
                        if self.wake_detector.wake_detected.wait(timeout=1.0):
                            self.wake_detector.wake_detected.clear()
                            
                            # Brief acknowledgment sound
                            self.tts_manager.speak("Yes?", blocking=True)
                            
                            # Start listening for command
                            self.speech_recognizer.listening_for_command.set()
                            command_text = self.speech_recognizer.listen_for_command(self.audio_manager)
                            self.speech_recognizer.listening_for_command.clear()
                            
                            if command_text.strip():
                                print(f"\n📥 Processing: '{command_text}'")
                                
                                # Process with embedding model
                                command, confidence, song, song_confidence = self.command_processor.process_command(command_text)
                                
                                print(f"🎯 Best match: '{command}' ({confidence:.1f}%)")
                                if song:
                                    print(f"🎵 Song match: '{song}' ({song_confidence:.1f}%)")
                                
                                # Execute command
                                self.command_executor.execute_command(command, song, confidence)
                            
                            else:
                                print("No command detected")
                                self.tts_manager.speak("I didn't hear anything. Please try again.")
                            
                            print(f"\n{'=' * 30}")
                            print("🔊 Ready for next wake word...")
                            print("=" * 30)
                    
                    except KeyboardInterrupt:
                        break
        
        except Exception as e:
            print(f"Main loop error: {e}")
        
        finally:
            self.running = False
            self.wake_detector.running = False
            self.tts_manager.stop()
            print("\n👋 Voice assistant stopped")


def main():
    parser = argparse.ArgumentParser(description="Modular Voice Assistant with TTS")
    parser.add_argument("-m", "--model", type=str, default="en-us",
                        help="VOSK language model (default: en-us)")
    parser.add_argument("-r", "--samplerate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("-l", "--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    # Create and run assistant
    assistant = VoiceAssistant(model_lang=args.model, samplerate=args.samplerate)
    
    if args.no_tts:
        # Disable TTS if requested
        assistant.tts_manager.speak = lambda text, blocking=False: None
    
    assistant.run()


if __name__ == "__main__":
    main()
