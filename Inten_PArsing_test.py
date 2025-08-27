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
from kittentts import KittenTTS
import pvrhino
from datetime import datetime, timedelta
import threading

class VoiceAssistant:
    def __init__(self, model_lang="en-us", samplerate=16000, picovoice_access_key=None):
        # Initialize models
        print("Loading models...")
        self.vosk_model = Model(lang=model_lang)
        self.wake_model = WakeWordModel(wakeword_models=["alexa", "hey_jarvis"], inference_framework="onnx")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize TTS model
        print("Loading TTS model...")
        self.tts_model = KittenTTS("KittenML/kitten-tts-nano-0.2")
        
        # TTS voices
        self.voices = ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
                      'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']
        self.current_voice = 'expr-voice-2-m'  # Default voice
        
        # Initialize Rhino for structured intent parsing
        self.picovoice_access_key = picovoice_access_key
        self.rhino = None
        if picovoice_access_key:
            try:
                print("Loading Rhino Speech-to-Intent...")
                # You'll need to create a context file on Picovoice Console
                # This is a basic context - you should create your own .rhn file
                self.rhino = pvrhino.create(
                    access_key=picovoice_access_key,
                    context_path="smart_home_context.rhn"  # You need to create this
                )
                print("Rhino loaded successfully!")
            except Exception as e:
                print(f"Failed to load Rhino: {e}")
                print("Falling back to embedding-based matching")
                
        # Fallback commands for embedding-based matching
        self.commands = [
            "turn on all lights", "turn on bedroom lights", "turn off bedroom lights",
            "turn off living room lamp", "turn on living room lamp", "turn off all lights",
            "get weather", "get time", "set timer", "play music or song", "call for help", 
            "search for bluetooth speaker"
        ]
        
        # Songs list
        self.songs = [
            "Manathe Chandanakkeeru", "Entammede Jimikki Kammal", "Aaro Viral Meeti",
            "Kizhakku Pookkum", "Mazhaye Thoomazhaye", "Oru Madhurakinavin",
            "Ponveene", "Thumbi Vaa", "Melle Melle", "Nee Himamazhayayi"
        ]
        
        # Pre-compute embeddings for fallback
        print("Computing command embeddings...")
        self.command_embeddings = self.sentence_model.encode(self.commands)
        self.song_embeddings = self.sentence_model.encode(self.songs)
        
        # Audio setup
        self.samplerate = samplerate
        self.audio_queue = queue.Queue()
        self.wake_queue = queue.Queue()
        self.rhino_queue = queue.Queue()
        
        # State management
        self.wake_detected = Event()
        self.listening_for_command = Event()
        self.running = True
        
        # Timer management
        self.active_timers = {}
        self.timer_counter = 0
        
        print("All models loaded successfully!")

    def speak(self, text):
        """Convert text to speech and play it"""
        try:
            print(f"🔊 Speaking: {text}")
            audio = self.tts_model.generate(text, voice=self.current_voice)
            sd.play(audio, samplerate=24000)
            sd.wait()  # Wait until audio finishes playing
        except Exception as e:
            print(f"TTS Error: {e}")

    def wake_word_callback(self, indata, frames, time, status):
        """Callback for wake word detection"""
        if status:
            print(status, file=sys.stderr)
        # Convert to int16 for openWakeWord
        audio_int16 = np.frombuffer(indata, dtype=np.float32) * 32767
        audio_int16 = audio_int16.astype(np.int16)
        self.wake_queue.put(audio_int16)

    def command_callback(self, indata, frames, time, status):
        """Callback for command recording (VOSK + Rhino)"""
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))
        
        # Also queue for Rhino if available
        if self.rhino:
            # Convert float32 to int16 for Rhino
            audio_int16 = np.frombuffer(indata, dtype=np.float32) * 32767
            audio_int16 = audio_int16.astype(np.int16)
            self.rhino_queue.put(audio_int16)

    def wake_word_thread(self):
        """Thread for continuous wake word detection"""
        print("Wake word detection started. Say 'computer' or 'alexa'...")
        while self.running:
            try:
                if not self.wake_queue.empty():
                    audio_data = self.wake_queue.get_nowait()
                    # Check for wake word
                    prediction = self.wake_model.predict(audio_data)
                    # Check if any wake word was detected
                    for wake_word, confidence in prediction.items():
                        if confidence > 0.5:  # Confidence threshold
                            print(f"\n🎯 Wake word '{wake_word}' detected! (confidence: {confidence:.2f})")
                            self.wake_detected.set()
                            break
            except Exception as e:
                print(f"Wake word detection error: {e}")
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def process_with_rhino(self, timeout=3.0):
        """Process audio with Rhino for structured intent parsing"""
        if not self.rhino:
            return None
            
        print("🎯 Using Rhino for intent parsing...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if not self.rhino_queue.empty():
                    audio_frame = self.rhino_queue.get_nowait()
                    is_finalized = self.rhino.process(audio_frame)
                    
                    if is_finalized:
                        inference = self.rhino.get_inference()
                        if inference.is_understood:
                            print(f"🎯 Rhino Intent: {inference.intent}")
                            print(f"🎯 Rhino Slots: {inference.slots}")
                            return {
                                'intent': inference.intent,
                                'slots': inference.slots,
                                'confidence': 100  # Rhino doesn't provide confidence score
                            }
                        else:
                            print("🤔 Rhino didn't understand the command")
                            return None
            except Exception as e:
                print(f"Rhino processing error: {e}")
            
            time.sleep(0.01)
        
        print("⏰ Rhino timeout")
        return None

    def process_command_fallback(self, text):
        """Fallback embedding-based processing"""
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
            best_command = "Lets ask LLM"
            
        return {
            'intent': best_command,
            'slots': {'song': song_match} if song_match else {},
            'confidence': command_confidence
        }

    def parse_timer_duration(self, duration_text):
        """Parse duration from text or slots"""
        if not duration_text:
            return None
            
        # Handle Rhino slot format (should be in seconds or structured)
        if isinstance(duration_text, (int, float)):
            return int(duration_text)
            
        # Fallback text parsing
        duration_text = duration_text.lower()
        
        # Simple regex parsing for common patterns
        import re
        
        # Pattern: "5 minutes", "10 seconds", "1 hour"
        pattern = r'(\d+)\s*(second|minute|hour)s?'
        match = re.search(pattern, duration_text)
        
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'second':
                return num
            elif unit == 'minute':
                return num * 60
            elif unit == 'hour':
                return num * 3600
                
        return None

    def start_timer(self, duration_seconds, timer_id=None):
        """Start a timer"""
        if timer_id is None:
            self.timer_counter += 1
            timer_id = f"timer_{self.timer_counter}"
            
        end_time = time.time() + duration_seconds
        self.active_timers[timer_id] = {
            'end_time': end_time,
            'duration': duration_seconds,
            'paused': False
        }
        
        # Start timer thread
        timer_thread = threading.Thread(target=self._timer_worker, args=(timer_id,))
        timer_thread.daemon = True
        timer_thread.start()
        
        return timer_id

    def _timer_worker(self, timer_id):
        """Worker thread for timer countdown"""
        while timer_id in self.active_timers:
            timer = self.active_timers[timer_id]
            
            if timer['paused']:
                time.sleep(1)
                continue
                
            remaining = timer['end_time'] - time.time()
            
            if remaining <= 0:
                print(f"⏰ Timer {timer_id} finished!")
                self.speak(f"Timer finished!")
                del self.active_timers[timer_id]
                break
                
            time.sleep(1)

    def execute_command(self, command_data):
        """Execute command from either Rhino or fallback processing"""
        intent = command_data['intent']
        slots = command_data.get('slots', {})
        
        print(f"\n🚀 Executing: {intent}")
        print(f"📝 Slots: {slots}")
        
        response_text = ""
        
        # Handle Rhino intents
        if intent == "setTimer":
            duration_slot = slots.get('duration') or slots.get('time')
            duration = self.parse_timer_duration(duration_slot)
            
            if duration:
                timer_id = self.start_timer(duration)
                minutes = duration // 60
                seconds = duration % 60
                if minutes > 0:
                    response_text = f"Timer set for {minutes} minutes and {seconds} seconds"
                else:
                    response_text = f"Timer set for {seconds} seconds"
            else:
                response_text = "I couldn't understand the timer duration. Please try again."
                
        elif intent == "lightControl":
            location = slots.get('location', 'all')
            state = slots.get('state', 'on')
            response_text = f"{location} lights are now {state}"
            print(f"💡 {location} lights -> {state}")
            
        elif intent == "getWeather":
            location = slots.get('location', 'here')
            response_text = f"The weather in {location} is sunny, 22 degrees celsius"
            
        elif intent == "getTime":
            current_time = time.strftime("%H:%M")
            response_text = f"The current time is {current_time}"
            
        elif intent == "playMusic":
            song = slots.get('song') or slots.get('title')
            if song:
                response_text = f"Now playing {song}"
            else:
                response_text = "Playing music"
                
        # Handle fallback intents
        elif intent == "turn on all lights":
            response_text = "All lights are now on"
        elif intent == "turn off all lights":
            response_text = "All lights are now off"
        elif "bedroom lights" in intent:
            action = "on" if "on" in intent else "off"
            response_text = f"Bedroom lights are now {action}"
        elif "living room lamp" in intent:
            action = "on" if "on" in intent else "off"
            response_text = f"Living room lamp is now {action}"
        elif intent == "get weather":
            response_text = "Today's weather is sunny, 22 degrees celsius"
        elif intent == "get time":
            current_time = time.strftime("%H:%M")
            response_text = f"The current time is {current_time}"
        elif intent == "set timer":
            # For fallback, ask for duration
            response_text = "For how long would you like to set the timer?"
        elif intent == "play music or song":
            song = slots.get('song')
            if song:
                response_text = f"Now playing {song}"
            else:
                response_text = "Playing music"
        elif intent == "call for help":
            response_text = "Emergency call initiated"
        elif intent == "search for bluetooth speaker":
            response_text = "Searching for bluetooth speaker"
        else:
            response_text = "I'll need to ask an AI assistant for that"
        
        # Speak the response
        if response_text:
            self.speak(response_text)

    def listen_for_command(self):
        """Listen for command using VOSK and optionally Rhino"""
        print("🎤 Listening for command... (speak now)")
        
        # Try Rhino first if available
        if self.rhino:
            rhino_result = self.process_with_rhino(timeout=3.0)
            if rhino_result:
                return rhino_result
        
        # Fallback to VOSK + embedding matching
        print("🎤 Using VOSK fallback...")
        rec = KaldiRecognizer(self.vosk_model, self.samplerate)
        
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000,
                               dtype="int16", channels=1, callback=self.command_callback):
            command_text = ""
            silence_start = None
            max_silence_duration = 2.0
            
            while self.listening_for_command.is_set():
                try:
                    data = self.audio_queue.get(timeout=0.1)
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
                            silence_start = None
                        else:
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
            
            # Process with fallback method
            return self.process_command_fallback(command_text)

    def run(self):
        """Main assistant loop"""
        try:
            print("\n" + "=" * 60)
            print("🎙️  VOICE ASSISTANT READY")
            print("🔊  Say 'computer' or 'alexa' to start")
            if self.rhino:
                print("🎯  Rhino Speech-to-Intent: ENABLED")
            else:
                print("⚠️   Rhino Speech-to-Intent: DISABLED (fallback mode)")
            print("⌨️   Press Ctrl+C to stop")
            print("=" * 60)
            
            # Start wake word detection thread
            wake_thread = Thread(target=self.wake_word_thread)
            wake_thread.daemon = True
            wake_thread.start()
            
            # Start wake word audio stream
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=1024,
                                   dtype="float32", channels=1, callback=self.wake_word_callback):
                while self.running:
                    try:
                        # Wait for wake word
                        if self.wake_detected.wait(timeout=1.0):
                            self.wake_detected.clear()
                            
                            # Acknowledge wake word detection
                            print("🎯 Wake word detected!")
                            self.speak("Yes, how can I help you?")
                            
                            # Start listening for command
                            self.listening_for_command.set()
                            command_data = self.listen_for_command()
                            self.listening_for_command.clear()
                            
                            if command_data and command_data['intent']:
                                print(f"\n📥 Processing command...")
                                # Execute command (includes voice response)
                                self.execute_command(command_data)
                            else:
                                self.speak("I didn't understand that. Please try again.")
                                
                            print(f"\n{'=' * 30}")
                            print("🔊 Ready for next wake word...")
                            print("=" * 30)
                            
                    except KeyboardInterrupt:
                        break
        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            self.running = False
            if self.rhino:
                self.rhino.delete()
            print("\n👋 Voice assistant stopped")

def main():
    parser = argparse.ArgumentParser(description="Voice Assistant with Rhino Speech-to-Intent")
    parser.add_argument("-m", "--model", type=str, default="en-us",
                        help="VOSK language model (default: en-us)")
    parser.add_argument("-r", "--samplerate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("-k", "--access-key", type=str, 
                        help="Picovoice Access Key (get from console.picovoice.ai)")
    parser.add_argument("-v", "--voice", type=str, default="expr-voice-2-m",
                        help="TTS voice (default: expr-voice-2-m)")
    parser.add_argument("-l", "--list-devices", action="store_true",
                        help="List audio devices and exit")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    if not args.access_key:
        print("⚠️  Warning: No Picovoice Access Key provided.")
        print("   Get one free from: https://console.picovoice.ai")
        print("   Running in fallback mode (embedding-based matching)")
    
    # Create and run assistant
    assistant = VoiceAssistant(
        model_lang=args.model, 
        samplerate=args.samplerate,
        picovoice_access_key=args.access_key
    )
    
    if args.voice:
        assistant.current_voice = args.voice
        
    assistant.run()

if __name__ == "__main__":
    main()
