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


class VoiceAssistant:
    def __init__(self, model_lang="en-us", samplerate=16000):
        # Initialize models
        print("Loading models...")
        self.vosk_model = Model(lang=model_lang)
        self.wake_model = WakeWordModel(wakeword_models=["alexa", "hey_jarvis"])
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Your commands
        self.commands = [
            "turn on all lights", "turn on bedroom lights", "turn off bedroom lights",
            "turn off living room lamp", "turn on living room lamp", "turn off all lights",
            "get weather", "get time", "set timer", "play music or song", "call for help", "Search for bluetooth speaker"
        ]

        # Songs list
        self.songs = [
            "Manathe Chandanakkeeru", "Entammede Jimikki Kammal", "Aaro Viral Meeti",
            "Kizhakku Pookkum", "Mazhaye Thoomazhaye", "Oru Madhurakinavin",
            "Ponveene", "Thumbi Vaa", "Melle Melle", "Nee Himamazhayayi"
        ]

        # Pre-compute embeddings
        print("Computing command embeddings...")
        self.command_embeddings = self.sentence_model.encode(self.commands)
        self.song_embeddings = self.sentence_model.encode(self.songs)

        # Audio setup
        self.samplerate = samplerate
        self.audio_queue = queue.Queue()
        self.wake_queue = queue.Queue()

        # State management
        self.wake_detected = Event()
        self.listening_for_command = Event()
        self.running = True

        print("Models loaded successfully!")

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

    def process_command(self, text):
        """Your existing embedding logic"""
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

        return best_command, command_confidence, song_match, song_confidence

    def execute_command(self, command, song=None):
        """Execute the matched command"""
        print(f"\n🚀 Executing: {command}")

        if command == "turn on all lights":
            print("💡 All lights are now ON")
        elif command == "turn off all lights":
            print("🔌 All lights are now OFF")
        elif "bedroom lights" in command:
            action = "ON" if "on" in command else "OFF"
            print(f"🛏️ Bedroom lights are now {action}")
        elif "living room lamp" in command:
            action = "ON" if "on" in command else "OFF"
            print(f"🏠 Living room lamp is now {action}")
        elif command == "get weather":
            print("🌤️ Today's weather: Sunny, 22°C")
        elif command == "get time":
            current_time = time.strftime("%H:%M:%S")
            print(f"🕐 Current time: {current_time}")
        elif command == "set timer":
            print("⏰ Timer set for 5 minutes")
        elif command == "play music or song":
            if song:
                print(f"🎵 Playing: {song}")
            else:
                print("🎵 Playing music...")
        elif command == "call for help":
            print("🚨 Emergency call initiated!")
        else:
            print("🤖 I'll need to ask an AI assistant for that...")

    def listen_for_command(self):
        """Listen for command using VOSK"""
        print("🎤 Listening for command... (speak now)")

        # Create recognizer
        rec = KaldiRecognizer(self.vosk_model, self.samplerate)

        # Start recording
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000,
                               dtype="int16", channels=1, callback=self.command_callback):

            command_text = ""
            silence_start = None
            max_silence_duration = 2.0  # Stop after 2 seconds of silence

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

    def run(self):
        """Main assistant loop"""
        try:
            # Start wake word detection thread
            wake_thread = Thread(target=self.wake_word_thread)
            wake_thread.daemon = True
            wake_thread.start()

            # Start wake word audio stream
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=1024,
                                   dtype="float32", channels=1, callback=self.wake_word_callback):

                print("\n" + "=" * 60)
                print("🎙️  VOICE ASSISTANT READY")
                print("🔊  Say 'computer' or 'alexa' to start")
                print("⌨️   Press Ctrl+C to stop")
                print("=" * 60)

                while self.running:
                    try:
                        # Wait for wake word
                        if self.wake_detected.wait(timeout=1.0):
                            self.wake_detected.clear()

                            # Start listening for command
                            self.listening_for_command.set()
                            command_text = self.listen_for_command()
                            self.listening_for_command.clear()

                            if command_text.strip():
                                print(f"\n📥 Processing: '{command_text}'")

                                # Process with embedding model
                                command, confidence, song, song_confidence = self.process_command(command_text)

                                print(f"🎯 Best match: '{command}' ({confidence:.1f}%)")
                                if song:
                                    print(f"🎵 Song match: '{song}' ({song_confidence:.1f}%)")

                                # Execute command
                                self.execute_command(command, song)

                            print(f"\n{'=' * 30}")
                            print("🔊 Ready for next wake word...")
                            print("=" * 30)

                    except KeyboardInterrupt:
                        break

        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            self.running = False
            print("\n👋 Voice assistant stopped")


def main():
    parser = argparse.ArgumentParser(description="Voice Assistant with Wake Word Detection")
    parser.add_argument("-m", "--model", type=str, default="en-us",
                        help="VOSK language model (default: en-us)")
    parser.add_argument("-r", "--samplerate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("-l", "--list-devices", action="store_true",
                        help="List audio devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Create and run assistant
    assistant = VoiceAssistant(model_lang=args.model, samplerate=args.samplerate)
    assistant.run()


if __name__ == "__main__":
    main()
