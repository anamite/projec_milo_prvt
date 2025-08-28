#!/usr/bin/env python3

import argparse
import queue
import sys
import json
import time
import numpy as np
import asyncio
import aiohttp
import os
import re
from threading import Thread, Event
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openwakeword.model import Model as WakeWordModel
from kittentts import KittenTTS


class VoiceAssistant:
    def __init__(self, model_lang="en-us", samplerate=16000):
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

        # OpenRouter API setup
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.openrouter_api_key:
            print("⚠️  Warning: OPENROUTER_API_KEY not found in environment variables")

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

        print("All models loaded successfully!")

    def play_beep(self, frequency=800, duration=0.2):
        """Play a beep tone to indicate activation"""
        try:
            sample_rate = 24000
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = 0.3 * np.sin(frequency * 2 * np.pi * t)
            sd.play(wave, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Beep error: {e}")

    def speak(self, text):
        """Convert text to speech and play it"""
        try:
            print(f"🔊 Speaking: {text}")
            audio = self.tts_model.generate(text, voice=self.current_voice)
            sd.play(audio, samplerate=24000)
            sd.wait()  # Wait until audio finishes playing
        except Exception as e:
            print(f"TTS Error: {e}")

    def speak_async(self, text):
        """Convert text to speech and play it asynchronously without waiting"""
        try:
            print(f"🔊 Speaking: {text}")
            audio = self.tts_model.generate(text, voice=self.current_voice)
            sd.play(audio, samplerate=24000)
            # Don't wait here - let it play in background
        except Exception as e:
            print(f"TTS Error: {e}")

    def split_into_speakable_chunks(self, text):
        """Split text into natural speaking chunks"""
        # Split on sentence boundaries and conjunctions
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If sentence is too long, split on commas or conjunctions
            if len(sentence) > 80:
                sub_chunks = re.split(r'[,;]|(?:\s+(?:and|but|or|so|yet|for|nor|because|since|although|however|therefore)\s+)', sentence)
                for chunk in sub_chunks:
                    chunk = chunk.strip()
                    if chunk and len(chunk.split()) >= 3:  # At least 3 words
                        chunks.append(chunk)
            else:
                if len(sentence.split()) >= 3:  # At least 3 words
                    chunks.append(sentence)
        
        return chunks

    async def stream_openrouter_api(self, user_message):
        """Stream API call to OpenRouter with chunked TTS output"""
        if not self.openrouter_api_key:
            self.speak("Sorry, I don't have access to AI assistance right now.")
            return
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Voice Assistant",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-4o-mini",  # Changed to a faster model for streaming
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful voice assistant. Give clear, conversational responses. Break your answers into natural sentences."
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": True  # Enable streaming
        }

        try:
            print("🤖 Streaming AI response...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.openrouter_base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API Error {response.status}: {error_text}")
                        self.speak("Sorry, I encountered an error while processing your request.")
                        return

                    accumulated_text = ""
                    chunk_buffer = ""
                    
                    async for line in response.content:
                        try:
                            line = line.decode('utf-8').strip()
                            
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                
                                if data == '[DONE]':
                                    break
                                
                                try:
                                    json_data = json.loads(data)
                                    
                                    if 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        
                                        if content:
                                            accumulated_text += content
                                            chunk_buffer += content
                                            
                                            # Check if we have a complete sentence or phrase to speak
                                            if any(punct in chunk_buffer for punct in ['.', '!', '?', ':', ';']):
                                                # Split buffer into speakable chunks
                                                chunks = self.split_into_speakable_chunks(chunk_buffer)
                                                
                                                for chunk in chunks:
                                                    if chunk.strip():
                                                        # Speak chunk immediately in background
                                                        loop = asyncio.get_event_loop()
                                                        loop.run_in_executor(None, self.speak_async, chunk)
                                                        # Small delay to prevent audio overlap
                                                        await asyncio.sleep(0.5)
                                                
                                                chunk_buffer = ""
                                            
                                            # Also check for natural pause points (commas, conjunctions)
                                            elif len(chunk_buffer.split()) >= 8 and any(marker in chunk_buffer.lower() for marker in [', ', ' and ', ' but ', ' so ', ' because ']):
                                                # Find the last good break point
                                                words = chunk_buffer.split()
                                                if len(words) >= 6:
                                                    break_point = len(words) // 2
                                                    chunk_to_speak = ' '.join(words[:break_point])
                                                    
                                                    if chunk_to_speak.strip():
                                                        loop = asyncio.get_event_loop()
                                                        loop.run_in_executor(None, self.speak_async, chunk_to_speak)
                                                        await asyncio.sleep(0.3)
                                                    
                                                    chunk_buffer = ' '.join(words[break_point:])

                                except json.JSONDecodeError:
                                    continue
                                    
                        except Exception as e:
                            print(f"Stream processing error: {e}")
                            continue
                    
                    # Speak any remaining text in buffer
                    if chunk_buffer.strip():
                        remaining_chunks = self.split_into_speakable_chunks(chunk_buffer)
                        for chunk in remaining_chunks:
                            if chunk.strip():
                                loop = asyncio.get_event_loop()
                                loop.run_in_executor(None, self.speak_async, chunk)
                                await asyncio.sleep(0.4)
                    
                    print(f"✅ Complete response: {accumulated_text}")
                        
        except asyncio.TimeoutError:
            print("API request timed out")
            self.speak("Sorry, the response took too long. Please try again.")
        except Exception as e:
            print(f"Streaming API call error: {e}")
            self.speak("Sorry, I couldn't process your request right now.")

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
        if command_confidence > 50 and best_command == "play music or song":
            song_similarities = cosine_similarity([user_embedding], self.song_embeddings)
            best_song_index = song_similarities.argmax()
            song_match = self.songs[best_song_index]
            song_confidence = song_similarities[0, best_song_index] * 100

        if command_confidence < 80:
            best_command = "Lets ask LLM"

        return best_command, command_confidence, song_match, song_confidence

    async def execute_command(self, command, song=None, original_text=""):
        """Execute the matched command and provide voice feedback"""
        print(f"\n🚀 Executing: {command}")

        if command == "turn on all lights":
            print("💡 All lights are now ON")
            self.speak("All lights are now on")
        elif command == "turn off all lights":
            print("🔌 All lights are now OFF")
            self.speak("All lights are now off")
        elif "bedroom lights" in command:
            action = "ON" if "on" in command else "OFF"
            print(f"🛏️ Bedroom lights are now {action}")
            self.speak(f"Bedroom lights are now {action.lower()}")
        elif "living room lamp" in command:
            action = "ON" if "on" in command else "OFF"
            print(f"🏠 Living room lamp is now {action}")
            self.speak(f"Living room lamp is now {action.lower()}")
        elif command == "get weather":
            print("🌤️ Today's weather: Sunny, 22°C")
            self.speak("Today's weather is sunny, 22 degrees celsius")
        elif command == "get time":
            current_time = time.strftime("%H:%M")
            print(f"🕐 Current time: {current_time}")
            self.speak(f"The current time is {current_time}")
        elif command == "set timer":
            print("⏰ Timer set for 5 minutes")
            self.speak("Timer set for 5 minutes")
        elif command == "play music or song":
            if song:
                print(f"🎵 Playing: {song}")
                self.speak(f"Now playing {song}")
            else:
                print("🎵 Playing music...")
                self.speak("Playing music")
        elif command == "call for help":
            print("🚨 Emergency call initiated!")
            self.speak("Emergency call initiated")
        elif command == "Search for bluetooth speaker":
            print("🔍 Searching for bluetooth speaker...")
            self.speak("Searching for bluetooth speaker")
        elif command == "Lets ask LLM":
            print("🤖 Asking AI assistant...")
            # Use streaming for LLM responses
            await self.stream_openrouter_api(original_text)
        else:
            print("🤖 I'll need to ask an AI assistant for that...")
            await self.stream_openrouter_api(original_text)

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

    async def process_command_async(self, command_text):
        """Process command asynchronously"""
        if command_text.strip():
            print(f"\n📥 Processing: '{command_text}'")

            # Process with embedding model
            command, confidence, song, song_confidence = self.process_command(command_text)

            print(f"🎯 Best match: '{command}' ({confidence:.1f}%)")
            if song:
                print(f"🎵 Song match: '{song}' ({song_confidence:.1f}%)")

            # Execute command (includes voice response)
            await self.execute_command(command, song, command_text)
        else:
            self.speak("I didn't hear anything. Please try again.")

    def run(self):
        """Main assistant loop with async support"""
        async def async_main():
            try:
                print("\n" + "=" * 60)
                print("🎙️  VOICE ASSISTANT READY")
                print("🔊  Say 'computer' or 'alexa' to start")
                print("⌨️   Press Ctrl+C to stop")
                print("🚀  Now with STREAMING responses for faster TTS!")
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
                            await asyncio.sleep(0.1)  # Non-blocking sleep
                            
                            if self.wake_detected.is_set():
                                self.wake_detected.clear()
                                
                                # Play beep instead of voice response
                                print("🎯 Wake word detected!")
                                self.play_beep()

                                # Start listening for command
                                self.listening_for_command.set()
                                
                                # Run command listening in thread to avoid blocking
                                loop = asyncio.get_event_loop()
                                command_text = await loop.run_in_executor(
                                    None, self.listen_for_command
                                )
                                
                                self.listening_for_command.clear()

                                # Process command asynchronously
                                await self.process_command_async(command_text)

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

        # Run the async main function
        try:
            asyncio.run(async_main())
        except KeyboardInterrupt:
            print("\n👋 Voice assistant stopped by user")


def main():
    parser = argparse.ArgumentParser(description="Voice Assistant with Streaming LLM-to-TTS")
    parser.add_argument("-m", "--model", type=str, default="en-us",
                        help="VOSK language model (default: en-us)")
    parser.add_argument("-r", "--samplerate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("-l", "--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("-v", "--voice", type=str, default="expr-voice-2-m",
                        help="TTS voice (default: expr-voice-2-m)")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Create and run assistant
    assistant = VoiceAssistant(model_lang=args.model, samplerate=args.samplerate)
    if args.voice:
        assistant.current_voice = args.voice
    assistant.run()


if __name__ == "__main__":
    main()
