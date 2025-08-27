#!/usr/bin/env python3
import argparse
import queue
import sys
import json
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openwakeword.model import Model as WakeWordModel
from kittentts import KittenTTS


# ---------------------------
# Config containers
# ---------------------------

@dataclass
class AudioConfig:
    mic_samplerate: int = 16000      # 16 kHz capture for wake + ASR
    mic_blocksize_wake: int = 1024   # small blocks for wake reactivity
    mic_blocksize_asr: int = 8000    # common Vosk example block size
    wake_threshold: float = 0.5      # openWakeWord confidence threshold


@dataclass
class ModelConfig:
    vosk_lang: str = "en-us"                           # dynamic language model loading
    sentence_model_name: str = "all-MiniLM-L6-v2"      # unchanged for perf parity
    wake_models: List[str] = None                      # e.g. ["alexa","hey_jarvis"]
    if wake_models is None:
        wake_models = ["alexa", "hey_jarvis"]

    # TTS
    tts_repo_id: str = "KittenML/kitten-tts-nano-0.2"  # KittenTTS model id
    tts_default_voice: Optional[str] = None            # if None -> random each speak()
    tts_samplerate: int = 24000                        # KittenTTS outputs 24 kHz


# ---------------------------
# Wake word detector
# ---------------------------

class WakeWordDetector:
    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig):
        print("[Init] Loading wake word model...", flush=True)
        # onnx inference is fine on CPU; models can be named or tflite paths as per openWakeWord
        self.model = WakeWordModel(
            wakeword_models=model_cfg.wake_models,
            inference_framework="onnx",
            enable_speex_noise_suppression=False,
        )
        self.q = queue.Queue()
        self.audio_cfg = audio_cfg
        print("[Init] Wake word model ready.", flush=True)

    def _callback_wake(self, indata, frames, t, status):
        if status:
            print(f"[Wake] Audio status: {status}", file=sys.stderr)
        # openWakeWord expects 16-bit int16 16k PCM frames
        # Convert float32 stream to int16
        audio_int16 = (np.frombuffer(indata, dtype=np.float32) * 32767).astype(np.int16)
        self.q.put(audio_int16)

    def detect_wake(self) -> Tuple[Optional[str], float]:
        print("[Wake] Waiting for wake word...", flush=True)
        detected_word = None
        detected_conf = 0.0

        # Single, synchronous stream; no background thread
        with sd.RawInputStream(
            samplerate=self.audio_cfg.mic_samplerate,
            blocksize=self.audio_cfg.mic_blocksize_wake,
            dtype="float32",
            channels=1,
            callback=self._callback_wake,
        ):
            silence_ticks = 0
            while True:
                try:
                    frame = self.q.get(timeout=0.5)
                except queue.Empty:
                    silence_ticks += 1
                    if silence_ticks % 10 == 0:
                        print("[Wake] ...still listening.", flush=True)
                    continue

                pred = self.model.predict(frame)  # returns dict{name: score}
                # Check any above threshold
                for w, conf in pred.items():
                    if conf >= self.audio_cfg.wake_threshold:
                        detected_word = w
                        detected_conf = float(conf)
                        print(f"[Wake] Detected '{w}' @ {conf:.2f}", flush=True)
                        return detected_word, detected_conf


# ---------------------------
# Streaming speech recognizer (Vosk)
# ---------------------------

class SpeechRecognizer:
    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig):
        print("[Init] Loading Vosk model...", flush=True)
        self.vosk_model = VoskModel(lang=model_cfg.vosk_lang)
        self.audio_cfg = audio_cfg
        self.q = queue.Queue()
        print("[Init] Vosk model ready.", flush=True)

    def _callback_asr(self, indata, frames, t, status):
        if status:
            print(f"[ASR] Audio status: {status}", file=sys.stderr)
        self.q.put(bytes(indata))

    def listen_for_command(
        self,
        max_silence_sec: float = 2.0,
        overall_timeout_sec: Optional[float] = None,
    ) -> str:
        print("[ASR] Listening for command...", flush=True)
        rec = KaldiRecognizer(self.vosk_model, self.audio_cfg.mic_samplerate)
        recognized = ""
        silence_start = None
        t_start = time.time()

        with sd.RawInputStream(
            samplerate=self.audio_cfg.mic_samplerate,
            blocksize=self.audio_cfg.mic_blocksize_asr,
            dtype="int16",
            channels=1,
            callback=self._callback_asr,
        ):
            while True:
                if overall_timeout_sec is not None and (time.time() - t_start) > overall_timeout_sec:
                    print("[ASR] Overall timeout reached.", flush=True)
                    break

                try:
                    data = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res.get("text"):
                        recognized = res["text"]
                        print(f"[ASR] Final: {recognized}", flush=True)
                        break
                else:
                    partial = json.loads(rec.PartialResult())
                    if partial.get("partial"):
                        print(f"[ASR] Partial: {partial['partial']}", end="\r", flush=True)
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > max_silence_sec:
                            print(f"\n[ASR] Silence timeout, using: '{recognized}'", flush=True)
                            break

        return recognized.strip()


# ---------------------------
# Command processor (embeddings)
# ---------------------------

class CommandProcessor:
    def __init__(self, model_cfg: ModelConfig):
        print("[Init] Loading sentence embedding model...", flush=True)
        self.sentence_model = SentenceTransformer(model_cfg.sentence_model_name)
        print("[Init] Encoding command/song intents...", flush=True)

        self.commands = [
            "turn on all lights", "turn on bedroom lights", "turn off bedroom lights",
            "turn off living room lamp", "turn on living room lamp", "turn off all lights",
            "get weather", "get time", "set timer", "play music or song", "call for help",
            "Search for bluetooth speaker"
        ]
        self.songs = [
            "Manathe Chandanakkeeru", "Entammede Jimikki Kammal", "Aaro Viral Meeti",
            "Kizhakku Pookkum", "Mazhaye Thoomazhaye", "Oru Madhurakinavin",
            "Ponveene", "Thumbi Vaa", "Melle Melle", "Nee Himamazhayayi"
        ]
        self.commands_emb = self.sentence_model.encode(self.commands)
        self.songs_emb = self.sentence_model.encode(self.songs)
        print("[Init] Embeddings ready.", flush=True)

    def process(self, text: str) -> Tuple[str, float, Optional[str], float]:
        if not text.strip():
            return "", 0.0, None, 0.0

        user_emb = self.sentence_model.encode(text)
        sims = cosine_similarity([user_emb], self.commands_emb)
        idx = int(sims.argmax())
        best_cmd = self.commands[idx]
        cmd_conf = float(sims[0, idx] * 100)

        song_match = None
        song_conf = 0.0
        if cmd_conf > 20 and best_cmd == "play music or song":
            ss = cosine_similarity([user_emb], self.songs_emb)
            sidx = int(ss.argmax())
            song_match = self.songs[sidx]
            song_conf = float(ss[0, sidx] * 100)

        if cmd_conf < 20:
            best_cmd = "Lets ask LLM"

        return best_cmd, cmd_conf, song_match, song_conf


# ---------------------------
# TTS Engine (KittenTTS)
# ---------------------------

class TTSEngine:
    # Reference voice set from KittenTTS quick-start
    AVAILABLE_VOICES = [
        'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
        'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
    ]

    def __init__(self, model_cfg: ModelConfig):
        print("[Init] Loading TTS model...", flush=True)
        self.model = KittenTTS(model_cfg.tts_repo_id)
        self.default_voice = model_cfg.tts_default_voice
        self.sr = model_cfg.tts_samplerate
        print("[Init] TTS model ready.", flush=True)

    def choose_voice(self) -> str:
        return self.default_voice or random.choice(self.AVAILABLE_VOICES)

    def speak(self, text: str, voice: Optional[str] = None):
        voice_to_use = voice or self.choose_voice()
        print(f"[TTS] Voice: {voice_to_use} | Text: {text}", flush=True)
        audio = self.model.generate(text, voice=voice_to_use)
        sd.play(audio, samplerate=self.sr)
        sd.wait()


# ---------------------------
# Actions / actuator layer
# ---------------------------

class ActionExecutor:
    def execute(self, command: str, song: Optional[str]) -> str:
        # Returns a human-friendly response for TTS
        if command == "turn on all lights":
            print("💡 All lights are now ON")
            return "All lights are now on."
        elif command == "turn off all lights":
            print("🔌 All lights are now OFF")
            return "All lights are now off."
        elif "bedroom lights" in command:
            action = "on" if "on" in command else "off"
            print(f"🛏️ Bedroom lights are now {action.upper()}")
            return f"Bedroom lights are now {action}."
        elif "living room lamp" in command:
            action = "on" if "on" in command else "off"
            print(f"🏠 Living room lamp is now {action.upper()}")
            return f"Living room lamp is now {action}."
        elif command == "get weather":
            print("🌤️ Today's weather: Sunny, 22°C")
            return "Today's weather is sunny with a high of twenty two degrees."
        elif command == "get time":
            now = time.strftime("%H:%M")
            print(f"🕐 Current time: {now}")
            return f"The time is {now}."
        elif command == "set timer":
            print("⏰ Timer set for 5 minutes")
            return "Timer set for five minutes."
        elif command == "play music or song":
            if song:
                print(f"🎵 Playing: {song}")
                return f"Playing the song {song}."
            else:
                print("🎵 Playing music...")
                return "Playing music."
        elif command == "call for help":
            print("🚨 Emergency call initiated!")
            return "Emergency call initiated."
        else:
            print("🤖 I'll need to ask an AI assistant for that...")
            return "I will ask the assistant for that."


# ---------------------------
# Orchestrator
# ---------------------------

class VoiceAssistant:
    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig):
        # Order matters for clear startup prints
        self.audio_cfg = audio_cfg
        self.model_cfg = model_cfg

        self.wake = WakeWordDetector(audio_cfg, model_cfg)
        self.asr = SpeechRecognizer(audio_cfg, model_cfg)
        self.proc = CommandProcessor(model_cfg)
        self.tts = TTSEngine(model_cfg)
        self.exec = ActionExecutor()

        self.running = True

    def run(self):
        print("\n" + "=" * 60)
        print("🎙️  VOICE ASSISTANT READY")
        print("🔊  Say a wake word (e.g., 'alexa' or 'hey jarvis').")
        print("⌨️   Press Ctrl+C to stop")
        print("=" * 60, flush=True)

        try:
            while self.running:
                # 1) Wake
                wword, wconf = self.wake.detect_wake()
                if not wword:
                    continue

                # 2) Speak a short acknowledgement
                self.tts.speak("Yes, I am listening.")

                # 3) Listen for command
                text = self.asr.listen_for_command(max_silence_sec=2.0, overall_timeout_sec=15.0)
                if not text:
                    print("[Main] No speech captured. Ready again.", flush=True)
                    self.tts.speak("I did not catch that.")
                    continue
                print(f"[Main] Heard: '{text}'", flush=True)

                # 4) Process
                print("[Main] Processing command via embeddings...", flush=True)
                cmd, conf, song, sconf = self.proc.process(text)
                print(f"[Main] Best match: '{cmd}' ({conf:.1f}%)", flush=True)
                if song:
                    print(f"[Main] Song match: '{song}' ({sconf:.1f}%)", flush=True)

                # 5) Execute + speak response
                reply = self.exec.execute(cmd, song)
                self.tts.speak(reply)

                print("\n" + "-" * 40, flush=True)
                print("🔊 Ready for next wake word...", flush=True)
                print("-" * 40, flush=True)

        except KeyboardInterrupt:
            print("\n[Main] Stopping assistant...", flush=True)
        finally:
            self.running = False
            print("👋 Voice assistant stopped.", flush=True)


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Local Voice Assistant (sequential wake → speak → process → output)")
    parser.add_argument("-m", "--model", type=str, default="en-us", help="VOSK language model (default: en-us)")
    parser.add_argument("-r", "--samplerate", type=int, default=16000, help="Microphone sample rate (default: 16000)")
    parser.add_argument("-l", "--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--tts-voice", type=str, default=None, help="Fixed TTS voice id (else random each time)")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    audio_cfg = AudioConfig(mic_samplerate=args.samplerate)
    model_cfg = ModelConfig(vosk_lang=args.model, tts_default_voice=args.tts_voice)

    assistant = VoiceAssistant(audio_cfg, model_cfg)
    assistant.run()


if __name__ == "__main__":
    main()
