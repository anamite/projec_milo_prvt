#!/usr/bin/env python3
import argparse
import json
import time
import numpy as np

import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openwakeword.model import Model as WakeWordModel

# TTS
# pip install per KittenTTS instructions, then:
from kittentts import KittenTTS


class WakeWordDetector:
    def __init__(self, wakewords=("alexa", "hey_jarvis"), samplerate=16000, blocksize=1280, threshold=0.5):
        self.sr = samplerate
        self.blocksize = blocksize  # 1280 samples @16k suggested by openWakeWord demo
        self.threshold = threshold
        self.model = WakeWordModel(wakeword_models=list(wakewords), inference_framework="onnx")

    def listen_until_wake(self):
        print("[wake] Starting microphone stream for wake-word detection ...")
        with sd.RawInputStream(samplerate=self.sr, blocksize=self.blocksize, dtype="float32", channels=1) as stream:
            print("[wake] Say a wake word (e.g., alexa / hey jarvis) ...")
            while True:
                data, _ = stream.read(self.blocksize)
                samples = np.frombuffer(data, dtype=np.float32)
                # Predict on 1 x 1280 chunk
                scores = self.model.predict(samples)
                # Check threshold across known models
                for name, score in scores.items():
                    if score >= self.threshold:
                        print(f"[wake] Wake word '{name}' detected (score={score:.2f})")
                        return name, float(score)


class STTEngine:
    def __init__(self, model_lang="en-us", samplerate=16000, command_blocksize=8000, silence_timeout=2.0):
        self.sr = samplerate
        self.blocksize = command_blocksize
        self.silence_timeout = silence_timeout
        print("[stt] Loading Vosk model ...")
        self.model = VoskModel(lang=model_lang)
        print("[stt] Vosk model loaded")

    def transcribe_once(self):
        print("[stt] Listening for command ...")
        rec = KaldiRecognizer(self.model, self.sr)

        # Use int16 input for recognizer to avoid conversions inside the loop
        with sd.RawInputStream(samplerate=self.sr, blocksize=self.blocksize, dtype="int16", channels=1) as stream:
            last_speech = time.time()
            final_text = ""
            while True:
                data, _ = stream.read(self.blocksize)
                # Feed bytes directly
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        final_text = result["text"]
                        print(f"[stt] Final: {final_text}")
                        break
                else:
                    partial = json.loads(rec.PartialResult())
                    part = partial.get("partial", "")
                    if part:
                        print(f"[stt] Partial: {part}", end="\r")
                        last_speech = time.time()
                    elif (time.time() - last_speech) > self.silence_timeout:
                        print("\n[stt] Silence timeout")
                        break
        return final_text.strip()


class IntentMatcher:
    def __init__(self):
        print("[intent] Loading sentence-transformer and precomputing embeddings ...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
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
        self.command_embeddings = self.embedder.encode(self.commands)
        self.song_embeddings = self.embedder.encode(self.songs)
        print("[intent] Embeddings ready")

    def match(self, text):
        if not text:
            return "Lets ask LLM", 0.0, None, 0.0
        user = self.embedder.encode(text)
        sims = cosine_similarity([user], self.command_embeddings)
        cmd_idx = int(np.argmax(sims))
        cmd = self.commands[cmd_idx]
        cmd_conf = float(sims[cmd_idx] * 100)

        song = None
        song_conf = 0.0
        if cmd == "play music or song" and cmd_conf > 20:
            song_sims = cosine_similarity([user], self.song_embeddings)
            s_idx = int(np.argmax(song_sims))
            song = self.songs[s_idx]
            song_conf = float(song_sims[s_idx] * 100)
        if cmd_conf < 20:
            cmd = "Lets ask LLM"
        return cmd, cmd_conf, song, song_conf


class TTSEngine:
    def __init__(self, model_id="KittenML/kitten-tts-nano-0.1", voice="expr-voice-2-f", out_sr=24000):
        print("[tts] Initializing KittenTTS ...")
        self.tts = KittenTTS(model_id)
        self.voice = voice
        self.out_sr = out_sr
        print(f"[tts] Ready (voice={self.voice}, sr={self.out_sr})")

    def speak(self, text):
        print(f"[tts] Synthesizing: {text}")
        audio = self.tts.generate(text, voice=self.voice)
        sd.play(audio, self.out_sr, blocking=True)
        sd.stop()
        print("[tts] Playback complete")


class CommandExecutor:
    def __init__(self, tts: TTSEngine):
        self.tts = tts

    def execute(self, command, song=None):
        # Return a human-readable response and also speak it
        if command == "turn on all lights":
            msg = "All lights are now on."
        elif command == "turn off all lights":
            msg = "All lights are now off."
        elif "bedroom lights" in command:
            msg = "Bedroom lights are now on." if "on" in command else "Bedroom lights are now off."
        elif "living room lamp" in command:
            msg = "Living room lamp is now on." if "on" in command else "Living room lamp is now off."
        elif command == "get weather":
            msg = "Today’s weather is sunny, 22 degrees."
        elif command == "get time":
            msg = f"The current time is {time.strftime('%H:%M:%S')}."
        elif command == "set timer":
            msg = "Setting a five minute timer."
        elif command == "play music or song":
            msg = f"Playing {song}." if song else "Playing music."
        elif command == "call for help":
            msg = "Emergency call initiated."
        else:
            msg = "I will ask the assistant for that."
        print(f"[exec] {msg}")
        self.tts.speak(msg)
        return msg


class AssistantApp:
    def __init__(self, model_lang="en-us", asr_sr=16000, wakewords=("alexa", "hey_jarvis"), wake_threshold=0.5,
                 tts_model="KittenML/kitten-tts-nano-0.1", tts_voice="expr-voice-2-f"):
        # Modules
        self.wake = WakeWordDetector(wakewords=wakewords, samplerate=asr_sr, blocksize=1280, threshold=wake_threshold)
        self.stt = STTEngine(model_lang=model_lang, samplerate=asr_sr, command_blocksize=8000, silence_timeout=2.0)
        self.intent = IntentMatcher()
        self.tts = TTSEngine(model_id=tts_model, voice=tts_voice, out_sr=24000)
        self.exec = CommandExecutor(self.tts)

    def run(self):
        print("=" * 60)
        print("[app] Voice assistant ready. Sequence: wake → prompt → transcribe/process → speak result")
        print("=" * 60)
        while True:
            # 1) Wake
            wake_name, score = self.wake.listen_until_wake()
            # 2) Speak prompt
            self.tts.speak("Yes? I'm listening.")
            # 3) Transcribe
            print("[app] Recording command now ...")
            text = self.stt.transcribe_once()
            if not text:
                print("[app] No speech detected, returning to wake mode.")
                continue
            print(f"[app] Heard: '{text}'")
            # 4) Process intent
            cmd, conf, song, sconf = self.intent.match(text)
            print(f"[app] Matched command: '{cmd}' ({conf:.1f}%)")
            if song:
                print(f"[app] Matched song: '{song}' ({sconf:.1f}%)")
            # 5) Execute and speak output
            self.exec.execute(cmd, song)
            print("[app] Cycle complete. Waiting for next wake word ...\n")


def main():
    parser = argparse.ArgumentParser(description="Sequential Voice Assistant (Wake → Speak → STT → Execute → TTS)")
    parser.add_argument("-m", "--model", type=str, default="en-us", help="VOSK language code (default: en-us)")
    parser.add_argument("-r", "--samplerate", type=int, default=16000, help="ASR sample rate (default: 16000)")
    parser.add_argument("-w", "--wakewords", type=str, default="alexa,hey_jarvis", help="Comma-separated wakewords")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Wake score threshold (default: 0.5)")
    parser.add_argument("-v", "--voice", type=str, default="expr-voice-2-f", help="KittenTTS voice id")
    parser.add_argument("-l", "--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    wakewords = tuple(x.strip() for x in args.wakewords.split(",") if x.strip())

    app = AssistantApp(
        model_lang=args.model,
        asr_sr=args.samplerate,
        wakewords=wakewords,
        wake_threshold=args.threshold,
        tts_model="KittenML/kitten-tts-nano-0.1",
        tts_voice=args.voice,
    )
    app.run()


if __name__ == "__main__":
    main()
