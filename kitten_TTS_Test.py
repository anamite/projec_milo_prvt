from kittentts import KittenTTS
import sounddevice as sd
import numpy as np

# Load model once for quick responses
print("Loading TTS model...")
m = KittenTTS("KittenML/kitten-tts-nano-0.2")
print("Model loaded! Ready for input.")

# Available voices
voices = ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
          'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

# Main loop
while True:
    try:
        # Get user input
        text = input("\nEnter text to speak (or 'quit' to exit): ")

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if text.strip():
            # Generate audio
            # choose a random voice out of the available ones
            voice_ = voices[np.random.randint(len(voices))]
            print(f"Using voice: {voice_}") # expr-voice-3-m # expr-voice-2-m
            audio = m.generate(text, voice='expr-voice-2-m')

            # Use the model's actual sample rate (usually 22050 for KittenTTS)
            sd.play(audio, samplerate=24000)
            sd.wait()  # Wait until audio finishes playing

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

print("Goodbye!")



# Model that cuts text into chunks and plays them as they are generated
# from kittentts import KittenTTS
# import sounddevice as sd
# import numpy as np
# import threading
# import queue
# import time
#
# # Load model once for quick responses
# print("Loading TTS model...")
# m = KittenTTS("KittenML/kitten-tts-nano-0.2")
# print("Model loaded! Ready for input.")
#
# # Available voices
# voices = ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
#           'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']
#
#
# def chunk_text(text, max_words=6):
#     """Split text into chunks of max_words"""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words):
#         chunk = ' '.join(words[i:i + max_words])
#         chunks.append(chunk)
#     return chunks
#
#
# def generate_audio_chunks(text_chunks, voice, audio_queue):
#     """Generate audio for each chunk and put in queue"""
#     for chunk in text_chunks:
#         try:
#             chunk_with_ellipsis = f"{chunk}   .."
#             audio = m.generate(chunk_with_ellipsis, voice=voice)
#             audio_queue.put(audio)
#         except Exception as e:
#             print(f"Error generating audio for chunk: {e}")
#             audio_queue.put(None)
#
# def play_audio_queue(audio_queue, total_chunks):
#     """Play audio chunks as they become available"""
#     played_chunks = 0
#     while played_chunks < total_chunks:
#         try:
#             audio = audio_queue.get(timeout=10)
#             if audio is not None:
#                 sd.play(audio, samplerate=24000)
#                 sd.wait()
#             played_chunks += 1
#         except queue.Empty:
#             print("Timeout waiting for audio chunk")
#             break
#
#
# # Main loop
# while True:
#     try:
#         # Get user input
#         text = input("\nEnter text to speak (or 'quit' to exit): ")
#
#         if text.lower() in ['quit', 'exit', 'q']:
#             break
#
#         if text.strip():
#             # Choose a random voice
#             voice_ = voices[np.random.randint(len(voices))]
#             print(f"Using voice: {voice_}")
#
#             # Split text into chunks
#             chunks = chunk_text(text, max_words=6)
#             print(f"Split into {len(chunks)} chunks")
#
#             # Create queue for audio chunks
#             audio_queue = queue.Queue()
#
#             # Start generation thread
#             gen_thread = threading.Thread(
#                 target=generate_audio_chunks,
#                 args=(chunks, voice_, audio_queue)
#             )
#             gen_thread.start()
#
#             # Start playing as soon as first chunk is ready
#             play_thread = threading.Thread(
#                 target=play_audio_queue,
#                 args=(audio_queue, len(chunks))
#             )
#             play_thread.start()
#
#             # Wait for both threads to complete
#             gen_thread.join()
#             play_thread.join()
#
#     except KeyboardInterrupt:
#         print("\nExiting...")
#         break
#     except Exception as e:
#         print(f"Error: {e}")
#
# print("Goodbye!")
