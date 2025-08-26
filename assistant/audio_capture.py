import logging
import queue
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except Exception:
    PYAUDIO_AVAILABLE = False

class AudioCapture:
    """Handle audio capture from microphone. If pyaudio is unavailable,
    this becomes a dummy capture that emits small placeholder byte chunks
    so the assistant can be started in environments without audio devices.
    """

    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_recording = False
        self._stream = None
        self._audio = None

        if PYAUDIO_AVAILABLE:
            self._audio = pyaudio.PyAudio()
            self.audio_format = pyaudio.paInt16
            self.channels = 1
        else:
            logger.info("pyaudio not available; AudioCapture will use dummy mode")

    def start_capture(self, audio_queue: queue.Queue):
        """Start continuous audio capture and push bytes into audio_queue."""
        self.is_recording = True

        if PYAUDIO_AVAILABLE:
            try:
                stream = self._audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                self._stream = stream
                logger.info("Audio capture started (pyaudio)")
                while self.is_recording:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_queue.put(data)
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        else:
            # Dummy capture - push small placeholders occasionally
            logger.info("Audio capture started (dummy mode)")
            try:
                while self.is_recording:
                    audio_queue.put(b"\x00" * 1024)
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Dummy audio capture error: {e}")

    def stop(self):
        """Stop audio capture and clean up."""
        self.is_recording = False
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._audio is not None:
                self._audio.terminate()
        except Exception:
            pass
