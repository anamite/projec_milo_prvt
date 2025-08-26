import logging
import random
from typing import List

logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Wake word detection (lightweight placeholder)

    This implementation is intentionally simple and dependency-free so imports
    don't fail if audio or 3rd-party libs are missing. Replace with Porcupine
    or other detector for production.
    """

    def __init__(self, wake_words: List[str]):
        self.wake_words = [w.lower() for w in wake_words]
        self.sensitivity = 0.5

    def detect(self, audio_data) -> bool:
        """Detect wake word in audio_data. audio_data may be bytes from mic.
        This placeholder randomly triggers rarely so the rest of the system can
        be exercised without real audio hardware.
        """
        try:
            # Lightweight random trigger to simulate detection in tests
            if random.random() > 0.997:
                return True
            return False
        except Exception as e:
            logger.debug(f"Wake word detection error: {e}")
            return False
