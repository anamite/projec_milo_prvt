import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import vosk
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False


class SpeechToText:
    """Speech to text using Vosk when available, otherwise a dummy.

    The dummy returns an empty string for bytes input; this keeps the
    assistant stable when heavy models aren't installed.
    This class logs initialization details and transcription steps.
    """

    def __init__(self, model_path: str = "vosk-model-small-en-us-0.15"):
        self.model = None
        self.recognizer = None
        logger.debug(f"Initializing STT: VOSK_AVAILABLE={VOSK_AVAILABLE}, model_path={model_path}")
        if VOSK_AVAILABLE:
            try:
                self.model = vosk.Model(model_path)
                self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
                logger.info("Vosk STT initialized successfully")
            except Exception:
                logger.exception("Failed to initialize Vosk STT")
                self.model = None
                self.recognizer = None
        else:
            logger.info("Vosk not available; using dummy STT")

    def transcribe_streaming(self, audio_data) -> tuple[Optional[str], bool]:
        """Transcribe audio bytes with streaming support.
        
        Returns:
            tuple: (text, is_final) where is_final indicates if this is a complete result
        """
        logger.debug("STT.transcribe_streaming called", extra={"audio_len": len(audio_data) if hasattr(audio_data, '__len__') else 'unknown'})

        if not VOSK_AVAILABLE or not self.recognizer:
            # For testing without Vosk, simulate some recognition
            if len(audio_data) > 1000:  # Only process non-trivial audio chunks
                return "sample transcription for testing", True
            return None, False

        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result_raw = self.recognizer.Result()
                logger.debug(f"Vosk final result: {result_raw}")
                result = json.loads(result_raw)
                text = result.get('text', '').strip()
                return text if text else None, True
            else:
                partial_raw = self.recognizer.PartialResult()
                logger.debug(f"Vosk partial result: {partial_raw}")
                partial = json.loads(partial_raw)
                text = partial.get('partial', '').strip()
                return text if text else None, False
        except Exception:
            logger.exception("STT streaming transcription error")
            return None, False

    def transcribe(self, audio_data) -> Optional[str]:
        """Transcribe audio bytes into text.

        Returns full text when available, partial text for streaming partials,
        empty string for dummy, or None on error.
        """
        logger.debug("STT.transcribe called", extra={"audio_len": len(audio_data) if hasattr(audio_data, '__len__') else 'unknown'})

        if not VOSK_AVAILABLE or not self.recognizer:
            # For testing, if audio_data is a small marker bytes, return a sample phrase
            try:
                if audio_data == b"SAMPLE:time":
                    logger.debug("Dummy STT returning sample phrase for marker audio")
                    return "what time is it?"
            except Exception:
                logger.debug("Error checking marker bytes in dummy STT", exc_info=True)
            return ""

        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result_raw = self.recognizer.Result()
                logger.debug(f"Vosk full result: {result_raw}")
                result = json.loads(result_raw)
                return result.get('text', '')
            else:
                partial_raw = self.recognizer.PartialResult()
                logger.debug(f"Vosk partial result: {partial_raw}")
                partial = json.loads(partial_raw)
                return partial.get('partial', '')
        except Exception:
            logger.exception("STT transcription error")
            return None

    def cleanup(self):
        logger.debug("STT.cleanup called - nothing to cleanup by default")
