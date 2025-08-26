import logging

logger = logging.getLogger(__name__)

try:
    import kokoro
    KOKORO_AVAILABLE = True
except Exception:
    KOKORO_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False


class TextToSpeech:
    """Text to speech using Kokoro or fallback to pyttsx3 or print

    Adds debug logs for engine selection and errors to help troubleshooting.
    """

    def __init__(self):
        self.use_kokoro = False
        self.use_pyttsx3 = False
        self.engine = None

        logger.debug(f"KOKORO_AVAILABLE={KOKORO_AVAILABLE}, PYTTSX3_AVAILABLE={PYTTSX3_AVAILABLE}")

        if KOKORO_AVAILABLE:
            try:
                self.engine = kokoro.Kokoro()
                self.use_kokoro = True
                logger.info("Using Kokoro TTS engine")
            except Exception:
                logger.exception("Kokoro TTS init failed")
                self.use_kokoro = False

        if not self.use_kokoro and PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.use_pyttsx3 = True
                logger.info("Using pyttsx3 TTS engine")
                # sensible defaults
                try:
                    self.engine.setProperty('rate', 150)
                    self.engine.setProperty('volume', 0.8)
                except Exception:
                    logger.debug("Failed to set pyttsx3 properties", exc_info=True)
            except Exception:
                logger.exception("pyttsx3 init failed")
                self.use_pyttsx3 = False

        if not self.engine:
            logger.info("No TTS engine available; TextToSpeech will print output to console")

    def speak(self, text: str):
        """Speak the provided text and log key events for debugging."""
        logger.debug("TTS.speak called", extra={"text_preview": text[:120]})
        try:
            logger.info(f"Speaking text (len={len(text)})")
            if self.use_kokoro:
                logger.debug("Using kokoro engine to speak")
                self.engine.speak(text)
            elif self.use_pyttsx3:
                logger.debug("Using pyttsx3 engine to speak")
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Fallback: print
                logger.debug("No engine: printing TTS output to stdout")
                print(f"TTS: {text}")
        except Exception:
            logger.exception("TTS error while speaking")
