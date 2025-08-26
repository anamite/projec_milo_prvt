from assistant.audio_assistant import AudioAssistant
import logging
import sys


def setup_logging(level: int = logging.DEBUG):
    """Set up a basic logging config that prints to stdout for debugging."""
    root = logging.getLogger()
    if root.handlers:
        # avoid duplicate handlers when running interactively
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


if __name__ == "__main__":
    setup_logging()
    logging.getLogger(__name__).debug("Starting AudioAssistant from main")
    assistant = AudioAssistant("config.json")
    assistant.start()
