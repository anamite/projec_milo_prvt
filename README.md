# AI Home Assistant (Python)

This project contains a modular AI home assistant architecture implemented in Python.

Files:
- `assistant/` package: modules for audio capture, wake word, STT, TTS, tool matcher and executor.
- `main.py`: entrypoint to start the assistant.
- `config.json`: tools and wake words configuration.
- `requirements.txt`: suggested Python dependencies.

Quick smoke test (no heavy deps required):

Create and run `test_smoke.py` which runs a minimal import and executes a couple of tool functions to verify the package loads.
