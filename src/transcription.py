import os
import tempfile
import whisper
from src.utils import measure_time


@measure_time()
def transcribe(audio: bytes, whisper_model: whisper.Whisper) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio)
        temp_path: str = f.name

    result = whisper_model.transcribe(temp_path, language="en")
    os.remove(temp_path)
    return result["text"]
