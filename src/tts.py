import io
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from src.utils import measure_time


@measure_time()
def generate_speech(
    text: str, kokoro_pipeline: KPipeline, voice: str = "af_heart"
) -> io.BytesIO:
    generator = kokoro_pipeline(text, voice=voice)
    audio_chunks: list[np.ndarray] = []
    for _, _, audio_chunk in generator:
        audio_chunks.append(audio_chunk)
    audio: np.ndarray = np.concatenate(audio_chunks)

    buffer: io.BytesIO = io.BytesIO()
    sampling_rate: int = 24000
    sf.write(buffer, audio, sampling_rate, format="WAV")
    buffer.seek(0)
    return buffer
