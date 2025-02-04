"""Audio processing service"""
from typing import List, Union, Any
import numpy as np
import sounddevice as sd
import soundfile as sf
from ..models import AudioConfig
from ..exceptions import AudioProcessingError
from ..state import audio_state
import logging
import numpy.typing as npt
import asyncio
from io import BytesIO

logger = logging.getLogger(__name__)

async def start_recording(audio_config: AudioConfig) -> None:
    """Start audio recording"""
    logger.info("Starting audio recording")
    try:
        audio_state.is_recording = True
        audio_state.audio_data = []

        def audio_callback(indata: npt.NDArray, frames: int, time: Any, status: Any) -> None:
            if status:
                logger.warning(f"Audio recording status: {status}")
            if audio_state.is_recording:
                audio_state.audio_data.append(indata.astype(np.float32).copy())

        audio_state.audio_stream = sd.InputStream(
            channels=audio_config.channels,
            samplerate=audio_config.sample_rate,
            dtype=audio_config.dtype,
            callback=audio_callback,
            blocksize=1024
        )
        audio_state.audio_stream.start()

    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        raise AudioProcessingError(f"Failed to start recording: {str(e)}")

async def play_audio(audio_data: Union[np.ndarray, bytes]) -> None:
    """Play audio data using sounddevice"""
    try:
        if isinstance(audio_data, bytes):
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
        
        audio_state.is_playing = True
        audio_state.stop_event = asyncio.Event()
        
        sd.play(audio_data, AudioConfig().sample_rate)
        sd.wait()
        
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        raise AudioProcessingError(f"Failed to play audio: {str(e)}")
    finally:
        audio_state.is_playing = False 