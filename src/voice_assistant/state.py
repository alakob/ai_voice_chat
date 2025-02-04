"""Global state management"""
from typing import Optional, List
import asyncio
import numpy.typing as npt

class AudioState:
    """Global state for audio processing"""
    def __init__(self):
        self.is_recording: bool = False
        self.is_playing: bool = False
        self.stop_event: Optional[asyncio.Event] = None
        self.current_audio: Optional[npt.NDArray] = None
        self.current_position: int = 0
        self.audio_data: List = []

audio_state = AudioState() 