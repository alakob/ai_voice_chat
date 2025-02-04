"""Tests for audio service functionality"""
import pytest
import numpy as np
from voice_assistant.services.audio_service import start_recording, play_audio, stop_audio_playback
from voice_assistant.models import AudioConfig
from voice_assistant.exceptions import AudioProcessingError
import asyncio

@pytest.mark.asyncio
async def test_start_recording():
    """Test audio recording initialization"""
    config = AudioConfig()
    
    # Test successful recording start
    await start_recording(config)
    from voice_assistant.state import audio_state
    assert audio_state.is_recording == True
    assert hasattr(audio_state, 'audio_stream')
    
    # Cleanup
    audio_state.audio_stream.stop()
    audio_state.audio_stream.close()

@pytest.mark.asyncio
async def test_play_audio():
    """Test audio playback"""
    # Create test audio data
    duration = 0.1
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    # Test playing numpy array
    await play_audio(audio)
    from voice_assistant.state import audio_state
    assert audio_state.is_playing == False  # Should be False after completion
    
    # Test playing bytes
    audio_bytes = audio.tobytes()
    await play_audio(audio_bytes)
    assert audio_state.is_playing == False

def test_stop_audio_playback():
    """Test stopping audio playback"""
    result = stop_audio_playback()
    assert isinstance(result, bool) 