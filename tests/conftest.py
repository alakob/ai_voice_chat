"""Test configuration and fixtures"""
import pytest
from openai import AsyncOpenAI
from voice_assistant.config import Settings
import numpy as np
import soundfile as sf
import os
from pathlib import Path

@pytest.fixture
def test_settings():
    """Test settings fixture"""
    return Settings(
        openai_api_key="test-key",
        google_api_key="test-google-key",
        sample_rate=16000,
        channels=1,
        dtype="float32"
    )

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client fixture"""
    class MockOpenAI:
        async def chat_completions_create(self, **kwargs):
            class MockResponse:
                choices = [type('obj', (), {'message': type('obj', (), {'content': "Test response"})()})]
            return MockResponse()
            
        async def audio_speech_create(self, **kwargs):
            # Create a simple sine wave for test audio
            duration = 1.0
            t = np.linspace(0, duration, int(16000 * duration))
            audio = np.sin(2 * np.pi * 440 * t)
            
            # Save to BytesIO
            from io import BytesIO
            buffer = BytesIO()
            sf.write(buffer, audio, 16000, format='WAV')
            buffer.seek(0)
            
            return type('obj', (), {'content': buffer.read()})()
    
    return MockOpenAI()

@pytest.fixture
def test_audio_file():
    """Create a test audio file"""
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    test_file = Path("tests/data/test_audio.wav")
    test_file.parent.mkdir(exist_ok=True)
    sf.write(test_file, audio, sample_rate)
    
    yield test_file
    
    # Cleanup
    os.unlink(test_file) 