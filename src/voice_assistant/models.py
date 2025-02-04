"""Data models for the voice assistant"""
from pydantic import BaseModel, Field
from typing import Optional

class AudioConfig(BaseModel):
    """Configuration for audio processing"""
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    dtype: str = Field(default="float32", description="Audio data type")

class TranscriptionResult(BaseModel):
    """Result of speech-to-text conversion"""
    text: str
    confidence: float = Field(default=1.0)
    language: str = Field(default="en")

class TTSRequest(BaseModel):
    """Request for text-to-speech conversion"""
    text: str
    voice: str = Field(default="alloy")
    speed: float = Field(default=1.25) 