"""Tests for OpenAI service functionality"""
import pytest
from voice_assistant.services.openai_service import generate_ai_response, text_to_speech

@pytest.mark.asyncio
async def test_generate_ai_response(mock_openai_client):
    """Test AI response generation"""
    text = "Hello, how are you?"
    response = await generate_ai_response(text, mock_openai_client)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_text_to_speech(mock_openai_client):
    """Test text-to-speech conversion"""
    text = "Hello, this is a test."
    await text_to_speech(text, mock_openai_client, speed=1.25)
    # Success is indicated by no exception being raised 