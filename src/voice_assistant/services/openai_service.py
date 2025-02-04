"""OpenAI service for AI responses and TTS"""
from openai import AsyncOpenAI
from ..config import settings
from ..exceptions import TTSError
from ..models import TTSRequest
import logging
from io import BytesIO
import soundfile as sf
import sounddevice as sd

logger = logging.getLogger(__name__)

async def generate_ai_response(text: str, client: AsyncOpenAI) -> str:
    """Generate AI response using OpenAI"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Format responses in markdown."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        raise

async def text_to_speech(text: str, client: AsyncOpenAI, speed: float = 1.0) -> None:
    """Convert text to speech using OpenAI's TTS
    
    Args:
        text: Text to convert to speech
        client: OpenAI client instance
        speed: Playback speed multiplier (default: 1.0)
    """
    try:
        response = await client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            speed=speed
        )
        
        # Convert response to audio data
        audio_data = BytesIO(response.content)
        data, samplerate = sf.read(audio_data)
        
        # Use the improved play_audio function
        from .audio_service import play_audio
        await play_audio(data)
        
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        raise TTSError(f"TTS failed: {str(e)}") 