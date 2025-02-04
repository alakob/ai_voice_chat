"""
Voice interaction module for the Technical QA System.
Handles voice input/output and integration with OpenAI's APIs.
"""

from typing import Tuple, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import gradio as gr
import speech_recognition as sr
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import logging
import asyncio
import tempfile
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from pydantic import BaseModel
from fastapi import HTTPException, Depends
import numpy.typing as npt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_interaction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom Exceptions
class AudioProcessingError(Exception):
    """Raised when there's an error processing audio"""
    pass

class TranscriptionError(Exception):
    """Raised when speech-to-text conversion fails"""
    pass

class TTSError(Exception):
    """Raised when text-to-speech conversion fails"""
    pass

# Pydantic Models
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

# Global state management
class AudioState:
    """Global state for audio processing"""
    def __init__(self):
        self.is_recording: bool = False
        self.is_playing: bool = False
        self.stop_event: Optional[asyncio.Event] = None
        self.current_audio: Optional[npt.NDArray] = None
        self.current_position: int = 0

audio_state = AudioState()

# Dependencies
async def get_openai_client() -> AsyncOpenAI:
    """FastAPI dependency for OpenAI client"""
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_speech_recognizer() -> sr.Recognizer:
    """FastAPI dependency for speech recognizer"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    return recognizer

# Core functionality
async def start_recording(
    audio_config: AudioConfig = AudioConfig()
) -> Tuple[gr.update, gr.update, str]:
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

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Recording... Speak clearly into the microphone"
        )

    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        raise AudioProcessingError(f"Failed to start recording: {str(e)}")

async def process_audio(
    audio_data: List[npt.NDArray],
    client: AsyncOpenAI = Depends(get_openai_client),
    recognizer: sr.Recognizer = Depends(get_speech_recognizer)
) -> TranscriptionResult:
    """Process recorded audio and convert to text"""
    if not audio_data:
        raise AudioProcessingError("No audio data recorded")

    try:
        # Combine and normalize audio
        combined_audio = np.concatenate(audio_data)
        normalized_audio = combined_audio / np.max(np.abs(combined_audio))

        # Save temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_path = temp_file.name
            sf.write(audio_path, normalized_audio, AudioConfig().sample_rate)

            try:
                # Try OpenAI Whisper
                with open(audio_path, "rb") as audio_file:
                    transcript = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                    return TranscriptionResult(text=transcript.text)
            except Exception as whisper_error:
                logger.error(f"Whisper transcription failed: {str(whisper_error)}")
                
                # Fallback to Google Speech Recognition
                with sr.AudioFile(audio_path) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
                    return TranscriptionResult(text=text)
    finally:
        if 'audio_path' in locals():
            os.unlink(audio_path)

async def generate_ai_response(
    text: str,
    client: AsyncOpenAI = Depends(get_openai_client)
) -> str:
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
        raise HTTPException(status_code=500, detail=str(e))

async def text_to_speech(text: str, client: AsyncOpenAI) -> None:
    """Convert text to speech using OpenAI's TTS"""
    try:
        response = await client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            speed=1.0
        )
        
        # Convert response to audio data
        audio_data = BytesIO(response.content)
        data, samplerate = sf.read(audio_data)
        
        # Play the audio
        sd.play(data, samplerate)
        sd.wait()
        
        except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        raise TTSError(f"TTS failed: {str(e)}")

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
        raise TTSError(f"Failed to play audio: {str(e)}")
    finally:
        audio_state.is_playing = False

# Gradio Interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    # Add state management for audio recording
    audio_data = []
    
    def safe_start_recording():
        """Synchronous wrapper for start_recording"""
        try:
            audio_state.is_recording = True
            audio_state.audio_data = []
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio recording status: {status}")
                if audio_state.is_recording:
                    audio_state.audio_data.append(indata.copy())
            
            stream = sd.InputStream(
                channels=1,
                samplerate=16000,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=1024
            )
            stream.start()
            audio_state.audio_stream = stream
            
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "Recording... Speak clearly into the microphone"
            )
        except Exception as e:
            logger.error(f"Error in start_recording: {e}")
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                f"Error: {str(e)}"
            )
    
    def safe_stop_recording():
        """Synchronous wrapper for stop_recording"""
        try:
            if hasattr(audio_state, 'audio_stream'):
                audio_state.is_recording = False
                audio_state.audio_stream.stop()
                audio_state.audio_stream.close()
                
                # Process the recorded audio
                if not audio_state.audio_data:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "No audio recorded",
                    "",
                        "No audio recorded"
                )
                
                # Process audio synchronously
                audio_data = np.concatenate(audio_state.audio_data)
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, 16000)
                    recognizer = get_speech_recognizer()
                    
                    try:
                        with sr.AudioFile(temp_file.name) as source:
                            audio = recognizer.record(source)
                            text = recognizer.recognize_google(audio)
                            
                            # Get API key from environment
                            api_key = os.getenv("OPENAI_API_KEY")
                            if not api_key:
                                raise ValueError("OPENAI_API_KEY not found in environment variables")
                            
                            # Generate response synchronously
                            client = AsyncOpenAI(api_key=api_key)
                            response = asyncio.run(generate_ai_response(text, client))
                            
                            # Use OpenAI TTS to speak the response
                            asyncio.run(text_to_speech(response, client))
                        
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            "Processing complete",
                            text,
                                response
                            )
                    finally:
                        os.unlink(temp_file.name)
                
        except Exception as e:
            logger.error(f"Error in stop_recording: {e}")
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                f"Error: {str(e)}",
                "",
                f"Error: {str(e)}"
            )
    
    def safe_stop_audio():
        """Synchronous wrapper for stop_audio"""
        try:
            if audio_state.is_playing:
                audio_state.is_playing = False
                if hasattr(audio_state, 'stop_event'):
                    audio_state.stop_event.set()
                return "Audio playback stopped"
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            return f"Error: {str(e)}"
        
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
        # Interface components
        status = gr.Textbox(label="Status", value="Ready", interactive=False)
        text_input = gr.Textbox(label="Recognized Speech", interactive=True)
            
            with gr.Row():
                record_button = gr.Button("🎤 Start Recording")
                stop_button = gr.Button("⏹️ Stop Recording", visible=False)
                stop_audio_button = gr.Button("🔇 Stop Audio")
            
        response_text = gr.Markdown(label="AI Response")
            
            # Event handlers
            record_button.click(
            fn=safe_start_recording,
                outputs=[record_button, stop_button, status]
            )
            
            stop_button.click(
            fn=safe_stop_recording,
                outputs=[record_button, stop_button, status, text_input, response_text]
            )
            
            stop_audio_button.click(
            fn=safe_stop_audio,
            outputs=[status]
        )

            return interface

if __name__ == "__main__":
    try:
        logger.info("Starting Voice AI Assistant")
        interface = create_interface()
        interface.launch(share=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise