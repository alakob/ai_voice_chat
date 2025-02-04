"""Gradio-based user interface for the voice assistant.

Provides:
- Recording controls
- Speech-to-text display
- AI response display
- Audio playback controls
"""
import gradio as gr
from ..services import audio_service, openai_service
from ..state import audio_state
from ..config import settings
from ..models import AudioConfig
from openai import AsyncOpenAI
import asyncio
import logging
import numpy as np
import soundfile as sf
import tempfile
import speech_recognition as sr
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

class AudioState:
    """Manages global audio processing state
    
    Attributes:
        is_recording (bool): Current recording status
        is_playing (bool): Current playback status
        stop_event (Optional[asyncio.Event]): Event for stopping audio
        current_audio (Optional[np.ndarray]): Currently processing audio
        current_position (int): Position in audio stream
        audio_data (List): Collected audio data chunks
    """

def create_interface():
    """Creates and configures the Gradio interface
    
    Components:
    - Status display
    - Recording buttons
    - Text input/output areas
    - Audio control buttons
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    def safe_start_recording():
        """Synchronous wrapper for start_recording"""
        try:
            audio_config = AudioConfig()
            asyncio.run(audio_service.start_recording(audio_config))
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
                
                if not audio_state.audio_data:
                    return (
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "No audio recorded",
                        "",
                        "No audio recorded"
                    )
                
                # First, process and display the recognized text
                audio_data = np.concatenate(audio_state.audio_data)
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, settings.sample_rate)
                    recognizer = sr.Recognizer()
                    
                    try:
                        # Step 1: Recognize and display text immediately
                        with sr.AudioFile(temp_file.name) as source:
                            audio = recognizer.record(source)
                            text = recognizer.recognize_google(audio)
                            
                            # Yield immediate update for recognized text
                            yield (
                                gr.update(visible=True),
                                gr.update(visible=False),
                                "Processing AI response...",
                                text,
                                "Generating response..."
                            )
                            
                            # Step 2: Generate AI response
                            client = AsyncOpenAI(api_key=settings.openai_api_key)
                            response = asyncio.run(openai_service.generate_ai_response(text, client))
                            
                            # Step 3: Update UI with response and start TTS
                            yield (
                                gr.update(visible=True),
                                gr.update(visible=False),
                                "Playing response...",
                                text,
                                response
                            )
                            
                            # Step 4: Play TTS with increased speed
                            asyncio.run(openai_service.text_to_speech(
                                text=response,
                                client=client,
                                speed=1  # Increased playback speed
                            ))
                            
                            # Final update
                            return (
                                gr.update(visible=True),
                                gr.update(visible=False),
                                "Ready",
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
        """Stop audio playback"""
        try:
            from ..services.audio_service import stop_audio_playback
            if stop_audio_playback():
                return "Audio playback stopped"
            return "No audio playing"
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            return f"Error stopping audio: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Voice Chat")
        
        with gr.Row():
            with gr.Column():
                status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                text_input = gr.Textbox(
                    label="Recognized Speech",
                    placeholder="Your speech will appear here...",
                    interactive=True
                )
                response_text = gr.Markdown(
                    label="AI Response",
                    value="Waiting for input..."
                )
        
        with gr.Row():
            record_button = gr.Button("🎤 Start Recording", variant="primary")
            stop_button = gr.Button("⏹️ Stop Recording", variant="stop", visible=False)
            stop_audio_button = gr.Button("🔇 Stop Audio", variant="secondary")

        # Event handlers
        record_event = record_button.click(
            fn=safe_start_recording,
            outputs=[record_button, stop_button, status],
            queue=False  # Immediate UI update
        )
        
        stop_event = stop_button.click(
            fn=safe_stop_recording,
            outputs=[record_button, stop_button, status, text_input, response_text],
            queue=True  # Enable queuing for progressive updates
        )
        
        stop_audio_button.click(
            fn=safe_stop_audio,
            outputs=[status],
            queue=False,
            cancels=[stop_event]  # Cancel the stop recording event instead of using string
        )

        return interface 

# Initialize and start the voice assistant
from voice_assistant.ui.gradio_interface import create_interface

def main():
    """Main entry point for the voice assistant application"""
    try:
        interface = create_interface()
        interface.launch(share=True)
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        raise 