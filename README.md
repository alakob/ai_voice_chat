# Voice Chat

A real-time voice interaction system that converts speech to text, generates AI responses, and provides text-to-speech output using OpenAI's APIs.

## Features

- 🎤 Real-time voice recording
- 🔄 Speech-to-text conversion
- 🤖 AI-powered responses
- 🔊 Text-to-speech playback
- 📊 Status monitoring and logging
- 🌐 Web-based interface

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- OpenAI API key
- Google API key (for speech recognition fallback)

## Installation

1. Clone the repository:
```
bash
git clone https://github.com/alakob/ai_voice_chat.git
cd ai_voice_chat
```
2. Create and activate a virtual environment:

```
bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HF_TOKEN=your_huggingface_token
GEMINI_APIKEY=your_gemini_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## Project Structure
```
src/
├── voice_assistant/
│ ├── init.py
│ ├── config.py # Configuration and environment settings
│ ├── models.py # Data models and schemas
│ ├── exceptions.py # Custom exception definitions
│ ├── state.py # Global state management
│ ├── services/
│ │ ├── init.py
│ │ ├── audio_service.py # Audio processing functionality
│ │ └── openai_service.py # OpenAI API integration
│ └── ui/
│ ├── init.py
│ └── gradio_interface.py # Web interface components
├── main.py # Application entry point
├── requirements.txt # Project dependencies
└── .env # Environment variables
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:7860`)

3. Use the interface:
   - Click "Start Recording" to begin voice capture
   - Speak clearly into your microphone
   - Click "Stop Recording" when finished
   - Wait for the AI response
   - Listen to the spoken response
   - Use "Stop Audio" to interrupt playback

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project follows PEP 8 guidelines. Format code using:
```bash
black src/
```

### Type Checking
```bash
mypy src/
```

## API Documentation

### Audio Service
- `start_recording()`: Initiates audio capture
- `play_audio()`: Handles audio playback
- `process_audio()`: Processes recorded audio

### OpenAI Service
- `generate_ai_response()`: Creates AI responses
- `text_to_speech()`: Converts text to speech

## Configuration

Key settings in `config.py`:
- Audio sample rate: 16000 Hz
- Audio channels: 1 (mono)
- Audio format: float32
- Model settings: GPT-4 for responses, TTS-1 for speech

## Error Handling

The application includes custom exceptions:
- `AudioProcessingError`
- `TranscriptionError`
- `TTSError`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT and TTS APIs
- Gradio for the web interface
- SoundDevice for audio processing

## Contact

Your Name - blaisealako@gmail.com
Project Link: https://github.com/alakob/ai_voice_chat