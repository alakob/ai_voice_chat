"""Main entry point for the voice assistant"""
import logging
from voice_assistant.ui.gradio_interface import create_interface
from voice_assistant.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting Voice AI Assistant")
        interface = create_interface()
        interface.launch(share=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 