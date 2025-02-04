"""Custom exceptions for the voice assistant"""
class AudioProcessingError(Exception):
    """Raised when there's an error processing audio"""
    pass

class TranscriptionError(Exception):
    """Raised when speech-to-text conversion fails"""
    pass

class TTSError(Exception):
    """Raised when text-to-speech conversion fails"""
    pass 