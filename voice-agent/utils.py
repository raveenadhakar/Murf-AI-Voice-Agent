"""
Utility functions and helpers
"""
import logging
import os
from typing import Dict, Any, Optional
from config import config
from schemas import ErrorTestResponse


def setup_logging() -> None:
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_upload_folder() -> None:
    """Create upload folder if it doesn't exist"""
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)


def validate_audio_file(content_type: str, file_size: int) -> Optional[str]:
    """
    Validate audio file type and size
    
    Args:
        content_type: MIME type of the file
        file_size: Size of the file in bytes
        
    Returns:
        Error message if validation fails, None if valid
    """
    if content_type not in config.ALLOWED_AUDIO_TYPES:
        return f"Unsupported file type. Allowed types: {', '.join(config.ALLOWED_AUDIO_TYPES)}"
    
    if file_size > config.MAX_FILE_SIZE:
        return f"File too large. Maximum size is {config.MAX_FILE_SIZE // (1024*1024)}MB"
    
    if file_size == 0:
        return "Empty file uploaded"
    
    return None


def split_text_for_tts(text: str, max_length: int = None) -> list[str]:
    """
    Split text into chunks that respect sentence boundaries for TTS
    
    Args:
        text: Text to split
        max_length: Maximum length per chunk
        
    Returns:
        List of text chunks
    """
    if max_length is None:
        max_length = config.TTS_CHUNK_SIZE
        
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence + '. ') <= max_length:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Fallback responses for different error scenarios
FALLBACK_RESPONSES = {
    "stt_error": "I'm having trouble understanding your audio right now. Could you try speaking again?",
    "llm_error": "I'm experiencing some technical difficulties. Let me try to help you anyway.",
    "tts_error": "I can respond, but I'm having trouble with voice generation right now.",
    "api_key_missing": "Some services are temporarily unavailable. Please try again later.",
    "no_speech": "I didn't detect any speech in your audio. Could you try recording again?",
    "general_error": "Something went wrong, but I'm here to help. Please try again."
}


def get_fallback_response(error_type: str, custom_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate fallback response for different error scenarios
    
    Args:
        error_type: Type of error
        custom_message: Custom error message to use instead of default
        
    Returns:
        Fallback response dictionary
    """
    message = custom_message or FALLBACK_RESPONSES.get(error_type, FALLBACK_RESPONSES["general_error"])
    
    return {
        "llm_response": message,
        "audio_url": None,
        "tts_error": True,
        "fallback": True,
        "error_type": error_type
    }


def create_error_test_response(error_type: str) -> ErrorTestResponse:
    """
    Create error test response for debugging
    
    Args:
        error_type: Type of error to simulate
        
    Returns:
        ErrorTestResponse object
    """
    fallback_data = get_fallback_response(error_type)
    
    return ErrorTestResponse(
        llm_response=fallback_data["llm_response"],
        audio_url=fallback_data["audio_url"],
        tts_error=fallback_data["tts_error"],
        fallback=fallback_data["fallback"],
        error_type=error_type
    )


def log_request_info(endpoint: str, **kwargs) -> None:
    """Log request information"""
    logger = logging.getLogger(__name__)
    extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Request to {endpoint} {extra_info}")


def log_response_info(endpoint: str, success: bool = True, **kwargs) -> None:
    """Log response information"""
    logger = logging.getLogger(__name__)
    status = "SUCCESS" if success else "ERROR"
    extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Response from {endpoint} [{status}] {extra_info}")