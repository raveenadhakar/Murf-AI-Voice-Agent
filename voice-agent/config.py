"""
Application configuration
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class"""
    
    # API Keys
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MURF_API_KEY = os.getenv("MURF_API_KEY")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # File Upload Configuration
    UPLOAD_FOLDER = "uploads"
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    ALLOWED_AUDIO_TYPES = [
        "audio/webm", 
        "audio/wav", 
        "audio/mp3", 
        "audio/m4a", 
        "audio/ogg"
    ]
    
    # AI Model Configuration
    GEMINI_MODELS = [
        'gemini-1.5-flash',
        'gemini-1.5-pro', 
        'gemini-pro',
        'models/gemini-1.5-flash'
    ]
    
    # TTS Configuration
    DEFAULT_VOICE_ID = "en-US-julia"
    MAX_TTS_LENGTH = 3000
    TTS_CHUNK_SIZE = 2800
    
    # Response Configuration
    MAX_RESPONSE_LENGTH = 2500
    CONTEXT_MESSAGES_LIMIT = 10
    
    @classmethod
    def get_service_status(cls) -> Dict[str, bool]:
        """Get status of all external services"""
        return {
            "assemblyai": bool(cls.ASSEMBLYAI_API_KEY),
            "gemini": bool(cls.GEMINI_API_KEY),
            "murf": bool(cls.MURF_API_KEY)
        }
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        missing_keys = []
        warnings = []
        
        if not cls.ASSEMBLYAI_API_KEY:
            missing_keys.append("ASSEMBLYAI_API_KEY")
            warnings.append("Speech-to-text service will be unavailable")
            
        if not cls.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
            warnings.append("AI conversation service will be unavailable")
            
        if not cls.MURF_API_KEY:
            missing_keys.append("MURF_API_KEY")
            warnings.append("Text-to-speech service will use browser fallback")
        
        return {
            "valid": len(missing_keys) == 0,
            "missing_keys": missing_keys,
            "warnings": warnings
        }


# Global config instance
config = Config()