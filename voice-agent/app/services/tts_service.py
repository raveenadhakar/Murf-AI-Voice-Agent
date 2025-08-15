"""
Text-to-Speech service using Murf AI
"""
import os
import logging
from typing import Optional
from murf import Murf
from app.schemas import TTSResponse

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service wrapper"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize TTS service with API key"""
        self.api_key = api_key or os.getenv("MURF_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                self.client = Murf(api_key=self.api_key)
                logger.info("TTS service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS service: {e}")
        else:
            logger.warning("TTS service initialized without API key")
    
    def is_available(self) -> bool:
        """Check if TTS service is available"""
        return bool(self.api_key and self.client)
    
    def _split_text_for_tts(self, text: str, max_length: int = 2800) -> List[str]:
        """
        Split text into chunks that respect sentence boundaries
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
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
    
    async def text_to_speech(
        self, 
        text: str, 
        voice_id: str = "en-US-julia"
    ) -> TTSResponse:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use for TTS
            
        Returns:
            TTSResponse with audio URL or error information
        """
        # Input validation
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return TTSResponse(
                success=False,
                error="Text cannot be empty",
                fallback_text=text
            )
        
        text = text.strip()
        
        if len(text) < 3:
            logger.warning("Text too short for TTS")
            return TTSResponse(
                success=False,
                error="Text must be at least 3 characters long",
                fallback_text=text
            )
        
        if not self.is_available():
            logger.error("TTS service not available")
            return TTSResponse(
                success=False,
                error="Voice service temporarily unavailable",
                fallback_text=text,
                use_browser_tts=True
            )
        
        try:
            # Handle long text by splitting or truncating
            if len(text) > 3000:
                logger.info(f"Text too long ({len(text)} chars), splitting for TTS")
                chunks = self._split_text_for_tts(text)
                # Use first chunk for TTS
                tts_text = chunks[0] + "..." if len(chunks) > 1 else chunks[0]
            else:
                tts_text = text
            
            logger.info(f"Converting text to speech: {len(tts_text)} characters")
            
            # Generate speech using Murf
            response = self.client.text_to_speech.generate(
                text=tts_text,
                voice_id=voice_id
            )
            
            if not response.audio_file:
                logger.error("No audio file returned from Murf")
                return TTSResponse(
                    success=False,
                    error="Failed to generate audio",
                    fallback_text=text,
                    use_browser_tts=True
                )
            
            logger.info(f"TTS successful: {response.audio_file}")
            return TTSResponse(
                audio_url=response.audio_file,
                success=True
            )
            
        except Exception as e:
            logger.error(f"TTS service error: {str(e)}")
            return TTSResponse(
                success=False,
                error=f"Voice generation failed: {str(e)}",
                fallback_text=text,
                use_browser_tts=True
            )


# Global TTS service instance
tts_service = TTSService()