"""
Audio processing routes - TTS, STT, and file upload
"""
import logging
import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
import assemblyai as aai
from murf import Murf

from config import config
from utils import (
    validate_audio_file, log_request_info, log_response_info
)
from schemas import (
    TTSRequest, TTSResponse, TranscriptionResponse, UploadResponse,
    EchoResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["audio"])


@router.post("/text-to-speech", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest) -> TTSResponse:
    """
    Convert text to speech using Murf TTS service
    
    Args:
        request: TTS request with text and voice_id
        
    Returns:
        TTSResponse with audio URL or error information
    """
    log_request_info("text_to_speech", text_length=len(request.text), voice_id=request.voice_id)
    
    try:
        if not config.MURF_API_KEY:
            return TTSResponse(
                error="Voice service temporarily unavailable",
                fallback_text=request.text
            )
        
        client = Murf(api_key=config.MURF_API_KEY)
        res = client.text_to_speech.generate(
            text=request.text.strip(),
            voice_id=request.voice_id,
        )
        
        log_response_info("text_to_speech", success=True)
        
        return TTSResponse(
            audio_url=res.audio_file,
            success=True
        )
        
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        log_response_info("text_to_speech", success=False, error=str(e))
        
        return TTSResponse(
            error=f"Voice generation failed: {str(e)}",
            fallback_text=request.text,
            use_browser_tts=True
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload audio file to server
    
    Args:
        file: Audio file to upload
        
    Returns:
        UploadResponse with file information
    """
    log_request_info("upload", filename=file.filename, content_type=file.content_type)
    
    try:
        file_location = os.path.join(config.UPLOAD_FOLDER, file.filename)

        # Save file to uploads folder
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get file size
        file_size = os.path.getsize(file_location)
        
        log_response_info("upload", success=True, file_size=file_size)

        return UploadResponse(
            filename=file.filename,
            content_type=file.content_type,
            size=f"{file_size} bytes",
            status="Upload successful"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        log_response_info("upload", success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)) -> TranscriptionResponse:
    """
    Transcribe audio file to text using AssemblyAI
    
    Args:
        file: Audio file to transcribe
        
    Returns:
        TranscriptionResponse with transcribed text
    """
    log_request_info("transcribe_file", filename=file.filename, content_type=file.content_type)
    
    # Validate file
    file_content = await file.read()
    validation_error = validate_audio_file(file.content_type, len(file_content))
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    temp_file_path: Optional[str] = None
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        if not config.ASSEMBLYAI_API_KEY:
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")

        # Transcribe using AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")

        result_text = transcript.text or "No speech detected in audio"
        log_response_info("transcribe_file", success=True, transcript_length=len(result_text))
        
        return TranscriptionResponse(transcript=result_text)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        log_response_info("transcribe_file", success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")


@router.post("/tts/echo", response_model=EchoResponse)
async def tts_echo(file: UploadFile = File(...)) -> EchoResponse:
    """
    Echo bot endpoint that transcribes audio and converts it back to speech using Murf TTS
    
    Args:
        file: Audio file to echo
        
    Returns:
        EchoResponse with transcript and echo audio URL
    """
    log_request_info("tts_echo", filename=file.filename, content_type=file.content_type)
    
    # Validate file
    file_content = await file.read()
    validation_error = validate_audio_file(file.content_type, len(file_content))
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    temp_file_path: Optional[str] = None
    
    try:
        # Save file temporarily for transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Check API keys
        if not config.ASSEMBLYAI_API_KEY:
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")
        
        if not config.MURF_API_KEY:
            raise HTTPException(status_code=500, detail="Murf API key not configured")

        # Step 1: Transcribe the audio
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
        
        transcribed_text = transcript.text
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Step 2: Generate TTS using Murf
        client = Murf(api_key=config.MURF_API_KEY)
        tts_response = client.text_to_speech.generate(
            text=transcribed_text.strip(),
            voice_id=config.DEFAULT_VOICE_ID,
        )
        
        log_response_info("tts_echo", success=True, transcript_length=len(transcribed_text))
        
        return EchoResponse(
            transcript=transcribed_text,
            audio_url=tts_response.audio_file
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Echo processing error: {e}")
        log_response_info("tts_echo", success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"Echo processing error: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")