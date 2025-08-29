"""
Chat and conversation routes
"""
import logging
import os
import tempfile
from typing import Optional, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File
import assemblyai as aai
import google.generativeai as genai
from murf import Murf

from config import config
from utils import (
    validate_audio_file, split_text_for_tts, get_fallback_response,
    log_request_info, log_response_info
)
from schemas import (
    AgentChatResponse, ChatHistoryResponse, ClearChatResponse,
    LLMQueryResponse, ChatMessage, MessageRole
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# Chat sessions storage
chat_sessions: Dict[str, List[Dict[str, str]]] = {}


@router.post("/agent/chat/{session_id}", response_model=AgentChatResponse)
async def agent_chat(session_id: str, file: UploadFile = File(...)) -> AgentChatResponse:
    """
    Conversational AI endpoint with chat history and robust error handling
    
    Args:
        session_id: Unique session identifier
        file: Audio file with user's message
        
    Returns:
        AgentChatResponse with AI response and audio
    """
    log_request_info("agent_chat", session_id=session_id, filename=file.filename)
    
    temp_file_path: Optional[str] = None
    transcribed_text: Optional[str] = None
    
    try:
        # Validate file
        file_content = await file.read()
        validation_error = validate_audio_file(file.content_type, len(file_content))
        if validation_error:
            fallback = get_fallback_response("general_error", 
                "I can only process audio files. Please try recording again.")
            return _create_agent_response(session_id, fallback, transcribed_text)
        
        # Save file temporarily for transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Step 1: Transcribe the audio with error handling
        try:
            if not config.ASSEMBLYAI_API_KEY:
                fallback = get_fallback_response("api_key_missing", 
                    "Speech recognition is temporarily unavailable. Please try again later.")
                return _create_agent_response(session_id, fallback, transcribed_text)
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(temp_file_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                fallback = get_fallback_response("stt_error", 
                    f"I couldn't understand your audio. {transcript.error}")
                return _create_agent_response(session_id, fallback, transcribed_text)
            
            transcribed_text = transcript.text
            if not transcribed_text or not transcribed_text.strip():
                fallback = get_fallback_response("no_speech")
                return _create_agent_response(session_id, fallback, transcribed_text)
                
        except Exception as e:
            logger.error(f"STT Error: {e}")
            fallback = get_fallback_response("stt_error", 
                "I'm having trouble with speech recognition right now. Please try again.")
            return _create_agent_response(session_id, fallback, transcribed_text)

        # Step 2: Get or create chat history for this session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_history = chat_sessions[session_id]
        
        # Add user message to history
        user_message = {"role": "user", "content": transcribed_text.strip()}
        chat_history.append(user_message)
        
        # Step 3: Build context for LLM with chat history
        try:
            context_messages = []
            for msg in chat_history[-config.CONTEXT_MESSAGES_LIMIT:]:  # Keep last N messages for context
                if msg["role"] == "user":
                    context_messages.append(f"User: {msg['content']}")
                else:
                    context_messages.append(f"Assistant: {msg['content']}")
            
            # Create prompt with conversation context
            conversation_context = "\n".join(context_messages[:-1])  # Exclude current message
            current_question = transcribed_text.strip()
            
            if conversation_context:
                prompt = f"""Previous conversation:
{conversation_context}

Current question: {current_question}

Please provide a natural, conversational response that takes into account our previous conversation. Keep your response concise (under {config.MAX_RESPONSE_LENGTH} characters) and engaging."""
            else:
                prompt = f"Please provide a concise, engaging response (under {config.MAX_RESPONSE_LENGTH} characters) to this question: {current_question}"
            
        except Exception as e:
            logger.error(f"Context building error: {e}")
            prompt = f"Please provide a concise response to: {transcribed_text.strip()}"

        # Step 4: Send to LLM with error handling
        response_text = None
        try:
            if not config.GEMINI_API_KEY:
                response_text = "I'm temporarily unable to process your request. My AI services are currently unavailable."
            else:
                model = None
                for model_name in config.GEMINI_MODELS:
                    try:
                        model = genai.GenerativeModel(model_name)
                        break
                    except Exception as model_error:
                        logger.warning(f"Failed to initialize model {model_name}: {model_error}")
                        continue
                
                if not model:
                    response_text = "I'm having trouble accessing my AI capabilities right now. Please try again in a moment."
                else:
                    llm_response = model.generate_content(prompt)
                    
                    if not llm_response.text:
                        response_text = "I'm drawing a blank right now. Could you rephrase your question?"
                    else:
                        response_text = llm_response.text.strip()
                        
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            response_text = get_fallback_response("llm_error")["llm_response"]
        
        # Add assistant response to history
        assistant_message = {"role": "assistant", "content": response_text}
        chat_history.append(assistant_message)
        
        # Step 5: Generate TTS using Murf with error handling
        audio_url = None
        tts_error = False
        
        try:
            if not config.MURF_API_KEY:
                tts_error = True
            else:
                client = Murf(api_key=config.MURF_API_KEY)
                
                if len(response_text) > config.MAX_TTS_LENGTH:
                    # Split into chunks and use first chunk only for now
                    chunks = split_text_for_tts(response_text)
                    tts_text = chunks[0] + "..." if len(chunks) > 1 else chunks[0]
                else:
                    tts_text = response_text
                
                tts_response = client.text_to_speech.generate(
                    text=tts_text,
                    voice_id=config.DEFAULT_VOICE_ID,
                )
                audio_url = tts_response.audio_file
                
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            tts_error = True
        
        log_response_info("agent_chat", success=True, session_id=session_id, 
                         message_count=len(chat_history), tts_error=tts_error)
        
        return AgentChatResponse(
            session_id=session_id,
            transcript=transcribed_text,
            llm_response=response_text,
            audio_url=audio_url,
            truncated=len(response_text) > config.MAX_TTS_LENGTH,
            message_count=len(chat_history),
            tts_error=tts_error,
            fallback=False
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in agent_chat: {e}")
        log_response_info("agent_chat", success=False, session_id=session_id, error=str(e))
        
        # Return fallback response instead of raising exception
        fallback = get_fallback_response("general_error")
        return _create_agent_response(session_id, fallback, transcribed_text)
        
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")


@router.get("/agent/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """
    Get chat history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        ChatHistoryResponse with message history
    """
    log_request_info("get_chat_history", session_id=session_id)
    
    if session_id not in chat_sessions:
        return ChatHistoryResponse(
            session_id=session_id, 
            messages=[], 
            message_count=0
        )
    
    messages = [
        ChatMessage(role=MessageRole(msg["role"]), content=msg["content"])
        for msg in chat_sessions[session_id]
    ]
    
    log_response_info("get_chat_history", success=True, session_id=session_id, 
                     message_count=len(messages))
    
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        message_count=len(messages)
    )


@router.delete("/agent/chat/{session_id}", response_model=ClearChatResponse)
async def clear_chat_history(session_id: str) -> ClearChatResponse:
    """
    Clear chat history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        ClearChatResponse with operation status
    """
    log_request_info("clear_chat_history", session_id=session_id)
    
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    
    log_response_info("clear_chat_history", success=True, session_id=session_id)
    
    return ClearChatResponse(session_id=session_id, status="cleared")


@router.post("/llm/query", response_model=LLMQueryResponse)
async def llm_query(file: UploadFile = File(...)) -> LLMQueryResponse:
    """
    LLM endpoint that accepts audio input, transcribes it, sends to LLM, and returns spoken response
    
    Args:
        file: Audio file with user query
        
    Returns:
        LLMQueryResponse with transcript, AI response, and audio
    """
    log_request_info("llm_query", filename=file.filename, content_type=file.content_type)
    
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
        
        if not config.GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
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
        
        # Step 2: Send to LLM
        model = None
        for model_name in config.GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception:
                continue
        
        if not model:
            raise HTTPException(status_code=500, detail="Failed to initialize Gemini model")
        
        # Add instruction to keep responses concise for TTS
        prompt = f"Please provide a concise response (under {config.MAX_RESPONSE_LENGTH} characters) to this question: {transcribed_text.strip()}"
        llm_response = model.generate_content(prompt)
        
        if not llm_response.text:
            raise HTTPException(status_code=500, detail="No response generated from LLM")
        
        response_text = llm_response.text.strip()
        
        # Step 3: Generate TTS using Murf
        client = Murf(api_key=config.MURF_API_KEY)
        
        if len(response_text) > config.MAX_TTS_LENGTH:
            # Split into chunks and use first chunk only for now
            chunks = split_text_for_tts(response_text)
            tts_text = chunks[0] + "..." if len(chunks) > 1 else chunks[0]
        else:
            tts_text = response_text
        
        tts_response = client.text_to_speech.generate(
            text=tts_text,
            voice_id=config.DEFAULT_VOICE_ID,
        )
        
        log_response_info("llm_query", success=True, transcript_length=len(transcribed_text),
                         response_length=len(response_text))
        
        return LLMQueryResponse(
            transcript=transcribed_text,
            llm_response=response_text,
            audio_url=tts_response.audio_file,
            truncated=len(response_text) > config.MAX_TTS_LENGTH
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM query processing error: {e}")
        log_response_info("llm_query", success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"LLM query processing error: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")


def _create_agent_response(session_id: str, fallback_data: dict, transcribed_text: Optional[str]) -> AgentChatResponse:
    """Helper function to create AgentChatResponse from fallback data"""
    message_count = len(chat_sessions.get(session_id, []))
    
    return AgentChatResponse(
        session_id=session_id,
        transcript=transcribed_text or "Audio processing failed",
        llm_response=fallback_data["llm_response"],
        audio_url=fallback_data["audio_url"],
        truncated=False,
        message_count=message_count,
        tts_error=fallback_data["tts_error"],
        fallback=fallback_data["fallback"]
    )