"""
AI Voice Chat Assistant - Main FastAPI Application
"""
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Third-party imports
import assemblyai as aai
import google.generativeai as genai
from murf import Murf

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
UPLOAD_FOLDER = "uploads"

# Configure APIs
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chat sessions storage
chat_sessions = {}

# Fallback responses for error scenarios
FALLBACK_RESPONSES = {
    "stt_error": "I'm having trouble understanding your audio right now. Could you try speaking again?",
    "llm_error": "I'm experiencing some technical difficulties. Let me try to help you anyway.",
    "tts_error": "I can respond, but I'm having trouble with voice generation right now.",
    "api_key_missing": "Some services are temporarily unavailable. Please try again later.",
    "no_speech": "I didn't detect any speech in your audio. Could you try recording again?",
    "general_error": "Something went wrong, but I'm here to help. Please try again."
}

def get_fallback_response(error_type, custom_message=None):
    """Generate fallback response for different error scenarios"""
    message = custom_message or FALLBACK_RESPONSES.get(error_type, FALLBACK_RESPONSES["general_error"])
    
    return {
        "llm_response": message,
        "audio_url": None,
        "tts_error": True,
        "fallback": True,
        "error_type": error_type
    }

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Chat Assistant",
    description="A conversational AI voice assistant with modern chat interface",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Serve index.html at root path
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# API Models
class TTSRequest(BaseModel):
    text: str
    voice_id: str

class LLMRequest(BaseModel):
    text: str

# API Route with validation
@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    try:
        # Input validation
        if not request.text or not request.text.strip():
            return {"error": "Text cannot be empty", "fallback_text": request.text}
        
        if len(request.text.strip()) < 3:
            return {"error": "Text must be at least 3 characters long", "fallback_text": request.text}
        
        if len(request.text) > 1000:
            return {"error": "Text must be less than 1000 characters", "fallback_text": request.text}
        
        if not MURF_API_KEY:
            return {"error": "Voice service temporarily unavailable", "fallback_text": request.text}
        
        client = Murf(api_key=MURF_API_KEY)
        res = client.text_to_speech.generate(
            text=request.text.strip(),
            voice_id=request.voice_id,
        )
        return {"audio_url": res.audio_file, "success": True}
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return {
            "error": f"Voice generation failed: {str(e)}", 
            "fallback_text": request.text,
            "use_browser_tts": True
        }

# New Upload Endpoint
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file to uploads folder
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get file size
        file_size = os.path.getsize(file_location)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": f"{file_size} bytes",
            "status": "Upload successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    # Validate file type
    allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Check file size (max 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    
    try:
        # Save file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            file_content = await file.read()
            
            if len(file_content) > max_size:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 25MB")
            
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        if not ASSEMBLYAI_API_KEY:
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")

        # Use file path instead of raw bytes
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")

        return {"transcript": transcript.text or "No speech detected in audio"}
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/llm/query")
async def llm_query(file: UploadFile = File(...)):
    """
    LLM endpoint that accepts audio input, transcribes it, sends to LLM, and returns spoken response
    """
    # Validate file type
    allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Check file size (max 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    
    try:
        # Save file temporarily for transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            file_content = await file.read()
            
            if len(file_content) > max_size:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 25MB")
            
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Check API keys
        if not ASSEMBLYAI_API_KEY:
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")
        
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        if not MURF_API_KEY:
            raise HTTPException(status_code=500, detail="Murf API key not configured")

        # Step 1: Transcribe the audio
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
        
        transcribed_text = transcript.text
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Step 2: Send to LLM
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'models/gemini-1.5-flash']
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if not model:
            raise HTTPException(status_code=500, detail="Failed to initialize Gemini model")
        
        # Add instruction to keep responses concise for TTS
        prompt = f"Please provide a concise response (under 2500 characters) to this question: {transcribed_text.strip()}"
        llm_response = model.generate_content(prompt)
        
        if not llm_response.text:
            raise HTTPException(status_code=500, detail="No response generated from LLM")
        
        response_text = llm_response.text.strip()
        
        # Step 3: Handle long responses (split if over 3000 chars)
        def split_text(text, max_length=2800):
            """Split text into chunks that respect sentence boundaries"""
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
        
        # Step 4: Generate TTS using Murf
        client = Murf(api_key=MURF_API_KEY)
        
        if len(response_text) > 3000:
            # Split into chunks and use first chunk only for now
            chunks = split_text(response_text)
            tts_text = chunks[0] + "..." if len(chunks) > 1 else chunks[0]
        else:
            tts_text = response_text
        
        tts_response = client.text_to_speech.generate(
            text=tts_text,
            voice_id="en-US-julia",
        )
        
        return {
            "transcript": transcribed_text,
            "llm_response": response_text,
            "audio_url": tts_response.audio_file,
            "truncated": len(response_text) > 3000
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"LLM query processing error: {str(e)}")

@app.get("/llm/models")
async def list_models():
    """Debug endpoint to list available Gemini models"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        models = list(genai.list_models())
        return {
            "models": [
                {
                    "name": model.name,
                    "display_name": model.display_name,
                    "supported_methods": model.supported_generation_methods
                }
                for model in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/agent/chat/{session_id}")
async def agent_chat(session_id: str, file: UploadFile = File(...)):
    """
    Conversational AI endpoint with chat history and robust error handling
    """
    temp_file_path = None
    transcribed_text = None
    
    try:
        # Validate file type
        allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg"]
        if file.content_type not in allowed_types:
            return get_fallback_response("general_error", 
                f"I can only process audio files. Please try recording again.")
        
        # Check file size (max 25MB)
        max_size = 25 * 1024 * 1024  # 25MB
        
        # Save file temporarily for transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            file_content = await file.read()
            
            if len(file_content) > max_size:
                return get_fallback_response("general_error", 
                    "Your audio file is too large. Please try a shorter recording.")
            
            if not file_content:
                return get_fallback_response("no_speech")
            
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Step 1: Transcribe the audio with error handling
        try:
            if not ASSEMBLYAI_API_KEY:
                return get_fallback_response("api_key_missing", 
                    "Speech recognition is temporarily unavailable. Please try again later.")
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(temp_file_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                return get_fallback_response("stt_error", 
                    f"I couldn't understand your audio. {transcript.error}")
            
            transcribed_text = transcript.text
            if not transcribed_text or not transcribed_text.strip():
                return get_fallback_response("no_speech")
                
        except Exception as e:
            print(f"STT Error: {e}")
            return get_fallback_response("stt_error", 
                "I'm having trouble with speech recognition right now. Please try again.")
        
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

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
            for msg in chat_history[-10:]:  # Keep last 10 messages for context
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

Please provide a natural, conversational response that takes into account our previous conversation. Keep your response concise (under 2500 characters) and engaging."""
            else:
                prompt = f"Please provide a concise, engaging response (under 2500 characters) to this question: {current_question}"
            
        except Exception as e:
            print(f"Context building error: {e}")
            prompt = f"Please provide a concise response to: {transcribed_text.strip()}"

        # Step 4: Send to LLM with error handling
        response_text = None
        try:
            if not GEMINI_API_KEY:
                response_text = "I'm temporarily unable to process your request. My AI services are currently unavailable."
            else:
                model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'models/gemini-1.5-flash']
                
                model = None
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        break
                    except Exception as model_error:
                        print(f"Failed to initialize model {model_name}: {model_error}")
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
            print(f"LLM Error: {e}")
            response_text = FALLBACK_RESPONSES["llm_error"]
        
        # Add assistant response to history
        assistant_message = {"role": "assistant", "content": response_text}
        chat_history.append(assistant_message)
        
        # Step 5: Handle long responses for TTS
        def split_text(text, max_length=2800):
            """Split text into chunks that respect sentence boundaries"""
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
        
        # Step 6: Generate TTS using Murf with error handling
        audio_url = None
        tts_error = False
        
        try:
            if not MURF_API_KEY:
                tts_error = True
            else:
                client = Murf(api_key=MURF_API_KEY)
                
                if len(response_text) > 3000:
                    # Split into chunks and use first chunk only for now
                    chunks = split_text(response_text)
                    tts_text = chunks[0] + "..." if len(chunks) > 1 else chunks[0]
                else:
                    tts_text = response_text
                
                tts_response = client.text_to_speech.generate(
                    text=tts_text,
                    voice_id="en-US-julia",
                )
                audio_url = tts_response.audio_file
                
        except Exception as e:
            print(f"TTS Error: {e}")
            tts_error = True
        
        return {
            "session_id": session_id,
            "transcript": transcribed_text,
            "llm_response": response_text,
            "audio_url": audio_url,
            "truncated": len(response_text) > 3000,
            "message_count": len(chat_history),
            "tts_error": tts_error,
            "fallback": False
        }
        
    except Exception as e:
        print(f"Unexpected error in agent_chat: {e}")
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        # Return fallback response instead of raising exception
        fallback = get_fallback_response("general_error")
        fallback.update({
            "session_id": session_id,
            "message_count": len(chat_sessions.get(session_id, [])),
            "transcript": transcribed_text or "Audio processing failed"
        })
        return fallback
    

@app.get("/agent/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        return {"session_id": session_id, "messages": [], "message_count": 0}
    
    return {
        "session_id": session_id,
        "messages": chat_sessions[session_id],
        "message_count": len(chat_sessions[session_id])
    }

@app.delete("/agent/chat/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    
    return {"session_id": session_id, "status": "cleared"}

@app.post("/tts/echo")
async def tts_echo(file: UploadFile = File(...)):
    """
    Echo bot endpoint that transcribes audio and converts it back to speech using Murf TTS
    """
    # Validate file type
    allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Check file size (max 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    
    try:
        # Save file temporarily for transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            file_content = await file.read()
            
            if len(file_content) > max_size:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 25MB")
            
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Check API keys
        if not ASSEMBLYAI_API_KEY:
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")
        
        if not MURF_API_KEY:
            raise HTTPException(status_code=500, detail="Murf API key not configured")

        # Step 1: Transcribe the audio
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
        
        transcribed_text = transcript.text
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Step 2: Generate TTS using Murf
        client = Murf(api_key=MURF_API_KEY)
        tts_response = client.text_to_speech.generate(
            text=transcribed_text.strip(),
            voice_id="en-US-julia",  # Using a default voice, can be made configurable
        )
        
        return {
            "transcript": transcribed_text,
            "audio_url": tts_response.audio_file
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Echo processing error: {str(e)}")

@app.post("/test/error/{error_type}")
async def test_error_scenario(error_type: str):
    """
    Test endpoint to simulate different error scenarios
    Usage: POST /test/error/stt or /test/error/llm or /test/error/tts
    """
    if error_type == "stt":
        return get_fallback_response("stt_error")
    elif error_type == "llm":
        return get_fallback_response("llm_error")
    elif error_type == "tts":
        response = get_fallback_response("tts_error")
        response["tts_error"] = True
        return response
    elif error_type == "api_keys":
        return get_fallback_response("api_key_missing")
    else:
        return get_fallback_response("general_error")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status"""
    status = {
        "status": "healthy",
        "services": {
            "assemblyai": bool(ASSEMBLYAI_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
            "murf": bool(MURF_API_KEY)
        },
        "timestamp": datetime.now().isoformat()
    }
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
