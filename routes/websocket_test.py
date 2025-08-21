"""
WebSocket endpoint for streaming audio data with AssemblyAI v3 transcription and LLM responses
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
import struct
import asyncio
import threading
import queue
from datetime import datetime
from typing import Type
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import google.generativeai as genai
import websockets
import base64
from config import config
import re
from tavily import TavilyClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Configure APIs
aai.settings.api_key = config.ASSEMBLYAI_API_KEY
genai.configure(api_key=config.GEMINI_API_KEY)

# Initialize Tavily client for real-time search
tavily_client = None
if config.TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
        logger.info("✅ [TAVILY]: Initialized search client")
    except Exception as e:
        logger.error(f"❌ [TAVILY]: Failed to initialize: {e}")
        tavily_client = None

# Initialize Gemini model
try:
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    logger.info("✅ [LLM]: Initialized Gemini model")
except Exception as e:
    logger.error(f"❌ [LLM]: Failed to initialize Gemini: {e}")
    model = None


def create_wav_header(sample_rate=16000, channels=1, bits_per_sample=16, data_length=0):
    """Create WAV file header for proper audio file format."""
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', data_length + 36, b'WAVE', b'fmt ',
        16, 1, channels, sample_rate,
        sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8, bits_per_sample,
        b'data', data_length
    )


async def search_weather(query: str, session_keys: dict = None) -> str:
    """Search for weather information using Tavily"""
    tavily_key = get_api_key('tavily', session_keys or {})
    if not tavily_key:
        return "Weather search is not available. Please configure your Tavily API key in settings."
    
    try:
        logger.info(f"🌤️ [WEATHER SEARCH]: Searching for: {query}")
        
        # Create Tavily client with session key
        client = TavilyClient(api_key=tavily_key)
        
        # Enhanced weather search query
        search_query = f"current weather {query} today temperature forecast"
        
        response = client.search(
            query=search_query,
            search_depth="basic",
            max_results=3,
            include_domains=["weather.com", "accuweather.com", "weather.gov", "bbc.com/weather"]
        )
        
        if response and response.get('results'):
            weather_info = []
            for result in response['results'][:2]:  # Top 2 results
                title = result.get('title', '')
                content = result.get('content', '')
                if content:
                    weather_info.append(f"• {title}: {content[:200]}...")
            
            if weather_info:
                return f"🌤️ **Current Weather Information:**\n" + "\n".join(weather_info)
        
        return "I couldn't find current weather information. Please try asking about a specific location."
        
    except Exception as e:
        logger.error(f"❌ [WEATHER SEARCH]: Error: {e}")
        return "I'm having trouble accessing weather information right now."


async def search_news(query: str, session_keys: dict = None) -> str:
    """Search for latest news using Tavily"""
    tavily_key = get_api_key('tavily', session_keys or {})
    if not tavily_key:
        return "News search is not available. Please configure your Tavily API key in settings."
    
    try:
        logger.info(f"📰 [NEWS SEARCH]: Searching for: {query}")
        
        # Create Tavily client with session key
        client = TavilyClient(api_key=tavily_key)
        
        # Enhanced news search query
        search_query = f"latest news {query} today breaking news recent"
        
        response = client.search(
            query=search_query,
            search_depth="basic",
            max_results=4,
            include_domains=["reuters.com", "bbc.com", "cnn.com", "apnews.com", "npr.org"]
        )
        
        if response and response.get('results'):
            news_items = []
            for result in response['results'][:3]:  # Top 3 results
                title = result.get('title', '')
                content = result.get('content', '')
                url = result.get('url', '')
                if title and content:
                    news_items.append(f"• **{title}**\n  {content[:150]}...")
            
            if news_items:
                return f"📰 **Latest News:**\n" + "\n\n".join(news_items)
        
        return "I couldn't find recent news on that topic. Please try a different search term."
        
    except Exception as e:
        logger.error(f"❌ [NEWS SEARCH]: Error: {e}")
        return "I'm having trouble accessing news information right now."


def get_api_key(service: str, session_keys: dict) -> str:
    """Get API key from session keys or fallback to config"""
    return session_keys.get(service) or getattr(config, f"{service.upper()}_API_KEY", None)


def detect_search_intent(user_input: str) -> tuple[bool, str, str]:
    """Detect if user wants weather or news information"""
    user_lower = user_input.lower()
    
    # Weather patterns
    weather_patterns = [
        r'\b(weather|temperature|forecast|rain|snow|sunny|cloudy|storm|climate)\b',
        r'\bwhat\'?s\s+the\s+weather\b',
        r'\bhow\'?s\s+the\s+weather\b',
        r'\bweather\s+(in|for|at)\b',
        r'\btemperature\s+(in|for|at)\b'
    ]
    
    # News patterns  
    news_patterns = [
        r'\b(news|headlines|breaking|latest|trending|happening)\b',
        r'\bwhat\'?s\s+(happening|trending|new)\b',
        r'\btell\s+me\s+(about\s+)?the\s+news\b',
        r'\blatest\s+(news|updates)\b',
        r'\bbreaking\s+news\b'
    ]
    
    # Check for weather intent
    for pattern in weather_patterns:
        if re.search(pattern, user_lower):
            return True, "weather", user_input
    
    # Check for news intent
    for pattern in news_patterns:
        if re.search(pattern, user_lower):
            return True, "news", user_input
    
    return False, "", user_input




async def stream_openai_tts_to_client(text: str, client_websocket: WebSocket):
    """
    Convert text to speech using OpenAI TTS and stream audio chunks to client
    """
    try:
        import openai
        from io import BytesIO
        
        logger.info(f"🎵 [OPENAI TTS]: Converting text to speech: '{text[:50]}...'")
        
        # Send TTS start notification to client
        await client_websocket.send_text(json.dumps({
            "type": "tts_start",
            "message": "Converting to speech..."
        }))
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
            input=text,
            response_format="mp3"
        )
        
        # Get audio data
        audio_data = response.content
        logger.info(f"🎵 [OPENAI TTS]: Generated {len(audio_data)} bytes of audio")
        
        # Split audio into chunks for streaming
        chunk_size = 8192  # 8KB chunks for smooth streaming
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunk_number = (i // chunk_size) + 1
            
            # Convert to base64 for JSON transmission
            chunk_base64 = base64.b64encode(chunk).decode('utf-8')
            
            # Send chunk to client
            await client_websocket.send_text(json.dumps({
                "type": "tts_chunk",
                "chunk": chunk_base64,
                "chunk_number": chunk_number,
                "total_chunks": total_chunks
            }))
            
            logger.info(f"🎵 [OPENAI TTS]: Sent chunk {chunk_number}/{total_chunks} ({len(chunk)} bytes)")
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        # Send completion notification
        await client_websocket.send_text(json.dumps({
            "type": "tts_complete",
            "total_chunks": total_chunks,
            "total_size": len(audio_data)
        }))
        
        logger.info(f"✅ [OPENAI TTS]: Streaming complete - {total_chunks} chunks, {len(audio_data)} bytes")
        
    except ImportError:
        logger.error("❌ [OPENAI TTS]: OpenAI library not installed")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": "OpenAI TTS service not available"
        }))
    except Exception as e:
        logger.error(f"❌ [OPENAI TTS]: Error during TTS: {e}")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": f"TTS error: {str(e)}"
        }))


async def stream_elevenlabs_tts_to_client(text: str, client_websocket: WebSocket):
    """
    Convert text to speech using ElevenLabs and stream audio chunks to client
    """
    try:
        import httpx
        
        logger.info(f"🎵 [ELEVENLABS TTS]: Converting text to speech: '{text[:50]}...'")
        
        # Send TTS start notification to client
        await client_websocket.send_text(json.dumps({
            "type": "tts_start",
            "message": "Converting to speech..."
        }))
        
        # ElevenLabs API configuration
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        # Stream audio from ElevenLabs
        async with httpx.AsyncClient() as http_client:
            async with http_client.stream("POST", url, json=data, headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"ElevenLabs API error: {response.status_code}")
                
                chunk_number = 0
                total_size = 0
                
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if chunk:
                        chunk_number += 1
                        total_size += len(chunk)
                        
                        # Convert to base64 for JSON transmission
                        chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                        
                        # Send chunk to client
                        await client_websocket.send_text(json.dumps({
                            "type": "tts_chunk",
                            "chunk": chunk_base64,
                            "chunk_number": chunk_number
                        }))
                        
                        logger.info(f"🎵 [ELEVENLABS TTS]: Sent chunk {chunk_number} ({len(chunk)} bytes)")
                        
                        # Small delay for smooth streaming
                        await asyncio.sleep(0.01)
        
        # Send completion notification
        await client_websocket.send_text(json.dumps({
            "type": "tts_complete",
            "total_chunks": chunk_number,
            "total_size": total_size
        }))
        
        logger.info(f"✅ [ELEVENLABS TTS]: Streaming complete - {chunk_number} chunks, {total_size} bytes")
        
    except Exception as e:
        logger.error(f"❌ [ELEVENLABS TTS]: Error during TTS: {e}")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": f"TTS error: {str(e)}"
        }))


async def stream_simple_tts_to_client(text: str, client_websocket: WebSocket):
    """
    Route to appropriate TTS service based on available API keys
    """
    # Use browser TTS for reliable audio playback
    logger.info(f"🎵 [TTS]: Converting text to speech: '{text[:50]}...'")
    await client_websocket.send_text(json.dumps({
        "type": "tts_text",
        "text": text,
        "voice": "default"
    }))
    return
    
    # Try OpenAI TTS first (best quality and reliability)
    if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
        await stream_openai_tts_to_client(text, client_websocket)
    # Try ElevenLabs as fallback
    elif hasattr(config, 'ELEVENLABS_API_KEY') and config.ELEVENLABS_API_KEY:
        await stream_elevenlabs_tts_to_client(text, client_websocket)
    # Try Murf as last resort
    elif config.MURF_API_KEY:
        await stream_murf_tts_to_client(text, client_websocket)
    else:
        # Fallback to browser TTS
        logger.warning("⚠️ [TTS]: No TTS API keys configured, using browser fallback")
        await client_websocket.send_text(json.dumps({
            "type": "tts_text",
            "text": text,
            "voice": "default"
        }))


async def stream_murf_tts_to_client(text: str, client_websocket: WebSocket, context_id: str = "voice_agent_static_context"):
    """
    Send text to Murf WebSocket TTS API and stream base64 audio chunks to client
    """
    if not config.MURF_API_KEY:
        logger.error("❌ [MURF]: API key not configured")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": "TTS service unavailable"
        }))
        return
    
    try:
        logger.info(f"🎵 [MURF]: Converting text to speech: '{text[:50]}...'")
        
        # Send TTS start notification to client
        await client_websocket.send_text(json.dumps({
            "type": "tts_start",
            "message": "Converting to speech..."
        }))
        
        # Murf WebSocket endpoint (using their streaming API)
        murf_ws_url = f"wss://api.murf.ai/v1/speech/generate-stream?api_key={config.MURF_API_KEY}"
        
        # Connect to Murf WebSocket
        async with websockets.connect(murf_ws_url) as murf_ws:
            logger.info("🔗 [MURF]: Connected to Murf WebSocket")
            
            # Send TTS request with static context_id
            request_payload = {
                "voice_id": config.DEFAULT_VOICE_ID,
                "text": text,
                "context_id": context_id,  # Static context ID to avoid limit issues
                "output_format": "mp3",
                "sample_rate": 22050,
                "speed": 1.0,
                "pitch": 1.0,
                "encoding": "base64",
                "stream": True  # Request streaming response
            }
            
            await murf_ws.send(json.dumps(request_payload))
            logger.info(f"📤 [MURF]: Sent TTS request with context_id: {context_id}")
            
            # Stream audio chunks to client
            chunk_count = 0
            total_audio_size = 0
            
            async for message in murf_ws:
                try:
                    response_data = json.loads(message)
                    
                    if response_data.get("type") == "audio_chunk":
                        chunk_count += 1
                        audio_chunk = response_data.get("data", "")
                        total_audio_size += len(audio_chunk)
                        
                        # Log chunk received
                        logger.info(f"🎵 [MURF CHUNK #{chunk_count}]: Received {len(audio_chunk)} chars base64")
                        
                        # Stream chunk to client
                        await client_websocket.send_text(json.dumps({
                            "type": "tts_chunk",
                            "chunk": audio_chunk,
                            "chunk_number": chunk_count
                        }))
                        
                    elif response_data.get("type") == "audio_complete":
                        # TTS streaming complete
                        logger.info(f"✅ [MURF]: TTS streaming complete - {chunk_count} chunks, {total_audio_size} total chars")
                        
                        await client_websocket.send_text(json.dumps({
                            "type": "tts_complete",
                            "total_chunks": chunk_count,
                            "total_size": total_audio_size
                        }))
                        break
                        
                    elif response_data.get("type") == "error":
                        logger.error(f"❌ [MURF]: TTS API error: {response_data.get('message', 'Unknown error')}")
                        await client_websocket.send_text(json.dumps({
                            "type": "tts_error",
                            "message": f"TTS error: {response_data.get('message', 'Unknown error')}"
                        }))
                        break
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ [MURF]: Invalid JSON in stream: {e}")
                    continue
                    
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"❌ [MURF]: WebSocket connection closed: {e}")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": "TTS connection lost"
        }))
    except websockets.exceptions.InvalidURI as e:
        logger.error(f"❌ [MURF]: Invalid WebSocket URI: {e}")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": "TTS service configuration error"
        }))
    except Exception as e:
        logger.error(f"❌ [MURF]: Error during TTS: {e}")
        await client_websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": f"TTS error: {str(e)}"
        }))


async def stream_llm_response(user_input: str, websocket: WebSocket, chat_history: list = None, session_keys: dict = None):
    """Stream LLM response to WebSocket client with professional AI assistant persona and real-time search"""
    
    # Use session API key for Gemini or fallback to config
    gemini_key = get_api_key('gemini', session_keys or {})
    if not gemini_key:
        logger.error("❌ [LLM]: No Gemini API key available")
        await websocket.send_text(json.dumps({
            "type": "llm_error",
            "message": "AI service is not configured. Please add your Gemini API key in settings."
        }))
        return
    
    # Initialize Gemini model with session key
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        session_model = genai.GenerativeModel("gemini-2.0-flash-exp")
    except Exception as e:
        logger.error(f"❌ [LLM]: Failed to initialize with session key: {e}")
        await websocket.send_text(json.dumps({
            "type": "llm_error",
            "message": "Failed to initialize AI service. Please check your API key."
        }))
        return

    try:
        logger.info(f"🤖 [MARVIS]: Processing user input: '{user_input}'")

        # Check if user wants real-time information
        needs_search, search_type, search_query = detect_search_intent(user_input)
        search_results = ""
        
        if needs_search:
            # Send search notification
            await websocket.send_text(json.dumps({
                "type": "llm_start", 
                "message": f"Searching for latest {search_type} information..."
            }))
            
            if search_type == "weather":
                search_results = await search_weather(search_query, session_keys)
            elif search_type == "news":
                search_results = await search_news(search_query, session_keys)
            
            logger.info(f"🔍 [SEARCH]: Found {search_type} information")

        # Build conversation context
        context = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]  # Last 6 messages for context
            context_parts = []
            for msg in recent_history:
                if msg["role"] == "user":
                    context_parts.append(f"User: {msg['content']}")
                else:
                    context_parts.append(f"Assistant: {msg['content']}")
            context = "\n".join(context_parts) + "\n\n"

        # Enhanced prompt with search results
        search_context = f"\n\nReal-time Information:\n{search_results}\n" if search_results else ""
        
        prompt = f"""You are MARVIS (Machine-based Assistant for Research, Voice, and Interactive Services), a professional AI assistant with access to real-time information.

Key characteristics:
- Professional yet friendly tone
- Clear and concise responses
- Helpful and informative
- Conversational and engaging
- Remember conversation context
- Provide accurate, up-to-date information
- When you have real-time search results, use them to provide current information

{context}User: {user_input}{search_context}

Respond as MARVIS using the real-time information if provided:"""

        # Send LLM processing notification
        await websocket.send_text(json.dumps({
            "type": "llm_start",
            "message": "Thinking..."
        }))

        # Generate streaming response
        response_stream = session_model.generate_content(prompt, stream=True)
        accumulated_response = ""
        chunk_count = 0

        for chunk in response_stream:
            if chunk.text:
                chunk_count += 1
                accumulated_response += chunk.text

                # Send chunk to client
                await websocket.send_text(json.dumps({
                    "type": "llm_chunk",
                    "chunk": chunk.text,
                    "chunk_number": chunk_count
                }))

        # Send completion notification
        await websocket.send_text(json.dumps({
            "type": "llm_complete",
            "full_response": accumulated_response,
            "total_chunks": chunk_count
        }))

        logger.info(f"✅ [MARVIS]: Response complete - {chunk_count} chunks, {len(accumulated_response)} characters")
        
        # Add to chat history
        if chat_history is not None:
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": accumulated_response})
            
            # Keep history manageable
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
        
        # Stream complete response to TTS
        if accumulated_response.strip():
            await stream_simple_tts_to_client(accumulated_response, websocket)
            
        return accumulated_response

    except Exception as e:
        logger.error(f"❌ [MARVIS]: Error during streaming: {e}")
        await websocket.send_text(json.dumps({
            "type": "llm_error",
            "message": f"I encountered an error: {str(e)}"
        }))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming with transcription and LLM responses"""
    await websocket.accept()
    logger.info("🔌 [WEBSOCKET]: Client connected")

    # Initialize variables
    audio_buffer = bytearray()
    audio_file_path = None
    chunk_count = 0
    transcriber = None
    transcript_queue = queue.Queue()
    current_turn_transcript = ""
    turn_timeout_task = None
    
    # Simple chat history for this session
    chat_history = []
    
    # User-provided API keys for this session
    session_api_keys = {}

    def on_begin(client: Type[StreamingClient], event: BeginEvent):
        """Handle session begin event"""
        logger.info(f"🎬 [TRANSCRIPTION]: Session started: {event.id}")
        transcript_queue.put(('session_begins', f'Session started: {event.id}'))

    def on_turn(client: Type[StreamingClient], event: TurnEvent):
        """Handle turn events with transcript data"""
        nonlocal current_turn_transcript
        
        logger.info(f"🎯 [TURN EVENT]: '{event.transcript}' (end_of_turn: {event.end_of_turn})")
        
        if event.transcript and event.transcript.strip():
            if event.end_of_turn:
                # This is a final transcript at end of turn
                logger.info(f"🔄 [TURN END]: '{event.transcript}'")
                current_turn_transcript = event.transcript
                transcript_queue.put(('final', event.transcript))
                transcript_queue.put(('turn_end', event.transcript))
            else:
                # This is a partial transcript
                logger.info(f"📝 [PARTIAL]: '{event.transcript}'")
                transcript_queue.put(('partial', event.transcript))
        elif event.end_of_turn:
            # Empty transcript but turn ended - might be silence or no speech detected
            logger.warning(f"⚠️ [EMPTY TURN]: Turn ended with empty transcript")
            # Send a helpful message for empty turns
            helpful_message = "I didn't catch that. Could you please speak a bit louder or closer to the microphone?"
            transcript_queue.put(('turn_end', helpful_message))
        
        # Enable turn formatting if not already enabled
        if event.end_of_turn and not event.turn_is_formatted:
            params = StreamingSessionParameters(format_turns=True)
            client.set_params(params)

    def on_terminated(client: Type[StreamingClient], event: TerminationEvent):
        """Handle session termination"""
        logger.info(f"🏁 [TRANSCRIPTION]: Session terminated: {event.audio_duration_seconds} seconds processed")
        transcript_queue.put(('session_terminated', f'Session ended: {event.audio_duration_seconds}s'))
        if current_turn_transcript:
            logger.info(f"🔄 [FINAL TURN]: Session ended with transcript: '{current_turn_transcript}'")
            transcript_queue.put(('turn_end', current_turn_transcript))

    def on_error(client: Type[StreamingClient], error: StreamingError):
        """Handle transcription errors"""
        logger.error(f"❌ [TRANSCRIPTION ERROR]: {error}")
        transcript_queue.put(('error', str(error)))



    try:
        while True:
            # Check for new transcripts
            try:
                while not transcript_queue.empty():
                    transcript_type, transcript_text = transcript_queue.get_nowait()

                    # Send transcript to client
                    response = {
                        "type": "transcript",
                        "transcript_type": transcript_type,
                        "text": transcript_text
                    }
                    await websocket.send_text(json.dumps(response))

                    if transcript_type == 'final':
                        logger.info(f"📤 [SENT TO CLIENT]: Final transcript: {transcript_text}")
                        # Don't trigger LLM on every final transcript, wait for turn_end

                    elif transcript_type == 'turn_end':
                        logger.info(f"🔄 [SENT TO CLIENT]: Turn ended with: {transcript_text}")
                        turn_end_response = {
                            "type": "turn_end",
                            "final_transcript": transcript_text,
                            "message": "User stopped talking"
                        }
                        await websocket.send_text(json.dumps(turn_end_response))

                        # Trigger LLM response only on turn end with meaningful text
                        if transcript_text.strip() and len(transcript_text.strip()) > 3:
                            logger.info(f"🚀 [TRIGGER LLM]: Starting LLM response for: '{transcript_text}'")
                            asyncio.create_task(stream_llm_response(transcript_text, websocket, chat_history, session_api_keys))

            except queue.Empty:
                pass

            # Receive message from client
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if "text" in message:
                text_data = message["text"]
                logger.info(f"📨 [CONTROL MESSAGE]: {text_data}")

                try:
                    data = json.loads(text_data)

                    if data.get("type") == "api_keys":
                        # Store user-provided API keys for this session
                        session_api_keys.update(data.get("keys", {}))
                        logger.info(f"🔑 [API KEYS]: Updated session keys: {list(session_api_keys.keys())}")
                        
                        # Send confirmation
                        response = {"type": "api_keys_received", "message": "API keys updated"}
                        await websocket.send_text(json.dumps(response))
                        
                    elif data.get("type") == "manual_test":
                        # Handle manual test for debugging
                        test_text = data.get("text", "Ahoy there! How be ye doing today?")
                        logger.info(f"🧪 [MANUAL TEST]: Received test: '{test_text}'")
                        
                        # Trigger LLM response directly
                        asyncio.create_task(stream_llm_response(test_text, websocket, chat_history, session_api_keys))
                        
                    elif data.get("type") == "browser_transcript":
                        # Handle browser speech recognition transcript
                        transcript_text = data.get("text", "").strip()
                        if transcript_text and len(transcript_text) > 2:
                            logger.info(f"🎤 [BROWSER SPEECH]: Processing transcript: '{transcript_text}'")
                            # Trigger LLM response for browser transcript
                            asyncio.create_task(stream_llm_response(transcript_text, websocket, chat_history, session_api_keys))
                        else:
                            logger.warning(f"⚠️ [BROWSER SPEECH]: Ignoring short transcript: '{transcript_text}'")
                        
                    elif data.get("type") == "start_recording":
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_file_path = f"streamed_audio_{timestamp}.wav"
                        audio_buffer.clear()
                        chunk_count = 0
                        logger.info(f"🎙️ [RECORDING]: Started session: {audio_file_path}")

                        try:
                            # Initialize AssemblyAI v3 Streaming Client
                            transcriber = StreamingClient(
                                StreamingClientOptions(
                                    api_key=config.ASSEMBLYAI_API_KEY,
                                    api_host="streaming.assemblyai.com"
                                )
                            )
                            
                            # Set up event handlers
                            transcriber.on(StreamingEvents.Begin, on_begin)
                            transcriber.on(StreamingEvents.Turn, on_turn)
                            transcriber.on(StreamingEvents.Termination, on_terminated)
                            transcriber.on(StreamingEvents.Error, on_error)
                            
                            # Connect with streaming parameters
                            transcriber.connect(
                                StreamingParameters(
                                    sample_rate=16000,
                                    format_turns=True,
                                    end_utterance_silence_threshold=1000,
                                    disable_partial_transcripts=False
                                )
                            )

                            logger.info("🎯 [TRANSCRIPTION]: Initialized AssemblyAI v3 streaming client")
                        except Exception as e:
                            logger.error(f"❌ [TRANSCRIPTION]: Failed to initialize: {e}")

                        response = {"type": "recording_started", "filename": audio_file_path}
                        await websocket.send_text(json.dumps(response))

                    elif data.get("type") == "stop_recording":
                        if transcriber:
                            try:
                                transcriber.disconnect(terminate=True)
                                logger.info("🛑 [TRANSCRIPTION]: Stopped AssemblyAI v3 streaming client")
                            except Exception as e:
                                logger.error(f"❌ [TRANSCRIPTION]: Error stopping: {e}")

                        if audio_buffer and audio_file_path:
                            wav_header = create_wav_header(sample_rate=16000, data_length=len(audio_buffer))
                            with open(audio_file_path, "wb") as f:
                                f.write(wav_header)
                                f.write(audio_buffer)

                            file_size = len(audio_buffer)
                            logger.info(f"💾 [RECORDING]: Saved {file_size} bytes to {audio_file_path}")

                            response = {
                                "type": "recording_stopped",
                                "filename": audio_file_path,
                                "size_bytes": file_size,
                                "chunks_received": chunk_count
                            }
                            await websocket.send_text(json.dumps(response))

                            audio_buffer.clear()
                            chunk_count = 0
                        else:
                            logger.warning("⚠️ [RECORDING]: Stop called but no active recording")

                except json.JSONDecodeError:
                    logger.info(f"📝 [TEXT MESSAGE]: {text_data}")

            elif "bytes" in message:
                audio_chunk = message["bytes"]
                chunk_count += 1
                audio_buffer.extend(audio_chunk)

                if transcriber and len(audio_chunk) > 0:
                    try:
                        # Log audio chunk info for debugging
                        if chunk_count % 50 == 0:  # Log every 50th chunk
                            logger.info(f"🎵 [AUDIO DEBUG]: Chunk #{chunk_count}, size: {len(audio_chunk)} bytes")
                        
                        # Stream audio data to AssemblyAI v3
                        transcriber.stream(audio_chunk)
                    except Exception as e:
                        logger.error(f"❌ [TRANSCRIPTION]: Error streaming audio chunk #{chunk_count}: {e}")

                if chunk_count % 20 == 0:
                    logger.info(f"🎵 [AUDIO]: Chunk #{chunk_count}, buffer: {len(audio_buffer)} bytes")

                if chunk_count % 100 == 0:
                    response = {
                        "type": "chunk_ack",
                        "chunks_received": chunk_count,
                        "buffer_size": len(audio_buffer)
                    }
                    await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("🔌 [WEBSOCKET]: Client disconnected")

        if transcriber:
            try:
                transcriber.disconnect(terminate=True)
                logger.info("🧹 [CLEANUP]: Closed AssemblyAI v3 streaming client")
            except Exception as e:
                logger.error(f"❌ [CLEANUP]: Error closing transcriber: {e}")

        if audio_buffer and audio_file_path:
            wav_header = create_wav_header(sample_rate=16000, data_length=len(audio_buffer))
            with open(audio_file_path, "wb") as f:
                f.write(wav_header)
                f.write(audio_buffer)
            logger.info(f"💾 [CLEANUP]: Saved remaining audio to {audio_file_path}")

    except Exception as e:
        logger.error(f"❌ [WEBSOCKET]: Error: {e}")

        if transcriber:
            try:
                transcriber.disconnect(terminate=True)
            except:
                pass

        try:
            await websocket.close()
        except:
            pass
