"""
Pydantic schemas for request and response models
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"


class ServiceStatus(str, Enum):
    """Service status types"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


# Request Models
class TTSRequest(BaseModel):
    """Text-to-speech request"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to convert to speech")
    voice_id: str = Field(default="en-US-julia", description="Voice ID for TTS")


# Response Models
class TTSResponse(BaseModel):
    """Text-to-speech response"""
    audio_url: Optional[str] = Field(None, description="URL to generated audio file")
    success: bool = Field(default=False, description="Whether TTS was successful")
    error: Optional[str] = Field(None, description="Error message if any")
    fallback_text: Optional[str] = Field(None, description="Original text for fallback")
    use_browser_tts: bool = Field(default=False, description="Whether to use browser TTS")


class TranscriptionResponse(BaseModel):
    """Audio transcription response"""
    transcript: str = Field(..., description="Transcribed text from audio")


class UploadResponse(BaseModel):
    """File upload response"""
    filename: str = Field(..., description="Name of uploaded file")
    content_type: str = Field(..., description="MIME type of uploaded file")
    size: str = Field(..., description="File size")
    status: str = Field(..., description="Upload status message")


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: MessageRole = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class AgentChatResponse(BaseModel):
    """Agent chat conversation response"""
    session_id: str = Field(..., description="Session identifier")
    transcript: str = Field(..., description="User's transcribed speech")
    llm_response: str = Field(..., description="AI assistant's response")
    audio_url: Optional[str] = Field(None, description="URL to AI response audio")
    truncated: bool = Field(default=False, description="Whether response was truncated")
    message_count: int = Field(..., description="Total messages in session")
    tts_error: bool = Field(default=False, description="Whether TTS failed")
    fallback: bool = Field(default=False, description="Whether using fallback response")


class ChatHistoryResponse(BaseModel):
    """Chat history response"""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default=[], description="List of chat messages")
    message_count: int = Field(..., description="Total number of messages")


class ClearChatResponse(BaseModel):
    """Clear chat response"""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Clear operation status")


class ModelInfo(BaseModel):
    """Information about an AI model"""
    name: str = Field(..., description="Model name")
    display_name: str = Field(..., description="Human-readable model name")
    supported_methods: List[str] = Field(default=[], description="Supported generation methods")


class ModelsResponse(BaseModel):
    """Available models response"""
    models: List[ModelInfo] = Field(default=[], description="List of available models")


class ServiceInfo(BaseModel):
    """Service status information"""
    assemblyai: bool = Field(..., description="AssemblyAI service status")
    gemini: bool = Field(..., description="Gemini AI service status")
    murf: bool = Field(..., description="Murf TTS service status")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: ServiceStatus = Field(..., description="Overall system status")
    services: ServiceInfo = Field(..., description="Individual service statuses")
    timestamp: str = Field(..., description="Health check timestamp")


class ErrorTestResponse(BaseModel):
    """Error test response"""
    llm_response: str = Field(..., description="Error message response")
    audio_url: Optional[str] = Field(None, description="Audio URL (usually None for errors)")
    tts_error: bool = Field(default=True, description="TTS error flag")
    fallback: bool = Field(default=True, description="Fallback response flag")
    error_type: str = Field(..., description="Type of error being tested")


class EchoResponse(BaseModel):
    """Echo TTS response"""
    transcript: str = Field(..., description="Transcribed text")
    audio_url: str = Field(..., description="URL to echo audio")


class LLMQueryResponse(BaseModel):
    """LLM query response"""
    transcript: str = Field(..., description="Transcribed user input")
    llm_response: str = Field(..., description="AI response")
    audio_url: str = Field(..., description="URL to response audio")
    truncated: bool = Field(default=False, description="Whether response was truncated")