"""
System and utility API routes
"""
import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
import google.generativeai as genai

from config import config
from utils import create_error_test_response, log_request_info, log_response_info
from schemas import HealthCheckResponse, ModelsResponse, ErrorTestResponse, ModelInfo, ServiceInfo, ServiceStatus

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Check health status of the application and services
    
    Returns:
        HealthCheckResponse with service status
    """
    log_request_info("health_check")
    
    services_status = config.get_service_status()
    overall_status = ServiceStatus.HEALTHY if all(services_status.values()) else ServiceStatus.DEGRADED
    
    service_info = ServiceInfo(
        assemblyai=services_status["assemblyai"],
        gemini=services_status["gemini"],
        murf=services_status["murf"]
    )
    
    log_response_info("health_check", success=True, status=overall_status.value)
    
    return HealthCheckResponse(
        status=overall_status,
        services=service_info,
        timestamp=datetime.now().isoformat()
    )


@router.get("/llm/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """
    List available LLM models
    
    Returns:
        ModelsResponse with available models
    """
    log_request_info("list_models")
    
    if not config.GEMINI_API_KEY:
        logger.warning("Gemini API key not configured")
        log_response_info("list_models", success=False, error="API key missing")
        return ModelsResponse(models=[])
    
    try:
        models_list = list(genai.list_models())
        models = [
            ModelInfo(
                name=model.name,
                display_name=model.display_name,
                supported_methods=model.supported_generation_methods
            )
            for model in models_list
        ]
        
        log_response_info("list_models", success=True, model_count=len(models))
        
        return ModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        log_response_info("list_models", success=False, error=str(e))
        return ModelsResponse(models=[])


@router.post("/test/error/{error_type}", response_model=ErrorTestResponse)
async def test_error_scenario(error_type: str) -> ErrorTestResponse:
    """
    Test different error scenarios for debugging
    
    Args:
        error_type: Type of error to simulate (stt, llm, tts, api_keys)
        
    Returns:
        ErrorTestResponse with simulated error
    """
    log_request_info("test_error", error_type=error_type)
    
    # Validate error type
    valid_error_types = ["stt", "llm", "tts", "api_keys", "general"]
    if error_type not in valid_error_types:
        error_type = "general"
    
    response = create_error_test_response(error_type)
    
    log_response_info("test_error", success=True, error_type=error_type)
    
    return response