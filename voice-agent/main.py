"""
AI Voice Chat Assistant - Main FastAPI Application
"""
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Third-party imports
import assemblyai as aai
import google.generativeai as genai

# Local imports
from config import config
from utils import setup_logging, create_upload_folder
from routes import audio, chat, system

# Setup logging and configuration
setup_logging()
logger = logging.getLogger(__name__)

# Validate configuration
validation_result = config.validate_config()
if not validation_result["valid"]:
    logger.error(f"Configuration validation failed: {validation_result['missing_keys']}")
    logger.warning("Some services may not be available")

for warning in validation_result.get("warnings", []):
    logger.warning(warning)

# Configure APIs
if config.ASSEMBLYAI_API_KEY:
    aai.settings.api_key = config.ASSEMBLYAI_API_KEY

if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)

# Create upload folder
create_upload_folder()

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Chat Assistant",
    description="A conversational AI voice assistant with modern chat interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include routers
app.include_router(audio.router)
app.include_router(chat.router)
app.include_router(system.router)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_FOLDER), name="uploads")

# Root endpoint - serve the chat interface
@app.get("/")
async def root():
    """Serve the main chat interface"""
    return FileResponse("static/index.html")


# Application startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 AI Voice Chat Assistant starting up...")
    logger.info(f"📊 Services status: {config.get_service_status()}")
    logger.info(f"🌐 Server will be available at http://{config.HOST}:{config.PORT}")
    logger.info("✅ Application startup complete")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🛑 AI Voice Chat Assistant shutting down...")
    logger.info("✅ Application shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )