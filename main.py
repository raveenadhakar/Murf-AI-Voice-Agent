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
from routes import audio, chat, system, websocket_test

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
app.include_router(websocket_test.router)


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_FOLDER), name="uploads")

# Root endpoint - serve the chat interface
@app.get("/")
async def root():
    """Serve the main chat interface"""
    return FileResponse("static/index.html")


# API key testing endpoint
@app.post("/test-api-keys")
async def test_api_keys(request: dict):
    """Test the validity of provided API keys"""
    results = {}
    
    # Test Gemini API key
    if request.get("gemini"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=request["gemini"])
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            # Try a simple generation to test the key
            response = model.generate_content("Hello")
            results["gemini"] = True
        except Exception as e:
            logger.error(f"Gemini API test failed: {e}")
            results["gemini"] = False
    
    # Test Tavily API key
    if request.get("tavily"):
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=request["tavily"])
            # Try a simple search to test the key
            response = client.search("test", max_results=1)
            results["tavily"] = True
        except Exception as e:
            logger.error(f"Tavily API test failed: {e}")
            results["tavily"] = False
    
    # Test AssemblyAI API key
    if request.get("assemblyai"):
        try:
            import assemblyai as aai
            aai.settings.api_key = request["assemblyai"]
            # Try to access the API to test the key
            transcriber = aai.Transcriber()
            results["assemblyai"] = True
        except Exception as e:
            logger.error(f"AssemblyAI API test failed: {e}")
            results["assemblyai"] = False
    
    # For other APIs, we'll just check if they're provided
    # (Testing them would require more complex setup)
    for key in ["openai", "elevenlabs", "murf"]:
        if request.get(key):
            results[key] = len(request[key].strip()) > 10  # Basic length check
    
    success = any(results.values()) if results else False
    
    return {
        "success": success,
        "results": results,
        "message": "API key validation complete"
    }


# Application startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🤖 MARVIS AI Assistant starting up...")
    logger.info(f"📊 Services status: {config.get_service_status()}")
    logger.info(f"🌐 MARVIS will be available at http://{config.HOST}:{config.PORT}")
    logger.info("🎤 Voice chat works best on HTTPS or localhost with microphone permissions!")
    logger.info("⌨️ Text input is always available as a backup option!")
    logger.info("✅ MARVIS is ready to assist!")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🤖 MARVIS is shutting down...")
    logger.info("👋 Goodbye! See you next time!")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting MARVIS on {config.HOST}:{config.PORT}")
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )