#!/usr/bin/env python3
"""
🏴‍☠️ Captain AI's Parrot - Startup Script
Simple script to launch the pirate voice assistant
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import fastapi
        import uvicorn
        import assemblyai
        import google.generativeai
        print("✅ All required packages found!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def check_env_file():
    """Check if .env file exists and has required keys"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found!")
        print("📝 Creating sample .env file...")
        
        sample_env = """# 🏴‍☠️ Captain AI's Parrot Configuration

# Required for AI responses
GEMINI_API_KEY=your_gemini_api_key_here

# Optional TTS services (browser fallback available)
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
MURF_API_KEY=your_murf_key_here

# Optional (AssemblyAI has issues, browser speech works)
ASSEMBLYAI_API_KEY=your_assemblyai_key_here

# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=true
"""
        
        with open(".env", "w") as f:
            f.write(sample_env)
        
        print("📄 Sample .env file created!")
        print("🔑 Please add your API keys to the .env file")
        print("🌟 At minimum, you need a GEMINI_API_KEY from https://makersuite.google.com/app/apikey")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🏴‍☠️ Ahoy! Starting Captain AI's Parrot...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("❌ Failed to install requirements")
        return
    
    # Check environment
    if not check_env_file():
        print("⚠️  Please configure your .env file and run again")
        return
    
    print("🚀 Launching the pirate ship...")
    print("🌐 The parrot will be available at: http://localhost:8000")
    print("🎤 Make sure to allow microphone permissions!")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🏴‍☠️ Farewell, matey! The parrot has sailed away...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()