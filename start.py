#!/usr/bin/env python3
"""
Simple startup script for MARVIS AI Assistant
"""
import uvicorn
import sys
import os

def main():
    """Start the MARVIS AI Assistant"""
    print("🤖 Starting MARVIS AI Assistant...")
    print("🌐 Will be available at: http://localhost:8000")
    print("🎤 Make sure to allow microphone permissions in your browser")
    print("⌨️ You can also use text input by clicking the toggle button")
    print("-" * 50)
    
    try:
        # Run the FastAPI application directly
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 MARVIS is shutting down. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting MARVIS: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())