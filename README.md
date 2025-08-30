# 🤖 MARVIS - AI Voice Assistant

**Machine-based Assistant for Research, Voice, and Interactive Services**

A modern, professional AI voice assistant with real-time conversation capabilities, weather updates, news search, and multiple TTS options. Built with FastAPI, WebSockets, and modern web technologies.

![MARVIS Interface](https://img.shields.io/badge/Interface-Modern%20Web%20UI-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange)

## ✨ Features

### 🎤 **Voice & Text Input**
- **Browser Speech Recognition** - Reliable voice input with real-time transcription
- **Text Input Mode** - Toggle between voice and text input seamlessly
- **Smart Speech Detection** - Automatic start/stop with timeout protection
- **Multi-language Support** - Configurable language settings

### 🤖 **AI Conversation**
- **Google Gemini Integration** - Powered by advanced AI models
- **Context Awareness** - Remembers conversation history for natural dialogue
- **Professional Persona** - Helpful, knowledgeable, and conversational responses
- **Streaming Responses** - Real-time response generation with typing indicators

### 🔍 **Real-time Information**
- **Weather Updates** 🌤️ - Current conditions and forecasts for any location
- **Latest News** 📰 - Breaking news and trending topics from trusted sources
- **Smart Intent Detection** - Automatically recognizes weather and news queries
- **Reliable Sources** - Weather.com, BBC, Reuters, CNN, and more

### 🎵 **Advanced Text-to-Speech**
- **Multiple TTS Engines** - OpenAI, ElevenLabs, Murf, and browser fallback
- **High-Quality Audio** - Premium voice synthesis options
- **Streaming Audio** - Real-time audio playback with chunked delivery
- **Automatic Fallback** - Graceful degradation to browser TTS if needed

### ⚙️ **User-Friendly Configuration**
- **Settings UI** - Easy-to-use sidebar for API key management
- **Local Storage** - Secure browser-based key storage
- **Connection Testing** - Validate API keys before use
- **Visual Status** - Clear indicators for service availability

### 📱 **Modern Interface**
- **Responsive Design** - Works perfectly on desktop and mobile
- **Clean UI** - Professional, minimalist design
- **Real-time Status** - Live connection and processing indicators
- **Smooth Animations** - Polished user experience

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** 
- **Modern web browser** with microphone support
- **API keys** for the services you want to use

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd marvis-ai-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the application**
   ```bash
   python start.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000`

6. **Configure API keys**
   - Click the ⚙️ settings button
   - Enter your API keys
   - Save and test the connection

## 🔑 API Keys Setup

### Required Services

| Service | Purpose | How to Get |
|---------|---------|------------|
| **Google Gemini** | AI Conversations | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| **AssemblyAI** | Voice Transcription | [AssemblyAI Console](https://www.assemblyai.com/app/signup) |

### Enhanced Features

| Service | Purpose | How to Get |
|---------|---------|------------|
| **Tavily** | Weather & News Search | [Tavily API](https://tavily.com/) |
| **OpenAI** | Premium TTS | [OpenAI Platform](https://platform.openai.com/api-keys) |
| **ElevenLabs** | Premium Voice Synthesis | [ElevenLabs](https://elevenlabs.io/) |
| **Murf** | Alternative TTS | [Murf AI](https://murf.ai/) |

### Configuration Options

#### Option 1: Settings UI (Recommended)
1. Click the ⚙️ settings button in the interface
2. Enter your API keys in the respective fields
3. Click "💾 Save Configuration"
4. Click "🔗 Test Connection" to verify

#### Option 2: Environment File
1. Copy `.env.example` to `.env`
2. Edit `.env` with your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   ASSEMBLYAI_API_KEY=your_assemblyai_key_here
   TAVILY_API_KEY=your_tavily_key_here
   OPENAI_API_KEY=your_openai_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_key_here
   MURF_API_KEY=your_murf_key_here
   ```

## 💬 How to Use

### 🎤 Voice Interaction
1. **Click the microphone button** 🎤
2. **Speak clearly** - MARVIS will show "Listening..."
3. **Stop speaking** - Your message appears in chat
4. **Get AI response** - MARVIS responds with voice and text

### ⌨️ Text Interaction
1. **Click the toggle button** 💬 to switch to text mode
2. **Type your message** in the input field
3. **Press Enter** or click Send
4. **Get AI response** - Same intelligent responses

### 🌤️ Weather Queries
- "What's the weather in New York?"
- "How's the weather today?"
- "Temperature in London?"
- "Is it going to rain?"

### 📰 News Queries
- "What's the latest news?"
- "Tell me what's happening today"
- "Breaking news about technology"
- "What's trending right now?"

### 💬 General Conversation
- Ask any question or have a natural conversation
- MARVIS remembers context throughout the session
- Professional, helpful, and engaging responses

## 🏗️ Project Structure

```
marvis-ai-assistant/
├── 📄 main.py                 # FastAPI application entry point
├── 🚀 start.py                # Startup script with uvicorn
├── ⚙️ config.py               # Configuration management
├── 📋 requirements.txt        # Python dependencies
├── 🔒 .env.example           # Environment variables template
├── 📁 static/                # Frontend assets
│   ├── 🌐 index.html         # Main web interface
│   └── 🎨 style.css          # Modern CSS styling
├── 📁 routes/                # API route handlers
│   ├── 🔌 websocket_test.py  # WebSocket communication
│   ├── 🎵 audio.py           # Audio processing
│   ├── 💬 chat.py            # Chat endpoints
│   └── 🖥️ system.py          # System endpoints
├── 📁 uploads/               # File upload directory
├── 📖 README.md              # This documentation
└── 🔧 utils.py               # Utility functions
```

## 🔌 API Endpoints

### HTTP Endpoints
- `GET /` - Main chat interface
- `POST /test-api-keys` - Validate API keys
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

### WebSocket Endpoints
- `WebSocket /ws` - Real-time communication
  - Voice transcription
  - AI conversation streaming
  - TTS audio streaming
  - API key management

## 🛠️ Development

### Running in Development Mode
```bash
# With auto-reload
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables
```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# API Keys (Optional - can be set via UI)
GEMINI_API_KEY=your_key_here
ASSEMBLYAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
MURF_API_KEY=your_key_here
```

### Adding New Features
1. **Backend**: Add routes in the `routes/` directory
2. **Frontend**: Modify `static/index.html` and `static/style.css`
3. **Configuration**: Update `config.py` for new settings
4. **Dependencies**: Add to `requirements.txt`

## 🔧 Troubleshooting

### Common Issues

#### Microphone Not Working
- **Check browser permissions** - Allow microphone access
- **Use HTTPS or localhost** - Required for microphone API
- **Try different browser** - Chrome/Edge work best

#### AI Not Responding
- **Check Gemini API key** - Required for AI responses
- **Verify internet connection** - Needed for API calls
- **Check browser console** - Look for error messages

#### Voice Transcription Issues
- **Check AssemblyAI key** - Required for voice-to-text
- **Speak clearly** - Ensure good audio quality
- **Check microphone** - Test with other applications

#### Search Features Not Working
- **Check Tavily API key** - Required for weather/news
- **Verify API limits** - Check your usage quotas
- **Try different queries** - Use clear, specific requests

### Debug Mode
Enable debug logging by setting `DEBUG=true` in your environment or `.env` file.

## 📊 Performance

### Optimizations
- **Streaming responses** - Real-time AI response delivery
- **Chunked audio** - Efficient TTS audio streaming
- **Connection pooling** - Optimized API calls
- **Local caching** - Browser-based settings storage

### Resource Usage
- **Memory**: ~50-100MB typical usage
- **CPU**: Low usage, spikes during AI processing
- **Network**: Depends on API usage and audio streaming

## 🔒 Security

### Data Privacy
- **API keys stored locally** - Never sent to external servers except target APIs
- **No conversation logging** - Chat history stays in browser session
- **HTTPS recommended** - For production deployments
- **No persistent storage** - Conversations not saved server-side

### Best Practices
- **Use environment variables** - For server deployments
- **Rotate API keys regularly** - Follow provider recommendations
- **Monitor API usage** - Track costs and limits
- **Use HTTPS in production** - Secure all communications

## 🚀 Deployment

### Local Development
```bash
python start.py
# Access at http://localhost:8000
```

### Production Deployment
```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000

# Using gunicorn (recommended)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Development Guidelines
- **Follow PEP 8** - Python code style
- **Add docstrings** - Document functions and classes
- **Write tests** - For new features
- **Update README** - For new features or changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini** - Advanced AI conversation capabilities
- **AssemblyAI** - High-quality speech recognition
- **Tavily** - Real-time search and information retrieval
- **FastAPI** - Modern, fast web framework
- **OpenAI, ElevenLabs, Murf** - Premium text-to-speech services

## 📞 Support

### Getting Help
- **GitHub Issues** - Bug reports and feature requests
- **Documentation** - Check this README and `/docs` endpoint
- **Community** - Join discussions in GitHub Discussions

### Reporting Issues
When reporting issues, please include:
- **Python version**
- **Browser and version**
- **Error messages**
- **Steps to reproduce**
- **API services being used**

---

**Made with ❤️ for the AI community**

*MARVIS - Your intelligent voice assistant for the modern world*