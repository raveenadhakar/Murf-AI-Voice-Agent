# 🤖 AI Voice Chat Assistant

A modern, conversational AI voice assistant with a sleek chat interface. Talk to AI naturally using your voice and get intelligent responses back in both text and speech.


## ✨ Features

### 🎙️ **Voice Interaction**
- **Real-time voice recording** with visual feedback
- **Speech-to-Text** transcription using AssemblyAI
- **Text-to-Speech** responses using Murf AI voices
- **Continuous conversation** flow with auto-recording

### 💬 **Modern Chat Interface**
- **WhatsApp-style chat bubbles** with proper alignment
- **Real-time message updates** with smooth animations
- **Session-based conversation** history
- **Responsive design** for desktop and mobile

### 🧠 **Intelligent AI**
- **Contextual conversations** using Google Gemini
- **Memory across sessions** - AI remembers your conversation
- **Smart response handling** with fallback mechanisms
- **Error recovery** with graceful degradation

### 🛡️ **Robust Error Handling**
- **Multiple fallback layers** for each service (STT, LLM, TTS)
- **Client-side TTS backup** using browser speech synthesis
- **Network error recovery** with retry mechanisms
- **User-friendly error messages** and visual feedback

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   External      │
│   (HTML/CSS/JS) │◄──►│   Backend        │◄──►│   APIs          │
│                 │    │                  │    │                 │
│ • Chat UI       │    │ • Session Mgmt   │    │ • AssemblyAI    │
│ • Voice Record  │    │ • Error Handling │    │ • Google Gemini │
│ • Audio Playback│    │ • API Routing    │    │ • Murf TTS      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Data Flow**
1. **User speaks** → Browser records audio
2. **Audio uploaded** → FastAPI receives file
3. **Speech-to-Text** → AssemblyAI transcribes audio
4. **Context building** → Previous conversation loaded
5. **AI Processing** → Google Gemini generates response
6. **Text-to-Speech** → Murf converts response to audio
7. **Response delivery** → Audio played + text displayed
8. **Session storage** → Conversation saved for context

## 🛠️ Technologies Used

### **Backend**
- **FastAPI** - Modern Python web framework
- **Python 3.11+** - Core programming language
- **Uvicorn** - ASGI server for FastAPI
- **Pydantic** - Data validation and serialization

### **AI Services**
- **AssemblyAI** - Speech-to-Text transcription
- **Google Gemini** - Large Language Model for conversations
- **Murf AI** - High-quality Text-to-Speech synthesis

### **Frontend**
- **Vanilla JavaScript** - No frameworks, pure JS
- **HTML5** - Modern web standards
- **CSS3** - Advanced styling with animations
- **Web Audio API** - Browser-based audio recording

### **Storage**
- **In-memory sessions** - Fast conversation history
- **File system** - Temporary audio file handling

## 🚀 Getting Started

### **Prerequisites**
- Python 3.11 or higher
- Modern web browser with microphone access
- API keys for external services

### **Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voice-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   MURF_API_KEY=your_murf_api_key_here
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

### **API Keys Setup**

#### **1. AssemblyAI (Speech-to-Text)**
- Visit [AssemblyAI](https://www.assemblyai.com/)
- Sign up for a free account
- Get your API key from the dashboard
- Add to `.env` as `ASSEMBLYAI_API_KEY`

#### **2. Google Gemini (AI Conversations)**
- Go to [Google AI Studio](https://makersuite.google.com/)
- Create a new project
- Generate an API key
- Add to `.env` as `GEMINI_API_KEY`

#### **3. Murf AI (Text-to-Speech)**
- Visit [Murf AI](https://murf.ai/)
- Sign up for an account
- Get your API key from settings
- Add to `.env` as `MURF_API_KEY`

### **Running the Application**

1. **Start the FastAPI server**
   ```bash
   uvicorn main:app --reload
   ```

2. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

3. **Grant microphone permissions**
   Allow microphone access when prompted

4. **Start chatting!**
   Click the blue microphone button and start talking

## 📁 Project Structure

```
voice-agent/
├── main.py                    # ✅ Clean 70-line main app
├── config.py                  # ✅ Configuration management
├── schemas.py                 # ✅ Pydantic models
├── utils.py                   # ✅ Helper functions
├── routes/                    # ✅ Organized API routes
│   ├── __init__.py           
│   ├── audio.py              # 🎵 Audio processing endpoints
│   ├── chat.py               # 💬 Chat & conversation endpoints
│   └── system.py             # ⚙️ System & utility endpoints
├── static/                    # ✅ Frontend files
│   ├── index.html
│   └── style.css
├── uploads/                   # ✅ File uploads
├── .env                       # ✅ Environment variables
├── requirements.txt           # ✅ Dependencies
└── README.md                 # ✅ Documentation

```

## 🎯 API Endpoints

### **Main Endpoints**
- `GET /` - Serve the chat interface
- `POST /agent/chat/{session_id}` - Process voice messages
- `GET /agent/chat/{session_id}/history` - Get conversation history
- `DELETE /agent/chat/{session_id}` - Clear conversation history

### **Utility Endpoints**
- `POST /upload` - Upload audio files
- `POST /transcribe/file` - Transcribe audio to text
- `POST /text-to-speech` - Convert text to speech
- `GET /health` - Health check
- `POST /test/error/{type}` - Test error scenarios

## 🔧 Configuration

### **Environment Variables**
| Variable | Description | Required |
|----------|-------------|----------|
| `MURF_API_KEY` | Murf AI API key for TTS | Yes |
| `ASSEMBLYAI_API_KEY` | AssemblyAI API key for STT | Yes |
| `GEMINI_API_KEY` | Google Gemini API key for AI | Yes |

### **Server Configuration**
- **Host**: `127.0.0.1` (localhost)
- **Port**: `8000` (default)
- **Reload**: Enabled in development
- **Workers**: 1 (for session consistency)

## 🛡️ Error Handling

The application includes comprehensive error handling:

### **Service Failures**
- **STT Failure**: Graceful error messages with retry options
- **LLM Failure**: Fallback responses maintain conversation flow
- **TTS Failure**: Browser-based speech synthesis as backup
- **Network Issues**: Retry mechanisms with user feedback

### **Fallback Chain**
1. **Primary**: External API services (AssemblyAI, Gemini, Murf)
2. **Secondary**: Browser-based alternatives (Web Speech API)
3. **Tertiary**: Text-only responses with clear error messages

## 🎨 UI Features

### **Chat Interface**
- **Modern chat bubbles** with tails and proper alignment
- **Teal user messages** on the right side
- **Dark AI responses** on the left side
- **Smooth animations** for new messages
- **Auto-scroll** to latest messages

### **Voice Controls**
- **Large record button** with visual feedback
- **Pulse animations** during recording
- **Color changes** (blue → red) for state indication
- **Hover effects** and smooth transitions

### **Responsive Design**
- **Mobile-first** approach
- **Touch-friendly** button sizes
- **Adaptive layouts** for different screen sizes
- **Cross-browser** compatibility

## 🧪 Testing

### **Manual Testing**
1. Test voice recording and transcription
2. Verify AI responses and conversation flow
3. Check error handling with invalid inputs
4. Test on different browsers and devices

### **Error Simulation**
Use the built-in error testing endpoints:
```bash
curl -X POST "http://127.0.0.1:8000/test/error/stt"
curl -X POST "http://127.0.0.1:8000/test/error/llm"
curl -X POST "http://127.0.0.1:8000/test/error/tts"
```

## 🚀 Deployment

### **Local Development**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Production**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Use only 1 worker to maintain session consistency.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AssemblyAI** for excellent speech recognition
- **Google** for the powerful Gemini AI model
- **Murf AI** for high-quality voice synthesis
- **FastAPI** for the amazing web framework

## 📞 Support

If you encounter any issues:
1. Check the console for error messages
2. Verify all API keys are correctly set
3. Ensure microphone permissions are granted
4. Check network connectivity

For additional help, please open an issue in the repository.

---

**Built with ❤️ using modern web technologies and AI services**