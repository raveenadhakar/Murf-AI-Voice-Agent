# 🏴‍☠️ Troubleshooting Captain AI's Parrot

## Speech Recognition Issues

### ❌ "Speech recognition error: network"

**Problem**: Browser speech recognition requires specific conditions to work properly.

**Solutions**:

1. **Use HTTPS or localhost**:
   - ✅ `https://yoursite.com` (production)
   - ✅ `http://localhost:8000` (development)
   - ✅ `http://127.0.0.1:8000` (development)
   - ❌ `http://yoursite.com` (won't work)

2. **Check browser support**:
   - ✅ Chrome/Chromium (best support)
   - ✅ Edge (good support)
   - ⚠️ Firefox (limited support)
   - ❌ Safari (no support)

3. **Grant microphone permissions**:
   - Click the microphone icon in address bar
   - Select "Allow" for microphone access
   - Refresh the page if needed

### 🔧 Quick Fixes

1. **Use Text Input Instead**:
   - Click "⌨️ Type Message" button
   - Type your message directly
   - Works without microphone/speech recognition

2. **Try Different Browser**:
   - Chrome works best for speech recognition
   - Edge is also reliable
   - Avoid Firefox/Safari for voice features

3. **Check Network Connection**:
   - Speech recognition needs internet
   - Try refreshing the page
   - Check if other sites work

### 🌐 HTTPS Setup (Production)

If you need HTTPS for production deployment:

1. **Using ngrok** (quick testing):
   ```bash
   # Install ngrok
   npm install -g ngrok
   
   # Run your app
   python main.py
   
   # In another terminal
   ngrok http 8000
   ```

2. **Using Cloudflare Tunnel**:
   ```bash
   # Install cloudflared
   # Then run:
   cloudflared tunnel --url http://localhost:8000
   ```

3. **Using Let's Encrypt** (proper deployment):
   - Set up SSL certificate
   - Configure reverse proxy (nginx/apache)
   - Update DNS records

### 🎤 Alternative Input Methods

The pirate parrot supports multiple ways to communicate:

1. **Voice Input** (primary):
   - Click microphone button
   - Speak clearly
   - Works on HTTPS/localhost

2. **Text Input** (fallback):
   - Click "⌨️ Type Message"
   - Type your message
   - Always works

3. **Auto-fallback**:
   - App automatically shows text input on speech errors
   - Seamless transition between methods

### 🏴‍☠️ Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `network` | No internet or HTTPS required | Use HTTPS or localhost |
| `not-allowed` | Microphone permission denied | Grant microphone access |
| `no-speech` | No voice detected | Speak louder/closer to mic |
| `audio-capture` | Microphone not working | Check microphone settings |
| `service-not-allowed` | Speech service blocked | Use HTTPS or different browser |

### 🧪 Testing Speech Recognition

Run the test script to check your setup:

```bash
python test_pirate.py
```

Or test in browser console:
```javascript
// Check if speech recognition is available
if ('webkitSpeechRecognition' in window) {
    console.log('✅ Speech recognition supported');
} else {
    console.log('❌ Speech recognition not supported');
}

// Check if HTTPS
if (location.protocol === 'https:' || location.hostname === 'localhost') {
    console.log('✅ Secure context available');
} else {
    console.log('❌ HTTPS required for speech recognition');
}
```

### 🚀 Best Practices

1. **Development**: Use `http://localhost:8000`
2. **Production**: Always use HTTPS
3. **Browser**: Chrome/Edge recommended
4. **Fallback**: Text input always available
5. **Permissions**: Grant microphone access when prompted

### 🆘 Still Having Issues?

1. **Try the text input method** - always works
2. **Use Chrome browser** - best compatibility
3. **Check browser console** for detailed error messages
4. **Ensure microphone works** in other apps
5. **Try different network** if connection issues

**Remember**: The pirate parrot is smart enough to work with or without voice! 🏴‍☠️⚓