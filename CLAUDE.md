# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Telnyx voice agent application that creates a WebSocket-based voice streaming service. This is just a simple POC to test Telnyx. Endgoal is to integrate realtime AI voice agent and talk to him over the real phone - using Telnyx as a service for phone calls. The system handles incoming phone calls, establishes WebSocket connections for real-time audio streaming, and can integrate with AI voice processing services.

## Key Architecture Components

### Refactored Modular Structure
- **Main App**: `voice_agent.py` - Clean FastAPI server (100 lines)
- **Models**: `models/call_session.py` - Call state management and session tracking
- **Services**: 
  - `services/telnyx_service.py` - Telnyx API wrapper
  - `services/audio_service.py` - Audio recording, playback, and format conversion
- **Handlers**:
  - `handlers/webhook_handler.py` - Telnyx event processing  
  - `handlers/websocket_handler.py` - Real-time audio streaming
- **Audio System**: Unified recording and MP3 playback with PCMU encoding

## Development Commands

### Running the Application
```bash
# Install dependencies
uv sync

# Run the WORKING simplified voice agent
uv run python voice_agent_simple.py
# Server starts on 0.0.0.0:8080

# Alternative: Run the refactored version (has issues)
uv run python voice_agent.py

# Create test audio file (optional)
uv run python create_test_audio.py
```

### Jupyter Notebooks
- `telnyx_setup.ipynb`: Telnyx API configuration and phone number setup
- `test.ipynb`: Testing and experimentation

## Environment Configuration

Required environment variables (create `.env` file):
- `TELNYX_API_KEY`: Your Telnyx API key for phone service integration

## Key Technical Details

### WebSocket Audio Protocol
- Uses PCMU (μ-law) encoding at 8kHz sample rate for Telnyx compatibility
- Audio data exchanged as base64-encoded payloads in JSON messages
- Supports bidirectional streaming (inbound/outbound tracks)

### Telnyx Integration Points
- Webhook endpoint at `/webhook` handles call events (initiated, answered, hangup, etc.)
- WebSocket endpoint at `/audio` manages real-time audio streaming
- Call Control API for answering calls and text-to-speech

### Audio Processing
- `MP3AudioLogger` class handles audio file creation with WAV fallback
- Automatic audio chunk buffering and file finalization
- Musical tone generation for testing outbound audio streams

## Project Dependencies

Core libraries:
- `fastapi`: Web framework for webhooks and WebSocket server
- `telnyx`: Official Telnyx SDK for call control and phone number management  
- `websockets`: WebSocket client/server implementation
- `uvicorn`: ASGI server for FastAPI application

The application is designed to work with ngrok tunnels for webhook delivery during development.

## Refactored Architecture

The codebase was refactored from a monolithic 435-line file into a clean modular structure:

### File Structure
```
voice_agent.py              # Main FastAPI app (100 lines)
models/call_session.py      # Call session management
services/audio_service.py   # Audio recording/playbook
services/telnyx_service.py  # Telnyx API interactions  
handlers/webhook_handler.py # Event processing
handlers/websocket_handler.py # WebSocket streaming
```

### Key Improvements
- **Clean separation of concerns** - each module has single responsibility
- **Unified audio recording** - single file per conversation
- **MP3 playback support** - can play audio files during calls
- **Session management** - proper call state tracking
- **Future AI-ready** - designed for easy AI integration

### Configuration
Update the ngrok URL in `voice_agent.py`:
```python
NGROK_URL = "your-ngrok-url.ngrok-free.app"
```

Add optional test audio as `test_audio.mp3` in root directory.

## ⚠️ IMPORTANT: Use voice_agent_simple.py

The refactored modular version (`voice_agent.py`) has critical issues:
- Audio streaming too slow (10Hz instead of 50Hz)  
- Transcription causing playback interference
- Over-engineered PCMU conversion breaking audio
- Complex session management slowing the loop

**Use `voice_agent_simple.py` for working audio streaming.**

## Current Issues and Status

### Working: `voice_agent_simple.py`
- ✅ Continuous audio streaming at 50Hz
- ✅ Simple PCMU tone generation
- ✅ Raw audio recording
- ✅ No transcription interference

### Broken: `voice_agent.py` (refactored version)  
- ❌ Audio streaming too slow (causes no audio after welcome)
- ❌ Empty recordings due to complex PCMU conversion
- ❌ Transcription triggers playback events that interfere
- ❌ Over-engineered architecture introduces bugs

The modular architecture needs rework to fix core audio streaming issues.