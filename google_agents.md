# Google GenAI Realtime Voice Agents Guide

## Overview

The `google-genai` Python package provides powerful realtime voice agent capabilities through the Live API, enabling bidirectional audio streaming with minimal latency and natural conversation flow.

## Installation

```bash
pip install google-genai
pip install soundfile librosa  # For audio processing
```

## Key Features

- **Bidirectional Audio Streaming**: Real-time voice conversations
- **Voice Activity Detection (VAD)**: Natural interruption support
- **Multi-modal Support**: Audio, video, and text processing
- **24 Language Support**: Multilingual conversations
- **Audio Formats**: Input 16kHz PCM, Output 24kHz PCM

## Basic Setup

```python
import asyncio
import io
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
import soundfile as sf
import librosa

# Initialize client
client = genai.Client(api_key="YOUR_API_KEY")

# Available models
MODEL = "gemini-2.0-flash-live-001"  # Primary live model
# Other options: "gemini-live-2.5-flash-preview", "gemini-2.5-flash-preview-native-audio-dialog"

# Configuration
CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    system_instruction="You are a helpful voice assistant."
)
```

## Realtime Voice Agent Implementation

### 1. Basic Live Session

```python
async def basic_voice_session():
    """Basic voice conversation with Google GenAI"""
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        # Send audio input
        await session.send("Hello, can you hear me?")
        
        # Listen for responses - USE session.receive()
        async for response in session.receive():
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data:
                        # Handle audio response
                        audio_data = part.inline_data.data
                        await play_audio(audio_data)

async def play_audio(audio_data: bytes):
    """Play received audio data"""
    # Convert and play audio
    import sounddevice as sd
    import numpy as np
    
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    sd.play(audio_array, samplerate=24000)
    sd.wait()
```

### 2. Advanced Audio Processing

```python
class GoogleVoiceAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-live-preview"
        self.session = None
        
    async def start_session(self):
        """Initialize live session"""
        config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
            system_instruction="You are a friendly voice assistant."
        )
        
        self.session = await self.client.aio.live.connect(
            model=self.model, 
            config=config
        ).__aenter__()
        
    async def send_audio(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Send audio to the agent"""
        if not self.session:
            await self.start_session()
            
        # Convert audio to proper format if needed
        if sample_rate != 16000:
            audio_bytes = await self.convert_audio_format(audio_bytes, sample_rate)
        
        # Send audio
        await self.session.send_realtime_input(
            media=types.Blob(
                data=audio_bytes, 
                mime_type=f'audio/pcm;rate={sample_rate}'
            )
        )
    
    async def get_audio_response(self):
        """Get audio response from agent"""
        if not self.session:
            return None
            
        async for response in self.session.receive():
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith('audio/'):
                        return part.inline_data.data
        return None
    
    async def convert_audio_format(self, audio_data: bytes, from_rate: int, to_rate: int = 16000):
        """Convert audio sample rate"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Resample
        resampled = librosa.resample(audio_array, orig_sr=from_rate, target_sr=to_rate)
        
        # Convert back to int16 bytes
        return (resampled * 32767).astype(np.int16).tobytes()
```

### 3. Integration with Audio Streams

```python
import pyaudio
import asyncio
from collections import deque

class StreamingVoiceAgent:
    def __init__(self, api_key: str):
        self.agent = GoogleVoiceAgent(api_key)
        self.audio_queue = deque()
        self.is_listening = False
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
    async def start_streaming(self):
        """Start continuous audio streaming"""
        await self.agent.start_session()
        
        # Start audio capture
        capture_task = asyncio.create_task(self.capture_audio())
        process_task = asyncio.create_task(self.process_audio())
        response_task = asyncio.create_task(self.handle_responses())
        
        await asyncio.gather(capture_task, process_task, response_task)
    
    async def capture_audio(self):
        """Capture audio from microphone"""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        
        try:
            while self.is_listening:
                data = await asyncio.to_thread(stream.read, self.CHUNK)
                self.audio_queue.append(data)
                await asyncio.sleep(0.001)  # Small delay
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    async def process_audio(self):
        """Process queued audio chunks"""
        while self.is_listening:
            if self.audio_queue:
                chunk = self.audio_queue.popleft()
                await self.agent.send_audio(chunk, self.RATE)
            await asyncio.sleep(0.01)
    
    async def handle_responses(self):
        """Handle incoming audio responses"""
        while self.is_listening:
            response_audio = await self.agent.get_audio_response()
            if response_audio:
                await self.play_response(response_audio)
    
    async def play_response(self, audio_data: bytes):
        """Play agent's audio response"""
        import sounddevice as sd
        import numpy as np
        
        # Convert to numpy array and play
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sd.play(audio_array, samplerate=24000)  # GenAI outputs at 24kHz
```

### 4. Telnyx Integration Example

```python
async def telnyx_google_bridge(telnyx_audio: bytes, sample_rate: int = 8000):
    """Bridge Telnyx audio (8kHz PCMU) with Google GenAI (16kHz PCM)"""
    
    # Initialize Google agent
    agent = GoogleVoiceAgent("YOUR_API_KEY")
    await agent.start_session()
    
    # Convert PCMU to PCM
    import audioop
    pcm_data = audioop.ulaw2lin(telnyx_audio, 2)  # Convert μ-law to linear PCM
    
    # Resample from 8kHz to 16kHz
    audio_16k = await agent.convert_audio_format(pcm_data, from_rate=8000, to_rate=16000)
    
    # Send to Google GenAI
    await agent.send_audio(audio_16k, sample_rate=16000)
    
    # Get response
    response_audio = await agent.get_audio_response()
    
    if response_audio:
        # Convert response from 24kHz PCM back to 8kHz PCMU for Telnyx
        downsampled = await agent.convert_audio_format(response_audio, from_rate=24000, to_rate=8000)
        pcmu_response = audioop.lin2ulaw(downsampled, 2)
        return pcmu_response
    
    return None
```

### 5. Complete Voice Agent Server

```python
from fastapi import FastAPI, WebSocket
import json
import base64

app = FastAPI()

class VoiceAgentServer:
    def __init__(self):
        self.agents = {}  # Track multiple sessions
        
    async def handle_telnyx_audio(self, call_id: str, audio_data: bytes):
        """Process audio from Telnyx and return AI response"""
        
        # Get or create agent for this call
        if call_id not in self.agents:
            self.agents[call_id] = GoogleVoiceAgent("YOUR_API_KEY")
            await self.agents[call_id].start_session()
        
        agent = self.agents[call_id]
        
        # Process audio through Google GenAI
        response = await telnyx_google_bridge(audio_data)
        return response

voice_server = VoiceAgentServer()

@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Telnyx audio streaming"""
    await websocket.accept()
    call_id = None
    
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            
            if data.get("event") == "start":
                call_id = data.get("start", {}).get("call_control_id")
                
            elif data.get("event") == "media" and call_id:
                # Get audio from Telnyx
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    audio_data = base64.b64decode(payload)
                    
                    # Process with Google GenAI
                    response_audio = await voice_server.handle_telnyx_audio(call_id, audio_data)
                    
                    # Send response back to Telnyx
                    if response_audio:
                        response_payload = base64.b64encode(response_audio).decode()
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "media": {"payload": response_payload}
                        }))
                        
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if call_id and call_id in voice_server.agents:
            del voice_server.agents[call_id]
```

## Key Configuration Options

### Live API Models (2025)
- `gemini-2.0-flash-live-001`: Main Gemini 2.0 Live model (recommended)
- `gemini-live-2.5-flash-preview`: Half-cascade audio model
- `gemini-2.5-flash-preview-native-audio-dialog`: Native audio output model
- `gemini-2.5-flash-preview-native-audio-thinking-dialog`: Native audio with thinking

### Voice Models
- `Puck`: Conversational, expressive
- `Charon`: Deep, authoritative
- `Kore`: Warm, friendly
- `Fenrir`: Energetic, dynamic

### Audio Processing
- **Input**: 16-bit PCM, 16kHz, mono
- **Output**: 16-bit PCM, 24kHz, mono
- **Latency**: ~300-500ms typical
- **Session Limit**: 10 concurrent, 30 minutes each

### Rate Limits
- 10 concurrent sessions per project
- 25 tokens/second for audio
- Regional availability: us-central1

## Best Practices

1. **Audio Format Management**: Always convert between Telnyx (8kHz PCMU) and GenAI (16kHz PCM)
2. **Session Management**: Properly cleanup sessions to avoid hitting rate limits
3. **Error Handling**: Implement robust error handling for network issues
4. **Buffering**: Use audio buffering for smooth real-time processing
5. **Voice Activity Detection**: Leverage built-in VAD for natural conversations

This guide provides everything needed to implement Google GenAI realtime voice agents with proper audio handling and integration capabilities.