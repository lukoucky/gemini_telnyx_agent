# Telnyx Python Library Guide for Bidirectional Realtime Phone Calls

## Overview

The Telnyx Python library provides comprehensive support for creating servers capable of bidirectional realtime phone calls using WebSocket media streaming. This guide focuses on audio streaming to and from Telnyx's network for AI-powered voice applications.

## Installation

```bash
pip install --upgrade telnyx
```

## Basic Configuration

```python
import telnyx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API key
telnyx.api_key = os.getenv("TELNYX_API_KEY")
```

## Core Architecture

### WebSocket Media Streaming

Telnyx provides real-time access to call media through WebSocket connections. The system forks call media without quality degradation, enabling simultaneous processing and analysis.

**Key Features:**
- Real-time bidirectional audio streaming
- Multiple codec support (PCMU, PCMA, G722, OPUS, AMR-WB)
- Base64-encoded audio payloads
- Support for both inbound and outbound tracks

### Audio Format Specifications

- **Sample Rate**: 8kHz for PCMU/PCMA, variable for other codecs
- **Encoding**: Base64-encoded RTP payloads
- **Frame Size**: Typically 20ms audio frames
- **Payload Limit**: Maximum 1 media payload per second for bidirectional streaming

## Implementing Bidirectional Streaming

### 1. Initialize Call with Streaming

```python
import telnyx

def initiate_call_with_streaming():
    """Start a call with bidirectional media streaming enabled"""
    try:
        call = telnyx.Call.create(
            connection_id="your-connection-id",
            to="+18005550199",
            from_="+18005550100",
            stream_url="wss://yourdomain.com/audio",
            stream_track="both_tracks",  # Options: inbound_track, outbound_track, both_tracks
            stream_bidirectional_mode="rtp",  # Enable bidirectional RTP streaming
            stream_bidirectional_codec="PCMU",  # Codec for bidirectional audio
            stream_bidirectional_target_legs="self"  # Target call legs
        )
        return call
    except Exception as e:
        print(f"Error initiating call: {e}")
        return None
```

### 2. WebSocket Server Implementation

```python
from fastapi import FastAPI, WebSocket
import json
import base64
import asyncio

app = FastAPI()

# Global call tracking
active_calls = {}

@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for audio streaming"""
    await websocket.accept()
    call_id = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("event") == "connected":
                print("WebSocket connected")
                
            elif message.get("event") == "start":
                call_id = message.get("start", {}).get("call_control_id")
                if call_id:
                    active_calls[call_id] = websocket
                    print(f"Started streaming for call: {call_id}")
                    
            elif message.get("event") == "media":
                # Process incoming audio
                await handle_incoming_audio(message, websocket)
                
            elif message.get("event") == "stop":
                print("Streaming stopped")
                if call_id and call_id in active_calls:
                    del active_calls[call_id]
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if call_id and call_id in active_calls:
            del active_calls[call_id]
```

### 3. Audio Processing Functions

```python
async def handle_incoming_audio(message, websocket):
    """Process incoming audio from Telnyx"""
    media_data = message.get("media", {})
    payload = media_data.get("payload")
    
    if payload:
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(payload)
            
            # Process audio (transcription, AI analysis, etc.)
            processed_audio = await process_audio_chunk(audio_data)
            
            # Send response audio back to call
            if processed_audio:
                await send_audio_to_call(websocket, processed_audio)
                
        except Exception as e:
            print(f"Error processing audio: {e}")

async def send_audio_to_call(websocket, audio_data):
    """Send audio back to the call via WebSocket"""
    try:
        # Encode audio as base64
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Prepare media message
        media_message = {
            "event": "media",
            "media": {
                "payload": encoded_audio
            }
        }
        
        # Send to WebSocket
        await websocket.send_text(json.dumps(media_message))
        
    except Exception as e:
        print(f"Error sending audio: {e}")

async def process_audio_chunk(audio_data):
    """Process raw audio data - implement your AI logic here"""
    # Example: Convert PCMU to WAV, run through AI, convert back
    # This is where you'd integrate with speech recognition,
    # AI processing, text-to-speech, etc.
    
    # For now, return silence or processed audio
    return audio_data  # Placeholder
```

### 4. Webhook Handler for Call Events

```python
@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle Telnyx webhook events"""
    try:
        data = await request.json()
        event_type = data.get("data", {}).get("event_type")
        
        if event_type == "call.initiated":
            # Answer the call with streaming enabled
            call_control_id = data["data"]["call_control_id"]
            
            telnyx.Call.answer(
                call_control_id,
                stream_url="wss://yourdomain.com/audio",
                stream_track="both_tracks",
                stream_bidirectional_mode="rtp",
                stream_bidirectional_codec="PCMU"
            )
            
        elif event_type == "call.answered":
            print("Call answered, streaming should begin")
            
        elif event_type == "call.hangup":
            print("Call ended")
            
        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        print(f"Webhook error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
```

## Audio Format Conversion

### PCMU Encoding/Decoding

```python
import audioop
import wave

def pcmu_to_wav(pcmu_data, sample_rate=8000):
    """Convert PCMU (μ-law) audio to WAV format"""
    try:
        # Decode μ-law to linear PCM
        linear_data = audioop.ulaw2lin(pcmu_data, 2)
        return linear_data
    except Exception as e:
        print(f"Error converting PCMU to WAV: {e}")
        return None

def wav_to_pcmu(wav_data):
    """Convert WAV audio to PCMU (μ-law) format"""
    try:
        # Convert linear PCM to μ-law
        pcmu_data = audioop.lin2ulaw(wav_data, 2)
        return pcmu_data
    except Exception as e:
        print(f"Error converting WAV to PCMU: {e}")
        return None

def generate_tone(frequency=440, duration=1.0, sample_rate=8000):
    """Generate a sine wave tone in PCMU format"""
    import math
    
    samples = int(sample_rate * duration)
    tone_data = []
    
    for i in range(samples):
        # Generate sine wave
        sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        tone_data.append(sample)
    
    # Convert to bytes and then to PCMU
    wav_bytes = bytes()
    for sample in tone_data:
        wav_bytes += sample.to_bytes(2, byteorder='little', signed=True)
    
    return wav_to_pcmu(wav_bytes)
```

## Complete Example: AI Voice Agent

```python
from fastapi import FastAPI, WebSocket, Request
import telnyx
import asyncio
import json
import base64
import os

app = FastAPI()
telnyx.api_key = os.getenv("TELNYX_API_KEY")

# WebSocket URL for your server
STREAM_URL = "wss://your-ngrok-url.ngrok-free.app/audio"

@app.post("/webhook")
async def webhook_handler(request: Request):
    """Handle incoming call events"""
    data = await request.json()
    event_type = data.get("data", {}).get("event_type")
    
    if event_type == "call.initiated":
        call_control_id = data["data"]["call_control_id"]
        
        # Answer with bidirectional streaming
        telnyx.Call.answer(
            call_control_id,
            stream_url=STREAM_URL,
            stream_track="both_tracks",
            stream_bidirectional_mode="rtp",
            stream_bidirectional_codec="PCMU",
            stream_bidirectional_target_legs="self"
        )
    
    return {"status": "ok"}

@app.websocket("/audio")
async def audio_handler(websocket: WebSocket):
    """Handle real-time audio streaming"""
    await websocket.accept()
    
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            
            if data.get("event") == "media":
                # Get incoming audio
                payload = data.get("media", {}).get("payload", "")
                
                if payload:
                    # Decode and process audio
                    audio_data = base64.b64decode(payload)
                    
                    # Your AI processing here
                    response_audio = await your_ai_process_function(audio_data)
                    
                    # Send response back
                    if response_audio:
                        response_payload = base64.b64encode(response_audio).decode()
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "media": {"payload": response_payload}
                        }))
                        
    except Exception as e:
        print(f"Audio handler error: {e}")

async def your_ai_process_function(audio_data):
    """Implement your AI logic here"""
    # 1. Convert PCMU to WAV
    # 2. Run speech recognition
    # 3. Process with AI/LLM
    # 4. Generate speech response
    # 5. Convert back to PCMU
    # 6. Return processed audio
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Best Practices

### Performance Optimization
- Use async/await for all I/O operations
- Implement audio buffering for smooth playback
- Process audio in chunks (typically 20ms frames)
- Minimize latency in audio processing pipeline

### Error Handling
- Implement robust WebSocket reconnection logic
- Handle audio format conversion errors gracefully
- Monitor call state and cleanup resources properly

### Security
- Validate webhook signatures from Telnyx
- Use secure WebSocket connections (WSS)
- Implement proper authentication for your endpoints
- Never log sensitive audio data

## Advanced Features

### Multiple Call Support
```python
# Track multiple concurrent calls
call_sessions = {}

async def handle_multiple_calls(call_id, websocket):
    call_sessions[call_id] = {
        'websocket': websocket,
        'start_time': time.time(),
        'audio_buffer': []
    }
```

### Real-time Transcription Integration
```python
async def transcribe_audio(audio_data):
    """Integrate with speech recognition service"""
    # Example: Google Speech-to-Text, Azure Speech, etc.
    pass
```

### AI Response Generation
```python
async def generate_ai_response(transcript):
    """Generate AI response from transcript"""
    # Example: OpenAI GPT, Claude, etc.
    pass
```

## Troubleshooting WebSocket Media Streaming

### Common Issue: Receiving "start" Events but No "media" Events

If your WebSocket receives "start" events but no "media" events, check the following:

#### 1. Verify Parameter Configuration

**Critical Parameters for Receiving Caller Audio:**

```python
telnyx.Call.answer(
    call_control_id,
    stream_url=STREAM_URL,
    stream_track="both_tracks",  # or "inbound_track" for caller audio only
    stream_bidirectional_mode="rtp",  # Required for bidirectional streaming
    stream_bidirectional_codec="PCMU",  # Case sensitive: PCMU, PCMA, G722, OPUS, AMR-WB
    stream_bidirectional_target_legs="self"  # Options: "self", "opposite", "both"
)
```

**Parameter Details:**
- `stream_track`: 
  - `"inbound_track"` (default) - caller audio only
  - `"outbound_track"` - Telnyx generated audio only  
  - `"both_tracks"` - both caller and Telnyx audio
- `stream_bidirectional_mode`:
  - `"mp3"` (default) - for sending MP3 audio back
  - `"rtp"` - required for receiving RTP audio streams
- `stream_bidirectional_codec`: 
  - `"PCMU"` (default, 8kHz)
  - `"PCMA"` (8kHz)
  - `"G722"` (8kHz)
  - `"OPUS"` (8/16kHz)
  - `"AMR-WB"` (8/16kHz)
- `stream_bidirectional_target_legs`:
  - `"self"` - audio sent to the leg that initiated streaming
  - `"opposite"` (default) - audio sent to the other party
  - `"both"` - audio sent to both call legs

#### 2. Event Flow Verification

Correct WebSocket event sequence:
1. `"connected"` - WebSocket connection established
2. `"start"` - Contains media format metadata (encoding, sample_rate, channels)
3. `"media"` - Base64-encoded RTP payload chunks (should arrive continuously)
4. `"stop"` - Streaming ended

#### 3. Common Configuration Issues

**Issue: No media events with `stream_bidirectional_mode="mp3"`**
- Solution: Use `stream_bidirectional_mode="rtp"` to receive incoming audio
- MP3 mode is primarily for sending audio back to the call

**Issue: Wrong track selection**
- For caller audio: Use `stream_track="inbound_track"` or `"both_tracks"`
- For Telnyx audio: Use `stream_track="outbound_track"` or `"both_tracks"`

**Issue: Case sensitivity**
- Codec values are case sensitive: use `"PCMU"` not `"pcmu"`

#### 4. Debugging Steps

```python
@app.websocket("/audio")
async def debug_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            event_type = message.get("event")
            
            print(f"Received event: {event_type}")
            
            if event_type == "start":
                # Verify media format in start event
                media_format = message.get("start", {}).get("media_format", {})
                print(f"Media format: {media_format}")
                
            elif event_type == "media":
                # Count media events received
                payload_size = len(message.get("media", {}).get("payload", ""))
                print(f"Media event received, payload size: {payload_size}")
                
    except Exception as e:
        print(f"WebSocket error: {e}")
```

#### 5. Network and Timing Issues

- **WebSocket URL**: Ensure your URL is accessible (test with tools like ngrok)
- **Rate limiting**: Media events arrive continuously, but sending is limited to 1 per second
- **Connection timing**: Consider using early WebSocket establishment if connection takes time

#### 6. Audio Processing Verification

```python
async def verify_audio_reception(message):
    """Verify incoming audio data"""
    media_data = message.get("media", {})
    payload = media_data.get("payload")
    
    if payload:
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(payload)
            print(f"Received {len(audio_data)} bytes of audio data")
            
            # For PCMU, expect ~160 bytes per 20ms frame (8kHz sample rate)
            expected_frame_size = 160
            if len(audio_data) != expected_frame_size:
                print(f"Warning: Unexpected frame size. Expected ~{expected_frame_size}, got {len(audio_data)}")
                
        except Exception as e:
            print(f"Error decoding audio: {e}")
```

This guide provides the foundation for building robust bidirectional voice applications with Telnyx's Python library. The WebSocket streaming capabilities enable real-time AI interactions with sub-second latency.

## Simple Answering Machine Implementation

### Core Components for Answering Machine

```python
from fastapi import FastAPI, Request
import telnyx
import os

app = FastAPI()
telnyx.api_key = os.getenv("TELNYX_API_KEY")

@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming call events for answering machine"""
    data = await request.json()
    event_type = data.get("data", {}).get("event_type")
    
    if event_type == "call.initiated":
        call_control_id = data["data"]["call_control_id"]
        
        # Answer the call
        call = telnyx.Call()
        call.call_control_id = call_control_id
        call.answer()
        
    elif event_type == "call.answered":
        call_control_id = data["data"]["call_control_id"]
        
        # Play greeting message using TTS
        call = telnyx.Call()
        call.call_control_id = call_control_id
        call.speak(
            payload="Hello! You've reached our answering machine. Please leave a message after the beep and we'll get back to you soon.",
            voice="female",
            language="en-US"
        )
        
    elif event_type == "call.speak.ended":
        call_control_id = data["data"]["call_control_id"]
        
        # Start recording after greeting finishes
        call = telnyx.Call()
        call.call_control_id = call_control_id
        call.record_start(
            format="mp3",
            channels="single",
            play_beep=True,
            max_length=300,  # 5 minutes max
            recording_track="inbound"  # Record caller only
        )
        
    elif event_type == "call.recording.saved":
        # Handle saved recording
        recording_id = data["data"]["recording_id"]
        recording_url = data["data"]["public_recording_url"]
        print(f"Recording saved: {recording_id} at {recording_url}")
        
    elif event_type == "call.hangup":
        print("Call ended")
        
    return {"status": "ok"}
```

### Text-to-Speech (TTS) API

The Telnyx Speak API converts text to speech and plays it on the call.

**API Endpoint**: `POST /calls/{call_control_id}/actions/speak`

**Key Parameters**:
- `payload` (required): Text to convert to speech (3,000 character limit)
- `voice` (required): Voice for speech synthesis
- `language`: Language code (default: en-US)
- `service_level`: "basic" or "premium" (default: premium)
- `payload_type`: "text" or "ssml" (default: text)

**Supported Languages**: arb, cmn-CN, cy-GB, da-DK, de-DE, en-AU, en-GB, en-GB-WLS, en-IN, en-US, es-ES, es-MX, es-US, fr-CA, fr-FR, hi-IN, is-IS, it-IT, ja-JP, ko-KR, nb-NO, nl-NL, pl-PL, pt-BR, pt-PT, ro-RO, ru-RU, sv-SE, tr-TR

**Python SDK Usage**:
```python
# Using telnyx.Call object
call = telnyx.Call()
call.call_control_id = "your_call_control_id"
call.speak(
    payload="Your message here",
    voice="female",
    language="en-US"
)

# Multiple TTS commands are queued automatically
call.speak(payload="First message")
call.speak(payload="Second message")  # Will play after first completes
```

**Webhook Events Triggered**:
- `call.speak.started`
- `call.speak.ended`

### Call Recording API

The Telnyx Recording API allows you to record calls and save them as audio files.

**API Endpoint**: `POST /calls/{call_control_id}/actions/record_start`

**Required Parameters**:
- `format`: Audio file format ("wav" or "mp3")
- `channels`: Recording channel configuration ("single" or "dual")

**Optional Parameters**:
- `play_beep`: Play beep at recording start (boolean)
- `max_length`: Maximum recording duration in seconds (0-43200, 0 = infinite)
- `recording_track`: Audio track to record ("both", "inbound", "outbound")
- `custom_file_name`: Custom filename for the recording
- `transcription`: Enable post-recording transcription (boolean)

**Python SDK Usage**:
```python
# Start recording
call = telnyx.Call()
call.call_control_id = "your_call_control_id"
call.record_start(
    format="mp3",
    channels="single",
    play_beep=True,
    max_length=300,  # 5 minutes
    recording_track="inbound",  # Record caller only
    transcription=True  # Enable transcription
)

# Stop recording (optional - stops automatically on hangup)
call.record_stop()
```

**Webhook Events Triggered**:
- `call.recording.saved`
- `call.recording.transcription.saved` (if transcription enabled)
- `call.recording.error`

### Complete Answering Machine Example

```python
from fastapi import FastAPI, Request
import telnyx
import os
from datetime import datetime

app = FastAPI()
telnyx.api_key = os.getenv("TELNYX_API_KEY")

# Store active calls
active_calls = {}

@app.post("/webhook")
async def answering_machine_webhook(request: Request):
    """Complete answering machine webhook handler"""
    try:
        data = await request.json()
        event_type = data.get("data", {}).get("event_type")
        call_control_id = data.get("data", {}).get("call_control_id")
        
        if not call_control_id:
            return {"status": "ok"}
            
        call = telnyx.Call()
        call.call_control_id = call_control_id
        
        if event_type == "call.initiated":
            # Answer incoming call
            call.answer()
            active_calls[call_control_id] = {
                "start_time": datetime.now(),
                "caller": data.get("data", {}).get("from")
            }
            
        elif event_type == "call.answered":
            # Play greeting message
            greeting = (
                "Hello! You've reached our answering machine. "
                "We're unable to take your call right now, but your call is important to us. "
                "Please leave a detailed message after the beep, and we'll get back to you as soon as possible. "
                "Thank you!"
            )
            call.speak(
                payload=greeting,
                voice="female",
                language="en-US"
            )
            
        elif event_type == "call.speak.ended":
            # Start recording after greeting
            call.record_start(
                format="mp3",
                channels="single",
                play_beep=True,
                max_length=300,  # 5 minutes max
                recording_track="inbound",
                transcription=True,
                custom_file_name=f"voicemail_{call_control_id}_{int(datetime.now().timestamp())}"
            )
            
        elif event_type == "call.recording.saved":
            # Handle saved voicemail
            recording_data = data.get("data", {})
            recording_id = recording_data.get("recording_id")
            recording_url = recording_data.get("public_recording_url")
            duration = recording_data.get("duration_millis", 0) / 1000
            
            caller_info = active_calls.get(call_control_id, {})
            
            print(f"New voicemail received:")
            print(f"  From: {caller_info.get('caller', 'Unknown')}")
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Recording ID: {recording_id}")
            print(f"  Download URL: {recording_url}")
            
            # Here you could:
            # - Save to database
            # - Send notification email/SMS
            # - Process with AI for transcription summary
            # - Upload to cloud storage
            
        elif event_type == "call.recording.transcription.saved":
            # Handle transcription if enabled
            transcription_data = data.get("data", {})
            transcription_text = transcription_data.get("transcription_text", "")
            
            print(f"Voicemail transcription: {transcription_text}")
            
        elif event_type == "call.hangup":
            # Clean up call tracking
            if call_control_id in active_calls:
                del active_calls[call_control_id]
            print(f"Call {call_control_id} ended")
            
        return {"status": "ok"}
        
    except Exception as e:
        print(f"Webhook error: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Downloading and Managing Recordings

```python
import requests
import telnyx

# Retrieve recording details
def get_recording_details(recording_id):
    """Get recording information"""
    try:
        response = requests.get(
            f"https://api.telnyx.com/v2/recordings/{recording_id}",
            headers={
                "Authorization": f"Bearer {telnyx.api_key}",
                "Content-Type": "application/json"
            }
        )
        return response.json()
    except Exception as e:
        print(f"Error retrieving recording: {e}")
        return None

# Download recording file
def download_recording(recording_url, local_filename):
    """Download recording to local file"""
    try:
        response = requests.get(recording_url)
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"Recording saved to {local_filename}")
        return True
    except Exception as e:
        print(f"Error downloading recording: {e}")
        return False

# Usage example
recording_info = get_recording_details("recording_id_here")
if recording_info:
    download_url = recording_info["data"]["download_url"]
    download_recording(download_url, f"voicemail_{recording_id}.mp3")
```

This answering machine implementation provides:
1. **Automatic call answering** with webhook handling
2. **Text-to-speech greeting** using Telnyx TTS
3. **Voice message recording** in MP3 format
4. **Optional transcription** of voicemails
5. **Call tracking and cleanup**
6. **Recording download and management**