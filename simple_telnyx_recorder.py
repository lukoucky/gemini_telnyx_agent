#!/usr/bin/env python3
"""
Simple Telnyx Audio Recorder
Just receives and saves audio from Telnyx calls - no AI agent
"""

import os
import json
import base64
import logging
import wave
from datetime import datetime
from typing import Dict

import telnyx
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Simple Telnyx Audio Recorder")

# Configuration
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
NGROK_URL = os.getenv("NGROK_URL", "26746fed832f.ngrok-free.app").replace("https://", "").replace("http://", "")
STREAM_URL = f"wss://{NGROK_URL}/audio"

# Initialize Telnyx
telnyx.api_key = TELNYX_API_KEY

# Track active recordings
active_recordings: Dict[str, wave.Wave_write] = {}


@app.post("/webhook")
async def handle_telnyx_webhook(request: Request):
    """Handle Telnyx webhook events"""
    try:
        webhook = await request.json()
        
        event_type = webhook["data"]["event_type"]
        payload = webhook["data"]["payload"]
        
        # Handle different payload structures
        if isinstance(payload, dict):
            call_control_id = payload["call_control_id"]
        else:
            call_control_id = payload
        
        logger.info(f"Received webhook: {event_type} for call {call_control_id}")
        
        if event_type == "call.initiated":
            # Answer the call with streaming
            logger.info(f"Answering call {call_control_id} with streaming")
            
            telnyx.Call.create_answer(
                call_control_id,
                stream_url=STREAM_URL,
                stream_track="inbound_track",  # Only receive caller audio
                stream_bidirectional_mode="rtp",
                stream_bidirectional_codec="PCMU"
            )
            
        elif event_type == "call.answered":
            logger.info(f"Call {call_control_id} answered, streaming should begin")
            
        elif event_type == "call.hangup":
            logger.info(f"Call {call_control_id} ended")
            
        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.websocket("/audio")
async def handle_audio_websocket(websocket: WebSocket):
    """Handle WebSocket audio streaming from Telnyx"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    call_id = None
    recording = None
    frame_count = 0
    
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")
            
            logger.info(f"WebSocket event: {event}")
            
            if event == "connected":
                logger.info("WebSocket connected successfully")
                
            elif event == "start":
                # Extract call info and media format
                start_data = data.get("start", {})
                call_id = start_data.get("call_control_id")
                media_format = start_data.get("media_format", {})
                
                logger.info(f"Started streaming for call: {call_id}")
                logger.info(f"Media format: {json.dumps(media_format, indent=2)}")
                
                # Create WAV file for recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"telnyx_recording_{timestamp}.wav"
                
                # Open WAV file with correct parameters
                recording = wave.open(filename, 'wb')
                recording.setnchannels(1)  # Mono
                recording.setsampwidth(2)  # 16-bit
                recording.setframerate(8000)  # 8kHz for PCMU
                
                active_recordings[call_id] = recording
                logger.info(f"Created recording file: {filename}")
                
            elif event == "media" and call_id:
                # Process incoming audio
                media_data = data.get("media", {})
                payload = media_data.get("payload", "")
                
                if payload:
                    try:
                        # Decode base64 audio data
                        audio_data = base64.b64decode(payload)
                        frame_count += 1
                        
                        # Log every 50 frames (~1 second)
                        if frame_count % 50 == 0:
                            logger.info(f"Received {frame_count} audio frames, latest size: {len(audio_data)} bytes")
                        
                        # Convert PCMU to PCM and save
                        if call_id in active_recordings:
                            import audioop
                            # Convert PCMU to linear PCM
                            pcm_data = audioop.ulaw2lin(audio_data, 2)
                            active_recordings[call_id].writeframes(pcm_data)
                            
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                else:
                    logger.warning("Received media event with empty payload")
                    
            elif event == "stop":
                logger.info(f"Streaming stopped for call: {call_id}")
                
                # Close recording
                if call_id in active_recordings:
                    active_recordings[call_id].close()
                    del active_recordings[call_id]
                    logger.info(f"Recording saved. Total frames: {frame_count}")
                
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if call_id and call_id in active_recordings:
            active_recordings[call_id].close()
            del active_recordings[call_id]
            
        logger.info("WebSocket connection closed")


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Simple Telnyx Audio Recorder",
        "webhook_url": f"https://{NGROK_URL}/webhook",
        "stream_url": STREAM_URL,
        "active_recordings": len(active_recordings),
        "status": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Verify environment
    if not TELNYX_API_KEY:
        logger.error("TELNYX_API_KEY not found in environment variables")
        exit(1)
    
    logger.info("Starting Simple Telnyx Audio Recorder")
    logger.info(f"Webhook URL: https://{NGROK_URL}/webhook")
    logger.info(f"Stream URL: {STREAM_URL}")
    logger.info("This will only record caller audio to WAV files")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )