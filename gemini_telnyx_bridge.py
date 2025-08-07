#!/usr/bin/env python3
"""
Telnyx + Google Gemini Realtime Voice Agent Bridge
Connects incoming phone calls to Google Gemini realtime voice agents
"""

import asyncio
import json
import base64
import logging
import os
import audioop
import numpy as np
import wave
from typing import Dict, Optional
from collections import deque
from datetime import datetime

import telnyx
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Google GenAI imports
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
import librosa
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Initialize FastAPI app
app = FastAPI(title="Telnyx-Gemini Voice Bridge")

# Configuration
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
NGROK_URL = os.getenv("NGROK_URL", "26746fed832f.ngrok-free.app").replace("https://", "").replace("http://", "")
STREAM_URL = f"wss://{NGROK_URL}/audio"

# Initialize Telnyx
telnyx.api_key = TELNYX_API_KEY

# Gemini configuration
GEMINI_MODEL = "gemini-2.0-flash-live-001"
GEMINI_CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    system_instruction="""You are a helpful phone assistant. You're speaking to someone who called in.
    Keep responses conversational, concise, and natural. Ask how you can help them today.
    If you are asked what can you do, you say to the user that you can tell him a funny joke.
    Don't mention the joke unles you are asked about what you can do.
    If you don't hear anything from the user for more then 20 seconds, tell him he is quiet for 20 seconds and ask if everithing is ok.
    """
)


class GeminiVoiceAgent:
    """Google Gemini realtime voice agent wrapper"""
    
    def __init__(self, api_key: str, call_id: str):
        self.client = genai.Client(api_key=api_key)
        self.call_id = call_id
        self.session_context = None
        self.session = None
        self.is_connected = False
        self.response_queue = deque()
        self.converter = AudioConverter()
        self.listener_task = None
        
    async def start_session(self):
        """Initialize Gemini live session using proper context manager"""
        try:
            logger.info(f"Starting Gemini session for call {self.call_id}...")
            self.session_context = self.client.aio.live.connect(
                model=GEMINI_MODEL, 
                config=GEMINI_CONFIG
            )
            self.session = await self.session_context.__aenter__()
            self.is_connected = True
            logger.info(f"Gemini session started successfully for call {self.call_id}")
            
            # Start response listener with proper error handling
            self.listener_task = asyncio.create_task(self._listen_for_responses())
            
            # Monitor the listener task
            asyncio.create_task(self._monitor_listener())
            
        except Exception as e:
            logger.error(f"Failed to start Gemini session for call {self.call_id}: {e}")
            raise
    
    async def _listen_for_responses(self):
        """Listen for responses from Gemini"""
        logger.info(f"[{self.call_id}] Starting Gemini response listener")
        response_count = 0
        last_response_time = asyncio.get_event_loop().time()
        
        try:
            # Use session.receive() with proper error handling
            while self.is_connected:
                try:
                    async for response in self.session.receive():
                        if not self.is_connected:
                            logger.info(f"[{self.call_id}] Listener stopping - session disconnected")
                            break
                        
                        response_count += 1
                        last_response_time = asyncio.get_event_loop().time()
                        logger.info(f"[{self.call_id}] Gemini response #{response_count}: {type(response).__name__}")
                        
                        if response.server_content and response.server_content.model_turn:
                            logger.info(f"[{self.call_id}] Gemini model turn detected")
                            audio_parts = 0
                            text_parts = 0
                            
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.mime_type.startswith('audio/'):
                                    audio_parts += 1
                                    self.response_queue.append(part.inline_data.data)
                                    logger.info(f"[{self.call_id}] Audio response #{audio_parts}: {len(part.inline_data.data)} bytes (queue size: {len(self.response_queue)})")
                                elif hasattr(part, 'text') and part.text:
                                    text_parts += 1
                                    logger.info(f"[{self.call_id}] Text response #{text_parts}: {part.text}")
                            
                            logger.info(f"[{self.call_id}] Model turn complete: {audio_parts} audio parts, {text_parts} text parts")
                        
                        # Check for other response types
                        if hasattr(response, 'text') and response.text:
                            logger.info(f"[{self.call_id}] Direct text response: {response.text}")
                    
                    # If we get here, the async for loop completed normally
                    logger.warning(f"[{self.call_id}] Gemini receive loop completed unexpectedly after {response_count} responses")
                    
                    # If session is still connected but loop ended, this is abnormal
                    if self.is_connected:
                        logger.error(f"[{self.call_id}] CRITICAL: Gemini receive loop ended while session still connected!")
                        logger.error(f"[{self.call_id}] This suggests Gemini session closed prematurely")
                        
                        # Try to detect if session is actually still alive
                        try:
                            # Send a small audio chunk to test if session is responsive
                            test_audio = b'\x00' * 320  # 20ms of silence at 16kHz
                            await self.session.send_realtime_input(
                                media=types.Blob(data=test_audio, mime_type='audio/pcm;rate=16000')
                            )
                            logger.info(f"[{self.call_id}] Session test send successful - session may still be alive")
                            
                            # Wait a bit and try to receive again
                            await asyncio.sleep(0.1)
                            continue
                            
                        except Exception as test_e:
                            logger.error(f"[{self.call_id}] Session test failed: {test_e}")
                            logger.error(f"[{self.call_id}] Session is definitively dead, breaking listener loop")
                            break
                    else:
                        logger.info(f"[{self.call_id}] Receive loop ended normally - session disconnected")
                        break
                        
                except StopAsyncIteration:
                    logger.warning(f"[{self.call_id}] Gemini session receive iterator exhausted")
                    if self.is_connected:
                        logger.error(f"[{self.call_id}] Session iterator exhausted while still connected - this is unexpected")
                    break
                except Exception as receive_e:
                    logger.error(f"[{self.call_id}] Error in receive loop: {type(receive_e).__name__}: {receive_e}")
                    if self.is_connected:
                        logger.info(f"[{self.call_id}] Attempting to continue after receive error...")
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        break
                        
        except asyncio.CancelledError:
            logger.info(f"[{self.call_id}] Listener cancelled")
        except Exception as e:
            logger.error(f"[{self.call_id}] CRITICAL: Listener crashed with error: {type(e).__name__}: {e}")
            if self.is_connected:
                logger.error(f"[{self.call_id}] Listener crashed while session still connected!")
        finally:
            logger.info(f"[{self.call_id}] Listener ended after {response_count} responses")
    
    async def send_audio(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Send audio to Gemini"""
        if not self.session or not self.is_connected:
            logger.warning(f"[{self.call_id}] Session not available for audio sending")
            return
        
        try:
            await self.session.send_realtime_input(
                media=types.Blob(
                    data=audio_bytes, 
                    mime_type=f'audio/pcm;rate={sample_rate}'
                )
            )
            # Log periodically to avoid spam
            if not hasattr(self, '_audio_send_count'):
                self._audio_send_count = 0
            self._audio_send_count += 1
            if self._audio_send_count % 100 == 1:
                logger.info(f"[{self.call_id}] Sent {self._audio_send_count} audio chunks to Gemini")
        except Exception as e:
            logger.error(f"[{self.call_id}] Error sending audio to Gemini: {e}")
    
    def get_response_audio(self) -> Optional[bytes]:
        """Get audio response from Gemini if available"""
        if self.response_queue:
            audio = self.response_queue.popleft()
            logger.debug(f"[{self.call_id}] Retrieved audio from queue: {len(audio)} bytes (queue remaining: {len(self.response_queue)})")
            return audio
        return None
    
    async def cleanup(self):
        """Cleanup session"""
        logger.info(f"[{self.call_id}] Starting Gemini cleanup")
        
        # Cancel listener task if running
        if hasattr(self, 'listener_task') and self.listener_task and not self.listener_task.done():
            logger.info(f"[{self.call_id}] Cancelling listener task")
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        
        if self.session_context and self.session:
            try:
                self.is_connected = False
                await self.session_context.__aexit__(None, None, None)
                logger.info(f"[{self.call_id}] Gemini session cleaned up")
            except Exception as e:
                logger.error(f"[{self.call_id}] Error cleaning up Gemini session: {e}")
            finally:
                self.session = None
                self.session_context = None
                
    async def _monitor_listener(self):
        """Monitor the listener task and restart if it crashes"""
        logger.info(f"[{self.call_id}] Starting listener monitor")
        restart_count = 0
        max_restarts = 3
        
        while self.is_connected:
            await asyncio.sleep(1)  # Check every second
            
            if self.listener_task and self.listener_task.done():
                # Listener task has ended - determine if it crashed or ended normally
                listener_exception = None
                try:
                    # Get the result to see if it ended with an exception
                    await self.listener_task
                    logger.info(f"[{self.call_id}] Listener task ended normally")
                except Exception as e:
                    listener_exception = e
                    logger.error(f"[{self.call_id}] Listener task crashed: {type(e).__name__}: {e}")
                
                # Always try to restart if we're still connected, whether it crashed or ended normally
                if self.is_connected and restart_count < max_restarts:
                    restart_count += 1
                    logger.info(f"[{self.call_id}] Attempting to restart listener (attempt {restart_count}/{max_restarts})...")
                    
                    # Restart the listener
                    if self.session:
                        try:
                            self.listener_task = asyncio.create_task(self._listen_for_responses())
                            logger.info(f"[{self.call_id}] Listener restarted successfully")
                            # Reset restart count on successful restart
                            await asyncio.sleep(2)  # Give it a moment to start
                            if not self.listener_task.done():
                                restart_count = 0  # Reset counter if restart seems successful
                        except Exception as restart_e:
                            logger.error(f"[{self.call_id}] Failed to restart listener: {restart_e}")
                    else:
                        logger.error(f"[{self.call_id}] Cannot restart listener - session not available")
                        break
                elif restart_count >= max_restarts:
                    logger.error(f"[{self.call_id}] Max restart attempts ({max_restarts}) reached, giving up")
                    break
                else:
                    logger.info(f"[{self.call_id}] Not restarting listener - session disconnected")
                    break
        
        logger.info(f"[{self.call_id}] Listener monitor stopped (restarts attempted: {restart_count})")


class AudioConverter:
    """Handle audio format conversions between Telnyx and Gemini"""
    
    @staticmethod
    def pcmu_to_pcm(pcmu_data: bytes) -> bytes:
        """Convert PCMU (μ-law) to linear PCM"""
        try:
            return audioop.ulaw2lin(pcmu_data, 2)
        except Exception as e:
            logger.error(f"Error converting PCMU to PCM: {e}")
            return b""
    
    @staticmethod
    def pcm_to_pcmu(pcm_data: bytes) -> bytes:
        """Convert linear PCM to PCMU (μ-law)"""
        try:
            return audioop.lin2ulaw(pcm_data, 2)
        except Exception as e:
            logger.error(f"Error converting PCM to PCMU: {e}")
            return b""
    
    @staticmethod
    def resample_audio(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio from one sample rate to another"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0
            
            # Resample
            resampled = librosa.resample(audio_array, orig_sr=from_rate, target_sr=to_rate)
            
            # Convert back to int16 bytes
            resampled = (resampled * 32767).astype(np.int16)
            return resampled.tobytes()
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return b""
    
    @staticmethod
    def m4a_to_pcm_16khz(file_path: str) -> bytes:
        """Convert M4A file to PCM 16kHz mono for Gemini"""
        try:
            # Load M4A file using pydub
            audio = AudioSegment.from_file(file_path, format="m4a")
            
            # Convert to 16kHz mono 16-bit
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Return raw PCM data
            return audio.raw_data
        except Exception as e:
            logger.error(f"Error converting M4A to PCM: {e}")
            return b""


class AudioRecorder:
    """Record inbound and outbound audio to WAV files with proper timing"""
    
    def __init__(self, call_id: str):
        self.call_id = call_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.inbound_file = f"call_{call_id.replace(':', '_').replace('/', '_')}_{timestamp}_inbound.wav"
        self.outbound_file = f"call_{call_id.replace(':', '_').replace('/', '_')}_{timestamp}_outbound.wav"
        
        # Initialize WAV files
        self.inbound_wav = None
        self.outbound_wav = None
        
        # Track timing for proper silence insertion
        self.call_start_time = None
        self.last_agent_audio_time = None
        self.total_agent_samples_written = 0  # Track total samples written to agent recording
        
        logger.info(f"Audio recording initialized for call {call_id}")
        logger.info(f"Inbound: {self.inbound_file}")
        logger.info(f"Outbound: {self.outbound_file}")
    
    def record_caller_audio(self, pcm_data: bytes, sample_rate: int = 8000):
        """Record caller audio (you speaking to agent) - THIS IS INBOUND"""
        try:
            # Set call start time on first audio
            if self.call_start_time is None:
                self.call_start_time = datetime.now()
                logger.info(f"[{self.call_id}] Call audio recording started at {self.call_start_time}")
            
            if not self.inbound_wav:
                # Initialize inbound WAV file  
                self.inbound_wav = wave.open(self.inbound_file, 'wb')
                self.inbound_wav.setnchannels(1)  # Mono
                self.inbound_wav.setsampwidth(2)  # 16-bit
                self.inbound_wav.setframerate(sample_rate)
                
            # Write PCM data directly
            self.inbound_wav.writeframes(pcm_data)
            
        except Exception as e:
            logger.error(f"Error recording caller audio: {e}")
    
    def record_agent_audio(self, pcm_data: bytes, sample_rate: int = 24000):
        """Record agent audio (agent speaking to caller) - THIS IS OUTBOUND with proper timing"""
        try:
            current_time = datetime.now()
            
            # Set call start time if not already set
            if self.call_start_time is None:
                self.call_start_time = current_time
                logger.info(f"[{self.call_id}] Call audio recording started at {self.call_start_time}")
            
            if not self.outbound_wav:
                # Initialize outbound WAV file
                self.outbound_wav = wave.open(self.outbound_file, 'wb')
                self.outbound_wav.setnchannels(1)  # Mono
                self.outbound_wav.setsampwidth(2)  # 16-bit
                self.outbound_wav.setframerate(sample_rate)
                self.last_agent_audio_time = current_time
                logger.info(f"[{self.call_id}] Started agent audio recording at sample rate {sample_rate}Hz")
            
            # Calculate how much time has passed since call started
            time_since_call_start = (current_time - self.call_start_time).total_seconds()
            
            # Calculate how many samples SHOULD have been written by now
            expected_total_samples = int(time_since_call_start * sample_rate)
            
            # Calculate silence gap to insert
            silence_samples_needed = expected_total_samples - self.total_agent_samples_written
            
            # Insert silence if there's a significant gap (more than 50ms worth of samples)
            min_gap_samples = int(0.05 * sample_rate)  # 50ms
            if silence_samples_needed > min_gap_samples:
                # Limit silence to reasonable amount (max 10 seconds of silence)
                max_silence_samples = int(10.0 * sample_rate)
                silence_samples_needed = min(silence_samples_needed, max_silence_samples)
                
                # Create silence (16-bit PCM zeros)
                silence_bytes = b'\x00\x00' * silence_samples_needed
                self.outbound_wav.writeframes(silence_bytes)
                self.total_agent_samples_written += silence_samples_needed
                
                silence_duration = silence_samples_needed / sample_rate
                logger.info(f"[{self.call_id}] Inserted {silence_duration:.2f}s of silence in agent recording")
            
            # Write the actual audio data
            audio_samples = len(pcm_data) // 2  # 16-bit samples = bytes / 2
            self.outbound_wav.writeframes(pcm_data)
            self.total_agent_samples_written += audio_samples
            self.last_agent_audio_time = current_time
            
        except Exception as e:
            logger.error(f"Error recording agent audio: {e}")
    
    def close(self):
        """Close WAV files and finalize recordings"""
        try:
            if self.inbound_wav:
                self.inbound_wav.close()
                self.inbound_wav = None
                logger.info(f"Inbound recording saved: {self.inbound_file}")
                
            if self.outbound_wav:
                self.outbound_wav.close()
                self.outbound_wav = None
                logger.info(f"Outbound recording saved: {self.outbound_file}")
                
        except Exception as e:
            logger.error(f"Error closing audio recordings: {e}")


class TelnyxGeminiBridge:
    """Bridge between Telnyx calls and Gemini voice agents"""
    
    def __init__(self):
        self.active_calls: Dict[str, GeminiVoiceAgent] = {}
        self.audio_recorders: Dict[str, AudioRecorder] = {}
        self.converter = AudioConverter()
    
    async def handle_new_call(self, call_id: str) -> Optional[GeminiVoiceAgent]:
        """Initialize new Gemini agent for incoming call"""
        if call_id not in self.active_calls:
            logger.info(f"Creating new Gemini agent for call: {call_id}")
            try:
                agent = GeminiVoiceAgent(GOOGLE_API_KEY, call_id)
                await agent.start_session()
                self.active_calls[call_id] = agent
                
                # Create audio recorder for this call
                self.audio_recorders[call_id] = AudioRecorder(call_id)
                
                return agent
            except Exception as e:
                logger.error(f"Failed to create Gemini agent for call {call_id}: {e}")
                return None
        return self.active_calls[call_id]
    
    async def process_telnyx_audio(self, call_id: str, telnyx_audio: bytes) -> Optional[bytes]:
        """Process audio from Telnyx through Gemini and return response"""
        try:
            # Get or create agent for this call
            agent = await self.handle_new_call(call_id)
            if not agent:
                logger.warning(f"No Gemini agent available for call {call_id}")
                return None
            
            # Convert Telnyx audio: PCMU 8kHz -> PCM 16kHz
            pcm_data = self.converter.pcmu_to_pcm(telnyx_audio)
            if not pcm_data:
                return None
            
            # Record caller audio (you speaking to agent)
            if call_id in self.audio_recorders:
                self.audio_recorders[call_id].record_caller_audio(pcm_data, sample_rate=8000)
            
            # Resample from 8kHz to 16kHz for Gemini
            audio_16k = self.converter.resample_audio(pcm_data, from_rate=8000, to_rate=16000)
            if not audio_16k:
                return None
            
            # Send to Gemini
            await agent.send_audio(audio_16k, sample_rate=16000)
            
            # Get ALL available responses from Gemini (24kHz PCM) 
            response_audio = agent.get_response_audio()
            if response_audio:
                # Record agent audio (agent speaking to caller)
                if call_id in self.audio_recorders:
                    self.audio_recorders[call_id].record_agent_audio(response_audio, sample_rate=24000)
                
                # Convert Gemini response: PCM 24kHz -> PCM 8kHz -> PCMU 8kHz
                downsampled = self.converter.resample_audio(response_audio, from_rate=24000, to_rate=8000)
                if downsampled:
                    pcmu_response = self.converter.pcm_to_pcmu(downsampled)
                    logger.info(f"Sending Gemini audio to caller: {len(pcmu_response)} bytes")
                    return pcmu_response
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio for call {call_id}: {e}")
            return None
    
    async def get_pending_audio(self, call_id: str) -> Optional[bytes]:
        """Get any pending audio responses from Gemini for this call"""
        try:
            if call_id not in self.active_calls:
                return None
                
            agent = self.active_calls[call_id]
            response_audio = agent.get_response_audio()
            
            if response_audio:
                # Record agent audio (agent speaking to caller)
                if call_id in self.audio_recorders:
                    self.audio_recorders[call_id].record_agent_audio(response_audio, sample_rate=24000)
                
                # Convert Gemini response: PCM 24kHz -> PCM 8kHz -> PCMU 8kHz
                downsampled = self.converter.resample_audio(response_audio, from_rate=24000, to_rate=8000)
                if downsampled:
                    pcmu_response = self.converter.pcm_to_pcmu(downsampled)
                    logger.info(f"Retrieved pending audio: {len(pcmu_response)} bytes")
                    return pcmu_response
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting pending audio for call {call_id}: {e}")
            return None
    
    async def cleanup_call(self, call_id: str):
        """Cleanup resources for ended call"""
        if call_id in self.active_calls:
            logger.info(f"Cleaning up call: {call_id}")
            await self.active_calls[call_id].cleanup()
            del self.active_calls[call_id]
            
        # Close audio recordings
        if call_id in self.audio_recorders:
            self.audio_recorders[call_id].close()
            del self.audio_recorders[call_id]


# Initialize the bridge
bridge = TelnyxGeminiBridge()


@app.post("/webhook")
async def handle_telnyx_webhook(request: Request):
    """Handle Telnyx webhook events"""
    try:
        webhook = await request.json()
        
        # Debug: Log the actual webhook structure
        logger.info(f"Full webhook data: {json.dumps(webhook, indent=2)}")
        
        event_type = webhook["data"]["event_type"]
        payload = webhook["data"]["payload"]
        
        # Debug: Check payload type and content
        logger.info(f"Payload type: {type(payload)}, content: {payload}")
        
        # Handle different payload structures
        if isinstance(payload, dict):
            call_control_id = payload["call_control_id"]
        else:
            # If payload is a string, it might be the call_control_id itself
            call_control_id = payload
        
        logger.info(f"Received webhook: {event_type} for call {call_control_id}")
        
        if event_type == "call.initiated":
            # Answer the call with bidirectional streaming
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
            await bridge.cleanup_call(call_control_id)
            
        elif event_type == "call.streaming.started":
            logger.info(f"Streaming started for call {call_control_id}")
            
        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.websocket("/audio")
async def handle_audio_websocket(websocket: WebSocket):
    """Handle WebSocket audio streaming from Telnyx"""
    await websocket.accept()
    call_id = None
    
    logger.info("WebSocket connection established")
    
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
                
            elif event == "media" and call_id:
                # Process incoming audio
                media_data = data.get("media", {})
                payload = media_data.get("payload", "")
                
                if payload:
                    try:
                        # Decode base64 audio data
                        telnyx_audio = base64.b64decode(payload)
                        frame_count += 1
                        
                        # Log every 50 frames (~1 second)
                        if frame_count % 50 == 0:
                            logger.info(f"Received {frame_count} audio frames, latest size: {len(telnyx_audio)} bytes")
                        
                        # Process through Gemini
                        response_audio = await bridge.process_telnyx_audio(call_id, telnyx_audio)
                        
                        # Send response back to Telnyx
                        if response_audio:
                            response_payload = base64.b64encode(response_audio).decode()
                            response_message = {
                                "event": "media",
                                "media": {
                                    "payload": response_payload
                                }
                            }
                            await websocket.send_text(json.dumps(response_message))
                            
                    except Exception as e:
                        logger.error(f"Error processing media for call {call_id}: {e}")
                else:
                    logger.warning("Received media event with empty payload")
                
                # Also check for any pending audio responses from Gemini
                try:
                    pending_audio = await bridge.get_pending_audio(call_id)
                    if pending_audio:
                        response_payload = base64.b64encode(pending_audio).decode()
                        response_message = {
                            "event": "media",
                            "media": {
                                "payload": response_payload
                            }
                        }
                        await websocket.send_text(json.dumps(response_message))
                except Exception as e:
                    logger.error(f"Error sending pending audio for call {call_id}: {e}")
                        
            elif event == "stop":
                logger.info(f"Streaming stopped for call: {call_id}")
                if call_id:
                    await bridge.cleanup_call(call_id)
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if call_id:
            await bridge.cleanup_call(call_id)
        logger.info("WebSocket connection closed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_calls": len(bridge.active_calls),
        "ngrok_url": NGROK_URL
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Telnyx-Gemini Voice Bridge",
        "webhook_url": f"https://{NGROK_URL}/webhook",
        "stream_url": STREAM_URL,
        "active_calls": len(bridge.active_calls),
        "endpoints": {
            "webhook": "/webhook",
            "audio": "/audio",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Verify environment variables
    if not TELNYX_API_KEY:
        logger.error("TELNYX_API_KEY not found in environment variables")
        exit(1)
        
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        exit(1)
    
    logger.info(f"Starting Telnyx-Gemini Bridge Server")
    logger.info(f"Webhook URL: https://{NGROK_URL}/webhook")
    logger.info(f"Stream URL: {STREAM_URL}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )