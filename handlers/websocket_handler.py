import json
import asyncio
import logging
import os
import time
from fastapi import WebSocket
import websockets

from models.call_session import session_manager
from services.audio_service import AudioRecorder, ContinuousAudioGenerator, TelnyxAudioParser

logger = logging.getLogger(__name__)

class WebSocketHandler:
    def __init__(self, test_audio_file: str = None):
        self.test_audio_file = test_audio_file
        
    async def handle_audio_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection for continuous audio streaming"""
        logger.info("🔌 WebSocket connection")
        
        try:
            await websocket.accept()
            logger.info("✅ WebSocket accepted")
            
            # Initialize session and audio components
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"audio_logs/conversation_{session_id}.wav"
            
            recorder = AudioRecorder(audio_filename)
            audio_generator = ContinuousAudioGenerator()
            
            # Session state
            stream_id = None
            connection_id = None
            call_active = False
            sequence_number = 0
            audio_chunk_count = 0
            incoming_count = 0
            last_audio_send_time = 0
            
            logger.info(f"📋 Session {session_id} created")
            
            try:
                logger.info("🎵 Starting CONTINUOUS audio streaming")
                
                while True:
                    try:
                        # Handle incoming WebSocket messages
                        try:
                            data = await asyncio.wait_for(websocket.receive(), timeout=0.05)
                            
                            if "text" in data:
                                text_data = data["text"]
                                
                                try:
                                    parsed = json.loads(text_data)
                                    event = parsed.get("event")
                                    
                                    if event == "connected":
                                        connection_id = parsed.get("connection_id")
                                        logger.info(f"📞 Telnyx connected - Connection ID: {connection_id}")
                                        
                                    elif event == "start":
                                        stream_id = parsed.get("stream_id")
                                        media_format = parsed.get("start", {}).get("media_format", {})
                                        logger.info(f"🎵 Stream started: {stream_id}")
                                        logger.info(f"🔍 Media format: {media_format}")
                                        call_active = True
                                        last_audio_send_time = time.time()  # Initialize timer
                                        
                                    elif event == "media":
                                        # Handle incoming RTP media
                                        media = parsed.get("media", {})
                                        payload = media.get("payload", "")
                                        track = media.get("track", "unknown")
                                        
                                        if payload and track == "inbound":
                                            recorder.add_incoming_chunk(payload)
                                            incoming_count += 1
                                            
                                            if incoming_count == 1:
                                                logger.info("🎉 FIRST INCOMING RTP PAYLOAD!")
                                            elif incoming_count % 100 == 0:
                                                logger.info(f"📥 Received {incoming_count} RTP payloads")
                                                
                                    elif event == "stop":
                                        logger.info("⏹️ Stream stopped")
                                        call_active = False
                                        break
                                        
                                except json.JSONDecodeError:
                                    pass
                                    
                        except asyncio.TimeoutError:
                            pass
                        
                        # CONTINUOUS AUDIO STREAMING - Send 1-second chunks every second
                        current_time = time.time()
                        if (stream_id and call_active and 
                            current_time - last_audio_send_time >= 1.0):
                            
                            # Generate next audio chunk
                            audio_payload = audio_generator.generate_audio_chunk(duration_ms=1000)
                            
                            # Create media message
                            sequence_number += 1
                            audio_chunk_count += 1
                            message = {
                                "event": "media",
                                "sequence_number": str(sequence_number),
                                "media": {
                                    "track": "outbound",
                                    "chunk": str(audio_chunk_count),
                                    "timestamp": str((audio_chunk_count - 1) * 1000),
                                    "payload": audio_payload
                                },
                                "stream_id": stream_id
                            }
                            
                            await websocket.send_text(json.dumps(message))
                            recorder.add_outgoing_chunk(audio_payload)
                            last_audio_send_time = current_time
                            
                            if audio_chunk_count == 1:
                                logger.info("🎵 FIRST continuous audio chunk sent!")
                                logger.info("🔊 You should now hear continuous rotating tones!")
                            elif audio_chunk_count % 5 == 0:
                                logger.info(f"🎵 Sent {audio_chunk_count} continuous audio chunks")
                        
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.02)
                        
                    except Exception as e:
                        logger.error(f"❌ WebSocket loop error: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
            
            finally:
                logger.info(f"🎬 Session {session_id} ended")
                logger.info(f"📊 Final stats: {audio_chunk_count} audio chunks sent, {incoming_count} received")
                
                # Save the conversation
                recorder.finalize_recording()
                
        except Exception as e:
            logger.error(f"❌ Failed to accept WebSocket: {e}")
        
        finally:
            try:
                await websocket.close()
            except:
                pass