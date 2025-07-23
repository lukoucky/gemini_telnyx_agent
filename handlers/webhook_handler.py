import json
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from models.call_session import session_manager
from services.telnyx_service import get_telnyx_service

logger = logging.getLogger(__name__)

class WebhookHandler:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        
    async def handle_webhook(self, request: Request) -> JSONResponse:
        """Process incoming Telnyx webhook events"""
        try:
            webhook = await request.json()
            logger.info(f"Received webhook: {json.dumps(webhook, indent=2)}")

            # Extract event data
            event_type = webhook["data"]["event_type"]
            payload = webhook["data"]["payload"]
            call_control_id = payload["call_control_id"]

            logger.info(f"Processing event: {event_type} for call: {call_control_id}")

            # Route to appropriate handler
            if event_type == "call.initiated":
                await self._handle_call_initiated(call_control_id)
            elif event_type == "call.answered":
                await self._handle_call_answered(call_control_id)
            elif event_type == "call.transcription":
                await self._handle_transcription(payload)
            elif event_type == "streaming.started":
                await self._handle_streaming_started(payload)
            elif event_type == "streaming.stopped":
                await self._handle_streaming_stopped(payload)
            elif event_type == "streaming.failed":
                await self._handle_streaming_failed(payload)
            elif event_type == "call.speak.ended":
                await self._handle_speak_ended(call_control_id)
            elif event_type == "call.playback.ended":
                await self._handle_playback_ended(payload)
            elif event_type == "call.hangup":
                await self._handle_call_hangup(call_control_id, payload)
            else:
                logger.info(f"Unhandled event type: {event_type}")

            return JSONResponse(content={"status": "ok"})

        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            try:
                body = await request.body()
                logger.error(f"Raw request body: {body}")
            except:
                logger.error("Could not read request body")
            return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    
    async def _handle_call_initiated(self, call_control_id: str):
        """Handle incoming call initiation"""
        logger.info("📞 Call initiated - answering with bidirectional RTP streaming")
        
        # Create new call session
        session = session_manager.create_session(call_control_id)
        
        # Answer the call with bidirectional RTP streaming (working configuration)
        telnyx_service = get_telnyx_service()
        success = telnyx_service.answer_call_with_bidirectional_rtp(
            call_control_id, 
            self.stream_url
        )
        
        if success:
            logger.info("✅ Call answered with bidirectional RTP streaming")
        else:
            logger.error(f"❌ Failed to answer call {call_control_id}")
    
    async def _handle_call_answered(self, call_control_id: str):
        """Handle call answered event"""
        logger.info("📞 Call answered event (bidirectional streaming active)")
        # Note: No welcome message needed - continuous audio will start automatically
    
    async def _handle_transcription(self, payload: dict):
        """Handle transcription events"""
        logger.info("Received transcription event")
        
        transcription_data = payload.get("transcription_data", {})
        transcription = transcription_data.get("text", "")
        
        if transcription:
            logger.info(f"Transcription: '{transcription}'")
            # TODO: Forward to AI service when ready
        else:
            logger.warning("Empty transcription received")
    
    async def _handle_streaming_started(self, payload: dict):
        """Handle streaming started event"""
        connection_id = payload.get("connection_id")
        call_control_id = payload.get("call_control_id")
        logger.info(f"🎵 Streaming started - Connection ID: {connection_id}")
        
        # Update session with connection info
        session = session_manager.get_session(call_control_id)
        if session and connection_id:
            session.set_connection_id(connection_id)
    
    async def _handle_streaming_stopped(self, payload: dict):
        """Handle streaming stopped event"""
        logger.info("Audio streaming stopped")
    
    async def _handle_streaming_failed(self, payload: dict):
        """Handle streaming failure"""
        logger.error("Audio streaming failed")
        logger.error(f"Streaming failure payload: {payload}")
        
        failure_reason = payload.get("failure_reason", "unknown")
        logger.error(f"Failure reason: {failure_reason}")

        if failure_reason == "connection_failed":
            logger.error("WebSocket connection failed - Check if the WebSocket URL is accessible")
            logger.error(f"Attempted to connect to: {self.stream_url}")
    
    async def _handle_speak_ended(self, call_control_id: str):
        """Handle TTS completion"""
        logger.info("Text-to-speech ended - call ready for audio streaming")
    
    async def _handle_playback_ended(self, payload: dict):
        """Handle audio playback completion"""
        logger.info("Audio playback ended")
        
        playback_status = payload.get("status", "unknown")
        if playback_status == "file_not_found":
            logger.warning("Playback failed: file not found")
    
    async def _handle_call_hangup(self, call_control_id: str, payload: dict):
        """Handle call hangup"""
        logger.info("Call hangup received - cleaning up")
        
        hangup_cause = payload.get("hangup_cause", "unknown")
        logger.info(f"Hangup cause: {hangup_cause}")
        
        # Clean up session
        session_manager.remove_session(call_control_id)