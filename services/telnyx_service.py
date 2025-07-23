import telnyx
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TelnyxService:
    def __init__(self):
        self.api_key = os.getenv("TELNYX_API_KEY")
        if not self.api_key:
            raise ValueError("TELNYX_API_KEY environment variable is required")
        
        telnyx.api_key = self.api_key
        logger.info("Telnyx service initialized")
    
    def answer_call(self, call_control_id: str, stream_url: str) -> bool:
        """Answer incoming call with audio streaming enabled"""
        try:
            logger.info(f"Answering call {call_control_id} with stream_url: {stream_url}")
            
            response = telnyx.Call.create_answer(
                call_control_id,
                stream_url=stream_url,
                stream_track="both_tracks",
                transcription=True,
                transcription_tracks="inbound"
            )
            
            logger.info(f"Call answered successfully: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error answering call {call_control_id}: {e}")
            return False
            
    def answer_call_with_bidirectional_rtp(self, call_control_id: str, stream_url: str) -> bool:
        """Answer incoming call with bidirectional RTP streaming (working configuration)"""
        try:
            logger.info(f"Answering call {call_control_id} with bidirectional RTP streaming")
            
            response = telnyx.Call.create_answer(
                call_control_id,
                stream_url=stream_url,
                stream_track="both_tracks",
                stream_bidirectional_mode="rtp",
                stream_bidirectional_codec="PCMU"
            )
            
            logger.info(f"Bidirectional RTP call answered successfully: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error answering call with bidirectional RTP {call_control_id}: {e}")
            return False
    
    def speak_text(self, call_control_id: str, text: str, voice: str = "female", language: str = "en-US") -> bool:
        """Send text-to-speech to call"""
        try:
            logger.info(f"Speaking text to call {call_control_id}: '{text}'")
            
            response = telnyx.Call.create_speak(
                call_control_id,
                text=text,
                voice=voice,
                language=language
            )
            
            logger.info(f"Speak command sent successfully: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending speak command to {call_control_id}: {e}")
            # Try alternative format
            try:
                logger.info("Trying alternative speak format...")
                response = telnyx.Call.create_speak(
                    call_control_id,
                    payload=text,
                    voice=voice,
                    language=language
                )
                logger.info(f"Alternative speak command sent successfully: {response}")
                return True
            except Exception as e2:
                logger.error(f"Alternative speak command also failed: {e2}")
                return False
    
    def hangup_call(self, call_control_id: str) -> bool:
        """Hangup the call"""
        try:
            logger.info(f"Hanging up call {call_control_id}")
            
            response = telnyx.Call.create_hangup(call_control_id)
            
            logger.info(f"Call hangup successful: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error hanging up call {call_control_id}: {e}")
            return False
    
    def play_audio_url(self, call_control_id: str, audio_url: str) -> bool:
        """Play audio file from URL to the call"""
        try:
            logger.info(f"Playing audio URL to call {call_control_id}: {audio_url}")
            
            response = telnyx.Call.create_playback(
                call_control_id,
                audio_url=audio_url
            )
            
            logger.info(f"Playback command sent successfully: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio to {call_control_id}: {e}")
            return False

# Global instance - will be initialized after environment is loaded
telnyx_service = None

def get_telnyx_service():
    """Get or create the global TelnyxService instance"""
    global telnyx_service
    if telnyx_service is None:
        telnyx_service = TelnyxService()
    return telnyx_service