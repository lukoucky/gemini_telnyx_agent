from datetime import datetime
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

class CallSession:
    def __init__(self, call_control_id: str):
        self.call_control_id = call_control_id
        self.stream_id: Optional[str] = None
        self.connection_id: Optional[str] = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.is_active = True
        self.call_active = False
        self.audio_chunks_received = 0
        self.audio_chunks_sent = 0
        self.sequence_number = 0
        self.last_audio_send_time = 0
        
        logger.info(f"Created new call session: {self.session_id} for call: {call_control_id}")
    
    def set_stream_id(self, stream_id: str):
        self.stream_id = stream_id
        self.call_active = True
        self.last_audio_send_time = time.time()
        logger.info(f"Stream ID set for session {self.session_id}: {stream_id}")
        
    def set_connection_id(self, connection_id: str):
        self.connection_id = connection_id
        logger.info(f"Connection ID set for session {self.session_id}: {connection_id}")
        
    def get_next_sequence(self):
        self.sequence_number += 1
        return str(self.sequence_number)
        
    def should_send_audio(self):
        """Check if it's time to send the next audio chunk (1-second intervals)"""
        current_time = time.time()
        return (self.stream_id and self.call_active and 
                current_time - self.last_audio_send_time >= 1.0)
                
    def mark_audio_sent(self):
        """Mark that audio was just sent"""
        self.last_audio_send_time = time.time()
        self.audio_chunks_sent += 1
    
    def increment_received_chunks(self):
        self.audio_chunks_received += 1
    
    def increment_sent_chunks(self):
        self.audio_chunks_sent += 1
    
    def stop_call(self):
        """Stop the active call streaming"""
        self.call_active = False
        logger.info(f"Call streaming stopped for session {self.session_id}")
        
    def end_session(self):
        self.is_active = False
        self.call_active = False
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Session {self.session_id} ended - Duration: {duration:.1f}s, "
                   f"Received: {self.audio_chunks_received}, Sent: {self.audio_chunks_sent}")
    
    def get_audio_filename(self) -> str:
        return f"audio_logs/conversation_{self.session_id}.wav"
        
    def get_incoming_filename(self) -> str:
        return f"audio_logs/incoming_{self.session_id}.wav"
        
    def get_outgoing_filename(self) -> str:
        return f"audio_logs/outgoing_{self.session_id}.wav"

# Global session manager
class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, call_control_id: str) -> CallSession:
        session = CallSession(call_control_id)
        self.sessions[call_control_id] = session
        return session
    
    def get_session(self, call_control_id: str) -> Optional[CallSession]:
        return self.sessions.get(call_control_id)
    
    def remove_session(self, call_control_id: str):
        if call_control_id in self.sessions:
            session = self.sessions[call_control_id]
            session.end_session()
            del self.sessions[call_control_id]
            logger.info(f"Removed session for call: {call_control_id}")

session_manager = SessionManager()