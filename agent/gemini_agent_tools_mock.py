"""
Mock Gemini Agent Tools

This is a mock implementation of the Gemini agent tools that doesn't depend on external 'gmr' module.
It provides the same interface for testing and basic functionality.
"""

import logging
import uuid
from typing import Any, Dict

log = logging.getLogger(__name__)

class MockPortalClient:
    """Mock portal client for testing."""
    
    async def generate_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:8]}"
    
    async def generate_request_id(self) -> str:
        return f"request_{uuid.uuid4().hex[:8]}"
    
    async def send_start_call_event(self, request_id: str) -> None:
        log.info(f"Mock: Start call event for request {request_id}")
    
    async def save_transport_request(self, data: Dict[str, Any]) -> str:
        log.info(f"Mock: Saving transport request: {data}")
        return f"saved_{uuid.uuid4().hex[:8]}"

class MockConversationContext:
    """Mock conversation context for testing."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data = {}
        self.current_request_id = None
    
    def new_transport_request(self, request_id: str):
        self.current_request_id = request_id
        self.data = {}
        log.info(f"Mock: New transport request {request_id}")
    
    def add_info(self, field_name: str, value: Any):
        self.data[field_name] = value
        log.info(f"Mock: Added {field_name} = {value}")
    
    def get_info(self, field_name: str) -> Any:
        return self.data.get(field_name)
    
    def get_all_data(self) -> Dict[str, Any]:
        return self.data.copy()

class GeminiAgentTools:
    """Mock tools for Gemini agent to interact with transport requests and portal."""

    def __init__(self):
        """Initialize the mock tools."""
        self.portal_client = MockPortalClient()
        self.context = MockConversationContext(session_id=str(uuid.uuid4()))

    async def initialize_session(self) -> str:
        """Initialize a new session and transport request."""
        # Generate session ID
        session_id = await self.portal_client.generate_session_id()
        self.context = MockConversationContext(session_id)

        # Start a new transport request
        request_id = await self.portal_client.generate_request_id()
        self.context.new_transport_request(request_id)

        try:
            await self.portal_client.send_start_call_event(request_id)
        except Exception as e:
            log.warning(f"Failed to send start call event: {e}")

        return session_id

    def store_info(self, field_name: str, field_value: str) -> str:
        """
        Store information gained during a call.

        Args:
            field_name: The name of the field to store
            field_value: The value to store

        Returns:
            Success message
        """
        try:
            self.context.add_info(field_name, field_value)
            return f"Successfully stored {field_name}: {field_value}"
        except Exception as e:
            log.error(f"Error storing info: {e}")
            return f"Error storing {field_name}: {str(e)}"

    def get_form_status(self) -> str:
        """Get the current status of the transport request form."""
        try:
            data = self.context.get_all_data()
            if not data:
                return "No information has been collected yet. Ready to start gathering transport request details."
            
            status_parts = []
            status_parts.append(f"Request ID: {self.context.current_request_id}")
            status_parts.append(f"Collected {len(data)} fields:")
            
            for field, value in data.items():
                status_parts.append(f"  • {field}: {value}")
                
            return "\n".join(status_parts)
        except Exception as e:
            log.error(f"Error getting form status: {e}")
            return f"Error retrieving form status: {str(e)}"

    async def submit_transport_request(self) -> str:
        """Submit the completed transport request."""
        try:
            data = self.context.get_all_data()
            if not data:
                return "No information to submit. Please provide transport request details first."
            
            # Mock submission
            result = await self.portal_client.save_transport_request(data)
            return f"Transport request submitted successfully. Reference: {result}"
        except Exception as e:
            log.error(f"Error submitting transport request: {e}")
            return f"Error submitting request: {str(e)}"

    def get_field_value(self, field_name: str) -> str:
        """Get the value of a specific field."""
        try:
            value = self.context.get_info(field_name)
            if value is None:
                return f"Field '{field_name}' has not been set yet."
            return f"{field_name}: {value}"
        except Exception as e:
            log.error(f"Error getting field value: {e}")
            return f"Error retrieving {field_name}: {str(e)}"

    def get_next_steps(self) -> str:
        """Get recommended next steps based on current form status."""
        try:
            data = self.context.get_all_data()
            
            # Mock next steps logic
            if not data:
                return "Next step: Start gathering basic transport request information (patient name, pickup location, destination)."
            
            required_fields = ["patient_name", "pickup_location", "destination", "transport_reason"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return f"Next steps: Please provide the following missing information: {', '.join(missing_fields)}"
            else:
                return "All basic information collected. Ready to submit transport request or gather additional details."
                
        except Exception as e:
            log.error(f"Error getting next steps: {e}")
            return f"Error determining next steps: {str(e)}"

    def end_call(self) -> str:
        """End the current call session."""
        try:
            session_id = self.context.session_id
            request_id = self.context.current_request_id
            log.info(f"Ending call session {session_id}, request {request_id}")
            return f"Call session {session_id} ended. Thank you for using our transport request service."
        except Exception as e:
            log.error(f"Error ending call: {e}")
            return f"Error ending call: {str(e)}"