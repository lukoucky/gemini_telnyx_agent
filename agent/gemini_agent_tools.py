"""
Gemini Agent Tools

This module provides function tools for the Gemini agent to communicate with the database
and portal, similar to how transport_requests_agent.py does it.
"""

import logging
import uuid
from typing import Any

from gmr.dependencies import get_dependencies
from gmr.schema.conversation_context import ConversationContext
from gmr.schema.ift import InterfacilityTransportRequestBuilder
from gmr.tools.facility_search import resolve_facility_tool

log = logging.getLogger(__name__)


class GeminiAgentTools:
    """Tools for Gemini agent to interact with transport requests and portal."""

    def __init__(self):
        """Initialize the tools with dependencies."""
        self.deps = get_dependencies()
        self.portal_client = self.deps.portal_client
        self.context = ConversationContext(session_id=str(uuid.uuid4()))

    async def initialize_session(self) -> str:
        """Initialize a new session and transport request."""
        # Generate session ID
        session_id = await self.portal_client.generate_session_id()
        self.context.session_id = session_id

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
            field_name: The name of the field to store, for example caller_name.
            field_value: The value of the field to store, for example John Doe.

        Returns:
            Status message
        """
        log.info(f"Storing {field_name}: {field_value}")

        # Resolve facility if needed
        if field_name in ("caller_facility", "destination_facility", "pickup_facility"):
            field_value = resolve_facility_tool(field_value)

        try:
            if field_name == "caller_name":
                self.context.set_caller_name(field_value)
            elif field_name == "caller_facility":
                self.context.set_caller_facility(field_value)
            else:
                self.context.set_transport_value(field_name, field_value)

            # Get validated value
            value = self.context.get_transport_value(field_name)
            if value is None:
                return f"Failed to set {field_name}"

            validated_value = str(value)
            log.info(f"Validated value: {validated_value}")

            # Send field update event to portal
            request_id = self.context.get_transport_request_id()
            if request_id:
                import asyncio

                loop = asyncio.get_event_loop()
                loop.create_task(
                    self.portal_client.send_field_update_event(
                        request_id, field_name, validated_value
                    )
                )

            missing_required_fields = self._get_ordered_missing_mandatory_fields()
            if missing_required_fields:
                return f"Field {field_name} set to {validated_value}. Still need: {', '.join(missing_required_fields)}"
            else:
                return f"Field {field_name} set to {validated_value}. All required fields complete!"

        except Exception as e:
            log.error(f"Error setting field {field_name}: {e}")
            return f"Error setting {field_name}: {e}"

    def _get_ordered_missing_mandatory_fields(self) -> list[str]:
        """Get the ordered list of missing mandatory fields."""
        missing_required_fields = self.context.get_missing_required_fields()
        field_order = self.deps.config.transport_requests_agent.field_order
        return [field for field in field_order if field in missing_required_fields]

    def get_form_status(self) -> str:
        """
        Get comprehensive status report of the transport request form.

        Returns:
            Detailed form status including filled fields, missing required fields and
            missing optional fields.
        """
        log.info("Checking comprehensive form status")

        try:
            # Get basic status
            is_complete = self.context.is_transport_complete()
            missing_required = self.context.get_missing_required_fields()
            missing_optional = self.context.get_missing_optional_fields()

            # Get all fields and their required status from the builder
            all_fields_status = (
                InterfacilityTransportRequestBuilder.get_available_fields_with_required_status()
            )

            # Determine filled fields
            filled_fields = {}
            for field_name in all_fields_status:
                if (
                    field_name not in missing_required
                    and field_name not in missing_optional
                ):
                    value = self.context.get_transport_value(field_name)
                    if value is not None:
                        filled_fields[field_name] = value

            # Build comprehensive status report
            status_lines = [
                "=== FORM STATUS REPORT ===",
                f"All required fields complete: {'YES' if is_complete else 'NO'}",
                "",
            ]

            # Show filled fields with their values and required status
            if filled_fields:
                status_lines.append("FILLED FIELDS:")
                # Sort by required first, then alphabetically
                sorted_filled = sorted(
                    filled_fields.items(),
                    key=lambda x: (not all_fields_status[x[0]], x[0]),
                )
                for field_name, field_value in sorted_filled:
                    field_type = (
                        "REQUIRED" if all_fields_status[field_name] else "OPTIONAL"
                    )
                    status_lines.append(
                        f"  - {field_name} ({field_type}): {field_value}"
                    )
                status_lines.append("")

            # Show missing required fields
            if missing_required:
                status_lines.append("MISSING REQUIRED FIELDS:")
                status_lines.extend(
                    f"  - {field}" for field in sorted(missing_required)
                )
                status_lines.append("")

            # Show missing optional fields
            if missing_optional:
                status_lines.append("MISSING OPTIONAL FIELDS:")
                status_lines.extend(
                    f"  - {field}" for field in sorted(missing_optional)
                )
                status_lines.append("")

            # Add completion summary
            if is_complete:
                status_lines.append("Ready to submit!")
            else:
                remaining = len(missing_required)
                status_lines.append(
                    f"Still need {remaining} required field{'s' if remaining != 1 else ''}"
                )

            return "\n".join(status_lines)

        except Exception as e:
            log.error(f"Error checking form status: {e}")
            return f"Error checking form status: {e}"

    async def submit_transport_request(self) -> str:
        """
        Submit the transport request. This will succeed only if all required fields have been
        set via store_info.

        Returns:
            Success message or error message
        """
        log.info("Processing transport request.")

        if not self.context.is_transport_complete():
            log.info("Transport request is incomplete.")
            missing_required_fields = ", ".join(
                self.context.get_missing_required_fields()
            )
            return (
                f"Transport request incomplete. Missing required fields: {missing_required_fields}. "
                "Please collect this information and try again."
            )

        try:
            # Create the transport request object
            request = self.context.build_transport_request()
            if request is None:
                return "Error: Transport request is not complete."

            log.info(f"Constructed IFT request: {request}")

            # Submit the request using the portal client
            response = await self.portal_client.submit_transport_request(request)

            # Remove the request builder from the context
            self.context.clear_transport_request_builder()

            return f"Transport request submitted successfully with ID: {response.request_id}"

        except Exception as e:
            error_msg = f"Error submitting transport request: {e}"
            log.error(error_msg, exc_info=True)
            return error_msg

    def get_field_value(self, field_name: str) -> str:
        """
        Return the current value of a specific field in the transport request form.

        Args:
            field_name: Name of the field to read

        Returns:
            The field value or message if not set
        """
        log.info(f"Returning field value: {field_name}")

        try:
            # Get the field value using the get_value method
            field_value = self.context.get_transport_value(field_name)

            # Return message to be spoken by the agent
            if field_value is not None:
                return str(field_value)
            return "There is no value set for this field."

        except ValueError as e:
            # Invalid field name
            error_msg = str(e)
            log.error(f"Error getting field value: {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            # Other unexpected errors
            error_msg = f"Unexpected error getting field value: {e}"
            log.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    def get_next_steps(self) -> str:
        """
        Get simple guidance about what to do next based on current form state.

        Returns:
            Brief, actionable guidance.
        """
        log.info("Getting next steps hint")

        try:
            missing_required_fields = self.context.get_missing_required_fields()
            # Order missing required fields based on configuration
            ordered_missing_required = self._get_ordered_missing_mandatory_fields()

            log.info(f"Missing required fields: {ordered_missing_required}")
            is_complete = self.context.is_transport_complete()

            if not is_complete:
                if len(ordered_missing_required) == 1:
                    field_name = ordered_missing_required[0]
                    return f"Ask for the remaining required field: {field_name}."

                return (
                    f"Remaining required fields in order: {', '.join(ordered_missing_required)}. "
                    f"Start by asking for {ordered_missing_required[0]}."
                )

            missing_optional_fields = self.context.get_missing_optional_fields()
            if len(missing_optional_fields) > 0:
                return (
                    "All required fields complete. You can submit the request now, "
                    f"or ask for optional information: {', '.join(sorted(missing_optional_fields))}."
                )
            return "All required fields complete. You can submit the request now."

        except Exception as e:
            log.error(f"Error getting next steps: {e}")
            return f"Error getting guidance: {e}"

    def end_call(self) -> str:
        """
        End the call and provide summary.

        Returns:
            Summary of the call
        """
        log.info("Ending call.")

        # Get final status
        status = self.get_form_status()

        # Check if request is complete
        if self.context.is_transport_complete():
            return (
                "Call ended. Transport request is complete and ready to submit. "
                + status
            )
        else:
            missing = self.context.get_missing_required_fields()
            return (
                f"Call ended. Transport request incomplete. Missing: {', '.join(missing)}. "
                + status
            )
