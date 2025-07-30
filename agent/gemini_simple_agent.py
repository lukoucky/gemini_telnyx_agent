import asyncio
from typing import Optional

import os
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig
from google.genai.live import AsyncSession

import pyaudio
import logging
import argparse

from websockets import ConnectionClosedOK

from agent.match_facility import match_facility
from agent.audio_devices import find_device_by_name, print_devices
from agent.gemini_agent_tools import GeminiAgentTools

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS="You are Harmony, an AI conversational assistant. Lead engaging convesation with the user."

class VoiceBot:

    def __init__(
        self,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        agent_tools: Optional[GeminiAgentTools] = None,
    ):
        """
        Initialize the voice bot.

        Args:
            input_device_index: The index of the input device to use.
            output_device_index: The index of the output device to use.
        """
        if agent_tools is None:
            self.agent_tools = GeminiAgentTools()
        else:
            self.agent_tools = agent_tools
        self.audio_queue = asyncio.Queue()
        self.model = "models/gemini-live-2.5-flash-preview"
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.model_speaking = False
        self.call_done = False
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.end_call = False
        self.introduction_sent = False

    async def start(self):
        """
        Start the voice bot by connecting to Gemini Live API and providing tool information.
        """
        logger.info("Connecting to Gemini Live API ...")

        # Initialize session and transport request
        try:
            session_id = await self.agent_tools.initialize_session()
            logger.info(f"Initialized session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            # Continue with default session

        config = LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            # TODO: Implement switch for system instructions based on agent type  # noqa: FIX002
            system_instruction=SYSTEM_INSTRUCTIONS,
            tools=[
                self.agent_tools.store_info,
                match_facility,
                self.agent_tools.get_form_status,
                self.agent_tools.submit_transport_request,
                self.agent_tools.get_field_value,
                self.agent_tools.get_next_steps,
                self.agent_tools.end_call,
            ],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                ),
            ),
            max_output_tokens=8192,
        )

        # Start audio streaming
        async with self.client.aio.live.connect(
            model=self.model, config=config
        ) as session:
            logger.info("Connected! Sending introduction message...")

            # Send introduction message to trigger the model to speak first
            await self.send_introduction(session)

            logger.info("Starting audio consumers/producers")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.capture_audio_from_caller(session))
                tg.create_task(self.consume_model_stream(session))
                tg.create_task(self.speak_model_response())

    async def send_introduction(self, session: AsyncSession):
        """
        Send an introduction message to the model to trigger it to speak first.
        """
        if self.introduction_sent:
            return

        introduction_prompt = (
            "Please introduce yourself to the caller. Let them know you're here to help with "
            "their inter-hospital transport request. Keep it extra brief just 2 sentences and "
            "professional, then ask how you can assist them today."
        )

        logger.info("Sending introduction prompt to model")
        await session.send_realtime_input(text=introduction_prompt)

        self.introduction_sent = True

        # Give the model a moment to process the introduction before starting audio capture
        await asyncio.sleep(0.5)

    async def send_good_bye(self, session: AsyncSession):
        """
        Send an good bye message to the model to end the call.
        """
        introduction_prompt = "Thank you for your time. Good bye."

        logger.info("Sending good bye prompt to model")
        await session.send_realtime_input(text=introduction_prompt)
        await asyncio.sleep(0.5)

    async def capture_audio_from_caller(self, session: AsyncSession):
        """
        Asyncio task which retrieves audio from the microphone and sends it onward to the Gemini model.
        """
        logger.info("Initiating audio capture from caller ...")
        audio = pyaudio.PyAudio()
        input_index = (
            self.input_device_index or audio.get_default_input_device_info()["index"]
        )

        logger.info(audio.get_device_info_by_index(input_index))

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=512,
            input_device_index=input_index,
        )

        logger.info(f"Audio capture loop active")
        await self.send_audio_realtime(stream, session)

    async def send_audio_realtime(self, stream, session: AsyncSession):
        total_sent = 0

        while not self.end_call:
            data = await asyncio.to_thread(
                stream.read, 512, exception_on_overflow=False
            )
            # Only send audio when model is not speaking
            if not self.model_speaking:
                total_sent += len(data)
                logger.debug(
                    f"Sending {len(data)} bytes of caller audio data to model (total: {total_sent})"
                )
                try:
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm")
                    )
                except ConnectionClosedOK:
                    logger.info("websocket closed while sending audio")

    async def consume_model_stream(self, session: AsyncSession):
        """
        Read stream of message from Gemini model.
        """
        while not self.end_call:
            logger.info("Initiating model stream consumption ...")
            async for message in session.receive():
                # Only log important messages, not audio data
                if message.tool_call is not None:
                    logger.info(f"Model requested a tool call: {message.tool_call}")
                elif message.setup_complete is not None:
                    logger.info("Model setup complete")
                elif message.server_content and message.server_content.turn_complete:
                    logger.info("Model turn complete")
                else:
                    # For other messages (like audio data), just log that we received something
                    logger.debug(
                        "Received message from model (audio data or other content)"
                    )

                if message.tool_call is not None:
                    for fun_call in message.tool_call.function_calls:
                        logger.info(f"Model requested a function call: {fun_call}")
                        if fun_call.name == "match_facility":
                            facility_name = fun_call.args["facility_name"]
                            city = fun_call.args.get("city")
                            state_abbrev = fun_call.args.get("state_abbrev")
                            logger.info(
                                f"Calling match_facility with arguments: name={facility_name}, city={city}, state_abbrev={state_abbrev}"
                            )
                            result = match_facility(facility_name, city, state_abbrev)
                            logger.info(f"Matched facilities: {result}")
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="match_facility",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                        elif fun_call.name == "store_info":
                            field_name = fun_call.args["field_name"]
                            field_value = fun_call.args["field_value"]
                            result = self.agent_tools.store_info(field_name, field_value)
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="store_info",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                        elif fun_call.name == "get_form_status":
                            result = self.agent_tools.get_form_status()
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="get_form_status",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                        elif fun_call.name == "submit_transport_request":
                            result = await self.agent_tools.submit_transport_request()
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="submit_transport_request",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                        elif fun_call.name == "get_field_value":
                            field_name = fun_call.args["field_name"]
                            result = self.agent_tools.get_field_value(field_name)
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="get_field_value",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                        elif fun_call.name == "get_next_steps":
                            result = self.agent_tools.get_next_steps()
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="get_next_steps",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                        elif fun_call.name == "end_call":
                            result = self.agent_tools.end_call()
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name="end_call",
                                    response={"result": result},
                                    id=fun_call.id,
                                )
                            )
                            await self.send_good_bye(session)
                            logger.info("Ending call now.")
                            self.end_call = True
                            await session.close()
                            return

                    continue

                content = message.server_content
                if content is None:
                    logger.warning("Message has no server content!")
                    continue

                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data:
                            # FIXME: here assuming inline data can only be audio
                            if not self.model_speaking:
                                self.model_speaking = True
                                logger.info("Model started speaking")
                            self.audio_queue.put_nowait(part.inline_data.data)
                            logger.debug(
                                f"Received {len(part.inline_data.data)} bytes of audio data from model"
                            )

                        elif part.executable_code:
                            logger.info("Model requested executable code")
                            # TODO: handle executable code

                if content.turn_complete:
                    if self.model_speaking:
                        self.model_speaking = False
                        logger.info("Model finished speaking, ready for caller")

    async def speak_model_response(self):
        """
        Retrieve audio from the queue produced by consume_model_stream and play it.
        """
        logger.info("Initiating model response playback ...")
        audio = pyaudio.PyAudio()
        output_index = (
            self.output_device_index or audio.get_default_output_device_info()["index"]
        )
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            output_device_index=output_index,
        )
        while not self.end_call:
            data = await self.audio_queue.get()
            logger.debug(f"Playing {len(data)} bytes of model audio data")
            await asyncio.to_thread(stream.write, data)


def run_agent():
    parser = argparse.ArgumentParser(
        description="Configure input and output device names"
    )
    parser.add_argument(
        "-i",
        "--input-device",
        dest="input_name",
        help="Input device name",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-device",
        dest="output_name",
        help="Output device name",
        default=None,
    )
    args = parser.parse_args()
    input_dev = (
        find_device_by_name(args.input_name, True, False) if args.input_name else None
    )
    output_dev = (
        find_device_by_name(args.output_name, False, True) if args.output_name else None
    )
    print_devices()

    bot = VoiceBot(input_device_index=input_dev, output_device_index=output_dev)
    asyncio.run(bot.start())


if __name__ == "__main__":
    run_agent()
