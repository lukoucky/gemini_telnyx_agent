from ctypes import ArgumentError
import pyaudio
import logging

logger = logging.getLogger(__name__)


def list_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")

    device_list = []
    for i in range(0, numdevices):
        dev_info = p.get_device_info_by_host_api_device_index(0, i)
        device_list.append(
            (
                i,
                dev_info.get("name"),
                dev_info.get("maxInputChannels"),
                dev_info.get("maxOutputChannels"),
            )
        )

    return device_list


def print_devices():
    device_list = list_devices()
    for i, name, max_input, max_output in device_list:
        print(
            f"Device id={i} name={name} maxInputChannels={max_input} maxOutputChannels={max_output}"
        )


def find_device_by_name(name: str, is_input: bool, is_output: bool) -> int:
    device_list = list_devices()
    for i, device_name, max_input, max_output in device_list:
        if device_name == name and is_input and max_input > 0:
            logger.info(f"Found input device: {name} (id={i})")
            return i
        elif device_name == name and is_output and max_output > 0:
            logger.info(f"Found output device: {name} (id={i})")
            return i
    logger.error(f"No such device found: {name}")
    raise ArgumentError(f"No such device found: {name}")
