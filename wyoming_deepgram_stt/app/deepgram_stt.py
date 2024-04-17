from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    PrerecordedOptions,
    FileSource,
)
import httpx
import logging

_LOGGER = logging.getLogger(__name__)


class DeepgramSTT:
    """Class to handle Deepgram STT."""

    def __init__(self, api_key, model) -> None:
        """Initialize."""
        self.client = DeepgramClient(api_key=api_key)
        self.speech_config = PrerecordedOptions(
            model=model,
            smart_format=True,
            utterances=True,
            diarize=True,
        )

    def transcribe(self, filename: str):
        """Transcribe a file."""

        with open(filename, "rb") as file:
            buffer_data = file.read()

        payload : FileSource = {
            "buffer": buffer_data,
        }
        
        try:
            response = self.client.listen.prerecorded.v("1").transcribe_file(
                payload,
                self.speech_config,
                # timeout=httpx.Timeout(300.0, connect=10.0),
            )
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript
        except Exception as e:
            _LOGGER.warning(f"Failed to transcribe: {e}")
            return None