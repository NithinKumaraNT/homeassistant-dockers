#!/usr/bin/env python3
import argparse
import asyncio
import logging
from functools import partial
import contextlib

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import DeepgramEventHandler
from .deepgram_stt import DeepgramSTT
import os

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of faster-whisper model to use",
    )
    parser.add_argument("--uri", default="tcp://0.0.0.0:10000", help="unix:// or tcp://")
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=logging.BASIC_FORMAT
    )
    _LOGGER.debug(args)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="Deepgram",
                description="Deepgram Speech Transcription Service",
                attribution=Attribution(
                    name="Nithin Teekaramanaa",
                    url="https://github.com/NithinKumaraNT/ha-custom-dockers",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="Deepgram STT",
                        description="Deepgram Speech Transcription Service",
                        attribution=Attribution(
                            name="Nithin Teekaramanaa",
                            url="https://github.com/NithinKumaraNT/ha-custom-dockers",
                        ),
                        installed=True,
                        version=__version__,
                        languages=["en"],
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.debug("Loading Deepgram STT")
    stt_model = DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"], model=args.model)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            DeepgramEventHandler,
            wyoming_info,
            args,
            stt_model,
            model_lock,
        )
    )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())