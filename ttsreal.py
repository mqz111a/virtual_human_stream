import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

from typing import AsyncIterator

import aiohttp

from io import BytesIO
from enum import Enum


class State(Enum):
    """
    Enumeration to represent the state of the TTS processing.

    Attributes:
        RUNNING (int): Indicates that TTS processing is active.
        PAUSE (int): Indicates that TTS processing is paused.
    """
    RUNNING = 0
    PAUSE = 1


class BaseTTS:
    """
    BaseTTS is an abstract base class for implementing different TTS (Text-to-Speech) engines.
    It manages the message queue, state, and provides a framework for rendering audio from text.

    Attributes:
        opt: Configuration options for TTS.
        parent: Reference to the parent object that handles audio frames.
        fps (int): Frames per second, determining the number of samples per chunk.
        sample_rate (int): Target sample rate for audio processing.
        chunk (int): Number of samples per audio chunk.
        input_stream (BytesIO): Buffer to store incoming audio data.
        msgqueue (asyncio.Queue): Queue to hold incoming text messages for TTS processing.
        state (State): Current state of TTS processing (RUNNING or PAUSE).
    """

    def __init__(self, opt, parent):
        """
        Initializes the BaseTTS instance with configuration and parent references.

        Args:
            opt: Configuration options for TTS.
            parent: Reference to the parent object that handles audio frames.
        """
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps  # Frames per second (e.g., 50 for 20ms per frame)
        self.sample_rate = 16000  # Target sample rate for audio
        self.chunk = self.sample_rate // self.fps  # Samples per chunk (e.g., 320 for 20ms)
        self.input_stream = BytesIO()

        self.msgqueue = asyncio.Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        """
        Pauses the TTS processing by clearing the message queue and updating the state.
        """
        while not self.msgqueue.empty():
            self.msgqueue.get_nowait()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str):
        """
        Adds a text message to the TTS processing queue.

        Args:
            msg (str): The text message to be converted to speech.
        """
        self.msgqueue.put_nowait(msg)

    def render(self, quit_event: asyncio.Event):
        """
        Starts the TTS processing coroutine as an asyncio Task.

        Args:
            quit_event (asyncio.Event): Event to signal the coroutine to stop processing.
        """
        asyncio.create_task(self.process_tts(quit_event))

    async def process_tts(self, quit_event: asyncio.Event):
        """
        Coroutine that continuously processes text messages from the queue and converts them to audio.

        Args:
            quit_event (asyncio.Event): Event to signal the coroutine to stop processing.
        """
        while not quit_event.is_set():
            try:
                msg = await asyncio.wait_for(self.msgqueue.get(), timeout=1)
                self.state = State.RUNNING
                await self.txt_to_audio(msg)
            except asyncio.TimeoutError:
                continue
        print('BaseTTS task stopped')

    async def txt_to_audio(self, msg: str):
        """
        Abstract method to convert text to audio. Must be implemented by subclasses.

        Args:
            msg (str): The text message to convert to audio.
        """
        raise NotImplementedError("Subclasses must implement txt_to_audio method")


###########################################################################################
class EdgeTTS(BaseTTS):
    """
    EdgeTTS is a subclass of BaseTTS that utilizes the Edge TTS engine to convert text to audio.

    Attributes:
        None additional beyond BaseTTS.
    """

    async def txt_to_audio(self, msg: str):
        """
        Converts text to audio using the Edge TTS engine and streams the audio in chunks.

        Args:
            msg (str): The text message to convert to audio.
        """
        voicename = "zh-CN-YunxiaNeural"
        text = msg
        start_time = time.time()
        await self.__main(voicename, text)
        print(f'-------EdgeTTS processing time: {time.time() - start_time:.4f}s')

        self.input_stream.seek(0)
        stream = await self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            self.parent.put_audio_frame(stream[idx:idx + self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate()

    async def __create_bytes_stream(self, byte_stream: BytesIO) -> np.ndarray:
        """
        Reads and processes the byte stream into a normalized numpy array suitable for streaming.

        Args:
            byte_stream (BytesIO): The byte stream containing audio data.

        Returns:
            np.ndarray: The processed audio stream.
        """
        loop = asyncio.get_event_loop()
        stream, sample_rate = await loop.run_in_executor(None, sf.read, byte_stream)
        print(f'[INFO] TTS audio stream sample rate: {sample_rate}, shape: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] Audio has {stream.shape[1]} channels, only using the first channel.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            print(f'[WARN] Audio sample rate is {sample_rate}, resampling to {self.sample_rate}.')
            stream = await loop.run_in_executor(None, resampy.resample, stream, sample_rate, self.sample_rate)

        return stream

    async def __main(self, voicename: str, text: str):
        """
        Main method to interact with the Edge TTS engine and stream audio data.

        Args:
            voicename (str): The name of the voice to use for TTS.
            text (str): The text message to convert to audio.
        """
        communicate = edge_tts.Communicate(text, voicename)
        first_chunk = True
        async for chunk in communicate.stream():
            if first_chunk:
                first_chunk = False
            if chunk["type"] == "audio" and self.state == State.RUNNING:
                self.input_stream.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass  # Handle word boundaries if needed


###########################################################################################
class VoitsTTS(BaseTTS):
    """
    VoitsTTS is a subclass of BaseTTS that utilizes the Voits TTS engine for converting text to audio.

    Attributes:
        None additional beyond BaseTTS.
    """

    async def txt_to_audio(self, msg: str):
        """
        Converts text to audio using the Voits TTS engine and streams the audio in chunks.

        Args:
            msg (str): The text message to convert to audio.
        """
        audio_stream = self.gpt_sovits(
            msg,
            self.opt.REF_FILE,
            self.opt.REF_TEXT,
            "zh",
            self.opt.TTS_SERVER,
        )
        await self.stream_tts(audio_stream)

    async def gpt_sovits(self, text: str, reffile: str, reftext: str, language: str, server_url: str) -> AsyncIterator[bytes]:
        """
        Sends a request to the Voits TTS server and yields audio chunks as they are received.

        Args:
            text (str): The text to convert to speech.
            reffile (str): Path to the reference audio file.
            reftext (str): Reference text for speaker cloning.
            language (str): Language code (e.g., "zh").
            server_url (str): URL of the Voits TTS server.

        Yields:
            bytes: Audio chunks received from the server.
        """
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': language,
            'ref_audio_path': reffile,
            'prompt_text': reftext,
            'prompt_lang': language,
            'media_type': 'raw',
            'streaming_mode': True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}/tts", json=req) as res:
                end = time.perf_counter()
                print(f"VoitsTTS: Time to make POST request: {end - start:.4f}s")

                if res.status != 200:
                    error_text = await res.text()
                    print(f"VoitsTTS Error: {error_text}")
                    return

                first = True
                async for chunk in res.content.iter_chunked(32000):
                    if first:
                        end = time.perf_counter()
                        print(f"VoitsTTS: Time to first chunk: {end - start:.4f}s")
                        first = False
                    if chunk and self.state == State.RUNNING:
                        yield chunk

                print(f"VoitsTTS response elapsed time: {res.headers.get('X-Response-Time')}")

    async def stream_tts(self, audio_stream: AsyncIterator[bytes]):
        """
        Streams the audio chunks by resampling and sending them to the parent handler.

        Args:
            audio_stream (AsyncIterator[bytes]): Asynchronous iterator of audio chunks.
        """
        async for chunk in audio_stream:
            if chunk and len(chunk) > 0:
                loop = asyncio.get_event_loop()
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = await loop.run_in_executor(None, resampy.resample, stream, 32000, self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk


###########################################################################################
class XTTS(BaseTTS):
    """
    XTTS is a subclass of BaseTTS that utilizes the XTTS engine for converting text to audio.

    Attributes:
        speaker (dict): Configuration for the speaker, obtained from the TTS server.
    """

    def __init__(self, opt, parent):
        """
        Initializes the XTTS instance and sets the speaker configuration to None.

        Args:
            opt: Configuration options for TTS.
            parent: Reference to the parent object that handles audio frames.
        """
        super().__init__(opt, parent)
        self.speaker = None

    async def txt_to_audio(self, msg: str):
        """
        Converts text to audio using the XTTS engine and streams the audio in chunks.

        Args:
            msg (str): The text message to convert to audio.
        """
        if not self.speaker:
            self.speaker = await self.get_speaker(self.opt.REF_FILE, self.opt.TTS_SERVER)
        audio_stream = self.xtts(
            msg,
            self.speaker,
            "zh-cn",
            self.opt.TTS_SERVER,
            "20"
        )
        await self.stream_tts(audio_stream)

    async def get_speaker(self, ref_audio: str, server_url: str) -> dict:
        """
        Obtains the speaker configuration by sending a reference audio file to the XTTS server.

        Args:
            ref_audio (str): Path to the reference audio file.
            server_url (str): URL of the XTTS server.

        Returns:
            dict: Speaker configuration obtained from the server.
        """
        data = aiohttp.FormData()
        try:
            with open(ref_audio, 'rb') as f:
                data.add_field('wav_file', f, filename='reference.wav')
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{server_url}/clone_speaker", data=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"XTTS Error: {error_text}")
                            return {}
                        return await response.json()
        except FileNotFoundError:
            print(f"XTTS Error: Reference audio file '{ref_audio}' not found.")
            return {}
        except Exception as e:
            print(f"XTTS Exception: {e}")
            return {}

    async def xtts(self, text: str, speaker: dict, language: str, server_url: str, stream_chunk_size: str) -> AsyncIterator[bytes]:
        """
        Sends a request to the XTTS server and yields audio chunks as they are received.

        Args:
            text (str): The text to convert to speech.
            speaker (dict): Speaker configuration.
            language (str): Language code (e.g., "zh-cn").
            server_url (str): URL of the XTTS server.
            stream_chunk_size (str): Size of each audio stream chunk.

        Yields:
            bytes: Audio chunks received from the server.
        """
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}/tts_stream", json=speaker) as res:
                end = time.perf_counter()
                print(f"XTTS: Time to make POST request: {end - start:.4f}s")

                if res.status != 200:
                    error_text = await res.text()
                    print(f"XTTS Error: {error_text}")
                    return

                first = True
                async for chunk in res.content.iter_chunked(960):
                    if first:
                        end = time.perf_counter()
                        print(f"XTTS: Time to first chunk: {end - start:.4f}s")
                        first = False
                    if chunk:
                        yield chunk

                print(f"XTTS response elapsed time: {res.headers.get('X-Response-Time')}")

    async def stream_tts(self, audio_stream: AsyncIterator[bytes]):
        """
        Streams the audio chunks by resampling and sending them to the parent handler.

        Args:
            audio_stream (AsyncIterator[bytes]): Asynchronous iterator of audio chunks.
        """
        async for chunk in audio_stream:
            if chunk and len(chunk) > 0:
                loop = asyncio.get_event_loop()
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = await loop.run_in_executor(None, resampy.resample, stream, 24000, self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk
