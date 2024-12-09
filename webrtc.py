import asyncio
import json
import logging
import time
from typing import Tuple, Union
from fractions import Fraction

import numpy as np
from av import AudioFrame, Frame
from aiortc import MediaStreamTrack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for audio and video packetization
AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000  # Clock rate for video
VIDEO_FPS = 25  # Frames per second for video
VIDEO_PTIME = 1 / VIDEO_FPS  # Packetization time for video
VIDEO_TIME_BASE = Fraction(1, VIDEO_CLOCK_RATE)  # Time base for video
SAMPLE_RATE = 16000  # Sample rate for audio
AUDIO_TIME_BASE = Fraction(1, SAMPLE_RATE)  # Time base for audio


class PlayerStreamTrack(MediaStreamTrack):
    """
    A media stream track that reads frames from a player.

    This class extends aiortc's MediaStreamTrack to provide audio or video frames
    from a player. It handles timing and synchronization of frames based on the
    specified packetization time.

    Attributes:
        kind (str): Type of media ('audio' or 'video').
        _player (HumanPlayer): Reference to the associated HumanPlayer instance.
        _queue (asyncio.Queue): Queue to hold incoming frames.
        _start (float): Start time of the stream.
        _timestamp (int): Current timestamp of the stream.
        framecount (int): Counter for video frames processed.
        lasttime (float): Last timestamp when a video frame was processed.
        totaltime (float): Total time elapsed for video frames.
    """

    def __init__(self, player: 'HumanPlayer', kind: str):
        """
        Initializes the PlayerStreamTrack.

        Args:
            player (HumanPlayer): The HumanPlayer instance managing this track.
            kind (str): The type of media ('audio' or 'video').
        """
        super().__init__()  # Initialize the base MediaStreamTrack
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self._start = None
        self._timestamp = None

        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    async def next_timestamp(self) -> Tuple[int, Fraction]:
        """
        Calculates the next timestamp for the frame based on the packetization time.

        Returns:
            Tuple[int, Fraction]: A tuple containing the new timestamp and the time base.

        Raises:
            Exception: If the track is not in the 'live' state.
        """
        if self.readyState != "live":
            raise Exception("Track is not in a live state")

        if self.kind == 'video':
            time_base = VIDEO_TIME_BASE
            ptime = VIDEO_PTIME
            clock_rate = VIDEO_CLOCK_RATE
        else:  # audio
            time_base = AUDIO_TIME_BASE
            ptime = AUDIO_PTIME
            clock_rate = SAMPLE_RATE

        if self._timestamp is not None:
            self._timestamp += int(ptime * clock_rate)
            wait_time = self._start + (self._timestamp / clock_rate) - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        else:
            self._start = time.time()
            self._timestamp = 0
            logger.info(f'{self.kind} start time: {self._start}')

        return self._timestamp, time_base

    async def recv(self) -> Frame:
        """
        Retrieves the next frame from the queue, updates its presentation timestamp (PTS),
        and returns it.

        Returns:
            Frame: The next audio or video frame.

        Raises:
            Exception: If no more frames are available.
        """
        self._player._start(self)  # Ensure the player is running
        frame = await self._queue.get()

        if frame is None:
            self.stop()
            raise Exception('No more frames')

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()

            if self.framecount == 100:
                avg_fps = self.framecount / self.totaltime
                logger.info(f"Actual average FPS: {avg_fps:.4f}")
                self.framecount = 0
                self.totaltime = 0

        return frame

    def stop(self):
        """
        Stops the media stream track and notifies the associated player to stop.
        """
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


class HumanPlayer:
    """
    HumanPlayer manages audio and video stream tracks and interfaces with the rendering container.

    This class handles the coordination between audio and video tracks, manages worker tasks
    for rendering, and facilitates the queuing of audio frames.

    Attributes:
        __started (set): Set of active PlayerStreamTrack instances.
        __audio (PlayerStreamTrack): Audio stream track.
        __video (PlayerStreamTrack): Video stream track.
        __container: Reference to the rendering container (e.g., nerfreal).
        __task (asyncio.Task): Asyncio task running the player worker.
        __quit_event (asyncio.Event): Event to signal the worker to stop.
    """

    def __init__(self, nerfreal):
        """
        Initializes the HumanPlayer with audio and video tracks.

        Args:
            nerfreal: The rendering container that provides audio and video frames.
        """
        self.__started = set()
        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")
        self.__container = nerfreal
        self.__task = None
        self.__quit_event = asyncio.Event()

    @property
    def audio(self) -> MediaStreamTrack:
        """
        Retrieves the audio media stream track.

        Returns:
            MediaStreamTrack: The audio stream track.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        Retrieves the video media stream track.

        Returns:
            MediaStreamTrack: The video stream track.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        """
        Starts the player worker if not already running.

        Args:
            track (PlayerStreamTrack): The track that initiated the start.
        """
        self.__started.add(track)
        if self.__task is None:
            self.__log_debug("Starting worker task")
            self.__task = asyncio.create_task(self.player_worker())

    def _stop(self, track: PlayerStreamTrack) -> None:
        """
        Stops the player worker if no tracks are active.

        Args:
            track (PlayerStreamTrack): The track that initiated the stop.
        """
        self.__started.discard(track)

        if not self.__started and self.__task is not None:
            self.__log_debug("Stopping worker task")
            self.__quit_event.set()
            self.__task = None

        if not self.__started and self.__container is not None:
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        """
        Logs a debug message.

        Args:
            msg (str): The debug message.
            *args: Additional arguments for formatting.
        """
        logger.debug(f"HumanPlayer {msg}", *args)

    async def player_worker(self):
        """
        Worker coroutine that interfaces with the rendering container to generate frames.

        This coroutine runs until the quit event is set, allowing continuous rendering.
        """
        await self.__container.render(self.__quit_event, self.__audio, self.__video)
        self.__log_debug("Player worker stopped")

    def put_audio_frame(self, frame_data: np.ndarray):
        """
        Converts raw audio data to an AudioFrame and queues it for streaming.

        Args:
            frame_data (np.ndarray): Raw audio data as a NumPy array.
        """
        frame = AudioFrame.from_ndarray(frame_data, layout='mono', format='s16')
        frame.sample_rate = SAMPLE_RATE
        if self.__audio.readyState == "live":
            asyncio.create_task(self.__audio._queue.put(frame))

    def put_video_frame(self, frame: Frame):
        """
        Queues a video frame for streaming.

        Args:
            frame (Frame): The video frame to be streamed.
        """
        if self.__video.readyState == "live":
            asyncio.create_task(self.__video._queue.put(frame))
