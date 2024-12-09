import argparse
import asyncio
import json
import os
import sys
from threading import Thread, Event

import aiohttp_cors
import numpy as np
import torch
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from llm.LLM import LLM
from webrtc import HumanPlayer

# Import model-specific classes
from ernerf.nerf_triplane.provider import NeRFDataset_Test
from ernerf.nerf_triplane.utils import seed_everything
from ernerf.nerf_triplane.network import NeRFNetwork
from nerfreal import NeRFReal
from musereal import MuseReal
from lipreal import LipReal

# Import Trainer class if it's defined elsewhere
# Assuming Trainer is defined in the same module or imported correctly
from trainer_module import Trainer  # Replace with actual import if different


class Server:
    """
    Server class encapsulates the WebRTC server functionalities, including handling WebSocket
    connections, managing peer connections, and integrating with a Language Model (LLM).

    Attributes:
        config (argparse.Namespace): Configuration options parsed from command-line arguments.
        nerfreals (list): List of model instances handling real-time rendering or audio-visual processing.
        statreals (list): List tracking the state of each session (0 for available, 1 for active).
        pcs (set): Set of active RTCPeerConnection instances.
        llm (LLM): Initialized Language Model instance for generating responses.
        loop (asyncio.AbstractEventLoop): Event loop for asynchronous operations.
    """

    def __init__(self, config):
        """
        Initializes the Server with the provided configuration.

        Args:
            config (argparse.Namespace): Configuration options parsed from command-line arguments.
        """
        self.config = config
        self.nerfreals = []
        self.statreals = []
        self.pcs = set()
        self.llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
        self.loop = asyncio.get_event_loop()

        self._initialize_models()
        self._initialize_statreals()

    def _initialize_models(self):
        """
        Initializes model instances based on the specified model type in the configuration.
        Supports 'ernerf', 'musetalk', and 'wav2lip' models.
        """
        if self.config.model == 'ernerf':
            self._initialize_ernerf()
        elif self.config.model == 'musetalk':
            self._initialize_musetalk()
        elif self.config.model == 'wav2lip':
            self._initialize_wav2lip()
        else:
            print(f"Unsupported model type: {self.config.model}")
            sys.exit(1)

    def _initialize_ernerf(self):
        """
        Initializes Ernerf model instances with the provided configuration.
        Sets up the neural network, trainer, and data loaders for real-time processing.
        """
        seed_everything(self.config.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(self.config)
        criterion = torch.nn.MSELoss(reduction='none')
        trainer = Trainer(
            'ngp',
            self.config,
            model,
            device=device,
            workspace=self.config.workspace,
            criterion=criterion,
            fp16=self.config.fp16,
            metrics=[],
            use_checkpoint=self.config.ckpt
        )
        test_loader = NeRFDataset_Test(self.config, device=device).dataloader()
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        for _ in range(self.config.max_session):
            nerfreal = NeRFReal(self.config, trainer, test_loader)
            self.nerfreals.append(nerfreal)

    def _initialize_musetalk(self):
        """
        Initializes MuseTalk model instances based on the configuration.
        """
        for _ in range(self.config.max_session):
            nerfreal = MuseReal(self.config)
            self.nerfreals.append(nerfreal)

    def _initialize_wav2lip(self):
        """
        Initializes Wav2Lip model instances based on the configuration.
        """
        for _ in range(self.config.max_session):
            nerfreal = LipReal(self.config)
            self.nerfreals.append(nerfreal)

    def _initialize_statreals(self):
        """
        Initializes the statreals list to track the state of each session.
        Sets all sessions to available (0) initially.
        """
        self.statreals = [0] * self.config.max_session

    async def humanecho_handler(self, request):
        """
        Handles WebSocket connections for the '/humanecho' route.
        Receives text messages from clients and forwards them to the corresponding model instance.

        Args:
            request (aiohttp.web.Request): Incoming HTTP request.

        Returns:
            aiohttp.web.WebSocketResponse: WebSocket response object.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        session_id = int(request.rel_url.query.get('sessionid', 0))
        nerfreal = self.nerfreals[session_id]
        print(f'WebSocket connection established for /humanecho (Session ID: {session_id})')

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    message = msg.data.strip()
                    if message:
                        nerfreal.put_msg_txt(message)
                    else:
                        await ws.send_str('Input message is empty')
                elif msg.type == web.WSMsgType.ERROR:
                    print(f'WebSocket error on /humanecho: {ws.exception()}')
        finally:
            print(f'WebSocket connection closed for /humanecho (Session ID: {session_id})')
        return ws

    async def humanchat_handler(self, request):
        """
        Handles WebSocket connections for the '/humanchat' route.
        Receives text messages, processes them using the LLM, and forwards responses to the client.

        Args:
            request (aiohttp.web.Request): Incoming HTTP request.

        Returns:
            aiohttp.web.WebSocketResponse: WebSocket response object.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        session_id = int(request.rel_url.query.get('sessionid', 0))
        nerfreal = self.nerfreals[session_id]
        print(f'WebSocket connection established for /humanchat (Session ID: {session_id})')

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    message = msg.data.strip()
                    if message:
                        response = await self.get_llm_response(message)
                        nerfreal.put_msg_txt(response)
                    else:
                        await ws.send_str('Input message is empty')
                elif msg.type == web.WSMsgType.ERROR:
                    print(f'WebSocket error on /humanchat: {ws.exception()}')
        finally:
            print(f'WebSocket connection closed for /humanchat (Session ID: {session_id})')
        return ws

    async def get_llm_response(self, message):
        """
        Generates a response from the Language Model (LLM) based on the input message.

        Args:
            message (str): Input message from the client.

        Returns:
            str: Response generated by the LLM.
        """
        response = await self.loop.run_in_executor(None, self.llm.chat, message)
        print(f"LLM Response: {response}")
        return response

    async def offer_handler(self, request):
        """
        Handles POST requests to the '/offer' route.
        Manages WebRTC offer processing, peer connection setup, and responds with an answer.

        Args:
            request (aiohttp.web.Request): Incoming HTTP request containing the WebRTC offer.

        Returns:
            aiohttp.web.Response: JSON response containing the SDP answer and session ID.
        """
        try:
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        except (KeyError, json.JSONDecodeError) as e:
            return web.Response(status=400, text='Invalid offer parameters')

        # Find available session
        try:
            session_id = self.statreals.index(0)
        except ValueError:
            return web.Response(status=500, text='Reached max session limit')

        self.statreals[session_id] = 1
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            """
            Callback for handling changes in the peer connection state.
            Cleans up resources when the connection is closed or failed.
            """
            print(f"Connection state changed to {pc.connectionState} for Session ID: {session_id}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.discard(pc)
                self.statreals[session_id] = 0

        player = HumanPlayer(self.nerfreals[session_id])
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        response_data = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "sessionid": session_id,
        }
        return web.json_response(response_data)

    async def human_handler(self, request):
        """
        Handles POST requests to the '/human' route.
        Processes human interactions such as echoing messages or generating chat responses.

        Args:
            request (aiohttp.web.Request): Incoming HTTP request containing interaction data.

        Returns:
            aiohttp.web.Response: JSON response indicating success or failure.
        """
        try:
            params = await request.json()
            session_id = int(params.get('sessionid', 0))
            nerfreal = self.nerfreals[session_id]
        except (ValueError, IndexError, json.JSONDecodeError):
            return web.json_response({"code": 1, "message": "Invalid session ID"}, status=400)

        if params.get('interrupt'):
            nerfreal.pause_talk()

        message_type = params.get('type')
        text = params.get('text', '').strip()

        if message_type == 'echo':
            if text:
                nerfreal.put_msg_txt(text)
            else:
                return web.json_response({"code": 1, "message": "Text cannot be empty"}, status=400)
        elif message_type == 'chat':
            if text:
                response = await self.get_llm_response(text)
                nerfreal.put_msg_txt(response)
            else:
                return web.json_response({"code": 1, "message": "Text cannot be empty"}, status=400)
        else:
            return web.json_response({"code": 1, "message": "Invalid message type"}, status=400)

        return web.json_response({"code": 0, "data": "ok"})

    async def set_audiotype_handler(self, request):
        """
        Handles POST requests to the '/set_audiotype' route.
        Updates the audio type and reinitialization state for a given session.

        Args:
            request (aiohttp.web.Request): Incoming HTTP request containing audio configuration data.

        Returns:
            aiohttp.web.Response: JSON response indicating success or failure.
        """
        try:
            params = await request.json()
            session_id = int(params.get('sessionid', 0))
            audiotype = params['audiotype']
            reinit = params['reinit']
            nerfreal = self.nerfreals[session_id]
        except (ValueError, IndexError, KeyError, json.JSONDecodeError):
            return web.json_response({"code": 1, "message": "Invalid parameters"}, status=400)

        nerfreal.set_curr_state(audiotype, reinit)
        return web.json_response({"code": 0, "data": "ok"})

    async def on_shutdown(self, app):
        """
        Shutdown handler to gracefully close all active peer connections when the server is stopping.

        Args:
            app (aiohttp.web.Application): The web application instance.
        """
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    def run_server(self):
        """
        Sets up and runs the aiohttp web server with all configured routes, CORS settings,
        and initiates any necessary background threads based on the transport protocol.
        """
        app = web.Application()
        app.on_shutdown.append(self.on_shutdown)

        # Define routes
        app.router.add_get('/humanecho', self.humanecho_handler)
        app.router.add_get('/humanchat', self.humanchat_handler)
        app.router.add_post('/offer', self.offer_handler)
        app.router.add_post('/human', self.human_handler)
        app.router.add_post('/set_audiotype', self.set_audiotype_handler)
        app.router.add_static('/', path='web', name='static')

        # Configure CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })

        # Apply CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)

        # Start rendering thread if transport is RTMP
        if self.config.transport == 'rtmp' and self.nerfreals:
            thread_quit = Event()
            render_thread = Thread(target=self.nerfreals[0].render, args=(thread_quit,))
            render_thread.start()

        # Run the web application
        web.run_app(app, host='0.0.0.0', port=self.config.listenport)


def parse_arguments():
    """
    Parses command-line arguments to configure the server.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Optimized WebRTC Server with LLM Integration")

    # Define command-line arguments with descriptions and default values
    parser.add_argument('--listenport', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--model', type=str, required=True, choices=['ernerf', 'musetalk', 'wav2lip'],
                        help='Model type to use')
    parser.add_argument('--max_session', type=int, default=10, help='Maximum number of sessions')
    parser.add_argument('--customvideo_config', type=str, default='', help='Path to custom video config JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--workspace', type=str, default='./workspace', help='Workspace directory')
    parser.add_argument('--ckpt', type=str, default='', help='Path to checkpoint file')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--transport', type=str, default='webrtc', choices=['webrtc', 'rtmp'],
                        help='Transport protocol')
    parser.add_argument('--audiotype', type=str, default='default', help='Audio type')
    parser.add_argument('--reinit', action='store_true', help='Reinitialize audio')

    # Add other arguments as needed

    return parser.parse_args()


def main():
    """
    Main function to initialize and run the server.
    Sets up multiprocessing, parses arguments, loads configurations,
    initializes the Server instance, and starts the server.
    """
    # Ensure the script is run as the main module
    if __name__ != '__main__':
        return

    # Set multiprocessing start method to 'spawn' for compatibility
    multiprocessing.set_start_method('spawn')

    # Parse command-line arguments
    config = parse_arguments()

    # Load custom video configuration if provided
    if config.customvideo_config:
        try:
            with open(config.customvideo_config, 'r') as file:
                custom_options = json.load(file)
                for key, value in custom_options.items():
                    setattr(config, key, value)
        except Exception as e:
            print(f"Error loading custom video config: {e}")
            sys.exit(1)

    # Initialize and run the server
    server = Server(config)
    server.run_server()


# Ensure the main function is called when the script is executed
if __name__ == '__main__':
    main()
