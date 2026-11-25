"""One controller as the receiver sees it: discovery, the GMsg queue, the display link and its game session.

The receiver creates a :class:`ControllerSession` the first time a controller identifies itself
and routes everything that controller does to it.  Each session owns its own UDP sockets (the
display link and the game session), ICE credentials, DTLS certificates, GMsg queue and display
RPC state machine, so controllers never share wire state; the receiver only shares the HTTP
listeners, the helper processes and the log.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from google.protobuf.message import DecodeError

from stadiacast import SERVER_NAME, signaling
from stadiacast.audio import AudioSender, Source, ffmpeg_path, source_from_spec
from stadiacast.config import AUDIO_APPLICATIONS
from stadiacast.errors import SetupError
from stadiacast.game_session import GameSession
from stadiacast.gamepad import Button, GamepadState
from stadiacast.http_server import HttpResponse
from stadiacast.input_protocol import InputDecoder
from stadiacast.microphone import MIC_PAYLOAD_TYPE, MicrophoneCapture, capture_path
from stadiacast.protos import PKG_RPC, decode_unknown
from stadiacast.rtc import (
    WIRE_ID_RTP,
    WIRE_IDS_DC_RPC,
    DtlsCertificate,
    RtcEndpoint,
    RtpPacket,
    pack_wire_message,
    parse_rtp,
    parse_wire_message,
)

if TYPE_CHECKING:
    from stadiacast.receiver import Receiver

#: Milestones of the post-DTLS phase, in order.  The display state never moves backwards.
DISPLAY_STATES = ("idle", "dtls-connected", "get-controller-state-sent", "controller-state-received", "ready", "input")
#: How long the controller has to answer ``set_microphone`` before it is asked again: its own
#: microphone takes a third of a second to start, and the answer comes with it.
MIC_ANSWER_TIMEOUT = 1.0
#: How often one request is sent in all, and how often a controller that sends unasked is told to stop.
MIC_TRIES = 4

#: Placeholder identity of this display in the discovery responses (the firmware only checks presence).
DISPLAY_ID = SERVER_NAME
#: The controller echoes the display token as ``GmsgPollRequest.subscription``, so a per-controller
#: token attributes GMsg polls.  Protocol maximum: 63 characters.
DISPLAY_TOKEN_PREFIX = f"{SERVER_NAME}-display"
#: Placeholder ``registration_token`` of RegisterControllerResponse (presence is all the firmware checks).
REGISTRATION_TOKEN = SERVER_NAME


def _clamp_volume(v: float) -> float:
    """The gain range that makes sense: below 0.02 is silence, above 4 Opus clips before it gains."""
    return max(0.02, min(4.0, float(v)))


def _as_bool(v: Any) -> bool | None:
    """A tri-state read of a decoded protobuf scalar: None when the field was absent."""
    return None if v is None else bool(v)


def _fresh_display_session() -> dict[str, Any]:
    return {
        "state": "idle",
        "rpc_in": 0,
        "rpc_out": 0,
        "requests": {},
        "notifications": {},
        "get_controller_state_sent": 0,
        "controller_state": None,
        "controller_state_errors": 0,
        "trigger": None,
        "time_replies": 0,
        "keepalive_replies": 0,
        "wire_in": {},
        "input_rpc": 0,
        # What the controller reports about itself (protocol.md §5): the battery gauge (#10)
        # and the audio configuration (#4, the headset going in or out).
        "battery": None,
        "battery_percent": None,
        "battery_samples": 0,
        "headset": None,
        "audio_mode": None,
        "audio_events": 0,
    }


class ControllerSession:
    """Everything the receiver keeps for one controller."""

    def __init__(self, receiver: Receiver, slot: int, key: str, ip: str | None) -> None:
        self.rx = receiver
        self.cfg = receiver.cfg
        self.protos = receiver.protos
        self.slot = slot
        self.label = f"c{slot}"  # short name in logs, /_status and ?controller=
        self.key = key  # the identifier the controller was first seen with (device id, or ip:<addr>)
        self.ip = ip
        self.device_id: str | None = None  # RegisterController / DiscoverDisplay body `device_id` (a UUID)
        self.serial: str | None = None  # the `controller-device-id` header (serial-like)
        self.display_token = f"{DISPLAY_TOKEN_PREFIX}-{slot}"
        self.last_seen = time.monotonic()
        self.log = receiver.log.bind(controller=self.label)
        self.cert = DtlsCertificate()
        self.endpoint = RtcEndpoint(
            cert=self.cert,
            controlling=True,
            dtls_role=self.cfg.dtls_role,
            log=self.log,
            on_rtp=self._on_rtp,
            on_rtcp=self._on_rtcp,
            on_app_data=self._on_app_data,
            on_state=self._on_rtc_state,
            name="input",
            fast_check_interval=self.cfg.ice_fast_checks,
        )
        self.input = InputDecoder(self.protos, on_event=self.emit_input_event)
        self.game = GameSession(self)
        self.session: dict[str, Any] = {
            "state": "idle",
            "role": None,
            "remote": None,
            "offer_served": 0,
            "answers_received": 0,
            "offers_received": 0,
            "input_series": 0,
            "last_ack": None,
            "input_payload_type": None,
            "registered": None,
            "discovery": None,
            "gmsg": {"pending_id": None, "sent": 0, "acked": 0, "polls": 0, "last_ack": None},
            "display": _fresh_display_session(),
            "game": self.game.state,
        }
        self._offer_msg: Any = None
        self._last_answer: Any = None
        self._gmsg_seq = 0
        self._gmsg_pending: dict[str, Any] | None = None
        self._gmsg_event = asyncio.Event()
        self._rtcp_task: asyncio.Task[None] | None = None
        self._display_task: asyncio.Task[None] | None = None
        self._display_fallback_task: asyncio.Task[None] | None = None
        self._dtls_at: float | None = None  # monotonic time of dtls-connected
        self._resolved_at: float | None = None  # monotonic time v1:ResolveDiscovery was answered
        self._rpc_ticket = 0
        self._rpc_pending: dict[int, str] = {}  # tickets of our requests -> purpose
        self._last_input: GamepadState | None = None  # previous ControllerToDisplay #3 report
        #: This boot's latest gamepad report from either channel, and when it came (monotonic).
        self.latest_input: tuple[GamepadState, float] | None = None
        self._audio: AudioSender | None = None  # headset jack, created when a stream starts
        # The stream's settings, kept across starts and stops (a sender only lives while it plays).
        self._audio_volume = _clamp_volume(self.cfg.audio_volume)
        self._audio_application = self.cfg.audio_application
        self._audio_task: asyncio.Task[None] | None = None  # this boot's --audio-source autostart
        #: The controller's microphone. ``mic_on`` is the switch as the receiver wants it and has
        #: told the controller. What then arrives goes to the receiver's device (``rx.mic_device``)
        #: when there is one, and with ``mic_record`` into a recording, which the first packet
        #: opens and the switch closes.
        self.mic_on = False
        self.mic_ptt = self.cfg.mic == "ptt"  # the Assistant button holds the switch
        self.mic_record = self.cfg.mic_record  # what arrives is written into files
        self.mic_pcm = self.cfg.mic_pcm  # the next recording gets a PCM file as well
        self._mic_heard = 0  # packets since the switch was thrown, for the log of a microphone that is not recorded
        self._mic: MicrophoneCapture | None = None  # the recording that is open
        self._mic_done: MicrophoneCapture | None = None  # the one before it, which /_mic still names
        self._mic_ending: list[MicrophoneCapture] = []  # closed, their decoders seeing the rest through
        self._mic_ticket: int | None = None  # the request that turned it on, whose answer may refuse
        self._mic_asked: tuple[int, bool, int] | None = None  # the request not answered yet: ticket, on, tries
        self._mic_watch: asyncio.TimerHandle | None = None  # asks again when no answer comes
        self._mic_told = float("-inf")  # when a request last went out (monotonic)
        self._mic_held = False  # the switch is on because the Assistant button is held
        self._mic_button = False  # the Assistant button as the last report had it
        self._mic_autostarted = False  # --mic on asks once per boot
        self._mic_stray = 0  # packets that came with the switch off
        self._mic_strays = 0  # those since the switch was last thrown
        self._mic_hushed = 0  # times a controller that sent them was told to stop, since then

    def __repr__(self) -> str:
        return f"ControllerSession({self.label}, key={self.key!r}, ip={self.ip!r})"

    @property
    def display_state(self) -> str:
        """The post-DTLS milestone reached (one of :data:`DISPLAY_STATES`)."""
        return str(self.session["display"]["state"])

    def now(self) -> float:
        return self.rx.now()

    def seen(self) -> None:
        """Note traffic on one of its links.  A controller that plays sends nothing over HTTP, and
        the session idle the longest is the one that makes room for a new controller."""
        self.last_seen = time.monotonic()

    def touch(self, ip: str | None, device_id: str | None = None, serial: str | None = None) -> None:
        """Note a sighting: the address it came from and any identifier it carried."""
        self.last_seen = time.monotonic()
        if ip:
            self.ip = ip
        if device_id:
            self.device_id = device_id
        if serial:
            self.serial = serial

    # ---- lifecycle -----------------------------------------------------------------------
    async def start(self, bind: str, udp_port: int, game_udp_port: int) -> None:
        await self.endpoint.start(bind, udp_port)
        await self.game.start(bind, game_udp_port)

    def stop(self) -> None:
        for task in (self._rtcp_task, self._display_task, self._display_fallback_task):
            if task:
                task.cancel()
        self._rtcp_task = self._display_task = self._display_fallback_task = None
        self._stop_audio()
        if self.mic_on:  # a controller left with its microphone on would go on sending to nobody
            self.mic_stop(reason="stop")
        self._forget_asked()  # the link ends here: nobody is left to wait for that answer
        self._close_mic()
        if self.rx.mic_device is not None:
            self.rx.mic_device.drop(self.label)
        if self.rx.gamepads is not None:
            self.rx.gamepads.release(self.label)
        self.game.close()
        self.endpoint.close()

    async def wait_closed(self) -> None:
        """After :meth:`stop`: wait until both UDP sockets are closed and their ports can be bound again."""
        await self.endpoint.wait_closed()
        await self.game.endpoint.wait_closed()

    def reset_boot_state(self, reason: str = "new offer") -> None:
        """A new offer from this controller = a new boot: per-boot timestamps and the post-DTLS tasks start over.

        Otherwise a second full boot inherits the first boot's dtls-connected time:
        ``--gmsg-ack-release`` then answers the new ACK poll at once (the controller powers off
        immediately) and the display RPC trigger never fires.  The same reset runs when the
        controller closes the link (:meth:`_on_rtc_state`).
        """
        self._reset_microphone()  # before the way out below, which only knows the link's own state
        if self._dtls_at is None and self._resolved_at is None and self._display_task is None:
            return
        for name in ("_rtcp_task", "_display_task", "_display_fallback_task"):
            task = getattr(self, name)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            setattr(self, name, None)
        self.log(
            "boot_state_reset",
            reason=reason,
            previous_dtls=self._dtls_at is not None,
            previous_resolve=self._resolved_at is not None,
        )
        self._dtls_at = None
        self._resolved_at = None
        self._rpc_pending.clear()
        self._rpc_ticket = 0
        self._last_input = None
        self.latest_input = None  # nothing it held is held any more
        self.session["display"] = _fresh_display_session()
        self._stop_audio()
        if self.rx.gamepads is not None:
            self.rx.gamepads.release(self.label)
        self.game.reset()
        self.rx.live_changed()

    def status(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "key": self.key,
            "device_id": self.device_id,
            "serial": self.serial,
            "ip": self.ip,
            "last_seen_s": round(time.monotonic() - self.last_seen, 1),
            "session": self.session,
            "rtc_stats": self.endpoint.stats,
            "input_stats": dict(self.input.stats),
            "local_ice": self.endpoint.local_params(),
            "udp": [self.cfg.advertise_ip, self.endpoint.local_port],
            "battery_percent": self.session["display"]["battery_percent"],
            "headset": self.session["display"]["headset"],
            "playback": self.audio_state(),
            "microphone": self.mic_state(),
            **self.game.status(),
        }

    # ---- headset audio -------------------------------------------------------
    def set_audio_mode(self, mode: int, f3: bool | None = None) -> dict[str, Any]:
        """``DisplayToController`` #16 ``SetAudioModeRequest``: open the firmware's audio path.

        The table is explicit in the firmware: mode 3 is rejected as
        ``Unsupported audio mode: UDP_PCM``, modes 1, 2 and 4 map to audio paths 1, 2 and 3.
        Only ``mode=2`` (the WebRTC Opus route) feeds the jack; without this request the
        controller takes the RTP and plays nothing.
        """
        p = self.protos
        msg = p.DisplayToController()
        msg.set_audio_mode.mode = int(mode)
        if f3 is not None:
            msg.set_audio_mode.f3 = bool(f3)
        ticket = self.send_rpc("request", msg, purpose="set_audio_mode")
        self.log("audio_mode_sent", mode=int(mode), f3=f3, ticket=ticket)
        return {"ok": ticket is not None, "ticket": ticket, "mode": int(mode)}

    async def audio_start(
        self,
        source: Source | None = None,
        *,
        srtp: bool = True,
        volume: float | None = None,
        application: str | None = None,
    ) -> dict[str, Any]:
        """Send audio to the headset jack until :meth:`audio_stop` (or the link goes away).

        ``source=None`` starts the pacer with no encoder: the caller supplies the Opus frames
        through :meth:`audio_feed`.  That is what the tests use, and what a caller with its
        own encoder would do.  ``volume`` and ``application`` (one of
        :data:`~stadiacast.config.AUDIO_APPLICATIONS`) become the session's, for this start and
        the later ones (see :meth:`audio_set_volume`).  A running stream keeps its sender, and so
        its SSRC and sequence numbers, across a restart.
        """
        if application is not None and application not in AUDIO_APPLICATIONS:
            raise ValueError(f"unknown application {application!r}; use {' or '.join(AUDIO_APPLICATIONS)}")
        if volume is not None:
            self._audio_volume = _clamp_volume(volume)
        if application is not None:
            self._audio_application = application
        if self._audio is None:
            self._audio = AudioSender(
                self.endpoint,
                log=self.log,
                srtp=srtp,
                bitrate_kbps=self.cfg.audio_bitrate_kbps,
                volume=self._audio_volume,
                application=self._audio_application,
                follow_system_volume=self.cfg.audio_follow_volume,
            )
        else:
            self._audio.volume = self._audio_volume
            self._audio.application = self._audio_application
        switched: dict[str, Any] | None = None
        if self.cfg.audio_mode:
            switched = self.set_audio_mode(self.cfg.audio_mode, f3=True)
        try:
            info = await self._audio.start(source)
        except Exception:
            self._audio = None  # ffmpeg missing or the sound card refused the source
            raise
        return {**info, "audio_mode": switched} if switched is not None else info

    async def audio_start_spec(
        self, spec: str, *, volume: float | None = None, application: str | None = None
    ) -> dict[str, Any]:
        """``/_audio``: start from a ``--audio-source`` spelling.

        A bad spelling is a ``ValueError``, a capture this computer cannot open a
        :class:`~stadiacast.errors.SetupError`.
        """
        source = await self._resolve_source(spec)
        return await self.audio_start(source, volume=volume, application=application)

    async def _resolve_source(self, spec: str) -> Source:
        """:func:`~stadiacast.audio.source_from_spec` in a worker thread.

        Resolving a capture enumerates the PortAudio devices, initialises COM on Windows or runs
        ``pactl`` on Linux; on the event loop that would stall every controller's DTLS and RTP.
        A capture of the device the microphone is played into (``--mic-device``) is refused: the
        headset would hear its own microphone.
        """
        device = self.rx.mic_device
        if device is None:
            return await asyncio.to_thread(source_from_spec, spec)
        # with the microphone's output closed meanwhile, where it would keep the devices as they were
        source = await asyncio.to_thread(device.fresh, lambda: source_from_spec(spec))
        if source.label == f"capture {device.output.loop}":
            raise SetupError(
                f"{device.output.loop!r} carries the controllers' microphones (--mic-device): captured for the"
                " headset, they would be heard in it. Capture another device for the headset"
            )
        return source

    async def audio_set_volume(self, volume: float) -> dict[str, Any]:
        """Change the gain of this controller's stream, the one playing and the ones started later.

        The gain belongs to the session, not to a sender: without a stream it waits for the next
        start (``--audio-source`` at READY included).  ffmpeg takes it at start, so a running
        source is restarted.
        """
        self._audio_volume = _clamp_volume(volume)
        sender = self._audio
        if sender is None:
            return {"playing": False, "volume": self._audio_volume}
        sender.volume = self._audio_volume
        if sender.playing and sender.source is not None:
            await sender.start(sender.source)
        return sender.state()

    async def audio_stop(self) -> dict[str, Any]:
        sender, self._audio = self._audio, None
        if sender is None:
            return {"playing": False}
        return await sender.stop()

    def audio_feed(self, payload: bytes, marker: bool = False) -> None:
        """One Opus frame into a running sender (only meaningful with ``audio_start(None)``)."""
        if self._audio is None:
            raise RuntimeError("audio is not running")
        self._audio.enqueue(payload, marker)

    def audio_state(self) -> dict[str, Any]:
        if self._audio is None:
            return {"playing": False}
        return self._audio.state()

    def _maybe_start_audio(self) -> None:
        """``--audio-source``: start the stream once per boot, when the display link reaches READY.

        A stream that is already playing (started through ``/_audio``) is left alone.
        """
        if not self.cfg.audio_source or self._audio_task is not None:
            return
        if self._audio is not None and self._audio.playing:
            return
        self._audio_task = asyncio.get_running_loop().create_task(self._audio_autostart(self.cfg.audio_source))

    async def _audio_autostart(self, spec: str) -> None:
        """Resolve ``--audio-source`` and start it.  Nothing here raises: this task has no caller to
        report to, so every failure is a log record and the display link carries on without audio."""
        try:
            source = await self._resolve_source(spec)
        except (SetupError, ValueError) as e:  # a bad spelling, a capture this computer cannot open
            self.log("audio_source_invalid", spec=spec, error=str(e))
            return
        except Exception as e:
            self.log("audio_start_failed", spec=spec, error=repr(e))
            return
        if self._audio is not None and self._audio.playing:
            return  # /_audio started a stream while the source was being resolved
        try:
            await self.audio_start(source)
        except Exception as e:  # no ffmpeg, a sound card that refused the source
            self.log("audio_start_failed", spec=spec, error=repr(e))

    def _stop_audio(self) -> None:
        if self._audio_task is not None:
            self._audio_task.cancel()
            self._audio_task = None
        sender, self._audio = self._audio, None
        if sender is not None:
            sender.kill_now()

    # ---- the controller's microphone -----------------------------------------------------
    def mic_start(self, *, reason: str = "api") -> dict[str, Any]:
        """Ask the controller for its microphone: ``DisplayToController.set_microphone`` true.

        It answers with an empty response and sends Opus on the display link from then on, one
        packet per 20 ms (``docs/protocol.md``); the first packet opens the recording. The request
        only goes to a link that is READY, where the controller takes requests: before that
        nothing is sent, nothing changes, and the result says so. Nor is anything sent while
        nothing takes the sound, neither the computer's device nor a recording (``nowhere``).
        """
        if not self.mic_record and self.rx.mic_device is None:
            self.log("mic_switch", on=True, reason=reason, ticket=None, nowhere=True)
            return {"ok": False, "on": self.mic_on, "ticket": None, "nowhere": True}
        return self._set_microphone(True, reason)

    def mic_stop(self, *, reason: str = "api") -> dict[str, Any]:
        """``set_microphone`` false, and the end of the recording that is open."""
        out = self._set_microphone(False, reason)
        self._close_mic()
        return out

    def _link_ready(self) -> bool:
        return DISPLAY_STATES.index(self.display_state) >= DISPLAY_STATES.index("ready")

    def _set_microphone(self, on: bool, reason: str) -> dict[str, Any]:
        ticket = self._ask_microphone(on) if self._link_ready() else None
        if ticket is not None or not on:  # a controller that cannot be told has its microphone off
            changed, self.mic_on = self.mic_on != on, on
            self._mic_ticket = ticket if on else None
            self._mic_held = on and reason == "ptt"
            self._mic_strays = self._mic_hushed = self._mic_heard = 0
            if changed:
                self.rx.live_changed()
        self.log("mic_switch", on=on, reason=reason, ticket=ticket)
        return {"ok": ticket is not None, "on": self.mic_on, "ticket": ticket}

    def _ask_microphone(self, on: bool, tries: int = 1) -> int | None:
        """One ``set_microphone`` request, and a watch on its answer. The request is a single
        datagram: lost, it would leave a controller sending for a receiver that thinks it silent,
        or silent for one that waits for it."""
        msg = self.protos.DisplayToController()
        msg.set_microphone.value = on
        ticket = self.send_rpc("request", msg, purpose="microphone_on" if on else "microphone_off")
        self._forget_asked()
        if ticket is not None:
            self._mic_asked = (ticket, on, tries)
            self._mic_told = time.monotonic()
            with contextlib.suppress(RuntimeError):  # no loop runs: nothing is left that could ask again
                self._mic_watch = asyncio.get_running_loop().call_later(
                    MIC_ANSWER_TIMEOUT, self._on_microphone_unanswered, ticket
                )
        return ticket

    def _forget_asked(self) -> None:
        if self._mic_watch is not None:
            self._mic_watch.cancel()
        self._mic_watch = self._mic_asked = None

    def _on_microphone_answer(self, ticket: int, on: bool, code: int, message: str | None) -> None:
        """The controller's response to a ``set_microphone`` request: nothing waits for it any
        more, and an error in answer to true is a refusal."""
        if self._mic_asked is not None and self._mic_asked[0] == ticket:
            self._forget_asked()
        if on and code:
            self._on_microphone_refused(ticket, code, message)

    def _on_microphone_unanswered(self, ticket: int) -> None:
        """No answer in :data:`MIC_ANSWER_TIMEOUT`, and for true no sound either: the request is
        sent again, :data:`MIC_TRIES` times in all. A controller that still says nothing to true
        is taken to be silent, and the switch is off again."""
        self._mic_watch = None
        if self._mic_asked is None or self._mic_asked[0] != ticket:
            return
        _, on, tries = self._mic_asked
        again = self._ask_microphone(on, tries + 1) if tries < MIC_TRIES and self._link_ready() else None
        self.log("mic_unanswered", on=on, ticket=ticket, tries=tries, again=again)
        if again is not None:
            if on:
                self._mic_ticket = again
            return
        self._forget_asked()
        if on:
            self.mic_on = self._mic_held = False
            self._mic_ticket = None
            self._close_mic()
            self.rx.live_changed()

    def _reset_microphone(self) -> None:
        """A controller that boots, or whose link has ended, has its microphone off."""
        self._close_mic()
        self._forget_asked()
        self._mic_autostarted = self._mic_held = self._mic_button = False
        self._mic_ticket = None
        self._mic_told = float("-inf")
        self._mic_strays = self._mic_hushed = 0
        if self.rx.mic_device is not None:
            self.rx.mic_device.drop(self.label)  # the next boot's stream begins anew
        if self.mic_on:
            self.mic_on = False
            self.rx.live_changed()

    def _on_microphone_refused(self, ticket: int, code: int, message: str | None) -> None:
        """The controller answered ``set_microphone`` true with an error: the switch is off,
        unless it has been thrown again since, and that request's answer decides."""
        stale = ticket != self._mic_ticket
        self.log("mic_refused", ticket=ticket, code=code, error_message=message, stale=stale)
        if stale:
            return
        self.mic_on = self._mic_held = False
        self._mic_ticket = None
        self._close_mic()
        self.rx.live_changed()

    def set_mic_pcm(self, on: bool) -> dict[str, Any]:
        """Decode the recordings to PCM as well, or stop doing so. A change closes the open
        recording: the next packet starts one whose two files cover the same sound. Without
        ffmpeg it is a :class:`~stadiacast.errors.SetupError`."""
        if on and ffmpeg_path() is None:
            raise SetupError("the PCM file is decoded by ffmpeg, and none was found on PATH")
        if on != self.mic_pcm:
            self._close_mic()
            self.mic_pcm = on
            self.log("mic_pcm", on=on)
        return {"ok": True, "pcm": self.mic_pcm}

    def set_mic_record(self, on: bool) -> dict[str, Any]:
        """Write what arrives into files, or stop doing so: the recording that is open ends. A
        microphone that then goes nowhere, for want of a device, is switched off."""
        if on != self.mic_record:
            self.mic_record = on
            self.log("mic_record", on=on)
            if not on:
                self._close_mic()
                if self.mic_on and self.rx.mic_device is None:
                    self.mic_stop(reason="nowhere")
        return {"ok": True, "record": self.mic_record}

    def set_mic_ptt(self, on: bool) -> dict[str, Any]:
        """Make the Assistant button the microphone's switch, or stop it being one."""
        if on != self.mic_ptt:
            self.mic_ptt = on
            self.log("mic_ptt", on=on)
        if not on and self._mic_held:  # nothing would let go of it any more
            self.mic_stop(reason="ptt")
        return {"ok": True, "ptt": self.mic_ptt}

    def mic_state(self) -> dict[str, Any]:
        """What ``/_mic`` and ``/_status`` say: the switch, the two settings, the recording that
        is open and the one that ended last."""
        return {
            "on": self.mic_on,
            "ptt": self.mic_ptt,
            "record": self.mic_record,
            "pcm": self.mic_pcm,
            "recording": None if self._mic is None else self._mic.state(),
            "last": None if self._mic_done is None else self._mic_done.state(),
            "stray": self._mic_stray,
        }

    def _on_mic_packet(self, pkt: RtpPacket, via: str) -> None:
        """One Opus packet of the microphone."""
        if not self.mic_on:
            self._on_stray_packet(pkt, via)
            return
        if self._mic_asked is not None and self._mic_asked[1]:
            self._forget_asked()  # the sound is the answer, should the OK have been lost
        if self.rx.mic_device is not None:
            self.rx.mic_device.feed(self.label, pkt.payload, pkt.timestamp, pkt.seq)
        if not self.mic_record:
            self._mic_heard += 1
            if self._mic_heard <= 3 or self._mic_heard % 500 == 0:
                self.log("mic_rtp", _quiet=self._mic_heard > 3, n=self._mic_heard, via=via, **pkt.summary())
            return
        if self._mic is None:
            assert self.cfg.log_dir is not None
            path = capture_path(self.cfg.log_dir, self.label)
            try:
                self._mic = MicrophoneCapture(path, pcm=self.mic_pcm, log=self.log)
            except OSError as e:
                self.log("mic_recording_failed", file=path, error=str(e))
                self._recording_failed("no file")
                return
        before = self._mic.packets
        try:
            self._mic.write(pkt.payload, pkt.timestamp, pkt.seq)
        except OSError as e:  # a full disk ends the recording, not the link
            self.log("mic_write_failed", file=self._mic.path, error=str(e))
            self._recording_failed("write failed")
            return
        n = self._mic.packets
        if n != before and (n <= 3 or n % 500 == 0):  # the first packets of a recording, then one in ten seconds
            self.log("mic_rtp", _quiet=n > 3, n=n, via=via, **pkt.summary())

    def _recording_failed(self, reason: str) -> None:
        """No more is recorded. A microphone that the computer's device carries stays on for it;
        one that was only recorded is switched off."""
        if self.rx.mic_device is None:
            self.mic_stop(reason=reason)
        else:
            self.set_mic_record(False)

    def _on_stray_packet(self, pkt: RtpPacket, via: str) -> None:
        """Sound with the switch off: still on its way when the switch went off, or sent by a
        controller that did not hear it go off. There is no file for it. When no request is
        under way that would end it, the controller is told to stop, a few times at most."""
        self._mic_stray += 1
        self._mic_strays += 1
        if self.rx.mic_device is not None:
            self.rx.mic_device.skip(self.label, pkt.payload, pkt.timestamp, pkt.seq)
        if self._mic_strays == 1 or self._mic_strays % 500 == 0:
            self.log("mic_stray", _quiet=self._mic_strays > 1, n=self._mic_strays, via=via, **pkt.summary())
        if (
            self._mic_asked is None
            and self._mic_hushed < MIC_TRIES
            and self._link_ready()
            and time.monotonic() - self._mic_told >= MIC_ANSWER_TIMEOUT
        ):
            self._mic_hushed += 1
            self.log("mic_switch", on=False, reason="stray", ticket=self._ask_microphone(False))

    def _close_mic(self) -> None:
        capture, self._mic = self._mic, None
        if capture is None:
            return
        capture.close()
        self._mic_done = capture
        self._mic_ending = [c for c in self._mic_ending if not c.ended] + [capture]

    async def wait_microphone_closed(self) -> None:
        """After :meth:`stop`: wait until no recording has a decoder running any more."""
        ending, self._mic_ending = self._mic_ending, []
        for capture in ending:
            await asyncio.to_thread(capture.join)

    def _maybe_start_mic(self) -> None:
        """``--mic on``: ask for the microphone once per boot, when the link is READY."""
        if self.cfg.mic != "on" or self._mic_autostarted or self.mic_on:
            return
        self._mic_autostarted = True
        self.mic_start(reason="autostart")

    def _push_to_talk(self, state: GamepadState) -> None:
        """The Assistant button as the switch: pressing it turns the microphone on, letting go
        turns it off. A microphone that was on already is not this button's to turn off, and one
        that was turned off during the hold stays off until the next press.

        Only the display link's own reports come here (``ControllerToDisplay`` #3), whose buttons
        are mapped. The input protocol's reports, a game session's among them, have a layout
        nobody has mapped: a microphone is not switched on a guess."""
        held = bool(state.buttons & Button.ASSISTANT)
        pressed, self._mic_button = held and not self._mic_button, held
        if not self.mic_ptt:
            return
        if pressed and not self.mic_on:
            self.mic_start(reason="ptt")
        elif not held and self._mic_held:
            self.mic_stop(reason="ptt")

    # ---- discovery RPCs ------------------------------------------------------------------
    def register(self, request: Any, extra: dict[str, Any]) -> tuple[Any, str]:
        """``v1:RegisterController``: the response message and the assumption note to log."""
        p = self.protos
        dev = request.device_id if request else ""
        self.session["registered"] = {
            "device_id": dev,
            "attempt_id": request.attempt_id if request else None,
            "cloud_device_id": request.cloud_device_id if request else None,
        }
        resp = p.RegisterControllerResponse()
        resp.registration_id = f"local-{(dev or 'controller')[:40]}"
        resp.poll_interval_ms = 1000
        resp.registration_token = REGISTRATION_TOKEN
        note = "RegisterControllerResponse: placeholder id/token, poll_interval_ms=1000 (firmware requires > 0)"
        if request and request.HasField("discovery"):
            # Autolinking boot (discovery.resume = true + SDP offer): the controller only continues
            # (GMsg poll, ICE) when the response carries the display oneof; without field 5 it logs
            # to Clearcut and powers down.  A registration without a discovery must not carry it:
            # the controller would poll GMsg and take an answer that is not for the peer connection
            # it then uses (protocol.md §11, the boot after provisioning).
            resp.display.display_id = DISPLAY_ID
            resp.display.display_token = self.display_token
            note += "; display{id,token} included because the request carries a (resumed) discovery"
            if request.discovery.sdp_offer_bundle:
                extra["discovery_offer"] = self._offer_from_request(request.discovery, "rpc:RegisterController")
        return resp, note

    def discover(self, request: Any, extra: dict[str, Any]) -> tuple[Any, str]:
        """``v1:DiscoverDisplay``: accept the offer in field 4, answer with placeholder display ids."""
        p = self.protos
        resp = p.DiscoverDisplayResponse()
        resp.display_id = DISPLAY_ID
        resp.display_token = self.display_token
        note = "DiscoverDisplayResponse: placeholder display id/token (display_token must be non-empty)"
        if request:
            self.session["discovery"] = {
                "device_id": request.device_id,
                "attempt_id": request.attempt_id,
                "typed_keys": list(request.typed_keys.key),
                "app_id": request.app_id,
                "has_offer": bool(request.sdp_offer_bundle),
            }
            if request.sdp_offer_bundle:
                extra["discovery_offer"] = self._offer_from_request(request, "rpc:DiscoverDisplay")
        return resp, note

    def _offer_from_request(self, disc: Any, via: str) -> dict[str, Any]:
        """``sdp_offer_bundle``: accept the controller's offer and queue our answer bundle for its next GMsg poll."""
        p = self.protos
        try:
            bundle = p.parse_sdp_bundle(disc.sdp_offer_bundle)
        except DecodeError as e:
            return {"error": f"sdp_offer_bundle undecodable: {e}", "hex": disc.sdp_offer_bundle[:64].hex()}
        info: dict[str, Any] = {"bundle": p.to_dict(bundle)}
        if bundle.sdp.type != p.SDP_OFFER:
            info["error"] = f"bundle sdp.type={bundle.sdp.type}, expected SDP_OFFER"
            return info
        if self.cfg.signaling_mode == "offer":
            info["note"] = "signaling-mode=offer: controller offer recorded but not answered"
            return info
        answer_bundle = self._accept_offer(bundle.sdp.sdp_offer, list(bundle.ice_candidate), via)
        info["answer"] = p.to_dict(self._last_answer)
        info["gmsg_id"] = self.queue_gmsg(answer_bundle, "sdp_answer")
        return info

    def _accept_offer(self, offer: Any, candidates: list[Any], via: str) -> bytes:
        """Shared by the RPC path and the bare-signaling path: become the answerer."""
        self.reset_boot_state()
        self.session.update(role="answerer", state="offer-received", remote=self.protos.to_dict(offer))
        self.session["offers_received"] += 1
        self.endpoint.controlling = False
        self.endpoint.set_remote(offer.ice_ufrag, offer.ice_password, offer.fingerprint or None)
        added = [c for c in (self._add_candidate(s) for s in candidates) if c]
        answer = self.build_answer(offer)
        self._last_answer = answer
        self.log("offer_accepted", via=via, rtc_session_id=offer.rtc_session_id, candidates=added)
        return self.bundle(answer)

    def _add_candidate(self, s: Any) -> tuple[str, int] | None:
        if s.type != self.protos.ICE_CANDIDATE:
            return None
        addr = signaling.candidate_address(s)
        if addr is None:
            return None
        ip, port = addr
        if ":" in ip:  # the input socket is IPv4; the firmware also offers link-local + ULA v6 candidates
            self.log("rtc_remote_candidate_skipped", ip=ip, port=port, reason="IPv6 candidate, IPv4 socket")
            return None
        self.endpoint.add_remote_candidate(ip, port)
        return ip, port

    def on_discovery_resolved(self) -> None:
        """``v1:ResolveDiscovery`` arrived.

        The controller sends it on entering S/UNREGISTERING, i.e. after the connection registered
        its RPC read callback; our 200 then leads to ACCEPT_COMMANDS.  A display RPC sent before
        that point gets no answer, so this is the default trigger.
        """
        first = self._resolved_at is None
        self._resolved_at = time.monotonic()
        if not first:
            return
        since = round(self._resolved_at - self._dtls_at, 3) if self._dtls_at is not None else None
        self.log("discovery_resolved", since_dtls_s=since)
        if self.cfg.display_rpc == "on" and self.cfg.display_rpc_trigger == "resolve" and self._dtls_at is not None:
            self._start_display_rpc("resolve-discovery")

    # ---- GMsg push channel ---------------------------------------------------------------
    def queue_gmsg(self, payload: bytes, kind: str) -> int:
        self._gmsg_seq += 1
        self._gmsg_pending = {"id": self._gmsg_seq, "payload": payload, "kind": kind, "sent": 0}
        self.session["gmsg"]["pending_id"] = self._gmsg_seq
        self._gmsg_event.set()
        return self._gmsg_seq

    async def gmsg_poll(self, poll: Any, extra: dict[str, Any]) -> HttpResponse:
        """One ``/gmsg`` poll from this controller: deliver the pending envelope (re-sent until acked),
        hold the poll while nothing is pending, or answer empty."""
        p = self.protos
        g = self.session["gmsg"]
        g["polls"] += 1
        is_ack = False
        if poll is not None and poll.HasField("last_message") and poll.last_message.HasField("id"):
            is_ack = True
            acked = poll.last_message.id
            g["last_ack"] = acked
            if self._gmsg_pending and self._gmsg_pending["id"] == acked:
                g["acked"] += 1
                g["pending_id"] = None
                self.log("gmsg_acked", id=acked, kind=self._gmsg_pending["kind"])
                self._gmsg_pending = None
        if self._gmsg_pending is None and self.cfg.signaling_mode == "offer" and self.session["offer_served"] == 0:
            # legacy experiment: we offer, inside the GMsg envelope
            self.endpoint.controlling = True
            self.session.update(role="offerer", state="offer-served")
            self.session["offer_served"] += 1
            self.queue_gmsg(self.bundle(self.build_offer()), "sdp_offer")
        if self._gmsg_pending is None and self.cfg.gmsg_hold > 0:
            self._gmsg_event.clear()
            # An ACK poll may be released early: the firmware discards its result, and its serial
            # HTTP client then serves v1:ResolveDiscovery immediately instead of racing the power-off.
            while self._gmsg_pending is None:
                now = time.monotonic()
                end = now + self.cfg.gmsg_hold
                if is_ack and self.cfg.gmsg_ack_release > 0 and self._dtls_at is not None:
                    end = min(end, self._dtls_at + self.cfg.gmsg_ack_release)
                if now >= end:
                    break
                try:
                    await asyncio.wait_for(self._gmsg_event.wait(), end - now)
                except TimeoutError:
                    break
                self._gmsg_event.clear()
        if self._gmsg_pending is None:
            return HttpResponse(
                200,
                b"",
                "application/x-protobuf",
                handler="gmsg:idle",
                extra_log=extra,
                note="nothing pending: empty body = no message (firmware event 10, polls again)",
            )
        pend = self._gmsg_pending
        pend["sent"] += 1
        g["sent"] += 1
        body = p.gmsg_envelope(pend["payload"], pend["id"])
        extra.update(gmsg_id=pend["id"], kind=pend["kind"], resend=pend["sent"] - 1)
        return HttpResponse(
            200,
            body,
            "application/x-protobuf",
            handler=f"gmsg:{pend['kind']}",
            extra_log=extra,
            note="GmsgEnvelope{message{id, type=42, controller_command{payload}}}; re-sent until acked",
        )

    # ---- signaling -----------------------------------------------------------------------
    def build_offer(self) -> Any:
        """Our RtcSdpOffer (legacy offerer mode), built once per controller."""
        if self._offer_msg is None:
            self._offer_msg = signaling.build_offer(self.protos, self.endpoint, self.cfg.offer_overrides)
        return self._offer_msg

    def candidate_message(self, endpoint: RtcEndpoint | None = None) -> Any:
        ep = endpoint or self.endpoint
        return signaling.candidate_message(self.protos, self.cfg.advertise_ip or "", ep.local_port or 0)

    def bundle(self, sdp_msg: Any) -> bytes:
        return signaling.bundle(self.protos, sdp_msg, self.candidate_message())

    def build_answer(self, offer: Any, endpoint: RtcEndpoint | None = None) -> Any:
        return signaling.build_answer(self.protos, offer, endpoint or self.endpoint, boreal=self.cfg.boreal)

    def serve_offer(self, via: str) -> HttpResponse:
        """Legacy: answer a request with our offer bundle (``--offer-on`` paths)."""
        if self.cfg.signaling_mode == "answer":
            return HttpResponse(
                200,
                b"",
                "application/x-protobuf",
                handler=f"{via}:no-offer",
                note="signaling-mode=answer: not offering",
            )
        offer = self.build_offer()
        self.endpoint.controlling = True
        self.session.update(role="offerer", state="offer-served")
        self.session["offer_served"] += 1
        return HttpResponse(
            200,
            self.bundle(offer),
            "application/x-protobuf",
            handler=f"{via}:offer",
            note="served RtcSdpOffer bundle (assumption: envelope + delivery path)",
            extra_log={"offer": self.protos.to_dict(offer)},
        )

    def handle_signaling(self, kind: str, msg: Any) -> HttpResponse:
        """A bare WebRtcSdp or bundle found in a request body (any path)."""
        p = self.protos
        sdps = [msg.sdp, *msg.ice_candidate] if kind == "bundle" else [msg]
        reply: bytes | None = None
        handler = "signaling"
        extra: dict[str, Any] = {"signaling_kind": kind}
        for s in sdps:
            if s.type == p.SDP_OFFER:
                o = s.sdp_offer
                reply = self._accept_offer(o, [], "signaling")
                handler = "signaling:answer"
                extra["offer"] = p.to_dict(o)
                extra["answer"] = p.to_dict(self._last_answer)
            elif s.type == p.SDP_ANSWER:
                a = s.sdp_answer
                self.session.update(role="offerer", state="answer-received", remote=p.to_dict(a))
                self.session["answers_received"] += 1
                self.endpoint.controlling = True
                self.endpoint.set_remote(a.ice_ufrag, a.ice_password, a.fingerprint or None)
                reply = self.candidate_message().SerializeToString()
                handler = "signaling:answer-accepted"
                extra["answer"] = p.to_dict(a)
            elif s.type == p.ICE_CANDIDATE:
                addr = self._add_candidate(s)
                if addr:
                    extra.setdefault("candidates", []).append(list(addr))
                if reply is None:
                    reply = b""
                    handler = "signaling:candidate"
        return HttpResponse(200, reply or b"", "application/x-protobuf", handler=handler, extra_log=extra)

    # ---- RTC callbacks (display link) ----------------------------------------------------
    def _on_rtc_state(self, state: str, info: dict[str, Any]) -> None:
        self.session["state"] = state
        self.session["rtc"] = info
        if state == "dtls-closed":
            # The controller closed the link: it is powering off.  Its next boot starts from a new
            # offer, so what that offer would reset is reset now, and nothing goes on reporting to,
            # streaming to or checking an address the controller has left.
            self.reset_boot_state("the controller closed the link")
            self.endpoint.clear_remote()
            return
        if state == "dtls-connected" and self.cfg.rtcp_interval > 0 and self._rtcp_task is None:
            self._rtcp_task = asyncio.get_running_loop().create_task(self._rtcp_loop())
        if state == "dtls-connected" and self._dtls_at is None:
            self._dtls_at = time.monotonic()
            self._gmsg_event.set()  # wake an ACK-poll hold so it can recompute its deadline
            if self.cfg.display_rpc == "on":
                self._set_display_state("dtls-connected")
                if self.cfg.display_rpc_trigger == "dtls" or self._resolved_at is not None:
                    self._start_display_rpc(
                        "dtls-connected" if self.cfg.display_rpc_trigger == "dtls" else "resolve-discovery"
                    )
                elif self.cfg.display_rpc_fallback > 0:
                    self._display_fallback_task = asyncio.get_running_loop().create_task(self._display_rpc_fallback())

    async def _rtcp_loop(self) -> None:
        """Display-side keepalive: an RTCP RR every ``cfg.rtcp_interval`` seconds.

        The real display, as RTP receiver of the controller's mic/input streams, sends RTCP;
        some controller state machines only count the link as live once display traffic arrives.
        """
        interval = self.cfg.rtcp_interval
        try:
            while True:
                await asyncio.sleep(interval)
                if self.endpoint.srtp_out is None:
                    continue
                if self.endpoint.send_rtcp_rr():
                    self.session["rtcp_out"] = self.session.get("rtcp_out", 0) + 1
                    if self.session["rtcp_out"] <= 3:
                        self.log("rtcp_rr_sent", name="input", count=self.session["rtcp_out"])
        except asyncio.CancelledError:
            pass

    def _on_rtp(self, pkt: RtpPacket, via: str = "srtp") -> None:
        """RTP from the controller: the microphone's Opus, which comes as SRTP on the UDP wire, or
        input, which comes that way or (raw_dtls) as a plain RTP packet on WireHeader2 sub-channel
        0x66 and whose ack goes back the way it came."""
        self.seen()
        if pkt.payload_type == MIC_PAYLOAD_TYPE:  # Opus from the microphone, not the input protocol
            self._on_mic_packet(pkt, via)
            return
        self.session["input_payload_type"] = pkt.payload_type
        try:
            decoded = self.input.decode(pkt.payload)
        except (DecodeError, ValueError) as e:
            self.log(
                "rtp_payload_undecodable",
                pt=pkt.payload_type,
                seq=pkt.seq,
                len=len(pkt.payload),
                ext=[(i, v.hex()) for i, v in pkt.extensions],
                error=str(e),
                head=pkt.payload[:48].hex(),
            )
            return
        self.session["input_series"] += 1
        if self.session["input_series"] == 1:
            self._set_display_state("input", via=via)
        self.log(
            "rtp_input",
            via=via,
            **pkt.summary(),
            n_events=len(decoded.get("events", [])),
            sequence_number=decoded.get("sequence_number"),
            session_id=decoded.get("session_id"),
            compressed=decoded.get("compressed", False),
            ack=decoded.get("ack_sequence_number"),
        )
        if self.cfg.ack_mode == "rtp" and decoded.get("sequence_number") is not None:
            ack = self.protos.InputProtocolMessage()
            ack.ack_sequence_number = decoded["sequence_number"]
            send = self.endpoint.send_rtp_wire if via == "dtls" else self.endpoint.send_rtp
            if send(ack.SerializeToString(), payload_type=pkt.payload_type):
                self.session["last_ack"] = decoded["sequence_number"]

    def _on_rtcp(self, data: bytes) -> None:
        self.log("rtcp", pt=data[1] if len(data) > 1 else None, len=len(data), head=data[:16].hex())

    def _on_app_data(self, data: bytes) -> None:
        """raw_dtls: every DTLS application-data record is one WireHeader2 message.

        On the display link the controller frames its RpcMessages with id 0x74 (0x75 from the
        other side); 0x66 RTP is kept for the legacy fake-controller path.
        """
        self.seen()
        try:
            msg_id, payload = parse_wire_message(data)
        except ValueError as e:
            try:  # not framed: the older guess (a bare InputProtocolMessage), else a dump
                decoded = self.input.decode(data)
                self.log(
                    "dtls_app_input",
                    len=len(data),
                    n_events=len(decoded.get("events", [])),
                    sequence_number=decoded.get("sequence_number"),
                )
            except (DecodeError, ValueError):
                self.log(
                    "dtls_app_data",
                    len=len(data),
                    head=data[:48].hex(),
                    wire_error=str(e),
                    decoded=decode_unknown(data),
                )
            return
        wire_in = self.session["display"]["wire_in"]
        key = f"{msg_id:#06x}"
        wire_in[key] = wire_in.get(key, 0) + 1
        if msg_id == WIRE_ID_RTP:
            if len(payload) > 1 and 192 <= payload[1] <= 223:  # RFC 5761 RTCP range
                self._on_rtcp(payload)
                return
            try:
                pkt = parse_rtp(payload)
            except ValueError as e:
                self.log("wire_rtp_unparsable", len=len(payload), error=str(e), head=payload[:48].hex())
                return
            self.endpoint.stats["rtp_in"] += 1
            self._on_rtp(pkt, via="dtls")
        elif msg_id in WIRE_IDS_DC_RPC or msg_id == self.cfg.rpc_wire_id:
            self._on_rpc(payload)
        else:
            self.log(
                "wire_message",
                message_id=key,
                len=len(payload),
                head=payload[:48].hex(),
                decoded=decode_unknown(payload),
            )

    # ---- display side of the connection --------------------------------------------------
    def _set_display_state(self, state: str, **info: Any) -> None:
        """Advance the post-DTLS milestone (never backwards); READY arms StartGame."""
        d = self.session["display"]
        if DISPLAY_STATES.index(state) > DISPLAY_STATES.index(d["state"]):
            d["state"] = state
            self.log("display_state", display_state=state, **info)
            self.rx.live_changed()
            if state == "ready":
                self._log_hardware_summary()
            if DISPLAY_STATES.index(state) >= DISPLAY_STATES.index("ready"):
                if self.rx.gamepads is not None:
                    self.rx.gamepads.prepare(self.label)  # the pad is there before the first press needs it
                self.game.maybe_start()
                self._maybe_start_audio()
                if self.rx.mic_device is not None:
                    self.rx.mic_device.voice(self.label)  # its ffmpeg runs before the first packet needs it
                self._maybe_start_mic()

    def mark_input(self, via: str) -> None:
        """Input has arrived over ``via`` (the game session reports through here)."""
        self._set_display_state("input", via=via)

    def _start_display_rpc(self, reason: str) -> None:
        if self._display_task is not None:
            return
        if self._display_fallback_task is not None and self._display_fallback_task is not asyncio.current_task():
            self._display_fallback_task.cancel()
        self.session["display"]["trigger"] = reason
        since = round(time.monotonic() - self._dtls_at, 3) if self._dtls_at is not None else None
        self.log("display_rpc_trigger", reason=reason, since_dtls_s=since, delay_s=self.cfg.display_rpc_delay)
        self._display_task = asyncio.get_running_loop().create_task(self._display_rpc_loop())

    async def _display_rpc_fallback(self) -> None:
        try:
            await asyncio.sleep(self.cfg.display_rpc_fallback)
            if self._display_task is None:
                self.log(
                    "display_rpc_fallback",
                    after_s=self.cfg.display_rpc_fallback,
                    note="no v1:ResolveDiscovery since DTLS; starting the display RPCs anyway",
                )
                self._start_display_rpc("fallback")
        except asyncio.CancelledError:
            pass

    async def _display_rpc_loop(self) -> None:
        """Ask for the controller state until it answers OK.

        The controller sits in R/C/NOT_READY or R/C/WAITING_GET_CONTROLLER_STATE until the display
        asks for its state; READY needs that request plus the controller's own ACCEPT_COMMANDS
        (posted when discovery ends, after our ResolveDiscovery 200).  The request is re-sent
        until an OK answer with the controller state (an INTERNAL reply, e.g. in
        R/C/WAITING_ACCEPT_COMMAND, does not count) or until the controller's READY-only requests
        show up.
        """
        d = self.session["display"]
        cfg = self.cfg
        try:
            await asyncio.sleep(cfg.display_rpc_delay)
            deadline = time.monotonic() + cfg.display_rpc_window
            while time.monotonic() < deadline:
                if d["controller_state"] is not None or DISPLAY_STATES.index(d["state"]) >= DISPLAY_STATES.index(
                    "ready"
                ):
                    return
                msg = self.protos.DisplayToController()
                msg.get_controller_state.display_supports_send_log_report = cfg.send_log_report
                n = d["get_controller_state_sent"]
                if self.send_rpc("request", msg, purpose="get_controller_state", quiet=n >= 3) is not None:
                    d["get_controller_state_sent"] += 1
                    self._set_display_state("get-controller-state-sent")
                await asyncio.sleep(cfg.display_rpc_interval)
            self.log(
                "display_rpc_unanswered",
                sent=d["get_controller_state_sent"],
                rpc_in=d["rpc_in"],
                controller_state_errors=d["controller_state_errors"],
                window_s=cfg.display_rpc_window,
            )
        except asyncio.CancelledError:
            pass

    def send_rpc(
        self,
        kind: str,
        payload: Any = None,
        *,
        ticket: int | None = None,
        code: int = 0,
        message: str = "",
        purpose: str | None = None,
        quiet: bool = False,
    ) -> int | None:
        """One RpcMessage to the controller in a WireHeader2 record with id ``cfg.rpc_wire_id``.

        ``kind`` is request / notification / response; ``payload`` is a DisplayToController (or
        None).  Returns the ticket, or None when there is no DTLS session to send on.
        """
        p = self.protos
        if ticket is None:
            self._rpc_ticket += 1
            ticket = self._rpc_ticket
        if kind == "request":
            data = p.rpc_request(ticket, payload)
        elif kind == "notification":
            data = p.rpc_notification(ticket, payload)
        else:
            data = p.rpc_response(ticket, payload, code=code, message=message)
        record = pack_wire_message(self.cfg.rpc_wire_id, data)
        if not self.endpoint.send_app_data(record):
            self.log("rpc_out_failed", kind=kind, ticket=ticket, purpose=purpose, reason="no DTLS session")
            return None
        if kind == "request" and purpose:
            self._rpc_pending[ticket] = purpose
        self.session["display"]["rpc_out"] += 1
        self.log(
            "rpc_out",
            _quiet=quiet,
            kind=kind,
            ticket=ticket,
            purpose=purpose,
            code=code if kind == "response" else None,
            len=len(data),
            record=record.hex(),
            msg=p.to_dict(payload) if payload is not None else None,
        )
        return ticket

    def _on_rpc(self, data: bytes) -> None:
        """One RpcMessage from the controller; its payloads are ControllerToDisplay."""
        p = self.protos
        d = self.session["display"]
        d["rpc_in"] += 1
        try:
            rpc = p.parse_rpc(data, p.ControllerToDisplay)
        except DecodeError as e:
            self.log(
                "rpc_in_undecodable", len=len(data), error=str(e), head=data[:48].hex(), decoded=decode_unknown(data)
            )
            return
        kind, ticket, msg = rpc["kind"], rpc["ticket"], rpc["payload"]
        variant = msg.WhichOneof("variant") if msg is not None else None
        rec: dict[str, Any] = {"kind": kind, "ticket": ticket, "variant": variant}
        if kind == "response":
            rec.update(
                purpose=self._rpc_pending.pop(ticket, None),
                code=rpc.get("code"),
                error_message=rpc.get("error_message") or None,
            )
        if rpc["payload_error"]:
            rec.update(payload_error=rpc["payload_error"], payload_head=rpc["payload_raw"][:64].hex())
        counter = {"request": "requests", "notification": "notifications"}.get(kind)
        if counter:
            d[counter][variant or "?"] = d[counter].get(variant or "?", 0) + 1
        # READY-only traffic repeats every 1.5-3 s: the console shows the first few of each
        repeats = d[counter].get(variant or "?", 0) if counter else 0
        self.log("rpc_in", _quiet=repeats > 3, **rec, msg=p.to_dict(msg) if msg is not None else None)
        if kind == "request":
            self._answer_request(ticket, variant, msg, quiet=repeats > 3)
        elif kind == "notification" and variant == "message3":
            self._on_input_rpc(msg.message3)
        elif kind == "notification" and variant == "notification10":
            self._on_battery(msg.notification10)
        elif kind == "notification" and variant == "audio_configuration_changed":
            self._on_audio_config(msg.audio_configuration_changed)
        elif kind == "response" and rec.get("purpose") == "start_game":
            self.game.on_start_game_response(ticket, rpc)
        elif kind == "response" and rec.get("purpose") in ("microphone_on", "microphone_off"):
            self._on_microphone_answer(
                ticket, rec["purpose"] == "microphone_on", int(rpc.get("code") or 0), rpc.get("error_message") or None
            )
        elif kind == "response" and rec.get("purpose") == "get_controller_state":
            if rpc.get("code") or variant != "controller_state":
                # e.g. 13 INTERNAL from the responder destructor (request in R/C/WAITING_ACCEPT_COMMAND)
                d["controller_state_errors"] += 1
                self.log(
                    "display_controller_state_error",
                    _quiet=d["controller_state_errors"] > 3,
                    code=rpc.get("code"),
                    error_message=rpc.get("error_message") or None,
                    variant=variant,
                )
                return
            state = p.to_dict(msg.controller_state)
            d["controller_state"] = state
            headset = _as_bool(state.get("f4"))  # a plug in the jack, as of this snapshot
            if headset is not None:
                d["headset"] = headset
                self.rx.live_changed()
            if not self.serial and isinstance(state.get("f17"), str) and state["f17"].strip():
                self.serial = state["f17"].strip()  # f17 is the serial too, for when the header never came
            self.log(
                "display_controller_state",
                code=rpc.get("code"),
                device_id=state.get("device_id"),
                n_fields=len(state),
                headset=headset,
                controller_state=state,
            )
            self._set_display_state("controller-state-received")

    def _on_input_rpc(self, m: Any) -> None:
        """ControllerToDisplay #3 as an RpcNotification: the controller's input on the display link
        (READY only, sent on change).  Layout and button names: :mod:`stadiacast.gamepad`."""
        self.session["display"]["input_rpc"] += 1
        state = GamepadState.from_message3(m)
        ev = state.as_event(self._last_input)
        self._last_input = state
        self._set_display_state("input", via="rpc")
        self.emit_input_event(ev)
        self._push_to_talk(state)

    # ---- what the controller reports about itself ----------------------------------
    def _log_hardware_summary(self) -> None:
        """One line at READY with the controller's self-description (GetControllerStateResponse).

        The battery gauge arrives on its own a few minutes later (``controller_battery``), so it
        is usually still unknown here.
        """
        d = self.session["display"]
        state = d.get("controller_state") or {}
        self.log(
            "controller_hardware",
            serial=self.serial,
            manufacturer=state.get("f12"),
            product=state.get("f14"),
            hardware=state.get("f15"),
            firmware=state.get("f16"),
            headset=d.get("headset"),
            battery_percent=d.get("battery_percent"),
        )

    def _on_battery(self, m: Any) -> None:
        """``ControllerToDisplay`` #10: the charge level as a fraction, 0.0 to 1.0.

        The first sample comes a few minutes after READY, then one step down per percent (0.76,
        0.75, 0.74 ... every few minutes on a live controller) and never up while it runs on the
        battery.
        """
        d = self.session["display"]
        level = max(0.0, min(1.0, float(m.value)))
        pct = round(level * 100)
        changed = d["battery_percent"] != pct
        d["battery"] = round(level, 4)
        d["battery_percent"] = pct
        d["battery_samples"] += 1
        self.log("controller_battery", _quiet=not changed, percent=pct, level=round(level, 4), n=d["battery_samples"])
        if changed:
            self.rx.live_changed()

    def _on_audio_config(self, m: Any) -> None:
        """``ControllerToDisplay`` #4: the audio configuration changed, which means the headset moved.

        ``mode`` 1 is a plug in the 3.5 mm jack, 0 none; the notification comes within a second of
        the plug going in or out.  ``device_info`` is the controller advertising an audio receiver
        of its own (its address, a UDP port and a key), see protocol.md §5.
        """
        d = self.session["display"]
        info = self.protos.to_dict(m)
        mode = info.get("mode")
        d["audio_mode"] = mode
        d["audio_events"] += 1
        if mode in (0, 1):
            d["headset"] = mode == 1
            self.rx.live_changed()
        self.log("controller_audio_config", n=d["audio_events"], mode=mode, headset=d["headset"], audio=info)

    def _answer_request(self, ticket: int | None, variant: str | None, msg: Any, quiet: bool = False) -> None:
        """Answers for the controller's requests.  The periodic ones only start on entry to
        R/C/READY, so the first of them marks READY."""
        if ticket is None:
            return  # the controller's own incoming_rpc_manager drops these too
        p = self.protos
        d = self.session["display"]
        if variant in ("get_current_time", "keepalive"):
            self._set_display_state("ready")
        if variant == "get_current_time":
            if self.cfg.time_sync == "unsupported":
                self.send_rpc(
                    "response",
                    None,
                    ticket=ticket,
                    code=p.enum(f"{PKG_RPC}.StatusCode", "NOT_FOUND"),
                    message="GetCurrentTimeRequest not supported",
                    quiet=quiet,
                )
                return
            reply = p.DisplayToController()
            # controller_time_us echoes the request: remote_clock.cc computes rtt = now - f1 and CHECK-fails on rtt < 0
            reply.current_time.controller_time_us = msg.get_current_time.controller_time_us
            reply.current_time.display_time_us = int(self.now() * 1_000_000)
            d["time_replies"] += 1
            self.send_rpc("response", reply, ticket=ticket, quiet=quiet)
        elif variant in ("keepalive", "disconnect"):
            if variant == "keepalive":
                d["keepalive_replies"] += 1
            self.send_rpc("response", None, ticket=ticket, quiet=quiet)
        elif variant == "proxy_request":
            self.game.on_proxy_request(ticket, msg.proxy_request)
        else:  # what the controller itself answers to requests it does not implement
            self.send_rpc(
                "response",
                None,
                ticket=ticket,
                code=p.enum(f"{PKG_RPC}.StatusCode", "UNIMPLEMENTED"),
                message="Unimplemented RPC in local receiver",
                quiet=quiet,
            )

    # ---- input events --------------------------------------------------------------------
    def emit_input_event(self, ev: dict[str, Any]) -> None:
        """Log one decoded input event, stamped with this controller, and hand it to the receiver's queue."""
        ev["controller"] = self.label
        self.log("input", **ev)
        if "axes" in ev:  # a gamepad report; the game session's other events carry none
            state = GamepadState.from_event(ev)
            self.latest_input = (state, time.monotonic())
            self.rx.live_changed()
        self.rx.publish_event(ev)
