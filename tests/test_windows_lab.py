"""The Windows-only paths against the real drivers: the ViGEm pad read back through XInput, plugged in
and unplugged with its controller, the game's rumble through ViGEm's notification, the WASAPI loopback
capture, and the controller's microphone played into a virtual cable and recorded from its other side.
Runs on a Windows machine with the ViGEmBus driver, when STADIACAST_WINDOWS_LAB is set (the Windows lab
workflow); skipped elsewhere."""

from __future__ import annotations

import array
import asyncio
import ctypes
import os
import subprocess
import sys
from typing import Any

import pytest

from stadiacast.audio import ffmpeg_path
from stadiacast.errors import SetupError
from tests.helpers import connect_firmware_flow, fake, make_receiver, single, wait_for
from tests.test_micdevice import loud, say
from tests.test_microphone import real_opus

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("STADIACAST_WINDOWS_LAB"),
    reason="needs Windows, the ViGEmBus driver and STADIACAST_WINDOWS_LAB=1",
)


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _XInputGamepad)]


class _XInputVibration(ctypes.Structure):
    _fields_ = [("wLeftMotorSpeed", ctypes.c_ushort), ("wRightMotorSpeed", ctypes.c_ushort)]


def _xinput():
    lib = ctypes.WinDLL("xinput1_4.dll")  # type: ignore[attr-defined]
    lib.XInputGetState.argtypes = [ctypes.c_ulong, ctypes.POINTER(_XInputState)]
    lib.XInputGetState.restype = ctypes.c_ulong
    lib.XInputSetState.argtypes = [ctypes.c_ulong, ctypes.POINTER(_XInputVibration)]
    lib.XInputSetState.restype = ctypes.c_ulong
    return lib


def _read(lib, slot: int) -> _XInputGamepad:
    state = _XInputState()
    assert lib.XInputGetState(slot, ctypes.byref(state)) == 0, f"no XInput pad on slot {slot}"
    return state.Gamepad


def _players(lib) -> set[int]:
    """The XInput players that have a pad right now."""
    return {slot for slot in range(4) if lib.XInputGetState(slot, ctypes.byref(_XInputState())) == 0}


def _needs_the_xbox_360_driver() -> None:
    if not os.path.exists(r"C:\Windows\INF\xusb22.inf"):
        # A ViGEm pad is a virtual Xbox 360 controller and binds to Windows' inbox driver for it.
        # Windows Server (the hosted runners) ships without that driver: the pad enumerates but never
        # plugs in (TARGET_NOT_PLUGGED_IN), on any software. Desktop Windows 10 and 11 have it.
        pytest.skip("no Xbox 360 controller driver (xusb22.inf): this Windows cannot host a ViGEm pad")


def test_a_real_vigem_pad_follows_the_controller_and_reports_the_rumble(tmp_path) -> None:
    _needs_the_xbox_360_driver()

    async def run() -> None:
        rx = make_receiver(tmp_path, gamepad="vigem", start_game="auto", start_game_delay=0.05)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert fc.send_input_rpc(buttons=1 << 12, lx=20000, ly=-20000, lt=255)  # A, left stick, left trigger
            try:
                await wait_for(lambda: rx.status()["gamepads"].get("c0", {}).get("frames", 0) >= 1, timeout=40)
            except AssertionError:
                records = [r for r in rx.log.records if str(r["event"]).startswith("gamepad")]
                pytest.fail(f"no frame reached a pad; gamepad records: {records}; status: {rx.status()['gamepads']}")
            pad = rx.status()["gamepads"]["c0"]
            assert pad["slot"] is not None, "ViGEm created the pad but XInput never showed it"
            lib = _xinput()
            await asyncio.sleep(0.3)
            seen = _read(lib, pad["slot"])
            assert seen.wButtons & 0x1000 and seen.bLeftTrigger == 255 and seen.sThumbLX > 0 and seen.sThumbLY < 0
            assert fc.send_input_rpc(buttons=0)
            await wait_for(lambda: rx.status()["gamepads"]["c0"]["frames"] >= 2)
            await asyncio.sleep(0.3)
            assert _read(lib, pad["slot"]).wButtons == 0

            vibration = _XInputVibration(65535, 30000)  # what a game does
            assert lib.XInputSetState(pad["slot"], ctypes.byref(vibration)) == 0

            # ViGEmBus also reports (0, 0) when the pad plugs in and right before each new level, and
            # the receiver forwards every report, so the game's level is not the first rumble record.
            def rumbles() -> list[dict[str, Any]]:
                return [r for r in rx.log.records if r["event"] == "gamepad_rumble" and r["left"] == 65535]

            await wait_for(rumbles, timeout=10)
            assert rumbles()[0]["right"] > 0
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_real_vigem_pads_are_plugged_in_and_unplugged_with_their_controllers(tmp_path) -> None:
    """Two controllers that connect at the same moment each get an Xbox pad with a player of its
    own, before either has sent any input. The one that is switched off takes its pad out of
    XInput, and takes the lowest free player when it comes back."""
    _needs_the_xbox_360_driver()

    async def connect(fc: Any) -> None:
        await fc.boot()
        await fc.start_udp()
        await fc.discover()
        await fc.poll_gmsg()
        await fc.poll_gmsg()
        fc.connect()
        await fc.wait_connected(10)
        await asyncio.wait_for(fc.display_ready.wait(), 5)
        await fc.keepalive()

    async def run() -> None:
        lib = _xinput()
        before = _players(lib)
        rx = make_receiver(tmp_path, gamepad="vigem", display_rpc_delay=0.05, display_rpc_interval=0.1)
        await rx.start()
        fa, fb = fake(rx, "fake-a"), fake(rx, "fake-b")

        def slots() -> dict[str, int | None]:
            return {label: pad.get("slot") for label, pad in rx.status()["gamepads"].items()}

        try:
            await asyncio.gather(connect(fa), connect(fb))
            a, b = rx.find("fake-a"), rx.find("fake-b")
            assert a is not None and b is not None
            await wait_for(lambda: len(slots()) == 2 and None not in slots().values(), timeout=40)
            first = slots()
            assert len(set(first.values())) == 2, f"two pads, one player: {first}"
            assert _players(lib) - before == set(first.values())

            fa.power_off()
            await wait_for(lambda: slots().keys() == {b.label}, timeout=10)
            await wait_for(lambda: _players(lib) - before == {first[b.label]}, timeout=10)

            fa = fake(rx, "fake-a")
            await connect(fa)
            await wait_for(lambda: slots().get(a.label) is not None, timeout=40)
            assert slots() == first  # the lowest free player is the one it had
            assert _players(lib) - before == set(first.values())
        finally:
            fa.close()
            fb.close()
            await rx.stop()
        await wait_for(lambda: _players(lib) == before, timeout=10)

    asyncio.run(run())


def test_the_windows_loopback_reaches_the_controller(tmp_path) -> None:
    from stadiacast import loopback

    if ffmpeg_path() is None:
        pytest.skip("needs ffmpeg (with libopus) on PATH")
    if not loopback.available():
        pytest.skip("pyaudiowpatch is not installed")
    try:
        device = loopback.describe(None)
    except Exception as e:  # no render device on this machine
        try:
            names = [d.name for d in loopback.devices()]
        except Exception as listing:  # the listing failed too
            names = [repr(listing)]
        pytest.skip(f"no WASAPI loopback device: {e!r}; loopback devices seen: {names}")

    async def run() -> None:
        rx = make_receiver(tmp_path, audio_source="system", audio_mode=2)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: rx.log.counts.get("audio_capture", 0) >= 1, timeout=15)
            started = next(r for r in rx.log.records if r["event"] == "audio_capture")
            assert not str(started["keepalive"]).startswith("failed"), started
            # Nothing plays on this machine: the silence the capture renders on the speaker is what
            # makes WASAPI hand over loopback data at all, so the frames below are its doing.
            await wait_for(lambda: len(fc.audio_frames) >= 25, timeout=15)  # half a second of Opus
            state = single(rx).audio_state()
            assert state["playing"] and state["source"] == f"capture {device.name}"
            assert state["stats"]["send_failed"] == 0
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


#: A program that records a WASAPI recording device, in the format it has, into a file: a line with
#: the rate and the channels once the device is open, then five seconds of raw samples. Started as a
#: process of its own, since that is what a program which listens to the microphone is.
RECORDER = """
import sys, time
import pyaudiowpatch as pa
p = pa.PyAudio()
wasapi = p.get_host_api_info_by_type(pa.paWASAPI)["index"]
devices = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
info = next(d for d in devices if d["hostApi"] == wasapi and d["name"] == sys.argv[1] and d["maxInputChannels"] > 0
            and not d.get("isLoopbackDevice"))
rate, channels = int(info["defaultSampleRate"]), int(info["maxInputChannels"])
stream = p.open(format=pa.paInt16, channels=channels, rate=rate, input=True, input_device_index=info["index"],
                frames_per_buffer=rate // 50)
with open(sys.argv[2], "wb") as out:
    out.write(f"{rate} {channels}\\n".encode())
    out.flush()
    end = time.monotonic() + 5
    while time.monotonic() < end:
        out.write(stream.read(rate // 50, exception_on_overflow=False))
stream.close()
p.terminate()
"""


def test_the_controllers_microphone_is_recorded_from_a_virtual_cable(tmp_path) -> None:
    """``--mic on --mic-device`` with a virtual cable (VB-CABLE): what the lab controller's microphone
    sends is played into the cable and recorded from its recording side by another program."""
    from stadiacast import loopback
    from stadiacast.micdevice import WasapiOutput

    if ffmpeg_path() is None:
        pytest.skip("needs ffmpeg (with libopus) on PATH")
    if not loopback.available():
        pytest.skip("pyaudiowpatch is not installed")
    cable = WasapiOutput()
    try:
        cable.start(lambda frames: bytes(2 * frames))
    except SetupError as e:
        pytest.skip(f"no virtual cable on this machine: {e}")
    finally:
        cable.stop()
    if not cable.microphone:
        pytest.skip(f"the recording side of {cable.name!r} was not found")
    tone = real_opus(2.0)
    heard = tmp_path / "heard.pcm"

    async def run() -> dict[str, Any]:
        rx = make_receiver(tmp_path, mic="on", mic_device="")
        await rx.start()
        recorder = fc = None
        try:
            assert rx.mic_device is not None
            state = rx.mic_device.state()
            assert (state["name"], state["microphone"], state["error"]) == (cable.name, cable.microphone, None), state
            fc = await connect_firmware_flow(rx)
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            recorder = subprocess.Popen(
                [sys.executable, "-c", RECORDER, cable.microphone, str(heard)], stderr=subprocess.PIPE, text=True
            )
            await wait_for(
                lambda: recorder.poll() is not None or (heard.exists() and heard.stat().st_size > 0), timeout=30
            )
            if recorder.poll() is not None:  # Windows lets no program record: its microphone privacy setting
                said = (recorder.stderr.read() if recorder.stderr else "").strip().splitlines()
                pytest.skip(f"no program can record {cable.microphone!r} here: {said[-1] if said else 'no message'}")
            await asyncio.sleep(0.5)
            await say(fc, tone)
            await asyncio.to_thread(recorder.wait, 30)
            assert recorder.returncode == 0
            state = rx.mic_device.state()
            # The devices as they are now, which an open output keeps from being seen: the cable
            # is closed while they are listed, and plays again afterwards.
            listed = await asyncio.to_thread(rx.mic_device.fresh, lambda: [d.name for d in loopback.devices()])
            assert cable.loop in listed, listed
            asked = rx.mic_device._asked
            await wait_for(lambda: rx.mic_device._asked > asked, timeout=5)
            assert rx.mic_device.state()["error"] is None
            return state
        finally:
            if recorder is not None and recorder.poll() is None:
                recorder.kill()
            if fc is not None:
                fc.close()
            await rx.stop()

    state = asyncio.run(run())
    header, _, recorded = heard.read_bytes().partition(b"\n")
    rate, channels = (int(n) for n in header.split())
    first = array.array("h", recorded[: len(recorded) // (2 * channels) * 2 * channels])[0::channels].tobytes()
    size = 2 * (rate // 50)
    pieces = [first[at : at + size] for at in range(0, len(first) - size + 1, size)]
    sounding = [n for n, piece in enumerate(pieces) if loud([piece], 440, rate=rate)]
    assert sounding, f"nothing of the tone in {len(pieces)} pieces at {rate} Hz, {channels} channels: {state}"
    holes = sounding[-1] - sounding[0] + 1 - len(sounding)
    print(f"the microphone through {cable.name} at {rate} Hz: {len(sounding)} of {len(tone)} pieces, {holes} holes")
    assert len(sounding) >= len(tone) - 10 and holes <= 4, (len(pieces), len(sounding), holes, state)
    assert state["voices"]["c0"]["packets"] == len(tone) and (state["rate"], state["channels"]) == (rate, channels)


def test_the_cable_is_not_taken_while_the_headsets_sound_is_captured_from_it(tmp_path) -> None:
    """A cable that Windows made the default output, as its installation does on a computer with
    no other, is what ``--audio-source system`` captures: the microphone played into it would
    be in the headset. The receiver does not start that way, and says what to change."""
    from stadiacast import loopback
    from stadiacast.micdevice import WasapiOutput

    if ffmpeg_path() is None:
        pytest.skip("needs ffmpeg (with libopus) on PATH")
    if not loopback.available():
        pytest.skip("pyaudiowpatch is not installed")
    cable = WasapiOutput()
    try:
        cable.start(lambda frames: bytes(2 * frames))
        captured = loopback.describe(None).name
    except SetupError as e:
        pytest.skip(f"no virtual cable, or no output, on this machine: {e}")
    finally:
        cable.stop()
    if captured != cable.loop:
        pytest.skip(f"this computer plays on {captured!r}, not on the cable")

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="on", mic_device="", audio_source="system")
        try:
            with pytest.raises(SetupError, match="is the device the headset's sound is captured from"):
                await rx.start()
        finally:
            await rx.stop()
        assert rx.mic_device is None

    asyncio.run(run())
