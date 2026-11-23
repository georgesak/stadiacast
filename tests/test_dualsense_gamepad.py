"""The Linux DualSense uhid pad without a kernel: deterministic MAC derivation, feature reports
(calibration 0x05, pairing MAC 0x09, firmware 0x20), the 64-byte USB input report as SDL 3 and
``hid-playstation`` read it, output report 0x02 rumble, and end-to-end multi-controller identity."""

from __future__ import annotations

import asyncio
import errno
import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from stadiacast import dualsense_gamepad as ds
from stadiacast import virtual_gamepad
from stadiacast.dualsense_gamepad import (
    CALIBRATION_REPORT,
    FEATURE_CALIBRATION,
    FEATURE_FIRMWARE_INFO,
    FEATURE_PAIRING_INFO,
    FIRMWARE_REPORT,
    INPUT_REPORT_ID,
    INPUT_REPORT_SIZE,
    OUTPUT_REPORT_ID,
    PAD_PRODUCT,
    PAD_VENDOR,
    REPORT_DESCRIPTOR,
    DualSenseRequests,
    battery_byte,
    input_report,
    mac_bytes,
    mac_string,
    output_rumble,
    pairing_report,
    stick_byte,
)
from stadiacast.errors import SetupError
from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import (
    UHID_CLOSE,
    UHID_FEATURE_REPORT,
    UHID_GET_REPORT,
    UHID_GET_REPORT_REPLY,
    UHID_OPEN,
    UHID_OUTPUT,
    UHID_OUTPUT_REPORT,
    UHID_SET_REPORT,
    UHID_SET_REPORT_REPLY,
    UHID_START,
    UHID_STOP,
)
from stadiacast.virtual_gamepad import VirtualGamepads
from tests.fake_controller import FakeController
from tests.helpers import connect_firmware_flow, fake, make_receiver, single, wait_for
from tests.test_uhid_gamepad import _walk


def _get_report(request: int, rnum: int, rtype: int = UHID_FEATURE_REPORT) -> bytes:
    return struct.pack("=IIBB", UHID_GET_REPORT, request, rnum, rtype)


def _output_usb(
    *,
    flag0: int = 0,
    flag1: int = 0,
    flag2: int = 0,
    left: int = 0,
    right: int = 0,
    size: int = 48,
) -> bytes:
    buf = bytearray(size)
    buf[0] = OUTPUT_REPORT_ID
    buf[1] = flag0
    buf[2] = flag1
    buf[3] = right
    buf[4] = left
    buf[39] = flag2
    return bytes(buf)


def _uhid_output(data: bytes, rtype: int = UHID_OUTPUT_REPORT) -> bytes:
    return struct.pack("=I", UHID_OUTPUT) + data.ljust(4096, b"\0") + struct.pack("=HB", len(data), rtype)


def _uhid_set_report(request: int, data: bytes, rtype: int = UHID_OUTPUT_REPORT) -> bytes:
    return struct.pack("=IIBBH", UHID_SET_REPORT, request, data[0] if data else 0, rtype, len(data)) + data


def test_descriptor_declares_usb_dualsense_input_output_and_feature_reports():
    items, depth = _walk(REPORT_DESCRIPTOR)
    assert depth == 0
    inputs = [item for item in items if item["kind"] == "input"]
    assert {item["report_id"] for item in inputs} == {INPUT_REPORT_ID}
    assert sum(item["size"] * item["count"] for item in inputs) == (INPUT_REPORT_SIZE - 1) * 8
    (out,) = [item for item in items if item["kind"] == "output"]
    assert (out["report_id"], out["size"], out["count"]) == (OUTPUT_REPORT_ID, 8, 47)
    assert (PAD_VENDOR, PAD_PRODUCT) == (0x054C, 0x0CE6)


def test_mac_derivation_is_deterministic_unicast_locally_administered_and_reversed_in_report_0x09():
    mac_a = mac_bytes("99181FFAZ00AAA")
    mac_b = mac_bytes("99181FFAZ00BBB")
    assert len(mac_a) == 6 and mac_a == mac_bytes("99181FFAZ00AAA") and mac_a != mac_b
    assert (mac_a[0] & 0x03) == 0x02  # locally administered (bit 1), unicast (bit 0 == 0)
    assert mac_bytes("02:aa:bb:cc:dd:ee") == bytes([0x02, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE])

    rep = pairing_report(mac_a)
    assert len(rep) == 20 and rep[0] == FEATURE_PAIRING_INFO
    # Both hid-playstation (%pMR) and SDL_hidapi_ps5.c read bytes 6..1 in reverse as the MAC.
    sdl_mac = ":".join(f"{rep[i]:02x}" for i in range(6, 0, -1))
    assert sdl_mac == mac_string(mac_a)


def test_feature_reports_answer_calibration_pairing_mac_and_v2_firmware():
    mac = mac_bytes("99181FFAZ00123")
    req = DualSenseRequests(mac)

    cal = req.handle(_get_report(11, FEATURE_CALIBRATION))
    assert cal is not None
    assert struct.unpack_from("=IIHH", cal) == (UHID_GET_REPORT_REPLY, 11, 0, 41)
    assert cal[12:] == CALIBRATION_REPORT and len(CALIBRATION_REPORT) == 41

    pair = req.handle(_get_report(12, FEATURE_PAIRING_INFO))
    assert pair is not None
    assert struct.unpack_from("=IIHH", pair) == (UHID_GET_REPORT_REPLY, 12, 0, 20)
    assert pair[12:] == pairing_report(mac)

    fw = req.handle(_get_report(13, FEATURE_FIRMWARE_INFO))
    assert fw is not None
    assert struct.unpack_from("=IIHH", fw) == (UHID_GET_REPORT_REPLY, 13, 0, 64)
    assert fw[12:] == FIRMWARE_REPORT and struct.unpack_from("<H", FIRMWARE_REPORT, 44)[0] >= 0x0224

    unknown = req.handle(_get_report(14, 0x99))
    assert unknown is not None and struct.unpack_from("=IIHH", unknown) == (UHID_GET_REPORT_REPLY, 14, errno.EINVAL, 0)


@pytest.mark.parametrize(
    ("button", "byte_idx", "mask"),
    [
        (Button.X, 8, 0x10),
        (Button.A, 8, 0x20),
        (Button.B, 8, 0x40),
        (Button.Y, 8, 0x80),
        (Button.L1, 9, 0x01),
        (Button.R1, 9, 0x02),
        (Button.OPTIONS, 9, 0x10),
        (Button.MENU, 9, 0x20),
        (Button.L3, 9, 0x40),
        (Button.R3, 9, 0x80),
        (Button.STADIA, 10, 0x01),
        (Button.ASSISTANT, 10, 0x02),
        (Button.CAPTURE, 10, 0x04),
    ],
)
def test_every_stadia_button_maps_to_its_dualsense_bit(button: Button, byte_idx: int, mask: int):
    rep = input_report(GamepadState(buttons=int(button)))
    assert len(rep) == INPUT_REPORT_SIZE and rep[0] == INPUT_REPORT_ID
    assert (rep[byte_idx] & mask) == mask


def test_neutral_sticks_triggers_inactive_touchpad_and_battery_encoding():
    neutral = input_report(GamepadState(), seq=7, timestamp=12345678, battery_percent=None)
    assert len(neutral) == 64
    assert list(neutral[1:7]) == [128, 128, 128, 128, 0, 0]
    assert neutral[7] == 7 and neutral[8] == 8  # released hat
    assert struct.unpack_from("<I", neutral, 28) == (12345678,)
    assert neutral[33] == 0x80 and neutral[37] == 0x80  # both touch points inactive
    assert neutral[53] == 0x2A  # full (status 2, level 10)

    assert [stick_byte(v) for v in (-32768, 0, 32767)] == [0, 128, 255]
    assert [stick_byte(v, invert=True) for v in (-32768, 0, 32767)] == [255, 128, 0]
    assert battery_byte(76) == 7 and battery_byte(100) == 10 and battery_byte(0) == 0

    triggers = input_report(GamepadState(l2=200, r2=29))
    assert (triggers[5], triggers[6]) == (200, 29)
    assert (triggers[9] & 0x0C) == 0x04  # L2 digital bit set, R2 below threshold


def test_output_report_rumble_handles_v1_v2_sdl_stop_and_led_only_packets():
    mac = mac_bytes("99181FFAZ00123")
    req = DualSenseRequests(mac)

    # SDL 3 enhanced rumble (v2: flag2 & 0x04)
    req.handle(_uhid_output(_output_usb(flag0=0x02, flag2=0x04, left=200, right=100)))
    assert req.levels == (200 * 257, 100 * 257)

    # LED-only update (flag1 != 0, no vibration bits): succeeds on SET_REPORT and preserves rumble
    reply = req.handle(_uhid_set_report(55, _output_usb(flag1=0x14, size=63)))
    assert reply is not None and struct.unpack_from("=IIH", reply) == (UHID_SET_REPORT_REPLY, 55, 0)
    assert req.levels == (200 * 257, 100 * 257)

    # SDL 3 RumbleJoystick(0, 0): all enable flags cleared -> stops rumble
    req.handle(_uhid_output(_output_usb()))
    assert req.levels == (0, 0)

    # Classic v1 rumble (flag0 & 0x01), then stopped when hidraw closes
    req.handle(_uhid_output(_output_usb(flag0=0x03, left=64, right=32)))
    assert req.levels == (64 * 257, 32 * 257)
    req.handle(struct.pack("=I", UHID_OPEN).ljust(ds.EVENT_SIZE, b"\0"))
    assert req.opens == 1
    req.handle(struct.pack("=I", UHID_CLOSE).ljust(ds.EVENT_SIZE, b"\0"))
    assert req.levels == (0, 0)

    assert output_rumble(b"\x02short") == (False, None)
    assert output_rumble(b"\x05" + b"\0" * 47) == (False, None)
    req.handle(struct.pack("=I", UHID_START).ljust(ds.EVENT_SIZE, b"\0"))
    req.handle(struct.pack("=I", UHID_STOP).ljust(ds.EVENT_SIZE, b"\0"))
    assert req.starts == 1


def test_non_linux_refuses_cleanly():
    if not sys.platform.startswith("linux"):
        with pytest.raises(SetupError, match="--gamepad dualsense is Linux only"):
            ds.check_available()
        with pytest.raises(SetupError, match="--gamepad dualsense is Linux only"):
            ds.DualSensePad("c0")


@pytest.fixture
def kernel(monkeypatch) -> Iterator[socket.socket]:
    """A ``SOCK_SEQPACKET`` pair standing in for ``/dev/uhid`` on Linux."""
    ours_end, pads_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    os.set_blocking(pads_end.fileno(), False)
    fd = os.dup(pads_end.fileno())
    pads_end.close()
    phys = f"stadiacast/{os.getpid()}/c0"
    monkeypatch.setattr(ds, "open_uhid", lambda *_a: fd)
    monkeypatch.setattr(ds, "find_hidraw", lambda p, _root="": "/dev/hidraw19" if p == phys else None)
    ours_end.settimeout(5)
    try:
        yield ours_end
    finally:
        ours_end.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux uhid event socketpair test")
def test_dualsense_pad_answers_feature_reports_streams_input_and_forwards_rumble(kernel: socket.socket):
    made: list[ds.DualSensePad] = []
    creating = threading.Thread(target=lambda: made.append(ds.DualSensePad("c0", "99181FFAZ00123")))
    creating.start()

    create = kernel.recv(ds.EVENT_SIZE * 2)
    assert len(create) == ds.EVENT_SIZE
    assert create[196:260].rstrip(b"\0") == mac_string(mac_bytes("99181FFAZ00123")).encode()
    assert struct.unpack_from("=II", create, 264) == (0x054C, 0x0CE6)
    kernel.send(struct.pack("=I", UHID_START).ljust(ds.EVENT_SIZE, b"\0"))
    creating.join(5)
    (pad,) = made
    assert pad.device == "/dev/hidraw19"

    def recv_until_reply(expected_type: int, expected_req: int) -> bytes:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            ev = kernel.recv(ds.EVENT_SIZE * 2)
            kind = struct.unpack_from("=I", ev)[0]
            if kind == expected_type and struct.unpack_from("=I", ev, 4)[0] == expected_req:
                return ev
        raise AssertionError(f"timed out waiting for reply {expected_req}")

    try:
        # Probe feature report 0x09 (MAC) on the pad's thread
        kernel.send(_get_report(21, FEATURE_PAIRING_INFO))
        reply = recv_until_reply(UHID_GET_REPORT_REPLY, 21)
        assert reply[12:] == pairing_report(pad.mac)

        # Rumble via SET_REPORT, then stopped on UHID_CLOSE
        levels: list[tuple[int, int]] = []
        pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
        kernel.send(struct.pack("=I", UHID_OPEN).ljust(ds.EVENT_SIZE, b"\0"))
        kernel.send(_uhid_set_report(22, _output_usb(flag0=0x01, left=180, right=90)))
        assert struct.unpack_from("=IIH", recv_until_reply(UHID_SET_REPORT_REPLY, 22)) == (
            UHID_SET_REPORT_REPLY,
            22,
            0,
        )
        deadline = time.monotonic() + 2
        while levels != [(180, 90)] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert levels == [(180, 90)]
        kernel.send(struct.pack("=I", UHID_CLOSE).ljust(ds.EVENT_SIZE, b"\0"))
        deadline = time.monotonic() + 2
        while levels != [(180, 90), (0, 0)] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert levels == [(180, 90), (0, 0)]

        pad.set_battery(83)
        pad.apply_state(GamepadState(buttons=int(Button.CAPTURE | Button.ASSISTANT)))
        seen_pressed = False
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not seen_pressed:
            ev = kernel.recv(ds.EVENT_SIZE * 2)
            if struct.unpack_from("=IH", ev) == (12, 64) and (ev[6 + 10] & 0x06) == 0x06 and ev[6 + 53] == 8:
                seen_pressed = True
        assert seen_pressed
    finally:
        pad.close()
        pad.close()


class _FakeDualSensePad:
    """In-memory stand-in for DualSensePad in cross-platform receiver tests."""

    created: ClassVar[list[_FakeDualSensePad]] = []
    slot = None
    takes_state = True

    def __init__(self, label: str, uniq: str | None = None) -> None:
        self.label = label
        self.mac = mac_bytes((uniq or "").strip() or label)
        self.uniq = mac_string(self.mac)
        self.device = f"/dev/hidraw{50 + len(self.created)}"
        self.states: list[GamepadState] = []
        self.batteries: list[int | None] = []
        self.rumble_cb: Any = None
        self.closed = False
        self.created.append(self)

    def set_battery(self, percent: int | None) -> None:
        self.batteries.append(percent)

    def apply_state(self, state: GamepadState) -> None:
        self.states.append(state)

    def apply(self, _xinput: dict[str, int]) -> None:
        raise AssertionError("DualSensePad takes GamepadState directly")

    def reset(self) -> None:
        raise AssertionError("DualSensePad is per-boot and closed on disconnect")

    def on_rumble(self, callback: Any) -> None:
        self.rumble_cb = callback

    def close(self) -> None:
        self.closed = True


def test_dualsense_pads_keep_unique_macs_across_reverse_wake_order_and_forward_battery_and_rumble(
    tmp_path, monkeypatch
):
    """End-to-end ``--gamepad dualsense``: two controllers waking in opposite order across restarts
    keep their own deterministic ``02:..`` MACs (so Steam's ``configset_DS<mac>.vdf`` never swaps),
    forward battery percentage and rumble, and unplug cleanly when powered off."""
    monkeypatch.setattr(virtual_gamepad, "check_uhid_available", lambda *_a: None)
    monkeypatch.setattr(virtual_gamepad, "DualSensePad", _FakeDualSensePad)

    async def run_pair(run_name: str, order: tuple[tuple[str, str], tuple[str, str]]) -> dict[str, str]:
        _FakeDualSensePad.created.clear()
        rx = make_receiver(
            tmp_path / run_name,
            gamepad="dualsense",
            display_rpc_delay=0.05,
            display_rpc_interval=0.1,
            start_game_delay=0.05,
        )
        await rx.start()
        assert isinstance(rx.gamepads, VirtualGamepads) and rx.gamepads.kind == "dualsense"
        fakes: list[FakeController] = []
        try:
            for index, (dev, serial) in enumerate(order):
                fc = fake(rx, dev)
                fc.serial_number = serial
                fc.rpc_headers = lambda s=serial, c=fc.cloud_device_id: [  # type: ignore[method-assign]
                    ("controller-device-id", s),
                    ("controller-cloud-device-id", c),
                ]
                fakes.append(fc)
                await fc.boot()
                await fc.start_udp()
                await fc.discover()
                await fc.poll_gmsg()
                await fc.poll_gmsg()
                fc.connect()
                await fc.wait_connected(10)
                await asyncio.wait_for(fc.display_ready.wait(), 5)
                await fc.keepalive()
                fc.send_battery(0.76)
                await wait_for(lambda d=dev: rx.find(d) is not None and rx.find(d).status()["battery_percent"] == 76)  # type: ignore[union-attr]
                assert fc.send_input_rpc(buttons=int(Button.CAPTURE | Button.ASSISTANT | Button.A) | (1 << index))
            await wait_for(
                lambda: len(_FakeDualSensePad.created) == 2 and all(p.states for p in _FakeDualSensePad.created)
            )
            pad0 = _FakeDualSensePad.created[0]
            assert pad0.batteries[-1] == 76
            status = rx.status()["gamepads"]
            assert status["c0"]["kind"] == "dualsense" and status["c0"]["uniq"] == pad0.uniq
            return {dev: rx.status()["gamepads"][rx.find(dev).label]["uniq"] for dev, _ in order}  # type: ignore[union-attr]
        finally:
            for fc in fakes:
                fc.close()
            await rx.stop()

    white = ("ctl-white", "99181FFAZ00AAA")
    wasabi = ("ctl-wasabi", "99181FFAZ00BBB")
    want = {
        "ctl-white": mac_string(mac_bytes("99181FFAZ00AAA")),
        "ctl-wasabi": mac_string(mac_bytes("99181FFAZ00BBB")),
    }
    assert want["ctl-white"] != want["ctl-wasabi"]
    assert asyncio.run(run_pair("run1", (white, wasabi))) == want
    assert asyncio.run(run_pair("run2", (wasabi, white))) == want

    # Rumble and clean power-off unplug
    _FakeDualSensePad.created.clear()

    async def single_session() -> None:
        rx = make_receiver(tmp_path / "single", gamepad="dualsense", start_game_delay=0.05)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        rumbles: list[tuple[int, int]] = []
        cs.game.set_rumble = lambda left, right, _ms=None, _framing=None: rumbles.append((left, right)) or 1  # type: ignore[method-assign]
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert fc.send_input_rpc(buttons=int(Button.A))
            await wait_for(lambda: _FakeDualSensePad.created and _FakeDualSensePad.created[0].states)
            pad = _FakeDualSensePad.created[0]
            pad.rumble_cb(200, 100)
            await wait_for(lambda: rumbles)
            assert rumbles == [(200 * 257, 100 * 257)]
            fc.power_off()
            await wait_for(lambda: single(rx).session["state"] == "dtls-closed")
            assert pad.closed and rx.status()["gamepads"] == {}
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(single_session())
