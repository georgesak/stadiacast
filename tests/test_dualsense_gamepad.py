"""The Linux DualSense pad without a kernel: the address it makes of a controller's serial, its
report descriptor and feature reports as their readers take them, the input report, rumble from
the output report, the pad against a socket pair that plays the kernel, and the pad in the
receiver, where each controller is the same DualSense whatever order they wake in."""

from __future__ import annotations

import asyncio
import itertools
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
from stadiacast import uhid_gamepad as uhid
from stadiacast import virtual_gamepad
from stadiacast.dualsense_gamepad import (
    CALIBRATION_REPORT,
    FEATURE_CALIBRATION,
    FEATURE_FIRMWARE,
    FEATURE_PAIRING,
    FIRMWARE_REPORT,
    INPUT_REPORT_SIZE,
    REPORT_DESCRIPTOR,
    DualSenseRequests,
    battery_byte,
    input_report,
    mac_bytes,
    mac_string,
    output_rumble,
    pairing_report,
    stamped,
    stick_byte,
)
from stadiacast.errors import SetupError
from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import (
    EVENT_SIZE,
    UHID_CLOSE,
    UHID_CREATE2,
    UHID_GET_REPORT_REPLY,
    UHID_INPUT2,
    UHID_SET_REPORT_REPLY,
    UHID_START,
    UHID_STOP,
)
from stadiacast.virtual_gamepad import VirtualGamepads
from tests.fake_controller import FakeController
from tests.helpers import connect_firmware_flow, fake, make_receiver, single, wait_for
from tests.test_uhid_gamepad import _output, _plain, _reply, _set_report, _walk, linux_only, wait_until

SERIAL = "99181FFAZ00123"


def _get_report(request: int, number: int, rtype: int = uhid.UHID_FEATURE_REPORT) -> bytes:
    return struct.pack("=IIBB", uhid.UHID_GET_REPORT, request, number, rtype)


def _effects(
    *, flags: int = 0, lights: int = 0, more: int = 0, strong: int = 0, weak: int = 0, size: int = 48
) -> bytes:
    """An output report as its writers build it: SDL at 48 bytes, the kernel's driver at 63."""
    data = bytearray(size)
    data[0], data[1], data[2], data[3], data[4] = 0x02, flags, lights, weak, strong
    if size > 39:
        data[39] = more
    return bytes(data)


# ---- the address -----------------------------------------------------------------------------
def test_a_controllers_address_is_made_of_its_serial_and_is_no_real_devices():
    mac = mac_bytes(SERIAL)
    assert len(mac) == 6 and mac == mac_bytes(f"  {SERIAL} ") and mac != mac_bytes("99181FFAZ00124")
    assert mac[0] & 0x03 == 0x02  # given locally, and one device's: no manufacturer's range
    # The address is what Steam keeps a controller's name and bindings by: a change to how it is
    # made would lose every user theirs.
    assert mac_string(mac_bytes("PROBE0SERIAL01")) == "32:90:f4:d9:4c:00"
    assert mac_string(mac_bytes("c0")) == "72:4d:f2:f0:b3:37"
    with pytest.raises(ValueError, match="6 bytes"):
        mac_string(b"\x02\x00")


# ---- what its readers take from the descriptor and the feature reports -----------------------
def test_the_descriptor_declares_the_reports_and_the_fields_its_readers_need():
    items, depth = _walk(REPORT_DESCRIPTOR)
    assert depth == 0 and REPORT_DESCRIPTOR[:4] == bytes([0x05, 0x01, 0x09, 0x05])  # a gamepad, by its first words
    inputs = [item for item in items if item["kind"] == "input"]
    assert {item["report_id"] for item in inputs} == {1}
    # The kernel's driver reads an input report of these ids only at its full 64 bytes.
    assert sum(item["size"] * item["count"] for item in inputs) == (INPUT_REPORT_SIZE - 1) * 8
    assert len(input_report(GamepadState())) == INPUT_REPORT_SIZE
    (effects,) = [item for item in items if item["kind"] == "output"]
    assert (effects["report_id"], effects["size"] * effects["count"]) == (2, 47 * 8)
    features = {item["report_id"]: item["size"] * item["count"] // 8 + 1 for item in items if item["kind"] == "feature"}
    assert features == {
        FEATURE_CALIBRATION: len(CALIBRATION_REPORT),
        FEATURE_PAIRING: len(pairing_report(mac_bytes(SERIAL))),
        FEATURE_FIRMWARE: len(FIRMWARE_REPORT),
    }


def test_the_generic_driver_makes_a_dualsenses_axes_and_buttons_of_the_descriptor():
    """Without the kernel's PlayStation driver the generic one makes the event device out of the
    fields, and a program reads it with the mapping it keeps for a DualSense that is read that
    way. SDL's goes by the numbers of the axes and of the buttons
    (``leftx:a0,lefty:a1,rightx:a2,lefttrigger:a3,righttrigger:a4,righty:a5`` and
    ``x:b0,a:b1,b:b2,y:b3,leftshoulder:b4,rightshoulder:b5,back:b8,start:b9,leftstick:b10,``
    ``rightstick:b11,guide:b12``): an axis usage is its code (X 0x30 is ABS_X, Rz 0x35 is ABS_RZ)
    and button usage n is the nth key, so the fields have to be those, in the report's order."""
    data = [item for item in _walk(REPORT_DESCRIPTOR)[0] if item["kind"] == "input" and not item["flags"] & 1]
    axes, hat, buttons = data
    assert (axes["page"], axes["usages"], axes["size"], axes["count"]) == (
        0x01,
        [0x30, 0x31, 0x32, 0x35, 0x33, 0x34],
        8,
        6,
    )
    assert (hat["usages"], hat["size"], hat["low"], hat["high"]) == ([0x39], 4, 0, 7)
    assert (buttons["page"], buttons["usages"], buttons["size"]) == (0x09, list(range(1, 16)), 1)
    # The report's bytes in the fields' order: both sticks, both triggers, the count, then the hat
    # in the low half of the byte its buttons start in.
    report = input_report(GamepadState(buttons=int(Button.X | Button.L1 | Button.STADIA), l2=9, r2=7))
    assert (report[5], report[6]) == (9, 7)
    bits = int.from_bytes(report[8:11], "little") >> 4  # past the hat: button usage 1 is the lowest bit
    assert [n + 1 for n in range(15) if bits & (1 << n)] == [1, 5, 13]  # Square, L1, PS


def test_the_calibration_is_one_both_readers_take_and_that_changes_nothing():
    """The kernel's driver and SDL's both make a scale for each sensor axis out of this report
    and divide by what it says. The pad's sensors report nothing, so the scale has only to be
    sound: the one either reader uses when it has no calibration at all."""
    assert len(CALIBRATION_REPORT) == 41 and CALIBRATION_REPORT[0] == FEATURE_CALIBRATION
    values = struct.unpack_from("<17h", CALIBRATION_REPORT, 1)
    bias, gyro, speed, accel = values[:3], values[3:9], values[9:11], values[11:]
    assert bias == (0, 0, 0) and not any(CALIBRATION_REPORT[35:])
    for plus, minus in zip(gyro[::2], gyro[1::2], strict=True):
        # hid-playstation: (speed+ + speed-) * 1024 over |plus - bias| + |minus - bias|; SDL the
        # same over plus - minus, and it throws out a scale that is not within a half of 64.
        scale = (speed[0] + speed[1]) * 1024 / (abs(plus) + abs(minus))
        assert plus - minus > 0 and abs(1 - scale / 64) < 0.01
    for plus, minus in zip(accel[::2], accel[1::2], strict=True):
        span = plus - minus  # the two g between an axis up and down, which SDL holds in 16 bits
        assert 0 < span <= 32767 and plus - span // 2 == 0 and 2 * 8192 / span == 1.0


def test_the_pairing_report_is_the_address_backwards_and_the_firmware_is_new_enough_for_full_rumble():
    mac = mac_bytes(SERIAL)
    report = pairing_report(mac)
    assert len(report) == 20 and report[0] == FEATURE_PAIRING and not any(report[7:])
    # Both readers take bytes 6 down to 1 for the address, and the kernel's driver gives it to
    # the pad as its serial number in the same spelling.
    assert ":".join(f"{report[i]:02x}" for i in range(6, 0, -1)) == mac_string(mac)
    with pytest.raises(ValueError, match="6 bytes"):
        pairing_report(b"\x02")

    assert len(FIRMWARE_REPORT) == 64 and FIRMWARE_REPORT[0] == FEATURE_FIRMWARE
    (version,) = struct.unpack_from("<H", FIRMWARE_REPORT, 44)
    assert version >= 0x0224  # SDL's line for rumble at full range
    assert version >= (2 << 8 | 21)  # hid-playstation's: DS_FEATURE_VERSION(2, 21)


def test_the_kernel_is_given_the_three_feature_reports_it_asks_for():
    requests = DualSenseRequests(mac_bytes(SERIAL))
    for request, (number, report) in enumerate(
        (
            (FEATURE_PAIRING, pairing_report(mac_bytes(SERIAL))),
            (FEATURE_FIRMWARE, FIRMWARE_REPORT),
            (FEATURE_CALIBRATION, CALIBRATION_REPORT),
        ),
        start=11,
    ):
        reply = requests.handle(_get_report(request, number))
        assert reply is not None
        assert struct.unpack_from("=IIHH", reply) == (UHID_GET_REPORT_REPLY, request, 0, len(report))
        assert reply[12:] == report
    for number, rtype in ((0x08, uhid.UHID_FEATURE_REPORT), (FEATURE_PAIRING, uhid.UHID_INPUT_REPORT)):
        refused = requests.handle(_get_report(20, number, rtype))
        assert refused is not None and struct.unpack_from("=IIHH", refused)[1:] == (20, 5, 0)  # EIO: no such report


# ---- the input report ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("button", "index", "mask"),
    [
        (Button.X, 8, 0x10),  # Square: the left one of the four
        (Button.A, 8, 0x20),  # Cross: the lower
        (Button.B, 8, 0x40),  # Circle: the right
        (Button.Y, 8, 0x80),  # Triangle: the upper
        (Button.L1, 9, 0x01),
        (Button.R1, 9, 0x02),
        (Button.OPTIONS, 9, 0x10),  # Create
        (Button.MENU, 9, 0x20),  # Options
        (Button.L3, 9, 0x40),
        (Button.R3, 9, 0x80),
        (Button.STADIA, 10, 0x01),  # PS
        (Button.ASSISTANT, 10, 0x02),  # the touchpad's click
        (Button.CAPTURE, 10, 0x04),  # the microphone button
    ],
)
def test_every_button_is_where_a_dualsense_has_it(button: Button, index: int, mask: int):
    """The bits as the kernel's driver and SDL's read them (``DS_BUTTONS0_SQUARE`` and the rest),
    written out here a second time: a change to the pad's table has to be meant."""
    rest, pressed = input_report(GamepadState()), input_report(GamepadState(buttons=int(button)))
    changed = [(i, a ^ b) for i, (a, b) in enumerate(zip(rest, pressed, strict=True)) if a != b]
    assert changed == [(index, mask)]


def test_the_report_at_rest_the_sticks_the_triggers_the_hat_and_what_a_reader_must_not_mistake():
    rest = input_report(GamepadState())
    assert len(rest) == 64 and rest[0] == 0x01
    assert list(rest[1:7]) == [128, 128, 128, 128, 0, 0] and rest[8] == 8  # the hat's 8 is nothing pressed
    assert rest[33] == rest[37] == 0x80  # no finger at either touch point: a zero would be one in a corner
    assert not any(rest[16:28])  # sensors at rest
    assert not any(rest[12:16]) and not any(rest[38:53]) and not any(rest[54:])

    assert [stick_byte(v) for v in (-32768, -1, 0, 1, 32767)] == [0, 127, 128, 128, 255]
    assert [stick_byte(v, invert=True) for v in (-32768, 0, 32767)] == [255, 128, 0]  # up is the low end
    moved = input_report(GamepadState(lx=32767, ly=32767, rx=-32768, ry=-32768, l2=200, r2=29))
    assert list(moved[1:7]) == [255, 0, 0, 255, 200, 29]
    assert moved[9] & 0x0C == 0x04  # L2 also as a button; R2 is short of where a trigger counts as pressed
    hats = {Button.DPAD_UP: 0, Button.DPAD_RIGHT: 2, Button.DPAD_DOWN: 4, Button.DPAD_LEFT: 6}
    assert {b: input_report(GamepadState(buttons=int(b)))[8] & 0x0F for b in hats} == hats
    assert input_report(GamepadState(buttons=int(Button.DPAD_UP | Button.DPAD_RIGHT | Button.A)))[8] == 0x21


def test_the_battery_is_a_tenth_and_how_it_is_charged_and_full_while_it_is_not_known():
    """Both readers take the low half of the byte for the level in tenths (and show the middle of
    the tenth), and the high half for how it is charged: 0 on the battery, 2 full."""
    assert [battery_byte(p) for p in (0, 9, 10, 76, 99, 100, 150, -5)] == [0, 0, 1, 7, 9, 10, 10, 0]
    assert battery_byte(None) == 0x2A
    assert input_report(GamepadState(), 76)[53] == 7 and input_report(GamepadState())[53] == 0x2A


def test_every_report_counts_itself_and_tells_its_time():
    report = input_report(GamepadState(buttons=int(Button.A)), 50)
    first, later = stamped(report, 7, 12345678), stamped(report, 263, 1 << 32 | 5)
    assert first[7] == 7 and struct.unpack_from("<I", first, 28) == (12345678,)
    assert later[7] == 7 and struct.unpack_from("<I", later, 28) == (5,)  # both go round
    assert first[:7] + first[8:28] + first[32:] == report[:7] + report[8:28] + report[32:]


# ---- rumble ----------------------------------------------------------------------------------
def test_the_motor_levels_are_read_from_an_output_report_that_sets_them():
    # the kernel's driver and SDL with the firmware the pad states: bit 2 of the third byte of flags
    assert output_rumble(_effects(flags=0x02, more=0x04, strong=200, weak=100, size=63)) == (
        True,
        (200 * 257, 100 * 257),
    )
    # the older way: bit 0 of the first
    assert output_rumble(_effects(flags=0x03, strong=64, weak=32)) == (True, (64 * 257, 32 * 257))
    assert output_rumble(_effects(flags=0x02, more=0x04)) == (True, (0, 0))  # a rumble set to nothing
    # SDL's stop is a report that sets nothing at all, and its start one that only asks for the motors
    assert output_rumble(_effects()) == (True, (0, 0))
    assert output_rumble(_effects(flags=0x02)) == (True, (0, 0))
    # lights and the like leave the motors as they are: the kernel's driver sends these when it takes the pad
    for lights_only in (_effects(more=0x02, size=63), _effects(lights=0x14, size=63), _effects(lights=0x03, size=63)):
        assert output_rumble(lights_only) == (True, None)
    # and what is no output report of the pad's
    assert output_rumble(b"\x02\x01\x00\xff\xff") == (False, None)
    assert output_rumble(b"\x05" + bytes(47)) == (False, None)
    assert output_rumble(b"") == (False, None)


def test_rumble_comes_both_ways_and_lights_do_not_touch_it():
    """A program's write to the hidraw node and the kernel's driver both arrive as an output
    event, which wants no answer; a SET_REPORT wants one, and gets a refusal for what is no
    output report of the pad's."""
    requests = DualSenseRequests(mac_bytes(SERIAL))
    assert requests.handle(_output(_effects(flags=0x02, more=0x04, strong=200, weak=100))) is None
    assert requests.levels == (200 * 257, 100 * 257)
    assert requests.handle(_output(_effects(lights=0x14, size=63))) is None
    assert requests.levels == (200 * 257, 100 * 257)
    assert _reply(requests.handle(_set_report(55, _effects(lights=0x04, size=63)))) == (UHID_SET_REPORT_REPLY, 55, 0)
    assert requests.levels == (200 * 257, 100 * 257)
    assert _reply(requests.handle(_set_report(56, _effects()))) == (UHID_SET_REPORT_REPLY, 56, 0)
    assert requests.levels == (0, 0)
    assert _reply(requests.handle(_set_report(57, b"\x05\x00\x80\x00\x40"))) == (UHID_SET_REPORT_REPLY, 57, 5)  # EIO
    assert _reply(requests.handle(_set_report(58, _effects(flags=0x01, strong=9), uhid.UHID_FEATURE_REPORT)))[2] == 5

    requests.handle(_output(_effects(flags=0x03, strong=64, weak=32)))
    requests.handle(_plain(UHID_START))
    requests.handle(_plain(UHID_STOP))  # the driver let go of the device: nobody is left to stop it
    assert requests.levels == (0, 0) and (requests.starts, requests.stops) == (1, 1)
    requests.handle(_output(_effects(flags=0x03, strong=64, weak=32)))
    requests.handle(_plain(UHID_CLOSE))
    assert requests.levels == (0, 0) and requests.stops == 1


def test_the_pad_is_linux_only_and_says_which_flag_asked_for_it(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SetupError, match="--gamepad dualsense is Linux only"):
        ds.DualSensePad("c0")
    with pytest.raises(SetupError, match="--gamepad dualsense is Linux only"):
        uhid.check_available("--gamepad dualsense")


# ---- the pad, with the kernel's part played --------------------------------------------------
PHYS = f"stadiacast/{os.getpid()}/c0"


@pytest.fixture
def node(tmp_path) -> str:
    """A file in place of the pad's hidraw node, which programs open and let go of."""
    path = tmp_path / "hidraw19"
    path.write_bytes(b"")
    return str(path)


@pytest.fixture
def kernel(monkeypatch, node) -> Iterator[socket.socket]:
    """A socket pair in place of ``/dev/uhid``, as in the Stadia pad's tests."""
    ours_end, pads_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    os.set_blocking(pads_end.fileno(), False)
    fd = os.dup(pads_end.fileno())
    pads_end.close()
    monkeypatch.setattr(uhid, "open_uhid", lambda _flag="": fd)
    monkeypatch.setattr(uhid, "find_hidraw", lambda phys, _root="": node if phys == PHYS else None)
    ours_end.settimeout(5)
    try:
        yield ours_end
    finally:
        ours_end.close()


def _create_pad(kernel: socket.socket) -> ds.DualSensePad:
    """A pad made the way the receiver makes one. The kernel's driver asks for the three feature
    reports while it takes the device, before the pad is anybody's: they are answered there."""
    made: list[ds.DualSensePad] = []
    creating = threading.Thread(target=lambda: made.append(ds.DualSensePad("c0", SERIAL)))
    creating.start()
    create = kernel.recv(EVENT_SIZE * 2)
    assert len(create) == EVENT_SIZE and struct.unpack_from("=I", create) == (UHID_CREATE2,)
    assert create[4:55] == b"Sony Interactive Entertainment Wireless Controller\0"
    assert create[132:196].rstrip(b"\0") == PHYS.encode()
    assert create[196:260].rstrip(b"\0") == mac_string(mac_bytes(SERIAL)).encode()  # what the kernel's driver says too
    assert struct.unpack_from("=HHIII", create, 260) == (len(REPORT_DESCRIPTOR), 0x03, 0x054C, 0x0CE6, 0x0111)
    kernel.send(_plain(UHID_START))
    creating.join(5)
    (pad,) = made
    return pad


def _next(kernel: socket.socket, kind: int) -> bytes:
    """The pad's next event of that kind; it reports all the while."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        event = kernel.recv(EVENT_SIZE * 2)
        if struct.unpack_from("=I", event) == (kind,):
            return event
    raise AssertionError(f"no event of type {kind} in time")


def _report(kernel: socket.socket) -> bytes:
    event = _next(kernel, UHID_INPUT2)
    assert struct.unpack_from("=IH", event) == (UHID_INPUT2, 64)
    return event[6:70]


@linux_only
def test_the_pad_says_it_is_a_dualsense_answers_the_driver_and_reports_all_the_time(kernel, node):
    pad = _create_pad(kernel)
    try:
        assert pad.device == node and pad.uniq == mac_string(pad.mac) == mac_string(mac_bytes(SERIAL))
        # the driver's three questions, on the pad's thread
        for request, (number, report) in enumerate(
            (
                (FEATURE_PAIRING, pairing_report(pad.mac)),
                (FEATURE_FIRMWARE, FIRMWARE_REPORT),
                (FEATURE_CALIBRATION, CALIBRATION_REPORT),
            ),
            start=21,
        ):
            kernel.send(_get_report(request, number))
            reply = _next(kernel, UHID_GET_REPORT_REPLY)
            assert struct.unpack_from("=IIHH", reply) == (UHID_GET_REPORT_REPLY, request, 0, len(report))
            assert reply[12 : 12 + len(report)] == report

        # at rest from the start, and a report every few milliseconds, each with its count and a later time
        began = time.monotonic()
        reports = [_report(kernel) for _ in range(60)]
        assert time.monotonic() - began < 2.0  # at its 8 ms half a second; at the Stadia pad's pace six
        assert {stamped(r, 0, 0) for r in reports} == {input_report(GamepadState())}  # but for count and time
        counts = [r[7] for r in reports]
        assert all((b - a) % 256 == 1 for a, b in itertools.pairwise(counts))
        times = [struct.unpack_from("<I", r, 28)[0] for r in reports]
        assert all(b > a for a, b in itertools.pairwise(times))

        # input, and the battery when it is known, are in the reports that follow
        pad.set_battery(83)
        pad.apply_state(GamepadState(buttons=int(Button.CAPTURE | Button.ASSISTANT), l2=255))
        wait_until(lambda: (r := _report(kernel))[10] & 0x06 == 0x06 and r[5] == 255 and r[53] == 8)
        pad.apply_state(GamepadState())
        pad.set_battery(None)
        wait_until(lambda: (r := _report(kernel))[10] == 0 and r[53] == 0x2A)
    finally:
        pad.close()
        pad.close()


@linux_only
def test_the_pads_rumble_reaches_the_controller_and_stops_with_the_program_that_left_it_on(kernel, node):
    """Rumble as a program writes it to the hidraw node and as the kernel's driver sends it. The
    kernel's driver holds the device open for as long as it has it, so no close comes from uhid
    when a program is killed: the pad hears its node let go of, and stops the motors then."""
    pad = _create_pad(kernel)
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    try:
        game = os.open(node, os.O_RDWR)
        kernel.send(_output(_effects(flags=0x02, more=0x04, strong=180, weak=90)))
        wait_until(lambda: levels == [(180, 90)])
        kernel.send(_output(_effects(lights=0x14, size=63)))  # the driver's lights: the motors stay
        kernel.send(_set_report(22, _effects(flags=0x01, strong=20, weak=10)))
        assert _reply(_next(kernel, UHID_SET_REPORT_REPLY)) == (UHID_SET_REPORT_REPLY, 22, 0)
        wait_until(lambda: levels == [(180, 90), (20, 10)])
        os.close(game)  # killed, with its rumble on
        wait_until(lambda: levels == [(180, 90), (20, 10), (0, 0)])
    finally:
        pad.close()


@linux_only
def test_a_program_that_opens_the_pad_has_a_report_at_once(kernel, node, monkeypatch):
    """SDL's DualSense driver waits 16 ms for a first report from a pad it has just
    opened, and reads neither the address nor the firmware of one that shows none. The pad does
    not leave that to where its own turn falls: it writes when a program opens the node."""
    monkeypatch.setattr(ds.DualSensePad, "REFRESH", 30.0)  # no turn of its own in this test
    pad = _create_pad(kernel)
    try:
        time.sleep(0.2)  # the hurry after the start is over
        kernel.settimeout(0.3)
        while True:
            try:
                kernel.recv(EVENT_SIZE * 2)
            except TimeoutError:
                break
        kernel.settimeout(2)
        began = time.monotonic()
        program = os.open(node, os.O_RDWR)
        try:
            _report(kernel)
            first = time.monotonic() - began
            for _ in range(4):
                _report(kernel)
            assert first < 0.5 and time.monotonic() - began < 1.0
        finally:
            os.close(program)
    finally:
        pad.close()


@linux_only
def test_a_pad_the_kernels_driver_refuses_says_so(kernel, monkeypatch):
    """Two pads with one address: the kernel's driver refuses the second after it started it, lets
    go of it, and no driver takes it. The receiver is told, where it would show a pad that is none."""
    monkeypatch.setattr(ds.DualSensePad, "DRIVER_WAIT", 0.2)
    pad = _create_pad(kernel)
    told: list[str] = []
    pad.on_error(told.append)
    try:
        kernel.send(_plain(UHID_CLOSE))
        kernel.send(_plain(UHID_STOP))
        wait_until(lambda: bool(told))
        assert "no other took it" in told[0]
    finally:
        pad.close()


# ---- in the receiver -------------------------------------------------------------------------
class _FakeDualSensePad:
    """A DualSense pad with a hidraw node, in place of the kernel's."""

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
        raise AssertionError("a pad that takes the state is given the state")

    def on_rumble(self, callback: Any) -> None:
        self.rumble_cb = callback

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def pads(monkeypatch) -> type[_FakeDualSensePad]:
    monkeypatch.setattr(virtual_gamepad, "check_uhid_available", lambda _flag="": None)
    monkeypatch.setattr(virtual_gamepad, "DualSensePad", _FakeDualSensePad)
    _FakeDualSensePad.created.clear()
    return _FakeDualSensePad


def test_each_controller_is_the_same_dualsense_whatever_order_they_wake_in(tmp_path, pads):
    """Two controllers that wake in one order and, after the receiver was started again, in the
    other: each keeps its address, which is its serial's and not its place in the queue's, and
    ``/_status`` names it."""

    async def wake(run: str, order: tuple[tuple[str, str], ...]) -> dict[str, str]:
        pads.created.clear()
        rx = make_receiver(tmp_path / run, gamepad="dualsense", display_rpc_delay=0.05, display_rpc_interval=0.1)
        await rx.start()
        assert isinstance(rx.gamepads, VirtualGamepads) and rx.gamepads.kind == "dualsense"
        fakes: list[FakeController] = []
        try:
            for count, (device, serial) in enumerate(order, 1):
                fc = fake(rx, device)
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
                await wait_for(lambda n=count: len(rx.status()["gamepads"]) == n)
            status = rx.status()["gamepads"]
            assert all(
                pad["kind"] == "dualsense" and pad["device"].startswith("/dev/hidraw") for pad in status.values()
            )
            created = {r["controller"]: r["uniq"] for r in rx.log.records if r["event"] == "gamepad_created"}
            assert created == {label: pad["uniq"] for label, pad in status.items()}
            return {device: status[rx.find(device).label]["uniq"] for device, _serial in order}  # type: ignore[union-attr]
        finally:
            for fc in fakes:
                fc.close()
            await rx.stop()

    white, wasabi = ("ctl-white", "99181FFAZ00AAA"), ("ctl-wasabi", "99181FFAZ00BBB")
    want = {device: mac_string(mac_bytes(serial)) for device, serial in (white, wasabi)}
    assert want["ctl-white"] != want["ctl-wasabi"]
    assert asyncio.run(wake("one", (white, wasabi))) == want
    assert asyncio.run(wake("two", (wasabi, white))) == want


def test_the_pad_has_the_battery_level_from_the_start_and_when_it_moves(tmp_path, pads):
    """The level the controller's state gave is the pad's when it is made, before any input, and
    the gauge's word reaches the pad when it comes, with no input to carry it."""

    async def run() -> None:
        rx = make_receiver(tmp_path, gamepad="dualsense", start_game_delay=0.05)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        fc.battery_level = 0.64
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: rx.status()["gamepads"])
            (pad,) = pads.created
            assert pad.batteries == [64] and not pad.states
            fc.send_battery(0.63)
            await wait_for(lambda: pad.batteries == [64, 63])
            fc.send_battery(0.63)  # the same again is no news
            await wait_for(lambda: single(rx).session["display"]["battery_samples"] == 2)
            assert pad.batteries == [64, 63] and not pad.states
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_the_pad_gets_the_whole_state_its_rumble_comes_back_and_it_goes_with_its_controller(tmp_path, pads):
    async def run() -> None:
        rx = make_receiver(tmp_path, gamepad="dualsense", start_game_delay=0.05)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        rumbles: list[tuple[int, int]] = []
        cs.game.set_rumble = lambda left, right, _ms=None, _framing=None: rumbles.append((left, right)) or 1  # type: ignore[method-assign]
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            held = int(Button.CAPTURE | Button.ASSISTANT | Button.A)
            assert fc.send_input_rpc(buttons=held, lt=200)
            await wait_for(lambda: pads.created and pads.created[0].states)
            (pad,) = pads.created
            assert pad.uniq == mac_string(mac_bytes("fake-device"))  # the serial its requests name
            assert (pad.states[-1].buttons, pad.states[-1].l2) == (held, 200)
            assert rx.status()["gamepads"] == {
                "c0": {
                    "slot": None,
                    "frames": 1,
                    "rumbles": 0,
                    "kind": "dualsense",
                    "device": "/dev/hidraw50",
                    "uniq": pad.uniq,
                }
            }
            pad.rumble_cb(200, 100)
            await wait_for(lambda: rumbles)
            assert rumbles == [(200 * 257, 100 * 257)]
            fc.power_off()
            await wait_for(lambda: pad.closed)
            assert rx.status()["gamepads"] == {}
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())
