"""The Linux Stadia pad without a kernel: the events it speaks to uhid with, its report
descriptor, the input report as SDL's Stadia driver reads it, and the requests it answers. The
pad's own thread runs against a socket pair in place of ``/dev/uhid`` on Linux."""

from __future__ import annotations

import asyncio
import errno
import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import pytest

from stadiacast import uhid_gamepad as uhid
from stadiacast import virtual_gamepad
from stadiacast.errors import SetupError
from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import (
    EVENT_SIZE,
    REPORT_DESCRIPTOR,
    UHID_CLOSE,
    UHID_CREATE2,
    UHID_FEATURE_REPORT,
    UHID_GET_REPORT,
    UHID_GET_REPORT_REPLY,
    UHID_INPUT2,
    UHID_OPEN,
    UHID_OUTPUT,
    UHID_OUTPUT_REPORT,
    UHID_SET_REPORT,
    UHID_SET_REPORT_REPLY,
    UHID_START,
    UHID_STOP,
    Requests,
    create2_event,
    find_hidraw,
    hat_value,
    input2_event,
    input_report,
    rumble_levels,
    state_from_xinput,
    stick_byte,
)
from stadiacast.virtual_gamepad import VirtualGamepads
from tests.helpers import connect_firmware_flow, make_receiver, single, wait_for


def wait_until(cond: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


# ---- the kernel's event layout ----------------------------------------------------------------
def test_events_have_the_size_and_offsets_of_linux_uhid_h():
    """``struct uhid_event`` is packed: the type, then a union as wide as ``uhid_create2_req``."""
    assert EVENT_SIZE == 4 + 128 + 64 + 64 + 2 + 2 + 4 * 4 + 4096 == 4376
    event = create2_event("name", "phys", "uniq", b"\x05\x01")
    assert len(event) == EVENT_SIZE
    assert struct.unpack_from("=I", event, 0) == (UHID_CREATE2,)
    assert event[4:8] == b"name" and event[132:136] == b"phys" and event[196:200] == b"uniq"
    rd_size, bus, vendor, product, version, country = struct.unpack_from("=HHIIII", event, 260)
    assert (rd_size, bus, vendor, product, version, country) == (2, 0x03, 0x18D1, 0x9400, 0x0100, 0)
    assert event[280:282] == b"\x05\x01"

    report = bytes(range(1, 11))
    event = input2_event(report)
    assert event == struct.pack("=IH", UHID_INPUT2, 10) + report  # no longer than what it carries

    # uhid_output_req puts the data first and its size and type after the 4096 bytes
    output = struct.pack("=I", UHID_OUTPUT) + b"\x05\x01\x02\x03\x04".ljust(4096, b"\0") + struct.pack("=HB", 5, 1)
    assert uhid.output_request(output) == (b"\x05\x01\x02\x03\x04", UHID_OUTPUT_REPORT)
    get = struct.pack("=IIBB", UHID_GET_REPORT, 0xDEAD, 3, UHID_FEATURE_REPORT)
    assert uhid.get_report_request(get) == (0xDEAD, 3, UHID_FEATURE_REPORT)
    set_ = struct.pack("=IIBBH", UHID_SET_REPORT, 9, 5, UHID_OUTPUT_REPORT, 3) + b"\x05\x06\x07"
    assert uhid.set_report_request(set_) == (9, 5, UHID_OUTPUT_REPORT, b"\x05\x06\x07")


def test_a_long_descriptor_is_refused_before_it_reaches_the_kernel():
    with pytest.raises(ValueError, match="too long"):
        create2_event("n", "p", "u", b"\0" * 4097)


# ---- the report descriptor --------------------------------------------------------------------
def _walk(descriptor: bytes) -> tuple[list[dict[str, Any]], int]:
    """The main items of a report descriptor with the globals and locals in force, and the depth
    of collections left open (0 when they all close)."""
    items: list[dict[str, Any]] = []
    glob: dict[str, int] = {"report_id": 0, "size": 0, "count": 0, "page": 0, "low": 0, "high": 0}
    usages: list[int] = []
    depth, i = 0, 0
    while i < len(descriptor):
        prefix = descriptor[i]
        size = (0, 1, 2, 4)[prefix & 3]
        kind, tag = (prefix >> 2) & 3, prefix >> 4
        value = int.from_bytes(descriptor[i + 1 : i + 1 + size], "little")
        i += 1 + size
        if kind == 1:  # global
            key = {0: "page", 1: "low", 2: "high", 7: "size", 8: "report_id", 9: "count"}.get(tag)
            if key:
                glob[key] = value
        elif kind == 2 and tag == 0:  # local: usage
            usages.append(value)
        elif kind == 0:  # main
            if tag == 0xA:
                depth += 1
            elif tag == 0xC:
                depth -= 1
            elif tag in (0x8, 0x9):
                items.append({"kind": "input" if tag == 0x8 else "output", "flags": value, **glob, "usages": usages})
            usages = []
    return items, depth


def test_the_descriptor_describes_the_ten_byte_report_and_the_rumble_report():
    items, depth = _walk(REPORT_DESCRIPTOR)
    assert depth == 0
    inputs = [item for item in items if item["kind"] == "input"]
    assert {item["report_id"] for item in inputs} == {3}
    assert sum(item["size"] * item["count"] for item in inputs) == 9 * 8  # the bytes after the report id
    assert len(input_report(GamepadState())) == 10
    # hid-google-stadiaff binds only to a device whose output report 5 has a first field of at
    # least two values; it writes the strong motor to the first and the weak to the second.
    (rumble,) = [item for item in items if item["kind"] == "output"]
    assert (rumble["report_id"], rumble["size"], rumble["count"]) == (5, 16, 2)


def test_the_event_device_gets_the_wired_controllers_key_codes_and_button_numbers():
    """On a gamepad hid-input maps button usage n to BTN_GAMEPAD + n - 1 up to 16 and to
    BTN_TRIGGER_HAPPY + n - 17 above. The codes have to be the wired controller's, because a
    program that reads the event device numbers the buttons by rising code, and the mappings kept
    for these ids go by those numbers: SDL's
    (``a:b0,b:b1,x:b2,y:b3,leftshoulder:b4,rightshoulder:b5,back:b6,start:b7,guide:b8,``
    ``leftstick:b9,rightstick:b10``) and Chromium's, which adds Assistant and Capture as 11 and 12."""
    (buttons,) = [item for item in _walk(REPORT_DESCRIPTOR)[0] if item["page"] == 0x09 and not item["flags"] & 1]
    btn_gamepad, btn_trigger_happy = 0x130, 0x2C0

    def code(usage: int) -> int:
        return btn_gamepad + usage - 1 if usage <= 16 else btn_trigger_happy + usage - 17

    report_order: list[Any] = [
        Button.CAPTURE,
        Button.ASSISTANT,
        "l2",
        "r2",
        Button.STADIA,
        Button.MENU,
        Button.OPTIONS,
        Button.R3,
        Button.L3,
        Button.R1,
        Button.L1,
        Button.Y,
        Button.X,
        Button.B,
        Button.A,
    ]
    assert buttons["count"] == len(buttons["usages"]) == 15
    codes = {name: code(usage) for name, usage in zip(report_order, buttons["usages"], strict=True)}
    assert codes == {
        Button.A: 0x130,  # BTN_A
        Button.B: 0x131,  # BTN_B
        Button.X: 0x133,  # BTN_X
        Button.Y: 0x134,  # BTN_Y
        Button.L1: 0x136,  # BTN_TL
        Button.R1: 0x137,  # BTN_TR
        Button.OPTIONS: 0x13A,  # BTN_SELECT
        Button.MENU: 0x13B,  # BTN_START
        Button.STADIA: 0x13C,  # BTN_MODE
        Button.L3: 0x13D,  # BTN_THUMBL
        Button.R3: 0x13E,  # BTN_THUMBR
        Button.ASSISTANT: 0x2C0,  # BTN_TRIGGER_HAPPY1
        Button.CAPTURE: 0x2C1,  # BTN_TRIGGER_HAPPY2
        "r2": 0x2C2,  # BTN_TRIGGER_HAPPY3
        "l2": 0x2C3,  # BTN_TRIGGER_HAPPY4
    }
    numbered = sorted(codes, key=lambda name: codes[name])
    assert numbered[:11] == [
        Button.A,
        Button.B,
        Button.X,
        Button.Y,
        Button.L1,
        Button.R1,
        Button.OPTIONS,  # back
        Button.MENU,  # start
        Button.STADIA,  # guide
        Button.L3,
        Button.R3,
    ]
    assert numbered[11:13] == [Button.ASSISTANT, Button.CAPTURE]


def test_the_sticks_and_the_triggers_declare_the_range_the_report_uses():
    """A stick's byte runs from 1 to 255 around 128: declared from 0, the event device would put
    its rest off centre and never reach the far left or the top."""
    items = _walk(REPORT_DESCRIPTOR)[0]
    (sticks,) = [item for item in items if 0x30 in item["usages"]]
    (triggers,) = [item for item in items if item["page"] == 0x02]
    assert (sticks["usages"], sticks["low"], sticks["high"]) == ([0x30, 0x31, 0x32, 0x35], 1, 255)  # X, Y, Z, Rz
    assert (triggers["usages"], triggers["low"], triggers["high"]) == ([0xC5, 0xC4], 0, 255)  # brake, accelerator
    assert [stick_byte(value) for value in (-32768, 0, 32767)] == [1, 128, 255]


# ---- the report as SDL reads it --------------------------------------------------------------
#: Where SDL's Stadia driver (src/joystick/hidapi/SDL_hidapi_stadia.c) reads each button, by its name.
SDL_BUTTONS: dict[str, tuple[int, int]] = {
    "back": (2, 0x40),
    "guide": (2, 0x10),
    "start": (2, 0x20),
    "rightstick": (2, 0x80),
    "misc1": (2, 0x01),  # SDL_GAMEPAD_BUTTON_STADIA_CAPTURE
    "misc2": (2, 0x02),  # SDL_GAMEPAD_BUTTON_STADIA_GOOGLE_ASSISTANT
    "south": (3, 0x40),
    "east": (3, 0x20),
    "west": (3, 0x10),
    "north": (3, 0x08),
    "leftshoulder": (3, 0x04),
    "rightshoulder": (3, 0x02),
    "leftstick": (3, 0x01),
}


def _sdl_reads(report: bytes) -> dict[str, Any]:
    """What SDL's Stadia driver makes of an input report: the buttons held, the sticks (SDL's Y
    points down) and the triggers, both as SDL's -32768..32767."""
    assert len(report) >= 10 and report[0] == 0x03  # shorter, or another id: SDL drops it

    def stick(b: int) -> int:
        return 0 if b == 0x80 else round(-32768 + (b - 1) * (32767 + 32768) / (0xFF - 1))

    return {
        "hat": report[1],
        "buttons": {name for name, (index, mask) in SDL_BUTTONS.items() if report[index] & mask},
        "axes": [stick(b) for b in report[4:8]],
        "triggers": [b * 257 - 32768 for b in report[8:10]],
    }


@pytest.mark.parametrize(
    ("button", "sdl_name"),
    [
        (Button.A, "south"),
        (Button.B, "east"),
        (Button.X, "west"),
        (Button.Y, "north"),
        (Button.L1, "leftshoulder"),
        (Button.R1, "rightshoulder"),
        (Button.L3, "leftstick"),
        (Button.R3, "rightstick"),
        (Button.MENU, "start"),
        (Button.OPTIONS, "back"),
        (Button.STADIA, "guide"),
        (Button.CAPTURE, "misc1"),
        (Button.ASSISTANT, "misc2"),
    ],
)
def test_every_button_is_the_one_sdl_reads_there(button, sdl_name):
    assert _sdl_reads(input_report(GamepadState(buttons=int(button))))["buttons"] == {sdl_name}


def test_a_neutral_report_is_released_and_centred():
    report = input_report(GamepadState())
    assert report == bytes([3, 8, 0, 0, 0x80, 0x80, 0x80, 0x80, 0, 0])
    assert _sdl_reads(report) == {"hat": 8, "buttons": set(), "axes": [0, 0, 0, 0], "triggers": [-32768, -32768]}


def test_the_sticks_keep_their_range_and_y_turns_to_point_down():
    full = input_report(GamepadState(lx=32767, ly=32767, rx=-32768, ry=-32768))  # left up-right, right down-left
    assert _sdl_reads(full)["axes"] == [32767, -32768, -32768, 32767]
    assert stick_byte(0) == 0x80 and stick_byte(16384) == 0x80 + 64 and stick_byte(16384, invert=True) == 0x80 - 64
    wide = input_report(GamepadState(lx=99999, ly=-99999))  # the game session's fields are wider
    assert wide[4:6] == bytes([0xFF, 0xFF])


def test_the_triggers_are_analog_and_press_their_digital_bit_past_xinputs_threshold():
    assert _sdl_reads(input_report(GamepadState(l2=255, r2=0)))["triggers"] == [32767, -32768]
    assert input_report(GamepadState(l2=29, r2=29))[2] == 0
    assert input_report(GamepadState(l2=30))[2] == 0x04 and input_report(GamepadState(r2=30))[2] == 0x08
    assert input_report(GamepadState(l2=999, r2=-5))[8:10] == bytes([255, 0])


def test_the_d_pad_is_a_hat_turning_clockwise_from_up():
    up, down, left, right = Button.DPAD_UP, Button.DPAD_DOWN, Button.DPAD_LEFT, Button.DPAD_RIGHT
    held = [up, up | right, right, down | right, down, down | left, left, up | left]
    assert [hat_value(int(b)) for b in held] == list(range(8))
    assert hat_value(0) == hat_value(int(up | down)) == hat_value(int(left | right)) == 8
    assert hat_value(int(up | down | right)) == 2  # the opposite pair cancels, the third stays


def test_an_xinput_field_set_reads_back_with_the_guide_as_the_stadia_button():
    state = GamepadState(buttons=int(Button.A | Button.STADIA | Button.CAPTURE), lx=5, ly=-6, rx=7, ry=-8, l2=9, r2=10)
    back = state_from_xinput(state.to_xinput())
    assert back == GamepadState(buttons=int(Button.A | Button.STADIA), lx=5, ly=-6, rx=7, ry=-8, l2=9, r2=10)


# ---- the kernel's requests -------------------------------------------------------------------
def _output(data: bytes, rtype: int = UHID_OUTPUT_REPORT) -> bytes:
    return struct.pack("=I", UHID_OUTPUT) + data.ljust(4096, b"\0") + struct.pack("=HB", len(data), rtype)


def _set_report(request: int, data: bytes, rtype: int = UHID_OUTPUT_REPORT) -> bytes:
    return struct.pack("=IIBBH", UHID_SET_REPORT, request, data[0] if data else 0, rtype, len(data)) + data


def _reply(event: bytes) -> tuple[int, int, int]:
    return struct.unpack_from("=IIH", event)


def _plain(kind: int) -> bytes:
    """An event that is its type alone, as the kernel hands it out: start, stop, open, close."""
    return struct.pack("=I", kind).ljust(EVENT_SIZE, b"\0")


def test_rumble_is_output_report_5_with_the_strong_motor_first():
    assert rumble_levels(b"\x05\x00\x80\xff\x00") == (0x8000, 0x00FF)
    assert rumble_levels(b"\x05\x00\x80\xff") is None  # cut short
    assert rumble_levels(b"\x03\x00\x80\xff\x00") is None  # another report


def test_a_hidraw_write_sets_the_motors_and_is_not_answered():
    """What SDL does: it writes report 5 to the hidraw node (first with both motors off, to learn
    whether the pad rumbles at all)."""
    requests = Requests()
    assert requests.handle(_output(b"\x05\x00\x00\x00\x00")) is None and requests.levels == (0, 0)
    assert requests.handle(_output(b"\x05\x34\x12\x78\x56")) is None and requests.levels == (0x1234, 0x5678)
    assert requests.handle(_output(b"\x05\xff\xff\xff\xff", UHID_FEATURE_REPORT)) is None  # not an output report
    assert requests.levels == (0x1234, 0x5678)


def test_a_set_report_sets_the_motors_and_is_answered_with_its_id():
    """What hid-google-stadiaff does for a game's force feedback; the kernel waits for the answer."""
    requests = Requests()
    reply = requests.handle(_set_report(77, b"\x05\x00\xff\x00\x10"))
    assert reply is not None and len(reply) == 10  # the type, the request and the error
    assert _reply(reply) == (UHID_SET_REPORT_REPLY, 77, 0) and requests.levels == (0xFF00, 0x1000)
    refused = requests.handle(_set_report(78, b"\x03\x01"))
    assert refused is not None and _reply(refused) == (UHID_SET_REPORT_REPLY, 78, errno.EIO)
    assert requests.levels == (0xFF00, 0x1000)


def test_a_get_report_is_refused_and_the_starts_are_counted():
    requests = Requests()
    reply = requests.handle(struct.pack("=IIBB", UHID_GET_REPORT, 5, 3, UHID_FEATURE_REPORT))
    assert reply is not None and struct.unpack_from("=IIHH", reply) == (UHID_GET_REPORT_REPLY, 5, errno.EIO, 0)
    assert requests.starts == 0
    for kind in (UHID_START, UHID_OPEN, UHID_CLOSE, UHID_STOP, UHID_START, UHID_OPEN):
        assert requests.handle(_plain(kind)) is None
    assert requests.starts == 2  # once for each driver that took the device
    assert requests.handle(b"") is None  # nothing to read: nothing done


@pytest.mark.parametrize("kind", [UHID_CLOSE, UHID_STOP])
def test_the_motors_stop_when_nobody_is_left_to_stop_them(kind):
    """A program that ends while its rumble runs never sends the report that stops it. The kernel
    says when the last one has closed the pad's nodes, and when a driver lets go of the device."""
    requests = Requests()
    requests.handle(_output(b"\x05\xff\xff\x00\x80"))
    assert requests.levels == (0xFFFF, 0x8000)
    requests.handle(_plain(UHID_OPEN))  # another one opening changes nothing
    assert requests.levels == (0xFFFF, 0x8000)
    requests.handle(_plain(kind))
    assert requests.levels == (0, 0)


# ---- the device ------------------------------------------------------------------------------
def test_the_pad_is_for_linux_and_says_what_keeps_the_device_shut(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        with pytest.raises(SetupError, match="--gamepad uhid is Linux only"):
            uhid.check_available()
        with pytest.raises(SetupError, match="--gamepad uhid is Linux only"):
            uhid.UhidPad("c0")
    if sys.platform == "win32":
        return  # the rest opens files the way Linux does
    monkeypatch.setattr(uhid.sys, "platform", "linux")
    monkeypatch.setattr(uhid, "UHID_PATH", str(tmp_path / "none"))
    with pytest.raises(SetupError, match=r"found none \(.*none\): load it with `sudo modprobe uhid`"):
        uhid.check_available()
    shut = tmp_path / "uhid"
    shut.write_bytes(b"")
    monkeypatch.setattr(uhid, "UHID_PATH", str(shut))
    uhid.check_available()
    if os.geteuid() != 0:  # root opens everything
        shut.chmod(0)
        with pytest.raises(SetupError, match=r"may not open .*uhid: run the receiver as root, or give your user"):
            uhid.check_available()


@pytest.mark.skipif(sys.platform == "win32", reason="the kernel names a device with colons, which no file has there")
def test_the_hidraw_node_is_found_by_the_pads_physical_path(tmp_path):
    for name, phys, nodes in (
        ("0003:18D1:9400.0001", "usb-0000:00:14.0-1/input1", ["hidraw2"]),
        ("0003:18D1:9400.0002", "stadiacast/c0", ["hidraw5"]),
        ("0003:18D1:9400.0003", "stadiacast/c1", []),
    ):
        device = tmp_path / name
        (device / "hidraw").mkdir(parents=True)
        for node in nodes:
            (device / "hidraw" / node).mkdir()
        (device / "uevent").write_text(f"HID_ID=0003:000018D1:00009400\nHID_PHYS={phys}\n")
    assert find_hidraw("stadiacast/c0", str(tmp_path)) == "/dev/hidraw5"
    assert find_hidraw("stadiacast/c1", str(tmp_path)) is None  # not made yet
    assert find_hidraw("stadiacast/c9", str(tmp_path)) is None


@pytest.fixture
def kernel(monkeypatch) -> Iterator[socket.socket]:
    """A socket pair in place of ``/dev/uhid``: the test holds the kernel's end and the pad opens
    the other. A sequenced-packet socket keeps each event whole, as the kernel's file does."""
    ours_end, pads_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    os.set_blocking(pads_end.fileno(), False)
    fd = os.dup(pads_end.fileno())
    pads_end.close()
    monkeypatch.setattr(uhid, "open_uhid", lambda: fd)
    monkeypatch.setattr(uhid, "find_hidraw", lambda phys, _root="": "/dev/hidraw9" if phys == PHYS else None)
    ours_end.settimeout(5)
    try:
        yield ours_end
    finally:
        ours_end.close()


#: The physical path of the pad these tests make, by which its nodes are found.
PHYS = f"stadiacast/{os.getpid()}/c0"


def _create_pad(kernel: socket.socket) -> uhid.UhidPad:
    """A pad made the way the receiver makes one, with the kernel's part played: it sends its
    create event and is ready once it is told the device has started."""
    made: list[uhid.UhidPad] = []
    creating = threading.Thread(target=lambda: made.append(uhid.UhidPad("c0")))
    creating.start()
    create = kernel.recv(EVENT_SIZE * 2)
    assert len(create) == EVENT_SIZE and struct.unpack_from("=I", create) == (UHID_CREATE2,)
    assert create[4:40] == b"Google LLC Stadia Controller rev. A\0"
    assert create[132:196].rstrip(b"\0") == PHYS.encode()  # the pad's own: this process, this controller
    assert struct.unpack_from("=II", create, 264) == (0x18D1, 0x9400)
    creating.join(0.2)
    assert creating.is_alive()  # nothing before a driver has taken the device
    kernel.send(_plain(UHID_START))
    creating.join(5)
    (pad,) = made
    return pad


REST = input_report(GamepadState())


def _reports(kernel: socket.socket, wanted: int, within: float = 3.0, after: bytes | None = None) -> list[bytes]:
    """The next ``wanted`` different input reports the pad writes; one written again counts once,
    and so does ``after``, the report it was writing before."""
    seen: list[bytes] = [] if after is None else [after]
    deadline = time.monotonic() + within
    while len(seen) < wanted + (after is not None):
        kernel.settimeout(max(0.01, deadline - time.monotonic()))
        event = kernel.recv(EVENT_SIZE * 2)
        assert struct.unpack_from("=IH", event) == (UHID_INPUT2, 10)
        if not seen or event[6:16] != seen[-1]:
            seen.append(event[6:16])
    return seen if after is None else seen[1:]


def _answer(kernel: socket.socket) -> bytes:
    """The pad's next event that is not an input report: it writes its state all the while."""
    while True:
        event = kernel.recv(EVENT_SIZE * 2)
        if struct.unpack_from("=I", event) != (UHID_INPUT2,):
            return event


linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="a socket pair that keeps each event whole, as Linux has it"
)


@linux_only
def test_the_pad_waits_for_its_start_answers_on_its_thread_and_reports(kernel):
    """The whole pad against a socket pair that plays the kernel: the device is created and
    started, a SET_REPORT is answered and its rumble handed on, a hidraw write too, input becomes
    INPUT2 events, and closing the file is the end of it."""
    pad = _create_pad(kernel)
    assert pad.device == "/dev/hidraw9"
    assert _reports(kernel, 1) == [REST]  # at rest from its start, before any input

    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    kernel.send(_set_report(7, b"\x05\x00\x80\x00\x40"))
    assert _reply(_answer(kernel)) == (UHID_SET_REPORT_REPLY, 7, 0)
    wait_until(lambda: levels == [(0x80, 0x40)])  # one level at a time: the pad hands on the newest
    kernel.send(_output(b"\x05\x00\x00\x00\x00"))
    wait_until(lambda: levels == [(0x80, 0x40), (0, 0)])

    pad.apply_state(GamepadState(buttons=int(Button.ASSISTANT)))
    pad.apply_state(GamepadState(buttons=int(Button.ASSISTANT)))  # nothing changed: nothing written
    pad.reset()
    assert _reports(kernel, 2, after=REST) == [input_report(GamepadState(buttons=int(Button.ASSISTANT))), REST]

    pad.close()
    pad.close()  # once is enough
    kernel.settimeout(5)
    while kernel.recv(EVENT_SIZE * 2):  # the state repeated meanwhile, then the end
        pass


@linux_only
def test_the_state_is_written_again_while_nothing_changes_and_a_new_start_names_its_node(kernel, monkeypatch):
    """The kernel drops the input written while a HID driver takes the device, a new driver's
    event device knows no report, and a program that opens the hidraw node has seen none: the
    pad repeats its state at a steady pace, so each has it a moment later. A change is written
    at once all the same, and a pad that was given no state says it is at rest."""
    monkeypatch.setattr(uhid.UhidPad, "REFRESH", 0.02)
    pad = _create_pad(kernel)

    def written(wait: float = 0.0) -> list[bytes]:
        """The reports the pad has written since the last look, after waiting up to that long for
        the first of them."""
        seen: list[bytes] = []
        kernel.settimeout(wait)  # 0 is no waiting at all
        while True:
            try:
                event = kernel.recv(EVENT_SIZE * 2)
            except (TimeoutError, BlockingIOError):
                return seen
            assert struct.unpack_from("=IH", event) == (UHID_INPUT2, 10)
            seen.append(event[6:16])
            kernel.settimeout(0)

    def repeats(report: bytes, count: int) -> None:
        """Wait until the pad has written that report that many times more, and nothing else."""
        seen: list[bytes] = []
        deadline = time.monotonic() + 5
        while len(seen) < count:
            assert time.monotonic() < deadline, f"{len(seen)} of {count} repeats in time"
            seen += written(0.5)
        assert set(seen) == {report}

    repeats(REST, 3)  # no input yet: at rest, as its controller is
    held = input_report(GamepadState(buttons=int(Button.A), lx=32767))
    pad.apply_state(GamepadState(buttons=int(Button.A), lx=32767))
    repeats(held, 5)

    pad.apply_state(GamepadState(buttons=int(Button.B)))
    changed = input_report(GamepadState(buttons=int(Button.B)))
    assert written()[-1] == changed  # written at once, not at the next turn
    repeats(changed, 3)

    assert pad.device == "/dev/hidraw9"
    monkeypatch.setattr(uhid, "find_hidraw", lambda _phys, _root="": "/dev/hidraw12")
    kernel.send(_plain(UHID_STOP))
    kernel.send(_plain(UHID_START))  # another driver has the device, with a node of its own
    wait_until(lambda: pad.device == "/dev/hidraw12")
    written()
    repeats(changed, 2)  # and is written the state
    pad.close()


@linux_only
def test_a_pad_no_driver_takes_says_so(kernel, monkeypatch):
    monkeypatch.setattr(uhid.UhidPad, "START_TIMEOUT", 0.1)
    with pytest.raises(SetupError, match=r"did not start the virtual pad within 0.1 s: no HID driver took it"):
        uhid.UhidPad("c0")
    kernel.recv(EVENT_SIZE * 2)  # the create event
    assert kernel.recv(EVENT_SIZE * 2) == b""  # and the file closed again


@linux_only
def test_a_pad_without_a_hidraw_node_names_none_and_one_the_kernel_dropped_says_so(kernel, monkeypatch):
    monkeypatch.setattr(uhid, "find_hidraw", lambda _phys, _root="": None)  # a kernel built without hidraw
    monkeypatch.setattr(uhid.UhidPad, "NODE_WAIT", 0.05)
    pad = _create_pad(kernel)
    assert pad.device is None
    pad.apply_state(GamepadState(buttons=int(Button.A)))
    assert _reports(kernel, 1, after=REST) == [input_report(GamepadState(buttons=int(Button.A)))]
    kernel.close()  # the device is gone under the pad

    def press() -> None:
        for _ in range(100):  # the first write may still land in the socket
            pad.apply_state(GamepadState(buttons=int(Button.B)))
            pad.apply_state(GamepadState())
            time.sleep(0.01)

    with pytest.raises(SetupError, match="the virtual pad no longer takes reports"):
        press()
    pad.close()


@linux_only
def test_a_rumble_left_running_stops_when_the_last_program_closes_the_pad(kernel):
    pad = _create_pad(kernel)
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    kernel.send(_plain(UHID_OPEN))
    kernel.send(_output(b"\x05\x00\xc0\x00\x20"))
    wait_until(lambda: levels == [(0xC0, 0x20)])
    kernel.send(_plain(UHID_CLOSE))  # the program is gone, its rumble still on
    wait_until(lambda: levels == [(0xC0, 0x20), (0, 0)])
    pad.close()


# ---- in the receiver -------------------------------------------------------------------------
class _FakeUhidPad:
    """A Stadia pad with a hidraw node, in place of the kernel's."""

    created: ClassVar[list[_FakeUhidPad]] = []
    slot = None
    takes_state = True

    def __init__(self, label: str) -> None:
        self.label = label
        self.device = f"/dev/hidraw{40 + len(self.created)}"
        self.states: list[GamepadState] = []
        self.rumble_cb: Any = None
        self.closed = False
        self.created.append(self)

    def apply_state(self, state: GamepadState) -> None:
        self.states.append(state)

    def apply(self, _xinput: dict[str, int]) -> None:
        raise AssertionError("a pad that takes the state is given the state")

    def reset(self) -> None:
        raise AssertionError("a pad that lives for one boot is closed, not reset")

    def on_rumble(self, callback) -> None:
        self.rumble_cb = callback

    def close(self) -> None:
        self.closed = True


def test_the_receiver_says_at_start_what_the_pad_lacks(tmp_path, monkeypatch):
    def refuse() -> None:
        raise SetupError("--gamepad uhid may not open /dev/uhid: run the receiver as root")

    monkeypatch.setattr(virtual_gamepad, "check_uhid_available", refuse)

    async def run() -> None:
        rx = make_receiver(tmp_path, gamepad="uhid")
        try:
            with pytest.raises(SetupError, match="may not open /dev/uhid"):
                await rx.start()
        finally:
            await rx.stop()

    asyncio.run(run())


def test_a_stadia_pad_gets_the_whole_state_and_the_games_rumble_comes_back(tmp_path, monkeypatch):
    """``--gamepad uhid`` end to end with the kernel's part replaced: the pad is given the state with
    Assistant and Capture (which XInput's fields would drop), deadzones applied, names its hidraw
    node, hands the game's rumble to the controller and goes with its controller."""
    monkeypatch.setattr(virtual_gamepad, "check_uhid_available", lambda: None)
    monkeypatch.setattr(virtual_gamepad, "UhidPad", _FakeUhidPad)
    _FakeUhidPad.created.clear()

    async def run() -> None:
        rx = make_receiver(tmp_path, gamepad="uhid", start_game_delay=0.05)
        await rx.start()
        assert isinstance(rx.gamepads, VirtualGamepads) and rx.gamepads.kind == "uhid" and rx.cfg.start_game == "auto"
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        rumbles: list[tuple[int, int]] = []
        cs.game.set_rumble = lambda left, right, _ms=None, _framing=None: rumbles.append((left, right)) or 7  # type: ignore[method-assign]
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            held = int(Button.ASSISTANT | Button.CAPTURE | Button.A)
            assert fc.send_input_rpc(buttons=held, lx=1000, ly=-1000, rx=30000, lt=200)
            await wait_for(lambda: _FakeUhidPad.created and _FakeUhidPad.created[0].states)
            pad = _FakeUhidPad.created[0]
            state = pad.states[-1]
            assert state.buttons == held and state.l2 == 200
            assert (state.lx, state.ly) == (0, 0) and state.rx > 0  # inside, and past, XInput's deadzones
            assert rx.status()["gamepads"] == {
                "c0": {"slot": None, "frames": 1, "rumbles": 0, "kind": "uhid", "device": "/dev/hidraw40"}
            }
            created = [r for r in rx.log.records if r["event"] == "gamepad_created"]
            assert [(r["kind"], r["device"]) for r in created] == [("uhid", "/dev/hidraw40")]

            pad.rumble_cb(255, 64)
            await wait_for(lambda: rumbles)
            assert rumbles == [(65535, 64 * 257)]

            fc.power_off()
            await wait_for(lambda: single(rx).session["state"] == "dtls-closed")
            assert pad.closed and rx.status()["gamepads"] == {}
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())
