"""Linux virtual Stadia Controllers through the kernel's uhid device.

``--gamepad uhid`` gives each controller a HID device with the wired Stadia Controller's identity
(USB ``18d1:9400``, its name, and the controller's serial number as the pad's own) and its input
report (report 3), instead of the Xbox 360 pad of ``--gamepad uinput``. A program then reads it the
way it reads that controller:

- Steam and SDL 3 through the pad's ``hidraw`` node, with their Stadia driver. It has the two
  buttons an Xbox pad has no place for, Assistant and Capture (SDL 3's ``misc2`` and ``misc1``),
  and writes its rumble to the same node. SDL 3 gives a game the pad's serial number, so a game
  can know one controller from another whatever order they wake in.
- SDL 2, whose HID code passes over a device that has no USB parent, browsers and everything
  else through the event device the kernel makes of the report descriptor, with the mapping they
  keep for these ids.

So the event device has to be the wired controller's too. The report descriptor is written here,
not taken from the controller, to describe the report layout SDL's driver reads with the button
usages and the ranges that give the event device the controller's key codes and axes: the eleven
keys of an Xbox pad (``BTN_A`` to ``BTN_THUMBR``), Assistant and Capture on
``BTN_TRIGGER_HAPPY1`` and ``2``, the triggers' digital bits on ``4`` and ``3``, sticks from 1 to
255. Those are the button numbers SDL's mapping database and Chromium know the controller by.

A game's rumble arrives on the uhid file in one of two ways, both as output report 5 (two 16-bit
motor levels, strong first): Steam and SDL 3 write it to the ``hidraw`` node (``UHID_OUTPUT``),
and the kernel's ``hid-google-stadiaff`` driver, which binds to these ids and gives the event
device force feedback, sends it as a SET_REPORT request (``UHID_SET_REPORT``), which waits for an
answer. Every request is answered on a thread of the pad's own, so none waits for the event loop.
A rumble is stopped when the last program that had the ``hidraw`` node open lets go of it, since
one that was killed while its rumble ran is not there to stop it.

:class:`UhidPad` is that pad, and what a pad of another identity is made of:
:mod:`stadiacast.dualsense_gamepad` gives it the reports of a DualSense. The packing of the
kernel's events and the report encoding are plain Python and run in the tests of every platform;
the pad itself needs Linux.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import glob
import os
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from stadiacast.errors import SetupError
from stadiacast.gamepad import XINPUT_GUIDE, XINPUT_SHARED_BITS, Button, GamepadState

# ---- linux/uhid.h ----------------------------------------------------------------------------
UHID_DESTROY = 1
UHID_START = 2
UHID_STOP = 3
UHID_OPEN = 4
UHID_CLOSE = 5
UHID_OUTPUT = 6
UHID_GET_REPORT = 9
UHID_GET_REPORT_REPLY = 10
UHID_CREATE2 = 11
UHID_INPUT2 = 12
UHID_SET_REPORT = 13
UHID_SET_REPORT_REPLY = 14
UHID_FEATURE_REPORT, UHID_OUTPUT_REPORT, UHID_INPUT_REPORT = 0, 1, 2
UHID_DATA_MAX = 4096
HID_MAX_DESCRIPTOR_SIZE = 4096
BUS_USB = 0x03

#: ``struct uhid_event`` is packed, in the machine's byte order (``=``): a ``__u32`` type and a
#: union whose widest member is ``struct uhid_create2_req`` (names, ids, then the descriptor). The
#: kernel hands out whole events. It takes a shorter one as if zeros followed, so an input report
#: and a reply are written as their head and what they carry.
CREATE2 = struct.Struct(f"=I128s64s64sHHIIII{HID_MAX_DESCRIPTOR_SIZE}s")
EVENT_SIZE = CREATE2.size
INPUT2 = struct.Struct("=IH")  # uhid_input2_req: the size, then that many bytes of report
OUTPUT = struct.Struct(f"=I{UHID_DATA_MAX}sHB")  # uhid_output_req: the data comes first
GET_REPORT = struct.Struct("=IIBB")
GET_REPORT_REPLY = struct.Struct("=IIHH")  # the request, an error, the size, then the report
SET_REPORT = struct.Struct(f"=IIBBH{UHID_DATA_MAX}s")
SET_REPORT_REPLY = struct.Struct("=IIH")

UHID_PATH = "/dev/uhid"

# ---- what the pad says it is -----------------------------------------------------------------
#: The wired controller's ids, and its name as the kernel has it over USB: the manufacturer and
#: the product the controller itself states.
PAD_NAME = "Google LLC Stadia Controller rev. A"
PAD_VENDOR, PAD_PRODUCT, PAD_VERSION = 0x18D1, 0x9400, 0x0100
INPUT_REPORT_ID = 3
RUMBLE_REPORT_ID = 5

#: The report's buttons: (byte, mask) per :class:`~stadiacast.gamepad.Button`, where SDL's Stadia
#: driver reads them. L2 and R2 are digital copies of the triggers (see ``TRIGGER_THRESHOLD``).
BUTTON_BITS: dict[Button, tuple[int, int]] = {
    Button.CAPTURE: (2, 0x01),
    Button.ASSISTANT: (2, 0x02),
    Button.STADIA: (2, 0x10),
    Button.MENU: (2, 0x20),
    Button.OPTIONS: (2, 0x40),
    Button.R3: (2, 0x80),
    Button.L3: (3, 0x01),
    Button.R1: (3, 0x02),
    Button.L1: (3, 0x04),
    Button.Y: (3, 0x08),
    Button.X: (3, 0x10),
    Button.B: (3, 0x20),
    Button.A: (3, 0x40),
}
L2_BIT, R2_BIT = (2, 0x04), (2, 0x08)
#: A trigger past this (of 255) also presses its digital bit: XInput's own threshold
#: (``XINPUT_GAMEPAD_TRIGGER_THRESHOLD``).
TRIGGER_THRESHOLD = 30
#: The hat's value per (x, y) of the D-pad, y down: 0 is up, then clockwise; 8 is released.
HAT: dict[tuple[int, int], int] = {
    (0, -1): 0,
    (1, -1): 1,
    (1, 0): 2,
    (1, 1): 3,
    (0, 1): 4,
    (-1, 1): 5,
    (-1, 0): 6,
    (-1, -1): 7,
}
HAT_RELEASED = 8

# Button page usages for the report's bits, in report order. On a gamepad the kernel maps usage n
# to BTN_GAMEPAD + n - 1 for n <= 16 and to BTN_TRIGGER_HAPPY + n - 17 above (hid-input.c). These
# give the event device the wired controller's key codes, and so its button numbers, which count
# the codes in rising order: a key more or less, or two swapped, and every program with a mapping
# for these ids reads the wrong buttons.
_USAGES_BYTE2 = (
    18,  # Capture   -> BTN_TRIGGER_HAPPY2
    17,  # Assistant -> BTN_TRIGGER_HAPPY1
    20,  # L2        -> BTN_TRIGGER_HAPPY4
    19,  # R2        -> BTN_TRIGGER_HAPPY3
    13,  # Stadia    -> BTN_MODE
    12,  # Menu      -> BTN_START
    11,  # Options   -> BTN_SELECT
    15,  # R3        -> BTN_THUMBR
)
_USAGES_BYTE3 = (
    14,  # L3 -> BTN_THUMBL
    8,  # R1  -> BTN_TR
    7,  # L1  -> BTN_TL
    5,  # Y   -> BTN_Y, as on an Xbox pad (the code the kernel also calls BTN_WEST)
    4,  # X   -> BTN_X (also BTN_NORTH)
    2,  # B   -> BTN_B
    1,  # A   -> BTN_A
)

#: Input report 3 (10 bytes: id, hat, two button bytes, four stick bytes, two trigger bytes) and
#: output report 5 (two 16-bit motor levels, which ``hid-google-stadiaff`` requires to bind).
REPORT_DESCRIPTOR = bytes(
    [
        0x05, 0x01,  # Usage Page (Generic Desktop)
        0x09, 0x05,  # Usage (Game Pad)
        0xA1, 0x01,  # Collection (Application)
        0x85, INPUT_REPORT_ID,  # Report ID (3)
        0x09, 0x39,  # Usage (Hat Switch)
        0x15, 0x00,  # Logical Minimum (0)
        0x25, 0x07,  # Logical Maximum (7)
        0x75, 0x04,  # Report Size (4)
        0x95, 0x01,  # Report Count (1)
        0x81, 0x42,  # Input (Data, Variable, Absolute, Null State)
        0x81, 0x03,  # Input (Constant): 4 bits of padding
        0x05, 0x09,  # Usage Page (Button)
        *[b for usage in _USAGES_BYTE2 + _USAGES_BYTE3 for b in (0x09, usage)],  # Usage (n), 15 times
        0x25, 0x01,  # Logical Maximum (1)
        0x75, 0x01,  # Report Size (1)
        0x95, 0x0F,  # Report Count (15)
        0x81, 0x02,  # Input (Data, Variable, Absolute)
        0x95, 0x01,  # Report Count (1)
        0x81, 0x03,  # Input (Constant): 1 bit of padding
        0x05, 0x01,  # Usage Page (Generic Desktop)
        0x09, 0x30,  # Usage (X)
        0x09, 0x31,  # Usage (Y)
        0x09, 0x32,  # Usage (Z)
        0x09, 0x35,  # Usage (Rz)
        0x15, 0x01,  # Logical Minimum (1): a stick rests at 128, the middle of 1..255
        0x26, 0xFF, 0x00,  # Logical Maximum (255)
        0x75, 0x08,  # Report Size (8)
        0x95, 0x04,  # Report Count (4)
        0x81, 0x02,  # Input (Data, Variable, Absolute)
        0x05, 0x02,  # Usage Page (Simulation Controls)
        0x09, 0xC5,  # Usage (Brake): L2
        0x09, 0xC4,  # Usage (Accelerator): R2
        0x15, 0x00,  # Logical Minimum (0)
        0x95, 0x02,  # Report Count (2)
        0x81, 0x02,  # Input (Data, Variable, Absolute)
        0x85, RUMBLE_REPORT_ID,  # Report ID (5)
        0x06, 0x00, 0xFF,  # Usage Page (Vendor Defined 0xFF00)
        0x09, 0x01,  # Usage (1)
        0x27, 0xFF, 0xFF, 0x00, 0x00,  # Logical Maximum (65535)
        0x75, 0x10,  # Report Size (16)
        0x95, 0x02,  # Report Count (2): strong motor, weak motor
        0x91, 0x02,  # Output (Data, Variable, Absolute)
        0xC0,  # End Collection
    ]
)  # fmt: skip


# ---- the report ------------------------------------------------------------------------------
def stick_byte(value: int, *, invert: bool = False) -> int:
    """A stick axis (-32768..32767, Y up) as the report's byte: 1..255 with 128 at rest, Y down."""
    value = max(-32768, min(32767, value))
    scaled = round(value * 127 / 32767)
    return max(1, min(255, 128 - scaled if invert else 128 + scaled))


def hat_value(buttons: int) -> int:
    """The D-pad bits as the report's hat; opposite directions held together cancel out."""
    x = (1 if buttons & Button.DPAD_RIGHT else 0) - (1 if buttons & Button.DPAD_LEFT else 0)
    y = (1 if buttons & Button.DPAD_DOWN else 0) - (1 if buttons & Button.DPAD_UP else 0)
    return HAT.get((x, y), HAT_RELEASED)


def input_report(state: GamepadState) -> bytes:
    """Input report 3 for a state: what a wired Stadia Controller sends for the same controls."""
    data = bytearray(10)
    data[0] = INPUT_REPORT_ID
    data[1] = hat_value(state.buttons)
    for button, (index, mask) in BUTTON_BITS.items():
        if state.buttons & button:
            data[index] |= mask
    l2, r2 = max(0, min(255, state.l2)), max(0, min(255, state.r2))
    for value, (index, mask) in ((l2, L2_BIT), (r2, R2_BIT)):
        if value >= TRIGGER_THRESHOLD:
            data[index] |= mask
    data[4] = stick_byte(state.lx)
    data[5] = stick_byte(state.ly, invert=True)
    data[6] = stick_byte(state.rx)
    data[7] = stick_byte(state.ry, invert=True)
    data[8], data[9] = l2, r2
    return bytes(data)


def state_from_xinput(xinput: Mapping[str, int]) -> GamepadState:
    """An XInput field set read back into a state, for a caller that only has those fields (the
    Assistant and Capture buttons are lost on the way, as XInput has none)."""
    mask = xinput["wButtons"]
    buttons = mask & XINPUT_SHARED_BITS
    if mask & XINPUT_GUIDE:
        buttons |= Button.STADIA
    return GamepadState(
        buttons=buttons,
        lx=xinput["sThumbLX"],
        ly=xinput["sThumbLY"],
        rx=xinput["sThumbRX"],
        ry=xinput["sThumbRY"],
        l2=xinput["bLeftTrigger"],
        r2=xinput["bRightTrigger"],
    )


def rumble_levels(data: bytes) -> tuple[int, int] | None:
    """The strong and the weak motor level (0..65535) of output report 5, or ``None`` for any
    other report. Both paths it arrives on, a ``hidraw`` write and the kernel's SET_REPORT, carry
    the report id as the first byte. A report's own numbers are little-endian on every machine."""
    if len(data) < 5 or data[0] != RUMBLE_REPORT_ID:
        return None
    strong, weak = struct.unpack_from("<HH", data, 1)
    return strong, weak


# ---- the kernel's events ---------------------------------------------------------------------
def _padded(event: bytes) -> bytes:
    return event.ljust(EVENT_SIZE, b"\0")


def create2_event(
    name: str,
    phys: str,
    uniq: str,
    descriptor: bytes = REPORT_DESCRIPTOR,
    *,
    vendor: int = PAD_VENDOR,
    product: int = PAD_PRODUCT,
    version: int = PAD_VERSION,
) -> bytes:
    """``UHID_CREATE2`` for a USB HID pad."""
    if len(descriptor) > HID_MAX_DESCRIPTOR_SIZE:
        raise ValueError("report descriptor too long")
    return CREATE2.pack(
        UHID_CREATE2,
        name.encode()[:127],
        phys.encode()[:63],
        uniq.encode()[:63],
        len(descriptor),
        BUS_USB,
        vendor,
        product,
        version,
        0,
        descriptor,
    )


def input2_event(report: bytes) -> bytes:
    return INPUT2.pack(UHID_INPUT2, len(report)) + report


def set_report_reply_event(request: int, err: int) -> bytes:
    return SET_REPORT_REPLY.pack(UHID_SET_REPORT_REPLY, request, err)


def get_report_reply_event(request: int, err: int, data: bytes = b"") -> bytes:
    if len(data) > UHID_DATA_MAX:
        raise ValueError("feature report reply too long")
    return GET_REPORT_REPLY.pack(UHID_GET_REPORT_REPLY, request, err, len(data)) + data


def event_type(event: bytes) -> int | None:
    return struct.unpack_from("=I", event)[0] if len(event) >= 4 else None


def output_request(event: bytes) -> tuple[bytes, int]:
    """The data and the report type of a ``UHID_OUTPUT`` event."""
    _type, data, size, rtype = OUTPUT.unpack_from(_padded(event))
    return data[: min(size, UHID_DATA_MAX)], rtype


def get_report_request(event: bytes) -> tuple[int, int, int]:
    """The request id, report number and report type of a ``UHID_GET_REPORT`` event."""
    _type, request, rnum, rtype = GET_REPORT.unpack_from(_padded(event))
    return request, rnum, rtype


def set_report_request(event: bytes) -> tuple[int, int, int, bytes]:
    """The request id, report number, report type and data of a ``UHID_SET_REPORT`` event."""
    _type, request, rnum, rtype, size, data = SET_REPORT.unpack_from(_padded(event))
    return request, rnum, rtype, data[: min(size, UHID_DATA_MAX)]


class Requests:
    """What the pad does with each event the kernel sends: the motor levels it last asked for, the
    answer a request waits for, and how often a driver took the device and let go of it. Kept
    apart from the device so it runs anywhere. A pad of another kind says what its output and its
    feature reports are (:meth:`output`, :meth:`feature`)."""

    def __init__(self) -> None:
        self.levels = (0, 0)  # strong, weak: 0..65535
        #: ``UHID_START``: a HID driver has taken the device and is making its nodes. More than
        #: once when the kernel gives the device to another driver.
        self.starts = 0
        #: ``UHID_STOP``: a driver has let go of the device. As many as the starts while none has it.
        self.stops = 0

    def output(self, data: bytes) -> bool:
        """An output report, whichever way it came: take the motor levels it sets. ``False`` for
        a report that is none of the pad's."""
        levels = rumble_levels(data)
        if levels is None:
            return False
        self.levels = levels
        return True

    def feature(self, number: int) -> bytes | None:  # noqa: ARG002 (a pad with feature reports has a use for it)
        """The feature report a driver or a program asks for. This pad has none to read back."""
        return None

    def handle(self, event: bytes) -> bytes | None:
        """One event read from the uhid file; returns the reply to write, if it needs one."""
        kind = event_type(event)
        if kind == UHID_OUTPUT:
            data, rtype = output_request(event)
            if rtype == UHID_OUTPUT_REPORT:
                self.output(data)
            return None
        if kind == UHID_SET_REPORT:
            request, _rnum, rtype, data = set_report_request(event)
            taken = rtype == UHID_OUTPUT_REPORT and self.output(data)
            return set_report_reply_event(request, 0 if taken else errno.EIO)
        if kind == UHID_GET_REPORT:
            request, number, rtype = get_report_request(event)
            report = self.feature(number) if rtype == UHID_FEATURE_REPORT else None
            if report is None:
                return get_report_reply_event(request, errno.EIO)
            return get_report_reply_event(request, 0, report)
        if kind == UHID_START:
            self.starts += 1
        elif kind in (UHID_CLOSE, UHID_STOP):
            # Nobody has the pad's nodes open any more, or the driver has let go of the device:
            # nobody is left to stop a rumble (one played through the event device is stopped by
            # the kernel). Where a driver holds the device open itself no close comes, and the
            # pad's own watch of its hidraw node tells it instead (:class:`NodeWatch`).
            self.levels = (0, 0)
            self.stops += kind == UHID_STOP
        return None


# ---- the device ------------------------------------------------------------------------------
def open_uhid(flag: str = "--gamepad uhid") -> int:
    """Open the kernel's uhid device, or say in a :class:`SetupError` what keeps it shut."""
    if not sys.platform.startswith("linux"):
        raise SetupError(f"{flag} is Linux only: it creates the pad through the kernel's uhid device")
    try:
        return os.open(UHID_PATH, os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        raise SetupError(
            f"{flag} needs the kernel's uhid device and found none ({UHID_PATH}): load it with `sudo modprobe uhid`"
        ) from None
    except PermissionError:
        raise SetupError(
            f"{flag} may not open {UHID_PATH}: run the receiver as root, or give your user the device"
            " with the udev rule in the README (Virtual Stadia Controller on Linux)"
        ) from None
    except OSError as e:
        raise SetupError(f"{flag} cannot open {UHID_PATH}: {e}") from e


def check_available(flag: str = "--gamepad uhid") -> None:
    """Fail at start-up, in plain words, where a pad could not be created later."""
    os.close(open_uhid(flag))


def find_hidraw(phys: str, root: str = "/sys/devices/virtual/misc/uhid") -> str | None:
    """The ``hidraw`` node of the uhid device whose physical path is ``phys``, once the kernel has
    made it."""
    for uevent in sorted(glob.glob(os.path.join(root, "*", "uevent"))):
        try:
            with open(uevent, encoding="utf-8") as f:
                if f"HID_PHYS={phys}" not in f.read().splitlines():
                    continue
            nodes = sorted(os.listdir(os.path.join(os.path.dirname(uevent), "hidraw")))
        except OSError:
            continue
        return next((f"/dev/{node}" for node in nodes if node.startswith("hidraw")), None)
    return None


# ---- who has the pad's node open -------------------------------------------------------------
IN_CLOSE_WRITE, IN_CLOSE_NOWRITE, IN_OPEN = 0x08, 0x10, 0x20
#: ``struct inotify_event``: the watch, what happened, a cookie, and the length of a name that
#: follows (none for a watch on a file).
NODE_EVENT = struct.Struct("=iIII")


def node_events(data: bytes, watch: int) -> tuple[bool, bool]:
    """Whether the watched node was opened and whether it was let go of, in what was read from an
    inotify file. The kernel makes one event of several alike that wait unread, so they are not
    counted. Events of another watch, an earlier node's, are none of this one's."""
    opened = closed = False
    at = 0
    while at + NODE_EVENT.size <= len(data):
        which, what, _cookie, name = NODE_EVENT.unpack_from(data, at)
        at += NODE_EVENT.size + name
        if which == watch:
            opened = opened or bool(what & IN_OPEN)
            closed = closed or bool(what & (IN_CLOSE_WRITE | IN_CLOSE_NOWRITE))
    return opened, closed


def node_is_held(node: str, proc: str = "/proc") -> bool:
    """Whether a process has ``node`` open, among those this one may look into: its own user's,
    or all of them for root. One it may not look into counts for nothing. An open file is read
    as the link Linux shows it as, and only one of the node's name is asked what it is: a file
    on a network that does not answer is never touched."""
    try:
        wanted = os.stat(node)
    except OSError:
        return False
    name = os.path.basename(node)
    try:
        processes = [entry.path for entry in os.scandir(proc) if entry.name.isdigit()]
    except OSError:
        return False
    for process in processes:
        try:
            files = [entry.path for entry in os.scandir(f"{process}/fd")]
        except OSError:  # another user's process, or one that has just ended
            continue
        for file in files:
            try:
                if os.path.basename(os.readlink(file)) == name and os.path.samestat(os.stat(file), wanted):
                    return True
            except OSError:
                continue
    return False


class NodeWatch:
    """What programs do with a pad's ``hidraw`` node, as the kernel's inotify tells it: when one
    opens the node and when one lets go of it, killed or not.

    uhid has a word of its own for that (``UHID_OPEN``, ``UHID_CLOSE``), which counts a HID driver
    among those that have the device open. The kernel's PlayStation driver holds its device for as
    long as it has it, and no close ever comes. A node that cannot be watched (no right to read
    it, no inotify) is one nothing is heard of.
    """

    def __init__(self) -> None:
        self.fd = -1
        self._watch = -1
        self._libc: Any = None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.inotify_init1(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        except (OSError, AttributeError, TypeError):  # a system without inotify (Windows has no such library)
            return
        if fd >= 0:
            self.fd, self._libc = fd, libc

    @property
    def watching(self) -> bool:
        return self._watch >= 0

    def watch(self, node: str | None) -> bool:
        """Watch ``node`` from now on, in place of the node watched before; whether it is watched.
        A node that udev has not yet given to the receiver's user is not: try again."""
        if self._libc is None:
            return False
        if self._watch >= 0:
            self._libc.inotify_rm_watch(self.fd, self._watch)
        self._watch = -1
        if node is not None:
            mask = IN_OPEN | IN_CLOSE_WRITE | IN_CLOSE_NOWRITE
            self._watch = self._libc.inotify_add_watch(self.fd, os.fsencode(node), mask)
        return self._watch >= 0

    def read(self) -> tuple[bool, bool]:
        """What happened since the last look: whether a program opened the node, and whether one
        let go of it."""
        opened = closed = False
        while self.fd >= 0:
            try:
                data = os.read(self.fd, 4096)
            except OSError:  # nothing more to read
                break
            if not data:
                break
            more_opened, more_closed = node_events(data, self._watch)
            opened, closed = opened or more_opened, closed or more_closed
        return opened, closed

    def close(self) -> None:
        if self.fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self.fd)
            self.fd = -1


class UhidPad:
    """One virtual pad made through uhid: a wired Stadia Controller, or what a subclass says it is
    (its name, ids and report descriptor, its reports and its requests). The kernel's requests
    are answered on a thread of the pad's own."""

    slot: int | None = None  # Linux has no XInput player slot
    #: Takes the whole :class:`GamepadState` (:meth:`apply_state`): XInput's fields have no room for
    #: the Assistant and Capture buttons.
    takes_state = True
    #: Closing the uhid file is what removes the device, and the kernel takes a tenth of a second
    #: or more over it: :class:`~stadiacast.virtual_gamepad.VirtualGamepads` closes the pad in a
    #: worker thread.
    close_blocks = True
    #: The flag that asks for this pad, for what it says when it cannot be made.
    FLAG = "--gamepad uhid"
    #: What the pad says it is.
    NAME = PAD_NAME
    IDS = (PAD_VENDOR, PAD_PRODUCT, PAD_VERSION)
    DESCRIPTOR = REPORT_DESCRIPTOR
    #: How long the kernel may take to start the device after it is created.
    START_TIMEOUT = 5.0
    #: The state is written again this often while nothing changes (seconds): see
    #: :meth:`_keep_up`. The kernel passes a report that changes nothing on to the programs that
    #: read the hidraw node, and makes no event of it on the event device.
    REFRESH = 0.1
    #: For this long after a program opened the hidraw node, or a driver took the device, the
    #: state is written this often instead (seconds): the one has seen no report yet and the
    #: other dropped what was written while it took the device.
    HURRY_FOR, HURRY = 0.05, 0.002
    #: How long after a start the hidraw node is looked for, and how often (seconds).
    NODE_WAIT, NODE_LOOK = 1.0, 0.02
    #: How long the device may be without a driver before the pad says it is none (seconds). A
    #: change of driver takes a few hundredths of a second.
    DRIVER_WAIT = 2.0
    label: str
    uniq: str  # what the pad states as its own serial number (``HID_UNIQ``)
    device: str | None  # the hidraw node, e.g. /dev/hidraw7
    _phys: str
    _poll: Any
    _requests: Requests
    _opens: NodeWatch
    _state: GamepadState
    _report: bytes
    _report_lock: threading.Lock  # the state, its report as last written, and the writing of it
    _written: float  # when it was last written, or would have been, on the monotonic clock
    _hurry_until: float
    _starts: int
    _node_until: float
    _node_looked: float
    _node_is_old: bool  # a driver has taken the device since the node was found
    _driverless: float | None  # since when no driver has the device
    _sent: tuple[int, int]
    _callback: Callable[[int, int], None] | None
    _error: str | None
    _error_callback: Callable[[str], None] | None
    _lock: threading.Lock  # the callbacks, what was last sent to the one and said to the other
    _closed: bool
    _fd: int  # the uhid file; -1 once it is closed, as the two ends of the pipe below
    _wake_r: int
    _wake_w: int
    _thread: threading.Thread

    def __init__(self, label: str, uniq: str | None = None) -> None:
        if not sys.platform.startswith("linux"):
            raise SetupError(f"{self.FLAG} is Linux only: it creates the pad through the kernel's uhid device")
        import select

        self.label = label
        self.uniq = self._identity((uniq or "").strip() or label)
        self.device = None
        self._phys = f"stadiacast/{os.getpid()}/{label}"  # what the pad's nodes are found by: no other's
        self._requests = self._new_requests()
        # A pad is made when its controller connects, before any input: it says from the start
        # that it is at rest. An event device that has had no report reads its sticks as held in
        # a corner, and SDL's Stadia driver the D-pad as held up.
        self._state = GamepadState()
        self._report = self._encode(self._state)
        self._report_lock = threading.Lock()
        self._written = time.monotonic()
        self._hurry_until = 0.0
        self._starts = 0
        self._node_until = self._node_looked = 0.0
        self._node_is_old = False
        self._driverless = None
        self._sent = (0, 0)
        self._callback = None
        self._error = None
        self._error_callback = None
        self._lock = threading.Lock()
        self._closed = False
        self._fd = self._wake_r = self._wake_w = -1
        self._opens = NodeWatch()
        try:
            self._fd = open_uhid(self.FLAG)
            self._wake_r, self._wake_w = os.pipe()
            self._poll = select.poll()
            for fd in (self._fd, self._wake_r, self._opens.fd):
                if fd >= 0:
                    self._poll.register(fd, select.POLLIN)
            vendor, product, version = self.IDS
            created = create2_event(
                self.NAME, self._phys, self.uniq, self.DESCRIPTOR, vendor=vendor, product=product, version=version
            )
            try:
                os.write(self._fd, created)
            except OSError as e:
                raise SetupError(f"the kernel refused the virtual pad: {e}") from e
            self._wait_started()
            self._starts = self._requests.starts
            self._follow(self._find_node())
            self._hurry_until = time.monotonic() + self.HURRY_FOR  # as after any start: see _keep_up
            self._thread = threading.Thread(target=self._feedback_loop, name=f"uhid-{label}", daemon=True)
            self._thread.start()
        except BaseException:  # whatever it was: no file is left open, and so no device behind
            self._close_files()
            raise

    # ---- what a pad of another kind says otherwise -----------------------------------------
    def _identity(self, serial: str) -> str:
        """What the pad states as its own serial number, of its controller's (or, for want of
        one, of the controller's label): here that serial itself, cut to what the kernel takes."""
        return serial.encode()[:63].decode("utf-8", "ignore") or self.label

    def _new_requests(self) -> Requests:
        return Requests()

    def _encode(self, state: GamepadState) -> bytes:
        """The input report for a state, less what changes with every writing of it."""
        return input_report(state)

    def _stamp(self, report: bytes) -> bytes:
        """The report as it is written this time: a pad whose reports count themselves counts here."""
        return report

    # ---- the device --------------------------------------------------------------------------
    def _wait_started(self) -> None:
        """Wait until a HID driver has taken the device: the kernel has no device to give input to
        before, and no node to name."""
        deadline = time.monotonic() + self.START_TIMEOUT
        while not self._requests.starts:
            left = deadline - time.monotonic()
            if left <= 0:
                raise SetupError(
                    f"the kernel did not start the virtual pad within {self.START_TIMEOUT:g} s: no HID driver"
                    " took it (`sudo modprobe hid-generic` loads the kernel's)"
                )
            if self._poll.poll(int(left * 1000) + 1):
                self._read_events()

    def _find_node(self) -> str | None:
        """The hidraw node, which follows the start by a moment. A driver that asks for a report
        while it takes the device is answered meanwhile: it makes nothing of the pad before."""
        deadline = time.monotonic() + self.NODE_WAIT
        while True:
            node = find_hidraw(self._phys)
            if node is not None or time.monotonic() >= deadline:
                return node
            if self._poll.poll(10) and not self._read_events():
                return None  # the device is gone already, which the pad's thread will find too

    def _follow(self, node: str | None) -> None:
        self.device = node
        self._node_is_old = False
        self._opens.watch(node)

    def _close_files(self) -> None:
        """Close the pad's files, which removes the device. Under the lock a report is written
        under: a write must not meet a number that is by then another file's."""
        self._opens.close()
        with self._report_lock:
            files, self._fd, self._wake_r, self._wake_w = (self._fd, self._wake_r, self._wake_w), -1, -1, -1
        for fd in files:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)

    # ---- input ---------------------------------------------------------------------------
    def apply_state(self, state: GamepadState) -> None:
        with self._report_lock:
            report = self._encode(state)
            self._state = state
            if report == self._report:
                return
            try:
                os.write(self._fd, input2_event(self._stamp(report)))
            except OSError as e:
                raise SetupError(f"the virtual pad no longer takes reports: {e}") from e
            self._report = report
            self._written = time.monotonic()

    def _write_again(self) -> None:
        with self._report_lock:
            with contextlib.suppress(OSError):
                os.write(self._fd, input2_event(self._stamp(self._report)))
            self._written = time.monotonic()

    def apply(self, xinput: dict[str, int]) -> None:
        self.apply_state(state_from_xinput(xinput))

    def reset(self) -> None:
        self.apply_state(GamepadState())

    # ---- requests and rumble ---------------------------------------------------------------
    def on_rumble(self, callback: Callable[[int, int], None]) -> None:
        """``callback`` gets the two motor levels as XInput has them, 0..255, on the pad's thread."""
        with self._lock:
            self._callback = callback
            self._sent = (0, 0)
        with contextlib.suppress(OSError):
            os.write(self._wake_w, b"r")  # a game may have started rumble before anyone listened

    def on_error(self, callback: Callable[[str], None]) -> None:
        """``callback`` is told, on the pad's thread, when the pad has stopped being one."""
        with self._lock:
            self._error_callback = callback
            error = self._error
        if error is not None:
            callback(error)

    def _read_events(self) -> bool:
        """Answer every event waiting on the uhid file; ``False`` once the file is gone."""
        while True:
            try:
                event = os.read(self._fd, EVENT_SIZE)
            except BlockingIOError:
                return True
            except OSError:
                return False
            if not event:
                return False
            reply = self._requests.handle(event)
            if reply is not None:
                with contextlib.suppress(OSError):  # a request the kernel has given up waiting for
                    os.write(self._fd, reply)

    def _feedback_loop(self) -> None:
        """The pad's thread, which owns the pad's files: it closes them when it ends, so that
        nothing closes them under it. One that ends before the pad was closed has lost its
        device, and says so."""
        try:
            self._serve()
        finally:
            self._close_files()
            if not self._closed:
                self._fail("the virtual pad's device is gone")

    def _serve(self) -> None:
        while True:
            now = time.monotonic()
            due = self._written + (self.HURRY if now < self._hurry_until else self.REFRESH)  # the state's next turn
            if now < self._node_until:
                due = min(due, self._node_looked + self.NODE_LOOK)  # a new node is looked for meanwhile
            try:
                ready = self._poll.poll(max(0.0, due - now) * 1000)
            except OSError:
                return
            node_news = False
            for fd, _what in ready:
                if fd == self._wake_r:
                    if b"x" in os.read(fd, 16):  # close() asks the thread to end
                        return
                elif fd == self._opens.fd:
                    node_news = True
                elif not self._read_events():
                    return
            if node_news:
                self._node_news()
            self._keep_up()
            self._tell()

    def _unwatched(self) -> bool:
        """Whether the pad has a node that it could watch and does not."""
        return self.device is not None and self._opens.fd >= 0 and not self._opens.watching

    def _node_news(self) -> None:
        """A program opened the pad's hidraw node, or let go of it."""
        opened, closed = self._opens.read()
        if opened:
            self._hurry_until = time.monotonic() + self.HURRY_FOR
            self._write_again()
        if closed:
            self._stop_a_rumble_left_on()

    def _stop_a_rumble_left_on(self) -> None:
        """Stop the motors when a rumble is on and no program has the pad's hidraw node: the one
        that wrote it there is gone, killed perhaps, and has left it behind.

        The kernel's events are read first. What a program wrote before it let go of the node is
        waiting there by the time its leaving is heard, and the two come by different ways: a
        rumble read only afterwards would find nobody listening for the end of its program.

        A rumble of the kernel's own, played through the event device, is cut short here when
        an unrelated program closes the node meanwhile and nobody else has it."""
        self._read_events()
        if self._requests.levels != (0, 0) and self.device and not node_is_held(self.device):
            self._requests.levels = (0, 0)

    def _keep_up(self) -> None:
        """Write the state again when it has not changed for a while, follow the device's node and
        see that the device has a driver.

        The kernel has no word for when a report was lost or never seen, so the state is repeated
        (:attr:`REFRESH`), as many controllers repeat theirs on a wire, and for a moment much more
        often where it is known to be missing (:attr:`HURRY`):

        - A HID driver that is taking the device drops the input written meanwhile, and the writer
          is told nothing. Its event device then reads the sticks as held in a corner until a
          report arrives. Every start is a driver taking the device: the first pad after a boot is
          started twice, for the generic driver and then, once udev has loaded it, for the
          kernel's driver for these ids.
        - A program that opens the hidraw node sees only what is written from then on. SDL's Stadia
          driver starts from a state of its own in which the D-pad is held up, and misses that
          press until it has read a report.

        A start also makes a new hidraw node, which is looked up again, and watched again even
        where it has the old one's name. A node that cannot be watched is tried again at every
        turn: a new one is not there yet when the kernel first names it, and then it is root's
        alone until udev has given it to the receiver's user, a moment later. A device that a
        driver let go of and none took is no pad: the kernel's driver refused it, which its log
        explains."""
        now = time.monotonic()
        if self._requests.starts != self._starts:
            self._starts = self._requests.starts
            self._node_until = now + self.NODE_WAIT
            self._node_is_old = True
            self._hurry_until = now + self.HURRY_FOR
        if now < self._node_until and now - self._node_looked >= self.NODE_LOOK:
            self._node_looked = now
            node = find_hidraw(self._phys)
            if node is not None and (self._node_is_old or node != self.device):
                self._follow(node)
        if self._unwatched() and self._opens.watch(self.device):
            # A program that opened the node before it was watched was not heard: it is written
            # to now. And one that came, rumbled and went in that time has left its rumble on.
            self._hurry_until = now + self.HURRY_FOR
            self._stop_a_rumble_left_on()
        if self._requests.stops < self._requests.starts:
            self._driverless = None
        elif self._driverless is None:
            self._driverless = now
        elif now - self._driverless >= self.DRIVER_WAIT:
            self._fail("the kernel's driver let go of the virtual pad and no other took it (`dmesg` says why)")
        if now - self._written >= (self.HURRY if now < self._hurry_until else self.REFRESH):
            self._write_again()

    def _fail(self, error: str) -> None:
        with self._lock:
            if self._error is not None:
                return
            self._error = error
            callback = self._error_callback
        if callback is not None:
            callback(error)

    def _tell(self) -> None:
        """Hand the motor levels to the callback when they are not the ones it got last."""
        strong, weak = self._requests.levels
        levels = (strong >> 8, weak >> 8)
        with self._lock:
            callback = self._callback
            if callback is None or levels == self._sent:
                return
            self._sent = levels
        callback(*levels)

    # ---- lifecycle -----------------------------------------------------------------------
    def close(self) -> None:
        """Remove the pad from the system: closing the uhid file destroys the device."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._callback = self._error_callback = None
        with contextlib.suppress(OSError):
            os.write(self._wake_w, b"x")
        if self._thread is not threading.current_thread():
            self._thread.join(2)  # it closes the files as it ends, and the device with them


__all__ = [
    "EVENT_SIZE",
    "PAD_NAME",
    "PAD_PRODUCT",
    "PAD_VENDOR",
    "REPORT_DESCRIPTOR",
    "NodeWatch",
    "Requests",
    "UhidPad",
    "check_available",
    "create2_event",
    "find_hidraw",
    "input_report",
    "node_events",
    "node_is_held",
    "open_uhid",
    "rumble_levels",
    "state_from_xinput",
]
