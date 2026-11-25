"""Linux virtual Stadia Controllers through the kernel's uhid device.

``--gamepad uhid`` gives each controller a HID device with the wired Stadia Controller's identity
(USB ``18d1:9400`` and its name) and its input report (report 3), instead of the Xbox 360 pad of
``--gamepad uinput``. A program then reads it the way it reads that controller:

- Steam and SDL 3 through the pad's ``hidraw`` node, with their Stadia driver. It has the two
  buttons an Xbox pad has no place for, Assistant and Capture (SDL 3's ``misc2`` and ``misc1``),
  and writes its rumble to the same node.
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

The packing of the kernel's events and the report encoding are plain Python and run in the tests
of every platform; :class:`UhidPad` needs Linux.
"""

from __future__ import annotations

import contextlib
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


def create2_event(name: str, phys: str, uniq: str, descriptor: bytes = REPORT_DESCRIPTOR) -> bytes:
    """``UHID_CREATE2`` for a pad with the controller's ids."""
    if len(descriptor) > HID_MAX_DESCRIPTOR_SIZE:
        raise ValueError("report descriptor too long")
    return CREATE2.pack(
        UHID_CREATE2,
        name.encode()[:127],
        phys.encode()[:63],
        uniq.encode()[:63],
        len(descriptor),
        BUS_USB,
        PAD_VENDOR,
        PAD_PRODUCT,
        PAD_VERSION,
        0,
        descriptor,
    )


def input2_event(report: bytes) -> bytes:
    return INPUT2.pack(UHID_INPUT2, len(report)) + report


def set_report_reply_event(request: int, err: int) -> bytes:
    return SET_REPORT_REPLY.pack(UHID_SET_REPORT_REPLY, request, err)


def get_report_reply_event(request: int, err: int) -> bytes:
    return GET_REPORT_REPLY.pack(UHID_GET_REPORT_REPLY, request, err, 0)


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
    answer a request waits for, and how often the device was started. Kept apart from the device
    so it runs anywhere."""

    def __init__(self) -> None:
        self.levels = (0, 0)  # strong, weak: 0..65535
        #: ``UHID_START``: a HID driver has taken the device and is making its nodes. More than
        #: once when the kernel gives the device to another driver.
        self.starts = 0

    def handle(self, event: bytes) -> bytes | None:
        """One event read from the uhid file; returns the reply to write, if it needs one."""
        kind = event_type(event)
        if kind == UHID_OUTPUT:
            data, rtype = output_request(event)
            levels = rumble_levels(data) if rtype == UHID_OUTPUT_REPORT else None
            if levels is not None:
                self.levels = levels
            return None
        if kind == UHID_SET_REPORT:
            request, _rnum, rtype, data = set_report_request(event)
            levels = rumble_levels(data) if rtype == UHID_OUTPUT_REPORT else None
            if levels is None:
                return set_report_reply_event(request, errno.EIO)
            self.levels = levels
            return set_report_reply_event(request, 0)
        if kind == UHID_GET_REPORT:  # the pad has no report to read back
            request, _rnum, _rtype = get_report_request(event)
            return get_report_reply_event(request, errno.EIO)
        if kind == UHID_START:
            self.starts += 1
        elif kind in (UHID_CLOSE, UHID_STOP):
            # The last program has closed the pad's nodes, or the driver has let go of the device:
            # nobody is left to stop a rumble. A program that ends while its rumble runs gets here
            # (one that plays an effect through the event device has it stopped by the kernel).
            self.levels = (0, 0)
        return None


# ---- the device ------------------------------------------------------------------------------
def open_uhid() -> int:
    """Open the kernel's uhid device, or say in a :class:`SetupError` what keeps it shut."""
    if not sys.platform.startswith("linux"):
        raise SetupError("--gamepad uhid is Linux only: it creates the pad through the kernel's uhid device")
    try:
        return os.open(UHID_PATH, os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        raise SetupError(
            f"--gamepad uhid needs the kernel's uhid device and found none ({UHID_PATH}): load it with"
            " `sudo modprobe uhid`"
        ) from None
    except PermissionError:
        raise SetupError(
            f"--gamepad uhid may not open {UHID_PATH}: run the receiver as root, or give your user the device"
            " with the udev rule in the README (Virtual Stadia Controller on Linux)"
        ) from None
    except OSError as e:
        raise SetupError(f"--gamepad uhid cannot open {UHID_PATH}: {e}") from e


def check_available() -> None:
    """Fail at start-up, in plain words, where a pad could not be created later."""
    os.close(open_uhid())


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


class UhidPad:
    """One virtual Stadia Controller. The kernel's requests are answered on a thread of its own."""

    slot: int | None = None  # Linux has no XInput player slot
    #: Takes the whole :class:`GamepadState` (:meth:`apply_state`): XInput's fields have no room for
    #: the Assistant and Capture buttons.
    takes_state = True
    #: Closing the uhid file is what removes the device, and the kernel takes a tenth of a second
    #: or more over it: :class:`~stadiacast.virtual_gamepad.VirtualGamepads` closes the pad in a
    #: worker thread.
    close_blocks = True
    #: How long the kernel may take to start the device after it is created.
    START_TIMEOUT = 5.0
    #: The state is written again this often while nothing changes (seconds): see
    #: :meth:`_keep_up`. The kernel passes a report that changes nothing on to the programs that
    #: read the hidraw node, and makes no event of it on the event device.
    REFRESH = 0.1
    #: How long after a start the hidraw node is looked for (seconds).
    NODE_WAIT = 1.0
    label: str
    device: str | None  # the hidraw node, e.g. /dev/hidraw7
    _phys: str
    _poll: Any
    _requests: Requests
    _report: bytes
    _report_lock: threading.Lock  # the report last written, and the writing of it
    _written: float  # when it was last written, or would have been, on the monotonic clock
    _starts: int
    _node_until: float
    _sent: tuple[int, int]
    _callback: Callable[[int, int], None] | None
    _lock: threading.Lock  # the callback and what was last sent to it
    _closed: bool
    _fd: int
    _wake_r: int
    _wake_w: int
    _thread: threading.Thread

    def __init__(self, label: str) -> None:
        if not sys.platform.startswith("linux"):
            raise SetupError("--gamepad uhid is Linux only: it creates the pad through the kernel's uhid device")
        import select

        self.label = label
        self.device = None
        self._phys = f"stadiacast/{os.getpid()}/{label}"  # what the pad's nodes are found by: no other's
        self._requests = Requests()
        self._report = b""
        self._report_lock = threading.Lock()
        self._written = time.monotonic()
        self._starts = 0
        self._node_until = 0.0
        self._sent = (0, 0)
        self._callback = None
        self._lock = threading.Lock()
        self._closed = False
        self._fd = open_uhid()
        self._wake_r, self._wake_w = os.pipe()
        self._poll = select.poll()
        self._poll.register(self._fd, select.POLLIN)
        self._poll.register(self._wake_r, select.POLLIN)
        try:
            os.write(self._fd, create2_event(PAD_NAME, self._phys, label))
            self._wait_started()
        except SetupError:
            self._close_files()
            raise
        except OSError as e:
            self._close_files()
            raise SetupError(f"the kernel refused the virtual pad: {e}") from e
        self._starts = self._requests.starts
        self.device = self._find_node()
        # A pad is made when its controller connects, before any input: it says from the start
        # that it is at rest. An event device that has had no report reads its sticks as held in
        # a corner, and SDL's Stadia driver the D-pad as held up.
        self._report = input_report(GamepadState())
        self._write_again()
        self._thread = threading.Thread(target=self._feedback_loop, name=f"uhid-{label}", daemon=True)
        self._thread.start()

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
        deadline = time.monotonic() + self.NODE_WAIT
        while True:  # the hidraw node follows the start by a moment
            node = find_hidraw(self._phys)
            if node is not None or time.monotonic() >= deadline:
                return node
            time.sleep(0.01)

    def _close_files(self) -> None:
        for fd in (self._fd, self._wake_r, self._wake_w):
            with contextlib.suppress(OSError):
                os.close(fd)

    # ---- input ---------------------------------------------------------------------------
    def apply_state(self, state: GamepadState) -> None:
        report = input_report(state)
        with self._report_lock:
            if report == self._report:
                return
            try:
                os.write(self._fd, input2_event(report))
            except OSError as e:
                raise SetupError(f"the virtual pad no longer takes reports: {e}") from e
            self._report = report
            self._written = time.monotonic()

    def _write_again(self) -> None:
        with self._report_lock:
            with contextlib.suppress(OSError):
                os.write(self._fd, input2_event(self._report))
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
        while True:
            left = self._written + self.REFRESH - time.monotonic()  # until the state is due again
            try:
                ready = self._poll.poll(max(0.0, left) * 1000)
            except OSError:
                return
            for fd, _what in ready:
                if fd == self._wake_r:
                    if b"x" in os.read(fd, 16):  # close() asks the thread to end
                        return
                elif not self._read_events():
                    return
            self._keep_up()
            self._tell()

    def _keep_up(self) -> None:
        """Write the state again when it has not changed for a while, and follow the device's node.

        The kernel has no word for when a report was lost or never seen, so the state is repeated
        (:attr:`REFRESH`), as many controllers repeat theirs on a wire:

        - A HID driver that is taking the device drops the input written meanwhile, and the writer
          is told nothing. Its event device then reads the sticks as held in a corner until a
          report arrives. Every start is a driver taking the device: the first pad after a boot is
          started twice, for the generic driver and then, once udev has loaded it, for the
          kernel's Stadia driver.
        - A program that opens the hidraw node sees only what is written from then on. SDL's Stadia
          driver starts from a state of its own in which the D-pad is held up, and misses that
          press until it has read a report.

        A start also makes a new hidraw node, which is looked up again."""
        now = time.monotonic()
        if self._requests.starts != self._starts:
            self._starts = self._requests.starts
            self._node_until = now + self.NODE_WAIT
        if now < self._node_until:
            node = find_hidraw(self._phys)
            if node is not None:
                self.device = node
        if now - self._written >= self.REFRESH:
            self._write_again()

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
            self._callback = None
        with contextlib.suppress(OSError):
            os.write(self._wake_w, b"x")
        if self._thread is not threading.current_thread():
            self._thread.join(2)
        self._close_files()


__all__ = [
    "EVENT_SIZE",
    "PAD_NAME",
    "PAD_PRODUCT",
    "PAD_VENDOR",
    "REPORT_DESCRIPTOR",
    "Requests",
    "UhidPad",
    "check_available",
    "create2_event",
    "find_hidraw",
    "input_report",
    "open_uhid",
    "rumble_levels",
    "state_from_xinput",
]
