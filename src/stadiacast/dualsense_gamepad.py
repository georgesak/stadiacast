"""Linux virtual DualSense controllers through the kernel's uhid device.

``--gamepad dualsense`` gives each Stadia Controller a USB DualSense HID device (``054c:0ce6``)
with a deterministic locally-administered MAC address derived from the controller's own serial.

Why this exists alongside ``--gamepad uhid``: Steam's ``sdl://`` wrapper on Linux currently drops
``SDL_GetGamepadSerial()`` (``HID_UNIQ``) for Stadia and Xbox pads and assigns synthetic profiles
in connection order, whereas Steam drives a DualSense with its own HIDAPI driver, reads Feature
Report ``0x09`` (pairing info) for the 6-byte MAC address, and stores each controller's name,
calibration and bindings under ``configset_DS<mac>.vdf``. Because SteamOS already ships a
``uaccess`` udev rule matching ``KERNELS=="*054C:0CE6*"``, only ``/dev/uhid`` itself needs access.

All seventeen Stadia controls map onto the DualSense USB input report (report ``0x01``, 64 bytes):
A/B/X/Y to Cross/Circle/Square/Triangle, Options (``⋯``) to Create, Menu (``≡``) to Options, the
Stadia button to PS, Capture to Mic Mute (SDL's ``misc1``), and Assistant to Touchpad click. Both
touch contacts stay marked inactive (``0x80``) so the cursor never drifts, and the controller's
battery gauge, when known, is reported in ``status[0]``. Feature reports ``0x05`` (calibration),
``0x09`` (MAC) and ``0x20`` (firmware ``0x0224``, which enables full-range v2 rumble in both SDL 3
and the kernel's ``hid-playstation`` driver) are answered on the pad's own thread, and USB output
report ``0x02`` carries rumble back to the Stadia Controller.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import re
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from stadiacast.errors import SetupError
from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import (
    EVENT_SIZE,
    TRIGGER_THRESHOLD,
    UHID_CLOSE,
    UHID_FEATURE_REPORT,
    UHID_GET_REPORT,
    UHID_OPEN,
    UHID_OUTPUT,
    UHID_OUTPUT_REPORT,
    UHID_SET_REPORT,
    UHID_START,
    UHID_STOP,
    create2_event,
    event_type,
    find_hidraw,
    get_report_reply_event,
    get_report_request,
    hat_value,
    input2_event,
    open_uhid,
    output_request,
    set_report_reply_event,
    set_report_request,
    state_from_xinput,
)
from stadiacast.uhid_gamepad import check_available as _check_uhid_available

PAD_NAME = "Sony Interactive Entertainment Wireless Controller"
PAD_VENDOR, PAD_PRODUCT, PAD_VERSION = 0x054C, 0x0CE6, 0x0100

INPUT_REPORT_ID = 0x01
INPUT_REPORT_SIZE = 64
OUTPUT_REPORT_ID = 0x02
OUTPUT_REPORT_MIN_SIZE = 40  # report id + up to valid_flag2 at byte 39 (SDL writes 48 B, kernel 63 B)

FEATURE_CALIBRATION = 0x05
FEATURE_PAIRING_INFO = 0x09
FEATURE_FIRMWARE_INFO = 0x20

#: Inactive DualSense touch contact bit (``DS_TOUCH_POINT_INACTIVE`` in ``hid-playstation.c``).
TOUCH_INACTIVE = 0x80

_MAC_RE = re.compile(
    r"^([0-9a-fA-F]{2}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})$"
)

#: The report descriptor, written for what reads one. SDL and Steam take the pad for a gamepad by
#: the collection it opens with. The kernel takes input report 1 only from a device that declares
#: one, and its PlayStation driver then reads the 64 bytes by itself. Its generic driver makes the
#: event device out of the fields: so they are the six axes, the hat and the fifteen buttons in
#: the order programs know a DualSense's by, and the rest, which only a reader of the whole report
#: knows what to make of, is declared as filling.
REPORT_DESCRIPTOR = bytes(
    [
        0x05, 0x01,  # Usage Page (Generic Desktop)
        0x09, 0x05,  # Usage (Game Pad)
        0xA1, 0x01,  # Collection (Application)
        0x85, INPUT_REPORT_ID,  # Report ID (1)
        0x09, 0x30,  # Usage (X): the left stick
        0x09, 0x31,  # Usage (Y)
        0x09, 0x32,  # Usage (Z): the right stick
        0x09, 0x35,  # Usage (Rz)
        0x09, 0x33,  # Usage (Rx): L2
        0x09, 0x34,  # Usage (Ry): R2
        0x15, 0x00,  # Logical Minimum (0)
        0x26, 0xFF, 0x00,  # Logical Maximum (255)
        0x75, 0x08,  # Report Size (8)
        0x95, 0x06,  # Report Count (6)
        0x81, 0x02,  # Input (Data, Variable, Absolute)
        0x95, 0x01,  # Report Count (1)
        0x81, 0x03,  # Input (Constant): the report's count of itself
        0x09, 0x39,  # Usage (Hat Switch)
        0x25, 0x07,  # Logical Maximum (7)
        0x75, 0x04,  # Report Size (4)
        0x81, 0x42,  # Input (Data, Variable, Absolute, Null State)
        0x05, 0x09,  # Usage Page (Button)
        0x19, 0x01,  # Usage Minimum (1): Square, Cross, Circle, Triangle, L1, R1, L2, R2,
        0x29, 0x0F,  # Usage Maximum (15): Create, Options, L3, R3, PS, touchpad, microphone
        0x25, 0x01,  # Logical Maximum (1)
        0x75, 0x01,  # Report Size (1)
        0x95, 0x0F,  # Report Count (15)
        0x81, 0x02,  # Input (Data, Variable, Absolute)
        0x75, 0x0D,  # Report Size (13)
        0x95, 0x01,  # Report Count (1)
        0x81, 0x03,  # Input (Constant): to the end of the fourth byte of buttons
        0x75, 0x08,  # Report Size (8)
        0x95, 0x34,  # Report Count (52)
        0x81, 0x03,  # Input (Constant): sensors, time, touch points, battery
        0x85, OUTPUT_REPORT_ID,  # Report ID (2)
        0x06, 0x00, 0xFF,  # Usage Page (Vendor Defined 0xFF00)
        0x09, 0x01,  # Usage (1)
        0x26, 0xFF, 0x00,  # Logical Maximum (255)
        0x95, 0x2F,  # Report Count (47)
        0x91, 0x02,  # Output (Data, Variable, Absolute): motors and lights
        0x85, FEATURE_CALIBRATION,  # Report ID (5)
        0x09, 0x02,  # Usage (2)
        0x95, 0x28,  # Report Count (40)
        0xB1, 0x02,  # Feature (Data, Variable, Absolute)
        0x85, FEATURE_PAIRING_INFO,  # Report ID (9)
        0x09, 0x03,  # Usage (3)
        0x95, 0x13,  # Report Count (19)
        0xB1, 0x02,  # Feature (Data, Variable, Absolute)
        0x85, FEATURE_FIRMWARE_INFO,  # Report ID (32)
        0x09, 0x04,  # Usage (4)
        0x95, 0x3F,  # Report Count (63)
        0xB1, 0x02,  # Feature (Data, Variable, Absolute)
        0xC0,  # End Collection
    ]
)  # fmt: skip

#: What an ideal sensor would read: the gyroscope its whole 16 bits at the 2048 degrees a second
#: the kernel's driver takes its range to be, the accelerometer the 8192 counts a g that both
#: readers reckon with. Either reader then scales by what it scales by with no calibration at
#: all, to within a part in thirty thousand, and neither divides by nothing.
GYRO_SPEED, GYRO_COUNTS, ACCEL_COUNTS = 2048, 32767, 8192
#: Feature report 5 (41 bytes): three gyroscope offsets, each gyroscope axis at the speed below
#: and at its negative, that speed twice, and each accelerometer axis at one g and at minus one.
CALIBRATION_REPORT = bytes([FEATURE_CALIBRATION]) + struct.pack(
    "<3h6h2h6h6x", 0, 0, 0, *(GYRO_COUNTS, -GYRO_COUNTS) * 3, GYRO_SPEED, GYRO_SPEED, *(ACCEL_COUNTS, -ACCEL_COUNTS) * 3
)


#: Synthetic feature report 0x20 (64 bytes): non-zero hardware/firmware version fields and
#: ``update_version = 0x0224`` at bytes 44..45 so both ``hid-playstation`` (>= 0x0221) and SDL 3
#: (>= 0x0224) select unshifted v2 rumble.
def _build_firmware_report() -> bytes:
    buf = bytearray(64)
    buf[0] = FEATURE_FIRMWARE_INFO
    struct.pack_into("<II", buf, 24, 1, 0x01000000)  # hw_version, fw_version
    struct.pack_into("<H", buf, 44, 0x0224)  # update_version
    return bytes(buf)


def check_available() -> None:
    """Fail at start-up, in plain words, where a DualSense pad could not be created later."""
    _check_uhid_available("--gamepad dualsense")


FIRMWARE_REPORT = _build_firmware_report()

#: DualSense buttons in ``buttons[0..2]`` of input report 0x01.
BUTTON_BITS: dict[Button, tuple[int, int]] = {
    Button.X: (8, 0x10),  # Square (West)
    Button.A: (8, 0x20),  # Cross (South)
    Button.B: (8, 0x40),  # Circle (East)
    Button.Y: (8, 0x80),  # Triangle (North)
    Button.L1: (9, 0x01),  # L1
    Button.R1: (9, 0x02),  # R1
    Button.OPTIONS: (9, 0x10),  # Create (Stadia ⋯)
    Button.MENU: (9, 0x20),  # Options (Stadia ≡)
    Button.L3: (9, 0x40),  # L3
    Button.R3: (9, 0x80),  # R3
    Button.STADIA: (10, 0x01),  # PS button
    Button.ASSISTANT: (10, 0x02),  # Touchpad click
    Button.CAPTURE: (10, 0x04),  # Mic Mute (SDL misc1)
}
L2_BIT, R2_BIT = (9, 0x04), (9, 0x08)


def mac_bytes(identity: str) -> bytes:
    """A 6-byte locally-administered unicast MAC derived deterministically from ``identity``
    (or parsed as-is when ``identity`` is already ``xx:xx:xx:xx:xx:xx``)."""
    matched = _MAC_RE.match(identity.strip())
    if matched:
        return bytes(int(part, 16) for part in matched.groups())
    digest = hashlib.sha256(f"stadiacast-dualsense:{identity.strip()}".encode()).digest()
    return bytes([(digest[0] & 0xFE) | 0x02, *digest[1:6]])


def mac_string(mac: bytes) -> str:
    """Lowercase colon-separated MAC address, as ``hid-playstation`` formats ``hdev->uniq``."""
    if len(mac) != 6:
        raise ValueError("MAC address must be 6 bytes")
    return ":".join(f"{b:02x}" for b in mac)


def pairing_report(mac: bytes) -> bytes:
    """Feature report 0x09 (20 bytes): bytes 1..6 hold the controller MAC in reverse byte order."""
    if len(mac) != 6:
        raise ValueError("MAC address must be 6 bytes")
    buf = bytearray(20)
    buf[0] = FEATURE_PAIRING_INFO
    buf[1:7] = mac[::-1]
    return bytes(buf)


def stick_byte(value: int, *, invert: bool = False) -> int:
    """A stick axis (-32768..32767, Y up) as DualSense's 0..255 byte with 128 at rest, Y down."""
    value = max(-32768, min(32767, value))
    if invert:
        value = -1 - value if value == -32768 else -value
    if value == 0:
        return 128
    scaled = round((value + 32768) * 255 / 65535)
    return max(0, min(255, scaled))


def battery_byte(percent: int | None) -> int:
    """``status[0]`` of input report 0x01: 100%/full when unknown, else discharging decile 0..10."""
    if percent is None:
        return (0x02 << 4) | 10  # POWER_SUPPLY_STATUS_FULL, 100%
    return min(10, max(0, int(percent) // 10))  # charging_status 0x0 = discharging


def input_report(
    state: GamepadState,
    *,
    seq: int = 0,
    timestamp: int = 0,
    battery_percent: int | None = None,
) -> bytes:
    """64-byte DualSense USB input report (report 0x01)."""
    data = bytearray(INPUT_REPORT_SIZE)
    data[0] = INPUT_REPORT_ID
    data[1] = stick_byte(state.lx)
    data[2] = stick_byte(state.ly, invert=True)
    data[3] = stick_byte(state.rx)
    data[4] = stick_byte(state.ry, invert=True)
    l2, r2 = max(0, min(255, state.l2)), max(0, min(255, state.r2))
    data[5], data[6] = l2, r2
    data[7] = seq & 0xFF
    data[8] = hat_value(state.buttons)
    for button, (index, mask) in BUTTON_BITS.items():
        if state.buttons & button:
            data[index] |= mask
    for value, (index, mask) in ((l2, L2_BIT), (r2, R2_BIT)):
        if value >= TRIGGER_THRESHOLD:
            data[index] |= mask
    struct.pack_into("<I", data, 28, timestamp & 0xFFFFFFFF)
    data[33] = TOUCH_INACTIVE
    data[37] = TOUCH_INACTIVE
    data[53] = battery_byte(battery_percent)
    return bytes(data)


def output_rumble(data: bytes) -> tuple[bool, tuple[int, int] | None]:
    """Inspect a DualSense output report.

    Returns ``(valid_report, levels)``, where ``levels`` is ``(strong, weak)`` in ``0..65535`` when
    the report updates vibration and ``None`` when the report only updates LEDs/audio/triggers (so
    ``SET_REPORT`` still succeeds without clobbering active rumble).
    """
    if len(data) < OUTPUT_REPORT_MIN_SIZE or data[0] != OUTPUT_REPORT_ID:
        return False, None
    flag0, flag1, motor_right, motor_left = data[1], data[2], data[3], data[4]
    flag2 = data[39]
    if (flag0 & 0x01) or (flag2 & 0x04):
        return True, (motor_left * 257, motor_right * 257)
    if (flag0 & 0x02) or (flag0 == 0 and flag1 == 0 and flag2 == 0):
        return True, (0, 0)
    return True, None


class DualSenseRequests:
    """Kernel event state machine for one virtual DualSense pad."""

    def __init__(self, mac: bytes) -> None:
        self._pairing = pairing_report(mac)
        self.levels = (0, 0)  # strong (left), weak (right): 0..65535
        self.starts = 0
        self.opens = 0

    def handle(self, event: bytes) -> bytes | None:
        kind = event_type(event)
        if kind == UHID_GET_REPORT:
            request, rnum, rtype = get_report_request(event)
            if rtype == UHID_FEATURE_REPORT:
                if rnum == FEATURE_CALIBRATION:
                    return get_report_reply_event(request, 0, CALIBRATION_REPORT)
                if rnum == FEATURE_PAIRING_INFO:
                    return get_report_reply_event(request, 0, self._pairing)
                if rnum == FEATURE_FIRMWARE_INFO:
                    return get_report_reply_event(request, 0, FIRMWARE_REPORT)
            return get_report_reply_event(request, errno.EINVAL)
        if kind == UHID_OUTPUT:
            data, rtype = output_request(event)
            if rtype == UHID_OUTPUT_REPORT:
                _ok, levels = output_rumble(data)
                if levels is not None:
                    self.levels = levels
            return None
        if kind == UHID_SET_REPORT:
            request, _rnum, rtype, data = set_report_request(event)
            if rtype != UHID_OUTPUT_REPORT:
                return set_report_reply_event(request, errno.EIO)
            ok, levels = output_rumble(data)
            if not ok:
                return set_report_reply_event(request, errno.EIO)
            if levels is not None:
                self.levels = levels
            return set_report_reply_event(request, 0)
        if kind == UHID_START:
            self.starts += 1
        elif kind == UHID_OPEN:
            self.opens += 1
        elif kind in (UHID_CLOSE, UHID_STOP):
            self.levels = (0, 0)
        return None


class DualSensePad:
    """One virtual Sony DualSense controller with a unique MAC derived from the Stadia serial."""

    slot: int | None = None
    takes_state = True
    close_blocks = True
    START_TIMEOUT = 5.0
    #: DualSense readers expect a steady ~125 Hz stream (SDL's USB mode detection waits 16 ms for
    #: a 64-byte packet on open).
    REFRESH = 0.008
    NODE_WAIT = 1.0

    label: str
    mac: bytes
    uniq: str
    device: str | None
    _phys: str
    _poll: Any
    _requests: DualSenseRequests
    _state: GamepadState
    _battery: int | None
    _seq: int
    _report_lock: threading.Lock
    _written: float
    _starts: int
    _opens: int
    _node_until: float
    _sent: tuple[int, int]
    _callback: Callable[[int, int], None] | None
    _lock: threading.Lock
    _closed: bool
    _fd: int
    _wake_r: int
    _wake_w: int
    _thread: threading.Thread

    def __init__(self, label: str, uniq: str | None = None) -> None:
        if not sys.platform.startswith("linux"):
            raise SetupError("--gamepad dualsense is Linux only: it creates the pad through the kernel's uhid device")
        import select

        self.label = label
        self.mac = mac_bytes((uniq or "").strip() or label)
        self.uniq = mac_string(self.mac)
        self.device = None
        self._phys = f"stadiacast/{os.getpid()}/{label}"
        self._requests = DualSenseRequests(self.mac)
        self._state = GamepadState()
        self._battery = None
        self._seq = 0
        self._report_lock = threading.Lock()
        self._written = time.monotonic()
        self._starts = 0
        self._opens = 0
        self._node_until = 0.0
        self._sent = (0, 0)
        self._callback = None
        self._lock = threading.Lock()
        self._closed = False
        self._fd = open_uhid("--gamepad dualsense")
        self._wake_r, self._wake_w = os.pipe()
        self._poll = select.poll()
        self._poll.register(self._fd, select.POLLIN)
        self._poll.register(self._wake_r, select.POLLIN)
        try:
            os.write(
                self._fd,
                create2_event(
                    PAD_NAME,
                    self._phys,
                    self.uniq,
                    REPORT_DESCRIPTOR,
                    vendor=PAD_VENDOR,
                    product=PAD_PRODUCT,
                    version=PAD_VERSION,
                ),
            )
            self._wait_started()
        except SetupError:
            self._close_files()
            raise
        except OSError as e:
            self._close_files()
            raise SetupError(f"the kernel refused the virtual pad: {e}") from e
        self._starts = self._requests.starts
        self._opens = self._requests.opens
        self._emit(ignore_errors=True)
        # Start the event thread before looking for hidraw: hid-playstation blocks its probe on
        # GET_REPORT (0x09 MAC, 0x20 firmware, 0x05 calibration) after UHID_START and creates
        # the hidraw node only once all three replies arrive.
        self._thread = threading.Thread(target=self._feedback_loop, name=f"dualsense-{label}", daemon=True)
        self._thread.start()
        self.device = self._find_node()

    def _wait_started(self) -> None:
        deadline = time.monotonic() + self.START_TIMEOUT
        while not self._requests.starts:
            left = deadline - time.monotonic()
            if left <= 0:
                raise SetupError(
                    f"the kernel did not start the virtual pad within {self.START_TIMEOUT:g} s: no HID driver"
                    " took it (`sudo modprobe hid-playstation` or `hid-generic`)"
                )
            if self._poll.poll(int(left * 1000) + 1):
                self._read_events()

    def _find_node(self) -> str | None:
        deadline = time.monotonic() + self.NODE_WAIT
        while True:
            node = find_hidraw(self._phys)
            if node is not None or time.monotonic() >= deadline:
                return node
            time.sleep(0.01)

    def _close_files(self) -> None:
        for fd in (self._fd, self._wake_r, self._wake_w):
            with contextlib.suppress(OSError):
                os.close(fd)

    # ---- input ---------------------------------------------------------------------------
    def set_battery(self, percent: int | None) -> None:
        with self._report_lock:
            self._battery = percent

    def apply_state(self, state: GamepadState) -> None:
        with self._report_lock:
            if state == self._state and self._seq > 0:
                return
            self._state = state
            self._write_locked(ignore_errors=False)

    def _emit(self, *, ignore_errors: bool) -> None:
        with self._report_lock:
            self._write_locked(ignore_errors=ignore_errors)

    def _write_locked(self, *, ignore_errors: bool) -> None:
        report = input_report(
            self._state,
            seq=self._seq,
            timestamp=time.monotonic_ns() // 333,
            battery_percent=self._battery,
        )
        self._seq = (self._seq + 1) & 0xFF
        try:
            os.write(self._fd, input2_event(report))
        except OSError as e:
            if not ignore_errors:
                raise SetupError(f"the virtual pad no longer takes reports: {e}") from e
        self._written = time.monotonic()

    def apply(self, xinput: Mapping[str, int]) -> None:
        self.apply_state(state_from_xinput(xinput))

    def reset(self) -> None:
        self.apply_state(GamepadState())

    # ---- requests and rumble -------------------------------------------------------------
    def on_rumble(self, callback: Callable[[int, int], None]) -> None:
        with self._lock:
            self._callback = callback
            self._sent = (0, 0)
        with contextlib.suppress(OSError):
            os.write(self._wake_w, b"r")

    def _read_events(self) -> bool:
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
                with contextlib.suppress(OSError):
                    os.write(self._fd, reply)

    def _feedback_loop(self) -> None:
        while True:
            left = self._written + self.REFRESH - time.monotonic()
            try:
                ready = self._poll.poll(max(0.0, left) * 1000)
            except OSError:
                return
            for fd, _what in ready:
                if fd == self._wake_r:
                    if b"x" in os.read(fd, 16):
                        return
                elif not self._read_events():
                    return
            self._keep_up()
            self._tell()

    def _keep_up(self) -> None:
        now = time.monotonic()
        restarted = self._requests.starts != self._starts
        reopened = self._requests.opens != self._opens
        if restarted:
            self._starts = self._requests.starts
            self._node_until = now + self.NODE_WAIT
        if reopened:
            self._opens = self._requests.opens
        if now < self._node_until:
            node = find_hidraw(self._phys)
            if node is not None:
                self.device = node
        if restarted or reopened or now - self._written >= self.REFRESH:
            self._emit(ignore_errors=True)

    def _tell(self) -> None:
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
    "CALIBRATION_REPORT",
    "FEATURE_CALIBRATION",
    "FEATURE_FIRMWARE_INFO",
    "FEATURE_PAIRING_INFO",
    "FIRMWARE_REPORT",
    "INPUT_REPORT_ID",
    "INPUT_REPORT_SIZE",
    "OUTPUT_REPORT_ID",
    "PAD_NAME",
    "PAD_PRODUCT",
    "PAD_VENDOR",
    "REPORT_DESCRIPTOR",
    "DualSensePad",
    "DualSenseRequests",
    "battery_byte",
    "check_available",
    "input_report",
    "mac_bytes",
    "mac_string",
    "output_rumble",
    "pairing_report",
    "stick_byte",
]
