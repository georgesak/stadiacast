"""Linux virtual DualSense controllers through the kernel's uhid device.

``--gamepad dualsense`` gives each controller a HID device with the identity of a Sony DualSense
on its cable (USB ``054c:0ce6``), where ``--gamepad uhid`` gives it a wired Stadia Controller's.
It is there for the pad's address. A DualSense states a Bluetooth address of its own in a feature
report, on the cable too, and that is what a program knows one from another by: Steam keeps each
one's name and bindings by it. The address of this pad is made from its controller's serial
number, so a controller is the same DualSense every time, whatever order the controllers wake in.

The pad is :class:`~stadiacast.uhid_gamepad.UhidPad` with other reports, and is read in three
ways:

- Steam and SDL 3 read the pad's ``hidraw`` node with their DualSense driver: every control, with
  Capture as the microphone button (SDL's ``misc1``) and Assistant as the click of the touchpad,
  the battery, and rumble written to the same node. SDL's driver waits 16 ms for a first report
  from a pad it has just opened, and takes one that shows none for a pad on Bluetooth, which it
  asks for neither its address nor its firmware. So the pad reports all the time, as a DualSense
  does, and at once when a program opens the node.
- The kernel's PlayStation driver (``hid-playstation``, since Linux 5.12) binds to these ids,
  asks for the three feature reports and makes what everything else reads: the gamepad's event
  device, with force feedback, a battery, and event devices of their own for motion sensors
  that never move and for the touchpad, whose click is Assistant (and since Linux 6.18 one for
  a headset, which the pad never has). Capture is no key there: the driver keeps the microphone
  button for itself.
- Where that driver is missing, the kernel's generic one makes one event device out of the report
  descriptor.

Nothing here is taken from a controller. The report descriptor declares what those readers need
of one. The layout of the input report, of the output report and of the three feature reports is
the one the kernel's driver and SDL's read, both of which are public. What the feature reports
say is made up to suit them: a calibration for sensors that report nothing, and a firmware new
enough for rumble at its full range.
"""

from __future__ import annotations

import hashlib
import struct
import time

from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import TRIGGER_THRESHOLD, Requests, UhidPad, hat_value

# ---- what the pad says it is -----------------------------------------------------------------
#: A DualSense's name as the kernel has it over USB: the manufacturer and the product it states.
PAD_NAME = "Sony Interactive Entertainment Wireless Controller"
#: Its ids, and the version of its HID interface. The kernel's driver sets the top bit of that
#: version on the event device it makes, and programs file their mappings for a DualSense under
#: the two numbers: 0x0111 for the generic driver's buttons and axes, 0x8111 for the PlayStation
#: driver's. A pad that says 0x0100 comes out of that driver as 0x8100, which SDL has no mapping
#: for: it reads the pad with the generic driver's, and buttons and sticks come out scrambled.
PAD_VENDOR, PAD_PRODUCT, PAD_VERSION = 0x054C, 0x0CE6, 0x0111

INPUT_REPORT_ID, INPUT_REPORT_SIZE = 0x01, 64
OUTPUT_REPORT_ID = 0x02
#: An output report is read up to its third byte of flags, at 39. SDL writes 48 bytes and the
#: kernel's driver 63.
OUTPUT_REPORT_MIN_SIZE = 40
FEATURE_CALIBRATION, FEATURE_PAIRING, FEATURE_FIRMWARE = 0x05, 0x09, 0x20

#: The report's buttons: (byte, mask) per :class:`~stadiacast.gamepad.Button`, by position on the
#: pad, so A is Cross and Y is Triangle. Options (``⋯``) is Create, Menu (``≡``) is Options and
#: the Stadia button is PS. L2 and R2 are digital copies of the triggers.
BUTTON_BITS: dict[Button, tuple[int, int]] = {
    Button.X: (8, 0x10),  # Square
    Button.A: (8, 0x20),  # Cross
    Button.B: (8, 0x40),  # Circle
    Button.Y: (8, 0x80),  # Triangle
    Button.L1: (9, 0x01),
    Button.R1: (9, 0x02),
    Button.OPTIONS: (9, 0x10),  # Create
    Button.MENU: (9, 0x20),  # Options
    Button.L3: (9, 0x40),
    Button.R3: (9, 0x80),
    Button.STADIA: (10, 0x01),  # PS
    Button.ASSISTANT: (10, 0x02),  # the touchpad's click
    Button.CAPTURE: (10, 0x04),  # the microphone button
}
L2_BIT, R2_BIT = (9, 0x04), (9, 0x08)
#: Where the report counts itself and tells its time, in thirds of a microsecond.
SEQUENCE, TIMESTAMP = 7, 28
#: The two touch points, whose first byte says with its top bit that no finger is there.
TOUCH_POINTS, NO_FINGER = (33, 37), 0x80
#: The battery: a level from 0 to 10 in the low half and how it is charged in the high half.
STATUS, DISCHARGING, FULL = 53, 0x0, 0x2

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
        0x85, FEATURE_PAIRING,  # Report ID (9)
        0x09, 0x03,  # Usage (3)
        0x95, 0x13,  # Report Count (19)
        0xB1, 0x02,  # Feature (Data, Variable, Absolute)
        0x85, FEATURE_FIRMWARE,  # Report ID (32)
        0x09, 0x04,  # Usage (4)
        0x95, 0x3F,  # Report Count (63)
        0xB1, 0x02,  # Feature (Data, Variable, Absolute)
        0xC0,  # End Collection
    ]
)  # fmt: skip

# ---- the feature reports ---------------------------------------------------------------------
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
#: The firmware the pad says it has: the first that SDL sends rumble at the motors' whole range
#: to (an older one gets half of every level), and past the 0x0215 from which the kernel's driver
#: sends it the same kind of report.
FIRMWARE_VERSION = 0x0224
#: Feature report 32 (64 bytes): numbers of its own for the hardware and the firmware, which the
#: kernel's driver prints, and the version above where both readers look for it.
FIRMWARE_REPORT = bytes([FEATURE_FIRMWARE]) + struct.pack("<23xII12xH18x", 1, 0x01000000, FIRMWARE_VERSION)


def mac_bytes(identity: str) -> bytes:
    """The pad's address for a controller's serial number: six bytes of its hash, marked as an
    address that was given locally and is one device's, which no real controller's is."""
    digest = hashlib.sha256(f"stadiacast-dualsense:{identity.strip()}".encode()).digest()
    return bytes([(digest[0] & 0xFE) | 0x02, *digest[1:6]])


def mac_string(mac: bytes) -> str:
    """An address as the kernel's driver writes it, which is what it gives the pad for its serial."""
    if len(mac) != 6:
        raise ValueError("an address is 6 bytes")
    return ":".join(f"{b:02x}" for b in mac)


def pairing_report(mac: bytes) -> bytes:
    """Feature report 9 (20 bytes): the pad's address, last byte first."""
    if len(mac) != 6:
        raise ValueError("an address is 6 bytes")
    return bytes([FEATURE_PAIRING]) + mac[::-1] + bytes(13)


# ---- the reports -----------------------------------------------------------------------------
def stick_byte(value: int, *, invert: bool = False) -> int:
    """A stick axis (-32768..32767, Y up) as the report's byte: 0..255 with 128 at rest, Y down."""
    value = max(-32768, min(32767, value))
    if invert:
        value = -1 - value if value == -32768 else -value
    if value == 0:
        return 128
    return max(0, min(255, round((value + 32768) * 255 / 65535)))


def battery_byte(percent: int | None) -> int:
    """The battery's byte: full for a level that is not known, or else the level in tenths, on
    the battery (whether a controller is being charged is not known)."""
    if percent is None:
        return (FULL << 4) | 10
    return (DISCHARGING << 4) | min(10, max(0, int(percent) // 10))


def input_report(state: GamepadState, battery_percent: int | None = None) -> bytes:
    """Input report 1 for a state, as a DualSense on its cable sends it for the same controls,
    with nobody at its touchpad and sensors at rest. :func:`stamped` adds what every report has
    of its own."""
    data = bytearray(INPUT_REPORT_SIZE)
    data[0] = INPUT_REPORT_ID
    data[1] = stick_byte(state.lx)
    data[2] = stick_byte(state.ly, invert=True)
    data[3] = stick_byte(state.rx)
    data[4] = stick_byte(state.ry, invert=True)
    l2, r2 = max(0, min(255, state.l2)), max(0, min(255, state.r2))
    data[5], data[6] = l2, r2
    data[8] = hat_value(state.buttons)
    for button, (index, mask) in BUTTON_BITS.items():
        if state.buttons & button:
            data[index] |= mask
    for value, (index, mask) in ((l2, L2_BIT), (r2, R2_BIT)):
        if value >= TRIGGER_THRESHOLD:
            data[index] |= mask
    for point in TOUCH_POINTS:
        data[point] = NO_FINGER
    data[STATUS] = battery_byte(battery_percent)
    return bytes(data)


def stamped(report: bytes, sequence: int, clock: int) -> bytes:
    """A report with its count of itself and its time."""
    data = bytearray(report)
    data[SEQUENCE] = sequence & 0xFF
    struct.pack_into("<I", data, TIMESTAMP, clock & 0xFFFFFFFF)
    return bytes(data)


def output_rumble(data: bytes) -> tuple[bool, tuple[int, int] | None]:
    """Whether ``data`` is an output report of the pad's, and the motor levels it sets (strong,
    weak: 0..65535), or ``None`` for one that only sets lights and the like and leaves the motors
    as they are.

    The report's flags say what it sets. Either of two says the motor levels count: bit 0 of the
    first byte of flags, the older way, or bit 2 of the third, which the kernel's driver and SDL
    use with the firmware this pad states. SDL stops a rumble with a report that sets neither and
    nothing else, and asks for the motors to be the pad's haptics with bit 1 of the first byte
    alone: both mean no rumble.
    """
    if len(data) < OUTPUT_REPORT_MIN_SIZE or data[0] != OUTPUT_REPORT_ID:
        return False, None
    flags, lights, weak, strong, more = data[1], data[2], data[3], data[4], data[39]
    if flags & 0x01 or more & 0x04:
        return True, (strong * 257, weak * 257)
    if flags & 0x02 or not (flags or lights or more):
        return True, (0, 0)
    return True, None


class DualSenseRequests(Requests):
    """What the kernel asks of a DualSense: its three feature reports, and its output report."""

    def __init__(self, mac: bytes) -> None:
        super().__init__()
        self._features = {
            FEATURE_CALIBRATION: CALIBRATION_REPORT,
            FEATURE_PAIRING: pairing_report(mac),
            FEATURE_FIRMWARE: FIRMWARE_REPORT,
        }

    def output(self, data: bytes) -> bool:
        taken, levels = output_rumble(data)
        if levels is not None:
            self.levels = levels
        return taken

    def feature(self, number: int) -> bytes | None:
        return self._features.get(number)


class DualSensePad(UhidPad):
    """One virtual DualSense, whose address is its controller's alone."""

    FLAG = "--gamepad dualsense"
    NAME = PAD_NAME
    IDS = (PAD_VENDOR, PAD_PRODUCT, PAD_VERSION)
    DESCRIPTOR = REPORT_DESCRIPTOR
    #: A DualSense reports all the time, and a program that opens one counts on it (seconds).
    REFRESH = 0.008
    mac: bytes
    _battery: int | None
    _sequence: int

    def __init__(self, label: str, uniq: str | None = None) -> None:
        self._battery = None
        self._sequence = 0
        super().__init__(label, uniq)

    def _identity(self, serial: str) -> str:
        self.mac = mac_bytes(serial)
        return mac_string(self.mac)

    def _new_requests(self) -> Requests:
        return DualSenseRequests(self.mac)

    def _encode(self, state: GamepadState) -> bytes:
        return input_report(state, self._battery)

    def _stamp(self, report: bytes) -> bytes:
        self._sequence += 1
        return stamped(report, self._sequence, time.monotonic_ns() // 333)

    def set_battery(self, percent: int | None) -> None:
        """The controller's battery level, which the pad's next report carries."""
        with self._report_lock:
            self._battery = percent
            self._report = self._encode(self._state)


__all__ = [
    "CALIBRATION_REPORT",
    "FEATURE_CALIBRATION",
    "FEATURE_FIRMWARE",
    "FEATURE_PAIRING",
    "FIRMWARE_REPORT",
    "FIRMWARE_VERSION",
    "INPUT_REPORT_ID",
    "INPUT_REPORT_SIZE",
    "OUTPUT_REPORT_ID",
    "PAD_NAME",
    "PAD_PRODUCT",
    "PAD_VENDOR",
    "PAD_VERSION",
    "REPORT_DESCRIPTOR",
    "DualSensePad",
    "DualSenseRequests",
    "battery_byte",
    "input_report",
    "mac_bytes",
    "mac_string",
    "output_rumble",
    "pairing_report",
    "stamped",
    "stick_byte",
]
