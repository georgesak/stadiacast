"""The Linux Stadia pad against the real kernel: the device uhid creates from the report
descriptor, read back through its hidraw node the way SDL 3 reads it and through its event node
the way everything else does, rumble written to the hidraw node the way SDL 3 writes it, and a
game's force feedback through the kernel's hid-google-stadiaff driver. Runs on Linux when
STADIACAST_LINUX_LAB is set, with the right to open ``/dev/uhid`` and the pad's nodes (as root, or
with the udev rules of the README); a Stadia Controller appears on the system while one runs."""

from __future__ import annotations

import glob
import os
import sys
import time
from collections.abc import Iterator

import pytest

from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import PAD_NAME, UhidPad, input_report
from tests.test_linux_lab import Game, wait_until


def _uhid_opens() -> bool:
    try:
        os.close(os.open("/dev/uhid", os.O_RDWR))
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.environ.get("STADIACAST_LINUX_LAB") or not _uhid_opens(),
    reason="needs Linux, the right to open /dev/uhid and STADIACAST_LINUX_LAB=1",
)

EV_KEY, EV_ABS, EV_FF = 0x01, 0x03, 0x15
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TL, BTN_TR = 0x130, 0x131, 0x133, 0x134, 0x136, 0x137
BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL, BTN_THUMBR = 0x13A, 0x13B, 0x13C, 0x13D, 0x13E
BTN_TRIGGER_HAPPY1, BTN_TRIGGER_HAPPY2, BTN_TRIGGER_HAPPY3, BTN_TRIGGER_HAPPY4 = 0x2C0, 0x2C1, 0x2C2, 0x2C3
ABS_X, ABS_Y, ABS_Z, ABS_RZ, ABS_GAS, ABS_BRAKE, ABS_HAT0X, ABS_HAT0Y = 0, 1, 2, 5, 9, 10, 0x10, 0x11
#: What the kernel makes of a wired Stadia Controller, less the three media keys of its headset
#: socket: the keys in the order programs number them, and the axes.
KEYS = [BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TL, BTN_TR, BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL, BTN_THUMBR]
KEYS += [BTN_TRIGGER_HAPPY1, BTN_TRIGGER_HAPPY2, BTN_TRIGGER_HAPPY3, BTN_TRIGGER_HAPPY4]
AXES = {ABS_X, ABS_Y, ABS_Z, ABS_RZ, ABS_GAS, ABS_BRAKE, ABS_HAT0X, ABS_HAT0Y}


@pytest.fixture(scope="module", autouse=True)
def stadia_driver_loaded() -> None:
    """The first Stadia pad after a boot is taken twice: by the generic HID driver, and again by
    the kernel's Stadia driver once udev has loaded it for these ids, with new nodes each time.
    Let that pass with a pad of its own, so that the tests look at pads that stay as they are."""
    first = UhidPad("lab-first")
    try:
        deadline = time.monotonic() + 2  # a kernel without the driver keeps the generic one
        while time.monotonic() < deadline and not (first.device and _driver(first.device) == "stadia"):
            time.sleep(0.05)
    finally:
        first.close()


@pytest.fixture
def pad() -> Iterator[UhidPad]:
    made = UhidPad("lab")
    try:
        yield made
    finally:
        made.close()


def _event_node(hidraw: str) -> str:
    """The event node of the HID device behind a hidraw node."""
    name = os.path.basename(hidraw)
    found: list[str] = []
    wait_until(lambda: bool(found.extend(glob.glob(f"/sys/class/hidraw/{name}/device/input/input*/event*")) or found))
    return f"/dev/input/{os.path.basename(min(found))}"


def _open_hidraw(path: str) -> int:
    deadline = time.monotonic() + 3
    while True:  # udev sets the node's permissions a moment after the kernel made it
        try:
            return os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def _driver(hidraw: str) -> str:
    """The HID driver bound to the pad: ``stadia`` (hid-google-stadiaff) where the kernel has it."""
    return os.path.basename(os.path.realpath(f"/sys/class/hidraw/{os.path.basename(hidraw)}/device/driver"))


def test_the_kernel_takes_the_pad_for_a_stadia_controller(pad: UhidPad) -> None:
    assert pad.device is not None and os.path.exists(pad.device)
    game = Game(_event_node(pad.device))
    try:
        assert game.identity() == (0x03, 0x18D1, 0x9400, 0x0100)
        assert game.name() == PAD_NAME
        assert sorted(game.codes(EV_KEY)) == KEYS  # one more or one less would renumber the buttons
        assert game.codes(EV_ABS) == AXES
        for stick in (ABS_X, ABS_Y, ABS_Z, ABS_RZ):
            assert game.axis(stick)[1:3] == (1, 255)
        for trigger in (ABS_GAS, ABS_BRAKE):
            assert game.axis(trigger)[1:3] == (0, 255)
        assert game.axis(ABS_HAT0X)[1:3] == game.axis(ABS_HAT0Y)[1:3] == (-1, 1)
        driver = _driver(pad.device)
        assert driver in ("stadia", "hid-generic")
        assert (EV_FF in game.codes(0)) == (driver == "stadia")  # force feedback comes with the kernel's driver
    finally:
        game.close()


def test_a_pad_that_was_given_no_input_is_at_rest(pad: UhidPad) -> None:
    """A pad exists from the moment its controller connects. Before the first input its sticks
    are in the middle on the event device, which knows no value until a report comes, and a
    program that opens the hidraw node is told the same, as SDL 3 needs it."""
    assert pad.device is not None
    game = Game(_event_node(pad.device))
    try:
        wait_until(lambda: game.axis(ABS_X)[0] == 128)
        assert [game.axis(axis)[0] for axis in (ABS_X, ABS_Y, ABS_Z, ABS_RZ, ABS_GAS, ABS_BRAKE)] == [128] * 4 + [0, 0]
        assert (game.axis(ABS_HAT0X)[0], game.axis(ABS_HAT0Y)[0]) == (0, 0) and not game.held() & set(KEYS)
    finally:
        game.close()
    raw = _open_hidraw(pad.device)
    try:
        seen: list[bytes] = []

        def told() -> bool:
            try:
                seen.append(os.read(raw, 64))
            except BlockingIOError:
                return False
            return True

        wait_until(told)
        assert seen[0] == input_report(GamepadState())
    finally:
        os.close(raw)


def test_every_control_is_the_key_and_the_axis_of_the_wired_controller(pad: UhidPad) -> None:
    """Each button alone, then the D-pad, the sticks and the triggers: the event device says what
    a wired Stadia Controller's says for the same control."""
    assert pad.device is not None
    game = Game(_event_node(pad.device))
    try:
        pad.apply_state(GamepadState())
        wait_until(lambda: game.axis(ABS_X)[0] == 128)
        assert [game.axis(axis)[0] for axis in (ABS_X, ABS_Y, ABS_Z, ABS_RZ, ABS_GAS, ABS_BRAKE)] == [128] * 4 + [0, 0]
        assert not game.held() & set(KEYS)
        buttons = {
            Button.A: BTN_A,
            Button.B: BTN_B,
            Button.X: BTN_X,
            Button.Y: BTN_Y,
            Button.L1: BTN_TL,
            Button.R1: BTN_TR,
            Button.OPTIONS: BTN_SELECT,
            Button.MENU: BTN_START,
            Button.STADIA: BTN_MODE,
            Button.L3: BTN_THUMBL,
            Button.R3: BTN_THUMBR,
            Button.ASSISTANT: BTN_TRIGGER_HAPPY1,
            Button.CAPTURE: BTN_TRIGGER_HAPPY2,
        }
        for button, key in buttons.items():
            pad.apply_state(GamepadState(buttons=int(button)))
            wait_until(lambda key=key: game.held() & set(KEYS) == {key})
            pad.apply_state(GamepadState())
            wait_until(lambda: not game.held() & set(KEYS))
        hats = {
            Button.DPAD_UP: (0, -1),
            Button.DPAD_DOWN: (0, 1),
            Button.DPAD_LEFT: (-1, 0),
            Button.DPAD_RIGHT: (1, 0),
            Button.DPAD_UP | Button.DPAD_RIGHT: (1, -1),
        }
        for held, hat in hats.items():
            pad.apply_state(GamepadState(buttons=int(held)))
            wait_until(lambda hat=hat: (game.axis(ABS_HAT0X)[0], game.axis(ABS_HAT0Y)[0]) == hat)
        moves = [
            (GamepadState(lx=32767), ABS_X, 255),
            (GamepadState(lx=-32768), ABS_X, 1),
            (GamepadState(ly=32767), ABS_Y, 1),  # up is the small end
            (GamepadState(ly=-32768), ABS_Y, 255),
            (GamepadState(rx=32767), ABS_Z, 255),
            (GamepadState(ry=32767), ABS_RZ, 1),
            (GamepadState(l2=255), ABS_BRAKE, 255),
            (GamepadState(r2=255), ABS_GAS, 255),
        ]
        for state, axis, value in moves:
            pad.apply_state(state)
            wait_until(lambda axis=axis, value=value: game.axis(axis)[0] == value)
        pad.apply_state(GamepadState(l2=255, r2=255))
        wait_until(lambda: game.held() & set(KEYS) == {BTN_TRIGGER_HAPPY3, BTN_TRIGGER_HAPPY4})
        pad.apply_state(GamepadState(l2=255))
        wait_until(lambda: game.held() & set(KEYS) == {BTN_TRIGGER_HAPPY4})  # 4 is the left one
    finally:
        game.close()


def test_input_reaches_the_hidraw_node_as_the_report_and_the_event_node_as_keys(pad: UhidPad) -> None:
    assert pad.device is not None
    raw = _open_hidraw(pad.device)
    game = Game(_event_node(pad.device))

    def waiting() -> list[bytes]:
        """Every report that waits on the hidraw node."""
        found: list[bytes] = []
        while True:
            try:
                found.append(os.read(raw, 64))
            except BlockingIOError:
                return found

    try:
        game.events()
        state = GamepadState(buttons=int(Button.ASSISTANT | Button.CAPTURE | Button.A | Button.X), lx=32767, l2=255)
        pad.apply_state(state)
        reports: list[bytes] = []
        wait_until(lambda: bool(reports.extend(waiting()) or reports))
        assert reports[0] == input_report(state)  # what SDL's Stadia driver reads
        events: list[tuple[int, int, int]] = []
        wait_until(lambda: bool(events.extend(game.events()) or (EV_KEY, BTN_TRIGGER_HAPPY1, 1) in events))
        pressed = {code for kind, code, value in events if kind == EV_KEY and value == 1}
        assert pressed == {BTN_TRIGGER_HAPPY1, BTN_TRIGGER_HAPPY2, BTN_A, BTN_X, BTN_TRIGGER_HAPPY4}  # 4: L2 held
        assert (EV_ABS, ABS_X, 255) in events and (EV_ABS, ABS_BRAKE, 255) in events

        # The pad repeats its state while nothing changes: whoever reads the hidraw node gets the
        # report again, and the event device has nothing new to say.
        game.events()
        waiting()
        again: list[bytes] = []
        wait_until(lambda: len(again.extend(waiting()) or again) >= 3)
        assert set(again) == {input_report(state)}
        assert game.events() == []
    finally:
        game.close()
        os.close(raw)


def test_a_hidraw_write_of_report_5_is_rumble(pad: UhidPad) -> None:
    """What SDL does to rumble a Stadia Controller."""
    assert pad.device is not None
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    raw = _open_hidraw(pad.device)
    try:
        assert os.write(raw, b"\x05\x00\x80\x00\x40") == 5
        wait_until(lambda: (0x80, 0x40) in levels)
        os.write(raw, b"\x05\x00\x00\x00\x00")
        wait_until(lambda: levels[-1] == (0, 0))
    finally:
        os.close(raw)


def test_a_rumble_left_running_stops_when_its_program_closes_the_pad(pad: UhidPad) -> None:
    """A program that ends while its rumble runs: nothing stops the motors but the kernel's word
    that the pad's nodes are closed. No other program may hold a node of the pad open here."""
    assert pad.device is not None
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    raw = _open_hidraw(pad.device)
    try:
        os.write(raw, b"\x05\x00\x80\x00\x40")
        wait_until(lambda: (0x80, 0x40) in levels)
    finally:
        os.close(raw)
    wait_until(lambda: levels[-1] == (0, 0))


def test_a_driver_that_takes_the_pad_anew_finds_the_state_it_had(pad: UhidPad) -> None:
    """What the first pad after a boot goes through, done by hand: the device is taken from its
    driver and given to one again, which makes a new event device. That one knows no report (its
    sticks read 0, held up and to the left) until the pad writes its state again."""
    if os.geteuid() != 0:
        pytest.skip("taking a device from its driver by hand takes root")
    assert pad.device is not None
    held = GamepadState(buttons=int(Button.A), lx=32767, ly=32767, r2=255)
    pad.apply_state(held)
    hid = os.path.realpath(f"/sys/class/hidraw/{os.path.basename(pad.device)}/device")
    with open(os.path.join(hid, "driver", "unbind"), "w", encoding="ascii") as f:
        f.write(os.path.basename(hid))
    with open("/sys/bus/hid/drivers_probe", "w", encoding="ascii") as f:
        f.write(os.path.basename(hid))

    def state() -> tuple[int, int, int, bool] | None:
        nodes = glob.glob(os.path.join(hid, "input", "input*", "event*"))
        if not nodes:
            return None
        try:
            game = Game(f"/dev/input/{os.path.basename(nodes[0])}")
        except OSError:
            return None
        try:
            return game.axis(ABS_X)[0], game.axis(ABS_Y)[0], game.axis(ABS_GAS)[0], BTN_A in game.held()
        finally:
            game.close()

    wait_until(lambda: state() == (255, 1, 255, True))
    assert pad.device is not None and os.path.exists(pad.device)
    assert os.path.realpath(f"/sys/class/hidraw/{os.path.basename(pad.device)}/device") == hid


def test_a_games_force_feedback_reaches_the_motors_through_the_kernels_driver(pad: UhidPad) -> None:
    """A game that rumbles through the event node: hid-google-stadiaff turns the effect into a
    SET_REPORT of report 5, which the pad answers."""
    assert pad.device is not None
    if _driver(pad.device) != "stadia":
        pytest.skip("hid-google-stadiaff did not take the pad (not built, or not loaded)")
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    game = Game(_event_node(pad.device))
    try:
        slot = game.upload(0x8000, 0x4000, 0)
        game.play(slot)
        wait_until(lambda: (0x80, 0x40) in levels)
        game.play(slot, 0)
        wait_until(lambda: levels[-1] == (0, 0))
    finally:
        game.close()
