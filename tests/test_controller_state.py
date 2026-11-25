"""What the controller reports about itself: the hardware fields, the battery level, the headset."""

from __future__ import annotations

import asyncio
import struct

import pytest

from stadiacast.controller import _as_level
from tests.fake_controller import FakeController
from tests.helpers import connect_firmware_flow, make_receiver, single, wait_for


def test_hardware_battery_and_headset_reach_the_status(tmp_path) -> None:
    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the first periodic request is what enters READY
            await wait_for(lambda: cs.session["display"]["state"] == "ready")
            hardware = [r for r in rx.log.records if r["event"] == "controller_hardware"]
            assert len(hardware) == 1
            seen = (hardware[0]["firmware"], hardware[0]["hardware"], hardware[0]["headset"])
            assert seen == ("282115", "gotham_pvt_3", False)
            assert cs.status()["headset"] is False  # the f4 snapshot: nothing in the jack
            # The level its state gave at connection: the gauge has not reported yet.
            assert cs.status()["battery_percent"] == hardware[0]["battery_percent"] == 80
            assert cs.session["display"]["battery"] == 0.8 and cs.session["display"]["battery_samples"] == 0

            fc.send_battery(0.76)
            await wait_for(lambda: cs.status()["battery_percent"] == 76)
            fc.send_battery(0.76)  # a repeat is counted but kept off the console
            await wait_for(lambda: cs.session["display"]["battery_samples"] == 2)
            assert rx.log.counts["controller_battery"] == 2
            assert cs.session["display"]["battery"] == 0.76

            fc.send_audio_config(1)
            await wait_for(lambda: cs.status()["headset"] is True)
            assert cs.session["display"]["audio_mode"] == 1
            fc.send_audio_config(0)
            await wait_for(lambda: cs.status()["headset"] is False)
            audio = [r for r in rx.log.records if r["event"] == "controller_audio_config"]
            assert len(audio) == 2  # other records may follow: the periodic ones are not ours to time
            assert audio[-1]["audio"]["device_info"]["f6"] == 9989
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_the_serial_falls_back_to_the_controller_state(tmp_path, monkeypatch) -> None:
    """A controller that sends no controller-device-id header is still named by its serial: f17."""
    monkeypatch.setattr(FakeController, "rpc_headers", lambda _self: [])

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the first periodic request is what enters READY
            await wait_for(lambda: cs.session["display"]["state"] == "ready")
            assert cs.serial == "FAKESERIAL0001"
            hardware = next(r for r in rx.log.records if r["event"] == "controller_hardware")
            assert hardware["serial"] == "FAKESERIAL0001"
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def _bits(level: float) -> int:
    return struct.unpack("<I", struct.pack("<f", level))[0]


@pytest.mark.parametrize(
    ("field", "level"),
    [
        (_bits(0.64), 0.64),
        (_bits(1.0), 1.0),
        (0, 0.0),
        (_bits(1.5), None),  # no level
        (_bits(-0.1), None),
        (0x7FC00000, None),  # not a number
        (None, None),  # a state without the field
        ("0.5", None),
    ],
)
def test_the_states_battery_field_is_a_floats_bits(field, level) -> None:
    got = _as_level(field)
    assert got == pytest.approx(level) if level is not None else got is None


def test_a_state_without_a_battery_level_leaves_it_to_the_gauge(tmp_path) -> None:
    """Where the controller's state gave no level, it is unknown until the gauge reports."""

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        fc.battery_level = None  # before the receiver asks for the state, a moment after the link is up
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: cs.session["display"]["state"] == "ready")
            assert cs.status()["battery_percent"] is None

            state = next(r for r in rx.log.records if r["event"] == "display_controller_state")
            assert state["battery_percent"] is None and "f10" not in state["controller_state"]

            fc.send_battery(0.41)
            await wait_for(lambda: cs.status()["battery_percent"] == 41)
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())
