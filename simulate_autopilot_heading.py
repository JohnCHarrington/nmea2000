from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from nmea2000.device import N2KDevice
from nmea2000.encoder import NMEA2000Encoder
from nmea2000.ioclient import State
from nmea2000.message import NMEA2000Field, NMEA2000Message

logger = logging.getLogger(__name__)

AUTOPILOT_DEVICE_CLASS = 40
AUTOPILOT_DEVICE_FUNCTION = 150
AUTOPILOT_TRANSMIT_PGNS = [59904, 60928, 126464, 126993, 126996, 126998, 127250]

StatusCallback = Callable[[State], Awaitable[None]]
MessageCallback = Callable[[NMEA2000Message], Awaitable[None]]


class DryRunClient:
    def __init__(self) -> None:
        self.state = State.DISCONNECTED
        self.receive_callback: MessageCallback | None = None
        self.status_callback: StatusCallback | None = None
        self.sent_messages: list[NMEA2000Message] = []
        self.encoder = NMEA2000Encoder()

    def set_receive_callback(self, callback: MessageCallback | None) -> None:
        self.receive_callback = callback

    def set_status_callback(self, callback: StatusCallback | None) -> None:
        self.status_callback = callback

    async def connect(self) -> None:
        self.state = State.CONNECTED
        if self.status_callback is not None:
            await self.status_callback(self.state)

    async def close(self) -> None:
        self.state = State.CLOSED
        if self.status_callback is not None:
            await self.status_callback(self.state)

    async def send(self, message: NMEA2000Message) -> None:
        self.sent_messages.append(message)


@dataclass
class SmoothHeadingGenerator:
    heading_deg: float
    max_step_deg: float = 3.0
    smoothing: float = 0.85
    delta_deg: float = 0.0

    def next(self) -> tuple[float, float]:
        target_delta = random.uniform(-self.max_step_deg, self.max_step_deg)
        self.delta_deg = (self.delta_deg * self.smoothing) + (
            target_delta * (1.0 - self.smoothing)
        )
        self.delta_deg = max(-self.max_step_deg, min(self.max_step_deg, self.delta_deg))
        self.heading_deg = (self.heading_deg + self.delta_deg) % 360.0
        return self.heading_deg, self.delta_deg


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value, 0)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_client_args(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --client-arg '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        result[key] = parse_scalar(raw_value)
    return result


def build_heading_message(sid: int, heading_deg: float) -> NMEA2000Message:
    heading_rad = math.radians(heading_deg % 360.0)
    unavailable_angle_raw = (1 << 15) - 1
    magnetic_reference_raw = 1
    reserved_na_raw = (1 << 6) - 1
    return NMEA2000Message(
        PGN=127250,
        id="vesselHeading",
        description="Vessel Heading",
        source=0,
        destination=255,
        priority=2,
        timestamp=datetime.now(),
        fields=[
            NMEA2000Field("sid", value=sid, raw_value=sid),
            NMEA2000Field("heading", value=heading_rad, raw_value=heading_rad),
            NMEA2000Field(
                "deviation",
                value=None,
                raw_value=unavailable_angle_raw,
                encoded_value=unavailable_angle_raw,
            ),
            NMEA2000Field(
                "variation",
                value=None,
                raw_value=unavailable_angle_raw,
                encoded_value=unavailable_angle_raw,
            ),
            NMEA2000Field("reference", value="Magnetic", raw_value=magnetic_reference_raw),
            NMEA2000Field("reserved_58", value=None, raw_value=reserved_na_raw),
        ],
    )


def build_device(args: argparse.Namespace) -> N2KDevice:
    device_options = {
        "preferred_address": args.address,
        "unique_number": args.unique_number,
        "manufacturer_code": args.manufacturer_code,
        "device_function": AUTOPILOT_DEVICE_FUNCTION,
        "device_class": AUTOPILOT_DEVICE_CLASS,
        "model_id": args.model_id,
        "model_version": args.model_version,
        "product_code": args.product_code,
        "manufacturer_information": args.manufacturer_information,
        "installation_description1": args.installation_description1,
        "installation_description2": args.installation_description2,
        "transmit_pgns": AUTOPILOT_TRANSMIT_PGNS,
        "address_claim_startup_delay": args.startup_delay,
        "address_claim_detection_time": args.claim_detection_time,
        "heartbeat_interval": args.heartbeat_interval,
        "persistence_path": args.persistence_path,
        "persistence_key": args.persistence_key,
    }
    client_options = parse_client_args(args.client_arg)

    if args.transport == "dry-run":
        return N2KDevice(DryRunClient(), **device_options)
    if args.transport == "ebyte":
        if args.host is None or args.port is None:
            raise ValueError("--host and --port are required for ebyte transport")
        return N2KDevice.for_ebyte(
            args.host, args.port, client_options=client_options, **device_options
        )
    if args.transport == "yacht-devices":
        if args.host is None or args.port is None:
            raise ValueError(
                "--host and --port are required for yacht-devices transport"
            )
        return N2KDevice.for_yacht_devices(
            args.host, args.port, client_options=client_options, **device_options
        )
    if args.transport == "waveshare":
        if args.serial_port is None:
            raise ValueError("--serial-port is required for waveshare transport")
        return N2KDevice.for_waveshare(
            args.serial_port, client_options=client_options, **device_options
        )
    if args.transport == "python-can":
        if args.interface is None or args.channel is None:
            raise ValueError(
                "--interface and --channel are required for python-can transport"
            )
        return N2KDevice.for_python_can(
            args.interface,
            args.channel,
            client_options=client_options,
            **device_options,
        )
    raise ValueError(f"Unsupported transport: {args.transport}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a demo NMEA 2000 autopilot device and emit smoothly changing heading values."
    )
    parser.add_argument(
        "--transport",
        choices=["dry-run", "ebyte", "yacht-devices", "waveshare", "python-can"],
        default="dry-run",
        help="Transport used to send the simulated autopilot data",
    )
    parser.add_argument("--host", help="Gateway host for ebyte/yacht-devices")
    parser.add_argument("--port", type=int, help="Gateway port for ebyte/yacht-devices")
    parser.add_argument("--serial-port", help="Serial port for waveshare transport")
    parser.add_argument("--interface", help="python-can interface, e.g. slcan or socketcan")
    parser.add_argument("--channel", help="python-can channel, e.g. /dev/ttyUSB0 or can0")
    parser.add_argument(
        "--client-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra client constructor option, may be repeated",
    )

    parser.add_argument("--address", type=int, default=100, help="Preferred source address")
    parser.add_argument("--unique-number", type=int, help="Override the generated unique number")
    parser.add_argument("--manufacturer-code", type=int, default=999)
    parser.add_argument("--product-code", type=int, default=12725)
    parser.add_argument("--model-id", default="Demo Autopilot")
    parser.add_argument("--model-version", default="heading-simulator")
    parser.add_argument("--manufacturer-information", default="nmea2000 autopilot heading simulator")
    parser.add_argument("--installation-description1", default="Autopilot demo")
    parser.add_argument("--installation-description2", default="")
    parser.add_argument("--persistence-path", help="Optional persistence file path for unique number/address")
    parser.add_argument("--persistence-key", default="autopilot-heading-demo")

    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between heading messages; 0.1s matches PGN 127250's nominal 10 Hz cadence",
    )
    parser.add_argument("--count", type=int, help="Number of heading messages to send before exiting")
    parser.add_argument("--start-heading", type=float, default=90.0, help="Initial heading in degrees")
    parser.add_argument("--max-step", type=float, default=1.5, help="Maximum per-sample heading change in degrees")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.85,
        help="Fraction of the previous delta to keep; higher values are smoother",
    )
    parser.add_argument("--seed", type=int, help="Random seed for repeatable heading movement")
    parser.add_argument("--heartbeat-interval", type=float, default=60.0)
    parser.add_argument("--startup-delay", type=float, default=0.1)
    parser.add_argument("--claim-detection-time", type=float, default=0.25)
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


async def heading_loop(device: N2KDevice, args: argparse.Namespace) -> None:
    generator = SmoothHeadingGenerator(
        heading_deg=args.start_heading,
        max_step_deg=args.max_step,
        smoothing=args.smoothing,
    )
    sid = 0
    sent = 0

    while args.count is None or sent < args.count:
        heading_deg, delta_deg = generator.next()
        await device.send(build_heading_message(sid, heading_deg))
        print(
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"src={device.address:03d} sid={sid:03d} "
            f"heading={heading_deg:7.2f} deg delta={delta_deg:+.2f} deg",
            flush=True,
        )
        sid = (sid + 1) % 253
        sent += 1
        await asyncio.sleep(args.interval)


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.max_step <= 0:
        raise SystemExit("--max-step must be positive")
    if not 0 <= args.smoothing < 1:
        raise SystemExit("--smoothing must be between 0 and 1")
    if args.count is not None and args.count <= 0:
        raise SystemExit("--count must be positive when provided")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.seed is not None:
        random.seed(args.seed)

    device = build_device(args)

    try:
        await device.start()
        await device.wait_ready(timeout=args.ready_timeout)
        print(
            f"Autopilot ready on source address {device.address} "
            f"(unique number {device.unique_number}, transport={args.transport})",
            flush=True,
        )
        await heading_loop(device, args)
    finally:
        await device.close()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
