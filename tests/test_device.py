import asyncio
from datetime import datetime

import pytest

from nmea2000.device import N2KDevice
from nmea2000.encoder import NMEA2000Encoder
from nmea2000.ioclient import State
from nmea2000.message import NMEA2000Field, NMEA2000Message


class FakeClient:
    def __init__(self):
        self.state = State.DISCONNECTED
        self.receive_callback = None
        self.status_callback = None
        self.sent_messages: list[NMEA2000Message] = []
        self.encoder = NMEA2000Encoder()

    def set_receive_callback(self, callback):
        self.receive_callback = callback

    def set_status_callback(self, callback):
        self.status_callback = callback

    async def connect(self):
        self.state = State.CONNECTED
        if self.status_callback is not None:
            await self.status_callback(self.state)

    async def close(self):
        self.state = State.CLOSED
        if self.status_callback is not None:
            await self.status_callback(self.state)

    async def send(self, message: NMEA2000Message):
        self.sent_messages.append(message)

    async def emit(self, message: NMEA2000Message):
        if self.receive_callback is not None:
            await self.receive_callback(message)


def _build_iso_request(requested_pgn: int, *, source: int = 10, destination: int = 255) -> NMEA2000Message:
    return NMEA2000Message(
        PGN=59904,
        id="isoRequest",
        description="ISO Request",
        source=source,
        destination=destination,
        priority=6,
        timestamp=datetime.now(),
        fields=[NMEA2000Field("pgn", value=requested_pgn, raw_value=requested_pgn)],
    )


def _build_address_claim(source: int, unique_number: int) -> NMEA2000Message:
    return NMEA2000Message(
        PGN=60928,
        id="isoAddressClaim",
        description="ISO Address Claim",
        source=source,
        destination=255,
        priority=6,
        timestamp=datetime.now(),
        fields=[
            NMEA2000Field("uniqueNumber", value=unique_number, raw_value=unique_number),
            NMEA2000Field("manufacturerCode", value=999, raw_value=999),
            NMEA2000Field("deviceInstanceLower", value=0, raw_value=0),
            NMEA2000Field("deviceInstanceUpper", value=0, raw_value=0),
            NMEA2000Field("deviceFunction", value=130, raw_value=130),
            NMEA2000Field("spare", value=1, raw_value=1),
            NMEA2000Field("deviceClass", value=25, raw_value=25),
            NMEA2000Field("systemInstance", value=0, raw_value=0),
            NMEA2000Field("industryGroup", value=4, raw_value=4),
            NMEA2000Field("arbitraryAddressCapable", value=1, raw_value=1),
        ],
    )


def _build_group_function_request(source: int, requested_pgn: int, destination: int) -> NMEA2000Message:
    return NMEA2000Message(
        PGN=126208,
        id="nmeaRequestGroupFunction",
        description="NMEA - Request group function",
        source=source,
        destination=destination,
        priority=3,
        timestamp=datetime.now(),
        fields=[
            NMEA2000Field("pgn", value=requested_pgn, raw_value=requested_pgn),
            NMEA2000Field("numberOfParameters", value=0, raw_value=0),
            NMEA2000Field("parameter", value=0, raw_value=0),
        ],
    )


@pytest.mark.asyncio
async def test_device_start_claims_address_and_filters_management_messages(tmp_path):
    client = FakeClient()
    device = N2KDevice(
        client,
        persistence_path=tmp_path / "device.json",
        address_claim_startup_delay=0,
        address_claim_detection_time=0.01,
        heartbeat_interval=3600,
    )

    data_messages = asyncio.Queue()
    raw_messages = asyncio.Queue()

    async def handle_data(message: NMEA2000Message):
        await data_messages.put(message)

    async def handle_raw(message: NMEA2000Message):
        await raw_messages.put(message)

    device.set_receive_callback(handle_data)
    device.set_raw_receive_callback(handle_raw)

    await device.start()
    await device.wait_ready(timeout=1)

    assert [message.PGN for message in client.sent_messages[:2]] == [59904, 60928]
    assert device.ready is True

    await client.emit(_build_iso_request(126996, source=31, destination=255))
    response = client.sent_messages[-1]
    assert response.PGN == 126996
    assert data_messages.empty()
    raw_message = await raw_messages.get()
    assert raw_message.PGN == 59904

    data_message = NMEA2000Message(PGN=127250, id="vesselHeading", source=44, destination=255, priority=2)
    await client.emit(data_message)
    forwarded = await asyncio.wait_for(data_messages.get(), timeout=1)
    assert forwarded.PGN == 127250


@pytest.mark.asyncio
async def test_device_conflict_increments_address_when_it_loses(tmp_path):
    client = FakeClient()
    device = N2KDevice(
        client,
        unique_number=10,
        persistence_path=tmp_path / "device.json",
        address_claim_startup_delay=0,
        address_claim_detection_time=0.01,
        heartbeat_interval=3600,
    )

    await device.start()
    await device.wait_ready(timeout=1)
    assert device.address == 100

    await client.emit(_build_address_claim(100, unique_number=1))
    await asyncio.sleep(0.05)

    assert device.address == 101
    assert client.sent_messages[-1].PGN == 60928
    assert client.sent_messages[-1].source == 101


@pytest.mark.asyncio
async def test_device_responds_with_iso_nak_and_group_function_ack(tmp_path):
    client = FakeClient()
    device = N2KDevice(
        client,
        persistence_path=tmp_path / "device.json",
        address_claim_startup_delay=0,
        address_claim_detection_time=0.01,
        heartbeat_interval=3600,
    )

    await device.start()
    await device.wait_ready(timeout=1)

    await client.emit(_build_iso_request(130000, source=22, destination=255))
    iso_nak = client.sent_messages[-1]
    assert iso_nak.PGN == 59392
    assert iso_nak.destination == 22

    await client.emit(_build_group_function_request(25, 130000, device.address))
    group_ack = client.sent_messages[-1]
    assert group_ack.PGN == 126208
    assert group_ack.id == "nmeaAcknowledgeGroupFunction"
    assert group_ack.destination == 25