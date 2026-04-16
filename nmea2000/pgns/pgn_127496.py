# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127496() -> bool:
    """Return True if PGN 127496 is a fast PGN."""
    return True
def decode_pgn_127496(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127496."""
    nmea2000Message = NMEA2000Message(PGN=127496, id='tripParametersVessel', description='Trip Parameters, Vessel', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:time_to_empty | Offset: 0, Length: 32, Signed: False Resolution: 0.001, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    time_to_empty = time_to_empty_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.001, 0, 4294967.292)
    nmea2000Message.fields.append(NMEA2000Field('timeToEmpty', 'Time to Empty', None, 's', time_to_empty, time_to_empty_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 2:distance_to_empty | Offset: 32, Length: 32, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    distance_to_empty = distance_to_empty_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.01, 0, 42949672.92)
    nmea2000Message.fields.append(NMEA2000Field('distanceToEmpty', 'Distance to Empty', None, 'm', distance_to_empty, distance_to_empty_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 3:estimated_fuel_remaining | Offset: 64, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    estimated_fuel_remaining = estimated_fuel_remaining_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('estimatedFuelRemaining', 'Estimated Fuel Remaining', None, 'L', estimated_fuel_remaining, estimated_fuel_remaining_raw, PhysicalQuantities.VOLUME, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:trip_run_time | Offset: 80, Length: 32, Signed: False Resolution: 0.001, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    trip_run_time = trip_run_time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.001, 0, 4294967.292)
    nmea2000Message.fields.append(NMEA2000Field('tripRunTime', 'Trip Run Time', None, 's', trip_run_time, trip_run_time_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_127496(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127496."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # timeToEmpty | Offset: 0, Length: 32, Resolution: 0.001, Field Type: DURATION
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeToEmpty")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.001)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time to Empty' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time to Empty' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time to Empty' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # distanceToEmpty | Offset: 32, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("distanceToEmpty")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, False, 0.01)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Distance to Empty' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Distance to Empty' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Distance to Empty' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # estimatedFuelRemaining | Offset: 64, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("estimatedFuelRemaining")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Estimated Fuel Remaining' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Estimated Fuel Remaining' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Estimated Fuel Remaining' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tripRunTime | Offset: 80, Length: 32, Resolution: 0.001, Field Type: DURATION
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tripRunTime")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.001)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Trip Run Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Trip Run Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Trip Run Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(14, byteorder="little")
