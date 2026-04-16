# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129033() -> bool:
    """Return True if PGN 129033 is a fast PGN."""
    return False
def decode_pgn_129033(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129033."""
    nmea2000Message = NMEA2000Message(PGN=129033, id='timeDate', description='Time & Date', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:date | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    date_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    date = decode_date(date_raw)
    nmea2000Message.fields.append(NMEA2000Field('date', 'Date', None, 'd', date, date_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 2:time | Offset: 16, Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time = decode_time(time_raw)
    nmea2000Message.fields.append(NMEA2000Field('time', 'Time', "Seconds since midnight", 's', time, time_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 3:local_offset | Offset: 48, Length: 16, Signed: True Resolution: 60, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    local_offset = local_offset_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 60, -1966020, 1965840)
    nmea2000Message.fields.append(NMEA2000Field('localOffset', 'Local Offset', None, 's', local_offset, local_offset_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_129033(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129033."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # date | Offset: 0, Length: 16, Resolution: 1, Field Type: DATE
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("date")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Date' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Date' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Date' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # time | Offset: 16, Length: 32, Resolution: 0.0001, Field Type: TIME
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("time")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.0001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # localOffset | Offset: 48, Length: 16, Resolution: 60, Field Type: DURATION
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("localOffset")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 60)):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 60)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 16, True, 60)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Local Offset' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Local Offset' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Local Offset' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
