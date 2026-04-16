# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128008() -> bool:
    """Return True if PGN 128008 is a fast PGN."""
    return False
def decode_pgn_128008(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128008."""
    nmea2000Message = NMEA2000Message(PGN=128008, id='thrusterMotorStatus', description='Thruster Motor Status')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:identifier | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    identifier = identifier_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('identifier', 'Identifier', None, None, identifier, identifier_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:motor_events | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    motor_events_raw = decode_int(_data_raw_, running_bit_offset, 8)
    motor_events = decode_bit_lookup(motor_events_raw, master_flags_dict['THRUSTER_MOTOR_EVENTS'])
    nmea2000Message.fields.append(NMEA2000Field('motorEvents', 'Motor Events', None, None, motor_events, motor_events_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 8

    # 4:current | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    current = current_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('current', 'Current', None, 'A', current, current_raw, PhysicalQuantities.ELECTRICAL_CURRENT, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:temperature | Offset: 32, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    temperature = temperature_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('temperature', 'Temperature', None, 'K', temperature, temperature_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:operating_time | Offset: 48, Length: 16, Signed: False Resolution: 60, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    operating_time = operating_time_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 60, 0, 3931920)
    nmea2000Message.fields.append(NMEA2000Field('operatingTime', 'Operating Time', None, 's', operating_time, operating_time_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_128008(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128008."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # sid | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sid")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, False, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'SID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # identifier | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("identifier")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, False, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Identifier' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Identifier' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Identifier' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorEvents | Offset: 16, Length: 8, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorEvents")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['THRUSTER_MOTOR_EVENTS'])
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Motor Events' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor Events' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor Events' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # current | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("current")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, False, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Current' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Current' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Current' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # temperature | Offset: 32, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("temperature")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Temperature' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Temperature' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Temperature' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # operatingTime | Offset: 48, Length: 16, Resolution: 60, Field Type: DURATION
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("operatingTime")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 60)):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 60)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 16, False, 60)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Operating Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Operating Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Operating Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
