# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127488() -> bool:
    """Return True if PGN 127488 is a fast PGN."""
    return False
def decode_pgn_127488(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127488."""
    nmea2000Message = NMEA2000Message(PGN=127488, id='engineParametersRapidUpdate', description='Engine Parameters, Rapid Update', ttl=timedelta(milliseconds=100))
    running_bit_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance_raw = decode_int(_data_raw_, running_bit_offset, 8)
    instance = master_dict['ENGINE_INSTANCE'].get(instance_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 8

    # 2:speed | Offset: 8, Length: 16, Signed: False Resolution: 0.25, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    speed = speed_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.25, 0, 16383)
    nmea2000Message.fields.append(NMEA2000Field('speed', 'Speed', None, 'rpm', speed, speed_raw, PhysicalQuantities.ANGULAR_VELOCITY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:boost_pressure | Offset: 24, Length: 16, Signed: False Resolution: 100, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    boost_pressure = boost_pressure_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 100, 0, 6553200)
    nmea2000Message.fields.append(NMEA2000Field('boostPressure', 'Boost Pressure', None, 'Pa', boost_pressure, boost_pressure_raw, PhysicalQuantities.PRESSURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:tilt_trim | Offset: 40, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    tilt_trim = tilt_trim_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('tiltTrim', 'Tilt/Trim', None, '%', tilt_trim, tilt_trim_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:reserved_48 | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    reserved_48 = reserved_48_raw = decode_int(_data_raw_, running_bit_offset, 16)
    nmea2000Message.fields.append(NMEA2000Field('reserved_48', 'Reserved', None, None, reserved_48, reserved_48_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_127488(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127488."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # instance | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("instance")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENGINE_INSTANCE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speed | Offset: 8, Length: 16, Resolution: 0.25, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speed")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.25):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.25)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.25)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Speed' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # boostPressure | Offset: 24, Length: 16, Resolution: 100, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("boostPressure")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 100):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 100)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 100)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Boost Pressure' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Boost Pressure' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Boost Pressure' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tiltTrim | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tiltTrim")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Tilt/Trim' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tilt/Trim' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tilt/Trim' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_48 | Offset: 48, Length: 16, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_48")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Reserved' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Reserved' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Reserved' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
