# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65350() -> bool:
    """Return True if PGN 65350 is a fast PGN."""
    return False
def decode_pgn_65350(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65350."""
    nmea2000Message = NMEA2000Message(PGN=65350, id='simnetMagneticField', description='Simnet: Magnetic Field')
    running_bit_offset = 0
    # 1:a | Offset: 0, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    a = a_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('a', 'A', None, 'rad', a, a_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:b | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    b = b_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('b', 'B', None, '%', b, b_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:c | Offset: 24, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    c = c_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('c', 'C', None, 'rad', c, c_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:d | Offset: 40, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    d = d_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('d', 'D', None, 'rad', d, d_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:reserved_56 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    reserved_56 = reserved_56_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_56', 'Reserved', None, None, reserved_56, reserved_56_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_65350(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65350."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # a | Offset: 0, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("a")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'A' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'A' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'A' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # b | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("b")

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
        raise ValueError("Cant encode this message, 'B' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'B' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'B' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # c | Offset: 24, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("c")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'C' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'C' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'C' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # d | Offset: 40, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("d")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'D' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'D' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'D' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_56 | Offset: 56, Length: 8, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_56")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 8
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
