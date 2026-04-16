# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_59392() -> bool:
    """Return True if PGN 59392 is a fast PGN."""
    return False
# ERROR: This PGN is corrupted. It has multiple fields but none of them have a match attribute.
def decode_pgn_59392(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 59392."""
    nmea2000Message = NMEA2000Message(PGN=59392, id='0xe8000xee00StandardizedSingleFrameAddressed', description='0xE800-0xEE00: Standardized single-frame addressed')
    running_bit_offset = 0
    # 1:data | Offset: 0, Length: 64, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    data = data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 64))
    nmea2000Message.fields.append(NMEA2000Field('data', 'Data', None, None, data, data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 64

    return nmea2000Message

def encode_pgn_59392(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 59392."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # data | Offset: 0, Length: 64, Resolution: 1, Field Type: BINARY
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("data")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 64
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Data' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_59392(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 59392."""
    nmea2000Message = NMEA2000Message(PGN=59392, id='isoAcknowledgement', description='ISO Acknowledgement')
    running_bit_offset = 0
    # 1:control | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    control_raw = decode_int(_data_raw_, running_bit_offset, 8)
    control = master_dict['ISO_CONTROL'].get(control_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('control', 'Control', None, None, control, control_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:group_function | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    group_function = group_function_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('groupFunction', 'Group Function', None, None, group_function, group_function_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:reserved_16 | Offset: 16, Length: 24, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    reserved_16 = reserved_16_raw = decode_int(_data_raw_, running_bit_offset, 24)
    nmea2000Message.fields.append(NMEA2000Field('reserved_16', 'Reserved', None, None, reserved_16, reserved_16_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 24

    # 4:pgn | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Parameter Group Number of requested information", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_59392(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 59392."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # control | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("control")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ISO_CONTROL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupFunction | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupFunction")

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
        raise ValueError("Cant encode this message, 'Group Function' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Function' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Function' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_16 | Offset: 16, Length: 24, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_16")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 24
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
    # pgn | Offset: 40, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pgn")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 1)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'PGN' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PGN' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PGN' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
