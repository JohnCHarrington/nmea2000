# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130576() -> bool:
    """Return True if PGN 130576 is a fast PGN."""
    return False
def decode_pgn_130576(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130576."""
    nmea2000Message = NMEA2000Message(PGN=130576, id='smallCraftStatus', description='Small Craft Status', ttl=timedelta(milliseconds=200))
    running_bit_offset = 0
    # 1:port_trim_tab | Offset: 0, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    port_trim_tab = port_trim_tab_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('portTrimTab', 'Port trim tab', None, '%', port_trim_tab, port_trim_tab_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:starboard_trim_tab | Offset: 8, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    starboard_trim_tab = starboard_trim_tab_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('starboardTrimTab', 'Starboard trim tab', None, '%', starboard_trim_tab, starboard_trim_tab_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:reserved_16 | Offset: 16, Length: 48, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    reserved_16 = reserved_16_raw = decode_int(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('reserved_16', 'Reserved', None, None, reserved_16, reserved_16_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 48

    return nmea2000Message

def encode_pgn_130576(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130576."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # portTrimTab | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("portTrimTab")

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
        raise ValueError("Cant encode this message, 'Port trim tab' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Port trim tab' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Port trim tab' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # starboardTrimTab | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("starboardTrimTab")

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
        raise ValueError("Cant encode this message, 'Starboard trim tab' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Starboard trim tab' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Starboard trim tab' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_16 | Offset: 16, Length: 48, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_16")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 48
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
