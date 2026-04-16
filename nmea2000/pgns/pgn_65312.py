# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65312() -> bool:
    """Return True if PGN 65312 is a fast PGN."""
    return False
def decode_pgn_65312(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65312."""
    nmea2000Message = NMEA2000Message(PGN=65312, id='navicoWirelessSignalStatus', description='Navico: Wireless Signal Status')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 275, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Navico", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 2:reserved_11 | Offset: 11, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    reserved_11 = reserved_11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_11', 'Reserved', None, None, reserved_11, reserved_11_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 3:industry_code | Offset: 13, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 13
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Marine Industry", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 4:unknown | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    unknown = unknown_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('unknown', 'Unknown', None, None, unknown, unknown_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:signal_strength | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    signal_strength = signal_strength_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('signalStrength', 'Signal Strength', None, '%', signal_strength, signal_strength_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:reserved_32 | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    reserved_32 = reserved_32_raw = decode_int(_data_raw_, running_bit_offset, 32)
    nmea2000Message.fields.append(NMEA2000Field('reserved_32', 'Reserved', None, None, reserved_32, reserved_32_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_65312(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65312."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # manufacturerCode | Offset: 0, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_11 | Offset: 11, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_11")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 2
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
    # industryCode | Offset: 13, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 13
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # unknown | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("unknown")

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
        raise ValueError("Cant encode this message, 'Unknown' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Unknown' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Unknown' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # signalStrength | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("signalStrength")

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
        raise ValueError("Cant encode this message, 'Signal Strength' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Signal Strength' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Signal Strength' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_32 | Offset: 32, Length: 32, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_32")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 32
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
