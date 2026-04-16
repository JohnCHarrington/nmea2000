# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65379() -> bool:
    """Return True if PGN 65379 is a fast PGN."""
    return False
def decode_pgn_65379(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65379."""
    nmea2000Message = NMEA2000Message(PGN=65379, id='seatalkPilotMode', description='Seatalk: Pilot Mode')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1851, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Raymarine", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:pilot_mode | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    pilot_mode_raw = decode_int(_data_raw_, running_bit_offset, 16)
    pilot_mode = master_dict['SEATALK_PILOT_MODE_16'].get(pilot_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('pilotMode', 'Pilot Mode', None, None, pilot_mode, pilot_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 16

    # 5:sub_mode | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    sub_mode = sub_mode_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 16))
    nmea2000Message.fields.append(NMEA2000Field('subMode', 'Sub Mode', None, None, sub_mode, sub_mode_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 16

    # 6:pilot_mode_data | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    pilot_mode_data = pilot_mode_data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 8))
    nmea2000Message.fields.append(NMEA2000Field('pilotModeData', 'Pilot Mode Data', None, None, pilot_mode_data, pilot_mode_data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 8

    # 7:reserved_56 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    reserved_56 = reserved_56_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_56', 'Reserved', None, None, reserved_56, reserved_56_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_65379(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65379."""
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
    # pilotMode | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pilotMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SEATALK_PILOT_MODE_16(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Pilot Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Pilot Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Pilot Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # subMode | Offset: 32, Length: 16, Resolution: 1, Field Type: BINARY
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("subMode")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Sub Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sub Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sub Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pilotModeData | Offset: 48, Length: 8, Resolution: 1, Field Type: BINARY
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pilotModeData")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Pilot Mode Data' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Pilot Mode Data' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Pilot Mode Data' exceeds the encoded bit length")
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
