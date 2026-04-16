# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128259() -> bool:
    """Return True if PGN 128259 is a fast PGN."""
    return False
def decode_pgn_128259(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128259."""
    nmea2000Message = NMEA2000Message(PGN=128259, id='speed', description='Speed', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:speed_water_referenced | Offset: 8, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    speed_water_referenced = speed_water_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('speedWaterReferenced', 'Speed Water Referenced', None, 'm/s', speed_water_referenced, speed_water_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:speed_ground_referenced | Offset: 24, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    speed_ground_referenced = speed_ground_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('speedGroundReferenced', 'Speed Ground Referenced', None, 'm/s', speed_ground_referenced, speed_ground_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:speed_water_referenced_type | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    speed_water_referenced_type_raw = decode_int(_data_raw_, running_bit_offset, 8)
    speed_water_referenced_type = master_dict['WATER_REFERENCE'].get(speed_water_referenced_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('speedWaterReferencedType', 'Speed Water Referenced Type', None, None, speed_water_referenced_type, speed_water_referenced_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:speed_direction | Offset: 48, Length: 4, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    speed_direction = speed_direction_raw = decode_number(_data_raw_, running_bit_offset, 4, False, 1, 0, 13)
    nmea2000Message.fields.append(NMEA2000Field('speedDirection', 'Speed Direction', None, None, speed_direction, speed_direction_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 4

    # 6:reserved_52 | Offset: 52, Length: 12, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 52
    reserved_52 = reserved_52_raw = decode_int(_data_raw_, running_bit_offset, 12)
    nmea2000Message.fields.append(NMEA2000Field('reserved_52', 'Reserved', None, None, reserved_52, reserved_52_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 12

    return nmea2000Message

def encode_pgn_128259(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128259."""
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
    # speedWaterReferenced | Offset: 8, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedWaterReferenced")

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
        raise ValueError("Cant encode this message, 'Speed Water Referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed Water Referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed Water Referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speedGroundReferenced | Offset: 24, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedGroundReferenced")

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
        raise ValueError("Cant encode this message, 'Speed Ground Referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed Ground Referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed Ground Referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speedWaterReferencedType | Offset: 40, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedWaterReferencedType")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_WATER_REFERENCE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Speed Water Referenced Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed Water Referenced Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed Water Referenced Type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speedDirection | Offset: 48, Length: 4, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedDirection")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 4, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 4, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 4, False, 1)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Speed Direction' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed Direction' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed Direction' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_52 | Offset: 52, Length: 12, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 52
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_52")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 12
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
