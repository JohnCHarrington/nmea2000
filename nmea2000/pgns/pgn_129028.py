# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129028() -> bool:
    """Return True if PGN 129028 is a fast PGN."""
    return False
def decode_pgn_129028(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129028."""
    nmea2000Message = NMEA2000Message(PGN=129028, id='altitudeDeltaRapidUpdate', description='Altitude Delta, Rapid Update', ttl=timedelta(milliseconds=100))
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:time_delta | Offset: 8, Length: 8, Signed: False Resolution: 0.005, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    time_delta = time_delta_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 0.005, 0, 1.26)
    nmea2000Message.fields.append(NMEA2000Field('timeDelta', 'Time Delta', None, 's', time_delta, time_delta_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 8

    # 3:gnss_quality | Offset: 16, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    gnss_quality_raw = decode_int(_data_raw_, running_bit_offset, 4)
    gnss_quality = master_dict['GNS_METHOD'].get(gnss_quality_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('gnssQuality', 'GNSS Quality', None, None, gnss_quality, gnss_quality_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:direction | Offset: 20, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    direction_raw = decode_int(_data_raw_, running_bit_offset, 2)
    direction = master_dict['DIRECTION_REFERENCE'].get(direction_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('direction', 'Direction', None, None, direction, direction_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:reserved_22 | Offset: 22, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 22
    reserved_22 = reserved_22_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_22', 'Reserved', None, None, reserved_22, reserved_22_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 6:cog | Offset: 24, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    cog = cog_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('cog', 'COG', None, 'rad', cog, cog_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:altitude_delta | Offset: 40, Length: 24, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    altitude_delta = altitude_delta_raw = decode_number(_data_raw_, running_bit_offset, 24, True, 0.001, -8388.607, 8388.604)
    nmea2000Message.fields.append(NMEA2000Field('altitudeDelta', 'Altitude Delta', None, 'm', altitude_delta, altitude_delta_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_129028(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129028."""
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
    # timeDelta | Offset: 8, Length: 8, Resolution: 0.005, Field Type: DURATION
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeDelta")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.005)):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 0.005)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 8, False, 0.005)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 8)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time Delta' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time Delta' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time Delta' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # gnssQuality | Offset: 16, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("gnssQuality")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GNS_METHOD(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'GNSS Quality' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'GNSS Quality' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'GNSS Quality' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # direction | Offset: 20, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("direction")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DIRECTION_REFERENCE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Direction' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Direction' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Direction' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_22 | Offset: 22, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 22
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_22")

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
    # cog | Offset: 24, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("cog")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'COG' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'COG' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'COG' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # altitudeDelta | Offset: 40, Length: 24, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("altitudeDelta")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 24, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, True, 0.001)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Altitude Delta' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Altitude Delta' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Altitude Delta' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
