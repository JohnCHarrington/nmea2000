# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130848() -> bool:
    """Return True if PGN 130848 is a fast PGN."""
    return True
def decode_pgn_130848(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130848."""
    nmea2000Message = NMEA2000Message(PGN=130848, id='seatalkWaypointInformation', description='SeaTalk: Waypoint Information', ttl=timedelta(milliseconds=1000))
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

    # 4:sid | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:waypoint_name | Offset: 24, Length: 128, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    waypoint_name, waypoint_name_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 128)
    nmea2000Message.fields.append(NMEA2000Field('waypointName', 'Waypoint Name', None, None, waypoint_name, waypoint_name_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 128

    # 6:waypoint_sequence | Offset: 152, Length: 32, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    waypoint_sequence, waypoint_sequence_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 32)
    nmea2000Message.fields.append(NMEA2000Field('waypointSequence', 'Waypoint Sequence', None, None, waypoint_sequence, waypoint_sequence_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 32

    # 7:bearing_to_waypoint__true | Offset: 184, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 184
    bearing_to_waypoint__true = bearing_to_waypoint__true_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('bearingToWaypointTrue', 'Bearing to Waypoint, True', None, 'rad', bearing_to_waypoint__true, bearing_to_waypoint__true_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:bearing_to_waypoint__magnetic | Offset: 200, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 200
    bearing_to_waypoint__magnetic = bearing_to_waypoint__magnetic_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('bearingToWaypointMagnetic', 'Bearing to Waypoint, Magnetic', None, 'rad', bearing_to_waypoint__magnetic, bearing_to_waypoint__magnetic_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:distance_to_waypoint | Offset: 216, Length: 32, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 216
    distance_to_waypoint = distance_to_waypoint_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.01, 0, 42949672.92)
    nmea2000Message.fields.append(NMEA2000Field('distanceToWaypoint', 'Distance to Waypoint', None, 'm', distance_to_waypoint, distance_to_waypoint_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130848(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130848."""
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
    # sid | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
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
    # waypointName | Offset: 24, Length: 128, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("waypointName")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 128)
    field_bit_length = 128
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Waypoint Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Waypoint Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Waypoint Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # waypointSequence | Offset: 152, Length: 32, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("waypointSequence")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Waypoint Sequence' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Waypoint Sequence' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Waypoint Sequence' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # bearingToWaypointTrue | Offset: 184, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 184
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("bearingToWaypointTrue")

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
        raise ValueError("Cant encode this message, 'Bearing to Waypoint, True' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Bearing to Waypoint, True' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Bearing to Waypoint, True' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # bearingToWaypointMagnetic | Offset: 200, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 200
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("bearingToWaypointMagnetic")

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
        raise ValueError("Cant encode this message, 'Bearing to Waypoint, Magnetic' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Bearing to Waypoint, Magnetic' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Bearing to Waypoint, Magnetic' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # distanceToWaypoint | Offset: 216, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 216
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("distanceToWaypoint")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, False, 0.01)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Distance to Waypoint' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Distance to Waypoint' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Distance to Waypoint' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(31, byteorder="little")
