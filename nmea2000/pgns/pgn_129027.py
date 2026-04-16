# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129027() -> bool:
    """Return True if PGN 129027 is a fast PGN."""
    return False
def decode_pgn_129027(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129027."""
    nmea2000Message = NMEA2000Message(PGN=129027, id='positionDeltaRapidUpdate', description='Position Delta, Rapid Update', ttl=timedelta(milliseconds=100))
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

    # 3:latitude_delta | Offset: 16, Length: 24, Signed: True Resolution: 2.77778e-09, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    latitude_delta = latitude_delta_raw = decode_number(_data_raw_, running_bit_offset, 24, True, 2.77778e-09, -0.0233016861111111, 0.0233016777777778)
    nmea2000Message.fields.append(NMEA2000Field('latitudeDelta', 'Latitude Delta', None, 'deg', latitude_delta, latitude_delta_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 4:longitude_delta | Offset: 40, Length: 24, Signed: True Resolution: 2.77778e-09, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    longitude_delta = longitude_delta_raw = decode_number(_data_raw_, running_bit_offset, 24, True, 2.77778e-09, -0.0233016861111111, 0.0233016777777778)
    nmea2000Message.fields.append(NMEA2000Field('longitudeDelta', 'Longitude Delta', None, 'deg', longitude_delta, longitude_delta_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_129027(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129027."""
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
    # latitudeDelta | Offset: 16, Length: 24, Resolution: 2.77778e-09, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("latitudeDelta")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 2.77778e-09):
        field_value = encode_number_raw(field.raw_value, 24, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, True, 2.77778e-09)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, True, 2.77778e-09)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Latitude Delta' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Latitude Delta' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Latitude Delta' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitudeDelta | Offset: 40, Length: 24, Resolution: 2.77778e-09, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitudeDelta")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 2.77778e-09):
        field_value = encode_number_raw(field.raw_value, 24, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, True, 2.77778e-09)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, True, 2.77778e-09)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitude Delta' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitude Delta' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitude Delta' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
