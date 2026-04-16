# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130578() -> bool:
    """Return True if PGN 130578 is a fast PGN."""
    return True
def decode_pgn_130578(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130578."""
    nmea2000Message = NMEA2000Message(PGN=130578, id='vesselSpeedComponents', description='Vessel Speed Components', ttl=timedelta(milliseconds=250))
    running_bit_offset = 0
    # 1:longitudinal_speed__water_referenced | Offset: 0, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    longitudinal_speed__water_referenced = longitudinal_speed__water_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('longitudinalSpeedWaterReferenced', 'Longitudinal Speed, Water-referenced', None, 'm/s', longitudinal_speed__water_referenced, longitudinal_speed__water_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:transverse_speed__water_referenced | Offset: 16, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    transverse_speed__water_referenced = transverse_speed__water_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('transverseSpeedWaterReferenced', 'Transverse Speed, Water-referenced', None, 'm/s', transverse_speed__water_referenced, transverse_speed__water_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:longitudinal_speed__ground_referenced | Offset: 32, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    longitudinal_speed__ground_referenced = longitudinal_speed__ground_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('longitudinalSpeedGroundReferenced', 'Longitudinal Speed, Ground-referenced', None, 'm/s', longitudinal_speed__ground_referenced, longitudinal_speed__ground_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:transverse_speed__ground_referenced | Offset: 48, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    transverse_speed__ground_referenced = transverse_speed__ground_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('transverseSpeedGroundReferenced', 'Transverse Speed, Ground-referenced', None, 'm/s', transverse_speed__ground_referenced, transverse_speed__ground_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:stern_speed__water_referenced | Offset: 64, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    stern_speed__water_referenced = stern_speed__water_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('sternSpeedWaterReferenced', 'Stern Speed, Water-referenced', None, 'm/s', stern_speed__water_referenced, stern_speed__water_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:stern_speed__ground_referenced | Offset: 80, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    stern_speed__ground_referenced = stern_speed__ground_referenced_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('sternSpeedGroundReferenced', 'Stern Speed, Ground-referenced', None, 'm/s', stern_speed__ground_referenced, stern_speed__ground_referenced_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_130578(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130578."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # longitudinalSpeedWaterReferenced | Offset: 0, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitudinalSpeedWaterReferenced")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitudinal Speed, Water-referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitudinal Speed, Water-referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitudinal Speed, Water-referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # transverseSpeedWaterReferenced | Offset: 16, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("transverseSpeedWaterReferenced")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Transverse Speed, Water-referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Transverse Speed, Water-referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Transverse Speed, Water-referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitudinalSpeedGroundReferenced | Offset: 32, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitudinalSpeedGroundReferenced")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitudinal Speed, Ground-referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitudinal Speed, Ground-referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitudinal Speed, Ground-referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # transverseSpeedGroundReferenced | Offset: 48, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("transverseSpeedGroundReferenced")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Transverse Speed, Ground-referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Transverse Speed, Ground-referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Transverse Speed, Ground-referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sternSpeedWaterReferenced | Offset: 64, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sternSpeedWaterReferenced")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Stern Speed, Water-referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Stern Speed, Water-referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Stern Speed, Water-referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sternSpeedGroundReferenced | Offset: 80, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sternSpeedGroundReferenced")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Stern Speed, Ground-referenced' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Stern Speed, Ground-referenced' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Stern Speed, Ground-referenced' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(12, byteorder="little")
