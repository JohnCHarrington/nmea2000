# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127498() -> bool:
    """Return True if PGN 127498 is a fast PGN."""
    return True
def decode_pgn_127498(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127498."""
    nmea2000Message = NMEA2000Message(PGN=127498, id='engineParametersStatic', description='Engine Parameters, Static')
    running_bit_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance_raw = decode_int(_data_raw_, running_bit_offset, 8)
    instance = master_dict['ENGINE_INSTANCE'].get(instance_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 8

    # 2:rated_engine_speed | Offset: 8, Length: 16, Signed: False Resolution: 0.25, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    rated_engine_speed = rated_engine_speed_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.25, 0, 16383)
    nmea2000Message.fields.append(NMEA2000Field('ratedEngineSpeed', 'Rated Engine Speed', None, 'rpm', rated_engine_speed, rated_engine_speed_raw, PhysicalQuantities.ANGULAR_VELOCITY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:vin | Offset: 24, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    vin_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    vin = vin_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('vin', 'VIN', None, None, vin, vin_raw, None, FieldTypes.STRING_LAU, False))
    

    # 4:software_id | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    software_id_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    software_id = software_id_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('softwareId', 'Software ID', None, None, software_id, software_id_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_127498(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127498."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # instance | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("instance")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENGINE_INSTANCE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # ratedEngineSpeed | Offset: 8, Length: 16, Resolution: 0.25, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("ratedEngineSpeed")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.25):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.25)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.25)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rated Engine Speed' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rated Engine Speed' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rated Engine Speed' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # vin | Offset: 24, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("vin")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'VIN' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'VIN' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'VIN' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # softwareId | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("softwareId")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Software ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Software ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Software ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
