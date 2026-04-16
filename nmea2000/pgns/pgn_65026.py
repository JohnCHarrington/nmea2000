# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65026() -> bool:
    """Return True if PGN 65026 is a fast PGN."""
    return False
def decode_pgn_65026(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65026."""
    nmea2000Message = NMEA2000Message(PGN=65026, id='generatorPhaseAAcPower', description='Generator Phase A AC Power')
    running_bit_offset = 0
    # 1:real_power | Offset: 0, Length: 32, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    real_power = real_power_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1, -2000000000, 2294967292)
    nmea2000Message.fields.append(NMEA2000Field('realPower', 'Real Power', None, 'W', real_power, real_power_raw, PhysicalQuantities.ELECTRICAL_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 2:apparent_power | Offset: 32, Length: 32, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    apparent_power = apparent_power_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1, -2000000000, 2294967292)
    nmea2000Message.fields.append(NMEA2000Field('apparentPower', 'Apparent Power', None, 'VA', apparent_power, apparent_power_raw, PhysicalQuantities.ELECTRICAL_APPARENT_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_65026(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65026."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # realPower | Offset: 0, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("realPower")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Real Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Real Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Real Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # apparentPower | Offset: 32, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("apparentPower")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Apparent Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Apparent Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Apparent Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
