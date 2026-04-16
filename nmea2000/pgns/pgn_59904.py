# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_59904() -> bool:
    """Return True if PGN 59904 is a fast PGN."""
    return False
def decode_pgn_59904(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 59904."""
    nmea2000Message = NMEA2000Message(PGN=59904, id='isoRequest', description='ISO Request')
    running_bit_offset = 0
    # 1:pgn | Offset: 0, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_59904(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 59904."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # pgn | Offset: 0, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pgn")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 1)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'PGN' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PGN' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PGN' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(3, byteorder="little")
