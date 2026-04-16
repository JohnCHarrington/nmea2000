# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126976() -> bool:
    """Return True if PGN 126976 is a fast PGN."""
    raise ValueError('PGEN type Mixed not supported')

def decode_pgn_126976(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126976."""
    nmea2000Message = NMEA2000Message(PGN=126976, id='0x1f0000x1feffStandardizedMixedSingleFastPacketNonAddressed', description='0x1F000-0x1FEFF: Standardized mixed single/fast packet non-addressed')
    running_bit_offset = 0
    # 1:data | Offset: 0, Length: 1784, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    data = data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 1784))
    nmea2000Message.fields.append(NMEA2000Field('data', 'Data', None, None, data, data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 1784

    return nmea2000Message

def encode_pgn_126976(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126976."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # data | Offset: 0, Length: 1784, Resolution: 1, Field Type: BINARY
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("data")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 1784
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Data' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(223, byteorder="little")
