# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_60160() -> bool:
    """Return True if PGN 60160 is a fast PGN."""
    return False
def decode_pgn_60160(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60160."""
    nmea2000Message = NMEA2000Message(PGN=60160, id='isoTransportProtocolDataTransfer', description='ISO Transport Protocol, Data Transfer')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', "Sequence number of multi-packet frame", None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:data | Offset: 8, Length: 56, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    data = data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 56))
    nmea2000Message.fields.append(NMEA2000Field('data', 'Data', "Multi-packet packetized data - 56 bits organized as seven 8-bit bytes, each of the bytes is transmitted in the same order as it would in a standard (i.e., single frame) message. Unused bytes shall be filled with 0xFF.", None, data, data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 56

    return nmea2000Message

def encode_pgn_60160(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60160."""
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
    # data | Offset: 8, Length: 56, Resolution: 1, Field Type: BINARY
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("data")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 56
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
    return data_raw.to_bytes(8, byteorder="little")
