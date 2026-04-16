# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126464() -> bool:
    """Return True if PGN 126464 is a fast PGN."""
    return True
def decode_pgn_126464(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126464."""
    nmea2000Message = NMEA2000Message(PGN=126464, id='pgnListTransmitAndReceive', description='PGN List (Transmit and Receive)')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['PGN_LIST_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', None, None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    _repeating_field_set_1_offset = running_bit_offset
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = None
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
        running_bit_offset += 24
        repeating_entry["pgn"] = _repeating_entry_value(pgn, pgn_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "pgn",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_126464(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126464."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "pgn",
    ))
    # functionCode | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("functionCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_PGN_LIST_FUNCTION(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Function Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Function Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Function Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 8
    for repeating_entry in repeating_field_set_1_entries:
        # pgn | Offset: 8, Length: 24, Resolution: 1, Field Type: PGN
        field = repeating_entry.get("pgn")
        if field is None:
            raise ValueError("Cant encode this message, missing 'PGN'")
        field_offset = running_bit_offset
    
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
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
