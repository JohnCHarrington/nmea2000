# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130566() -> bool:
    """Return True if PGN 130566 is a fast PGN."""
    return True
def decode_pgn_130566(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130566."""
    nmea2000Message = NMEA2000Message(PGN=130566, id='lightingProgram', description='Lighting Program')
    running_bit_offset = 0
    # 1:program_id | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    program_id = program_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('programId', 'Program ID', None, None, program_id, program_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:name_of_program | Offset: 8, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    name_of_program_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    name_of_program = name_of_program_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('nameOfProgram', 'Name of Program', None, None, name_of_program, name_of_program_raw, None, FieldTypes.STRING_LAU, False))
    

    # 3:description | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    description_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    description = description_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('description', 'Description', None, None, description, description_raw, None, FieldTypes.STRING_LAU, False))
    

    # 4:program_capabilities | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    program_capabilities = program_capabilities_raw = decode_number(_data_raw_, running_bit_offset, 4, False, 1, 0, 13)
    nmea2000Message.fields.append(NMEA2000Field('programCapabilities', 'Program Capabilities', None, None, program_capabilities, program_capabilities_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 4

    # 5:reserved_ | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', None, None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    return nmea2000Message

def encode_pgn_130566(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130566."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # programId | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("programId")

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
        raise ValueError("Cant encode this message, 'Program ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Program ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Program ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # nameOfProgram | Offset: 8, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("nameOfProgram")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name of Program' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name of Program' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name of Program' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # description | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("description")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Description' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Description' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Description' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # programCapabilities | Offset: , Length: 4, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("programCapabilities")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 4, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 4, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 4, False, 1)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Program Capabilities' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Program Capabilities' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Program Capabilities' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 4, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 4
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
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
