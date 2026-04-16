# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130572() -> bool:
    """Return True if PGN 130572 is a fast PGN."""
    return True
def decode_pgn_130572(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130572."""
    nmea2000Message = NMEA2000Message(PGN=130572, id='libraryDataSearch', description='Library Data Search')
    running_bit_offset = 0
    # 1:source | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    source_raw = decode_int(_data_raw_, running_bit_offset, 8)
    source = master_dict['ENTERTAINMENT_SOURCE'].get(source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('source', 'Source', None, None, source, source_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:number | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    number = number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('number', 'Number', "Source number per type", None, number, number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:group_id | Offset: 16, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    group_id = group_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('groupId', 'Group ID', "Unique group ID", None, group_id, group_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 4:group_type_1 | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    group_type_1_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_type_1 = master_dict['ENTERTAINMENT_GROUP'].get(group_type_1_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupType1', 'Group type 1', None, None, group_type_1, group_type_1_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:group_name_1 | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    group_name_1_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    group_name_1 = group_name_1_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('groupName1', 'Group name 1', None, None, group_name_1, group_name_1_raw, None, FieldTypes.STRING_LAU, False))
    

    # 6:group_type_2 | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    group_type_2_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_type_2 = master_dict['ENTERTAINMENT_GROUP'].get(group_type_2_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupType2', 'Group type 2', None, None, group_type_2, group_type_2_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 7:group_name_2 | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    group_name_2_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    group_name_2 = group_name_2_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('groupName2', 'Group name 2', None, None, group_name_2, group_name_2_raw, None, FieldTypes.STRING_LAU, False))
    

    # 8:group_type_3 | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    group_type_3_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_type_3 = master_dict['ENTERTAINMENT_GROUP'].get(group_type_3_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupType3', 'Group type 3', None, None, group_type_3, group_type_3_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 9:group_name_3 | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    group_name_3_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    group_name_3 = group_name_3_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('groupName3', 'Group name 3', None, None, group_name_3, group_name_3_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_130572(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130572."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # source | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("source")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_SOURCE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Source' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # number | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("number")

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
        raise ValueError("Cant encode this message, 'Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupId | Offset: 16, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupId")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, False, 1)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupType1 | Offset: 48, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupType1")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_GROUP(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group type 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group type 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group type 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupName1 | Offset: 56, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupName1")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group name 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group name 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group name 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupType2 | Offset: , Length: 8, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupType2")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_GROUP(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group type 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group type 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group type 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupName2 | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupName2")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group name 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group name 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group name 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupType3 | Offset: , Length: 8, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupType3")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_GROUP(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group type 3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group type 3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group type 3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # groupName3 | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupName3")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group name 3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group name 3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group name 3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
