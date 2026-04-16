# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130060() -> bool:
    """Return True if PGN 130060 is a fast PGN."""
    return True
def decode_pgn_130060(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130060."""
    nmea2000Message = NMEA2000Message(PGN=130060, id='label', description='Label')
    running_bit_offset = 0
    # 1:hardware_channel_id | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    hardware_channel_id = hardware_channel_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('hardwareChannelId', 'Hardware Channel ID', None, None, hardware_channel_id, hardware_channel_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:data_source_instance_field_number | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    data_source_instance_field_number = data_source_instance_field_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('dataSourceInstanceFieldNumber', 'Data Source Instance Field Number', None, None, data_source_instance_field_number, data_source_instance_field_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:data_source_instance_value | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    data_source_instance_value = data_source_instance_value_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('dataSourceInstanceValue', 'Data Source Instance Value', None, None, data_source_instance_value, data_source_instance_value_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:secondary_enumeration_field_number | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    secondary_enumeration_field_number = secondary_enumeration_field_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('secondaryEnumerationFieldNumber', 'Secondary Enumeration Field Number', None, None, secondary_enumeration_field_number, secondary_enumeration_field_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:secondary_enumeration_field_value | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    secondary_enumeration_field_value = secondary_enumeration_field_value_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('secondaryEnumerationFieldValue', 'Secondary Enumeration Field Value', None, None, secondary_enumeration_field_value, secondary_enumeration_field_value_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:parameter_field_number | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    parameter_field_number = parameter_field_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameterFieldNumber', 'Parameter Field Number', None, None, parameter_field_number, parameter_field_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:label | Offset: 72, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    label_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    label = label_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('label', 'Label', None, None, label, label_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_130060(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130060."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # hardwareChannelId | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("hardwareChannelId")

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
        raise ValueError("Cant encode this message, 'Hardware Channel ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Hardware Channel ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Hardware Channel ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pgn | Offset: 8, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 8
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
    # dataSourceInstanceFieldNumber | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dataSourceInstanceFieldNumber")

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
        raise ValueError("Cant encode this message, 'Data Source Instance Field Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data Source Instance Field Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data Source Instance Field Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dataSourceInstanceValue | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dataSourceInstanceValue")

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
        raise ValueError("Cant encode this message, 'Data Source Instance Value' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data Source Instance Value' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data Source Instance Value' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # secondaryEnumerationFieldNumber | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("secondaryEnumerationFieldNumber")

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
        raise ValueError("Cant encode this message, 'Secondary Enumeration Field Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Secondary Enumeration Field Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Secondary Enumeration Field Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # secondaryEnumerationFieldValue | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("secondaryEnumerationFieldValue")

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
        raise ValueError("Cant encode this message, 'Secondary Enumeration Field Value' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Secondary Enumeration Field Value' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Secondary Enumeration Field Value' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # parameterFieldNumber | Offset: 64, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("parameterFieldNumber")

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
        raise ValueError("Cant encode this message, 'Parameter Field Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Parameter Field Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Parameter Field Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # label | Offset: 72, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("label")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Label' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Label' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Label' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
