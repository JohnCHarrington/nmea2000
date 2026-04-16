# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129809() -> bool:
    """Return True if PGN 129809 is a fast PGN."""
    return True
def decode_pgn_129809(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129809."""
    nmea2000Message = NMEA2000Message(PGN=129809, id='aisClassBStaticDataMsg24PartA', description='AIS Class B static data (msg 24 Part A)')
    running_bit_offset = 0
    # 1:message_id | Offset: 0, Length: 6, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 6)
    message_id = master_dict['AIS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', None, None, message_id, message_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 6

    # 2:repeat_indicator | Offset: 6, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 6
    repeat_indicator_raw = decode_int(_data_raw_, running_bit_offset, 2)
    repeat_indicator = master_dict['REPEAT_INDICATOR'].get(repeat_indicator_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('repeatIndicator', 'Repeat Indicator', None, None, repeat_indicator, repeat_indicator_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:user_id | Offset: 8, Length: 32, Signed: False Resolution: 1, Field Type: MMSI, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 8
    user_id = user_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 2000000, 999999999)
    nmea2000Message.fields.append(NMEA2000Field('userId', 'User ID', None, None, user_id, user_id_raw, None, FieldTypes.MMSI, True))
    running_bit_offset += 32

    # 4:name | Offset: 40, Length: 160, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    name, name_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 160)
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 160

    # 5:ais_transceiver_information | Offset: 200, Length: 5, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 200
    ais_transceiver_information_raw = decode_int(_data_raw_, running_bit_offset, 5)
    ais_transceiver_information = master_dict['AIS_TRANSCEIVER'].get(ais_transceiver_information_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('aisTransceiverInformation', 'AIS Transceiver information', None, None, ais_transceiver_information, ais_transceiver_information_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 5

    # 6:reserved_205 | Offset: 205, Length: 3, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 205
    reserved_205 = reserved_205_raw = decode_int(_data_raw_, running_bit_offset, 3)
    nmea2000Message.fields.append(NMEA2000Field('reserved_205', 'Reserved', None, None, reserved_205, reserved_205_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 3

    # 7:sequence_id | Offset: 208, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 208
    sequence_id = sequence_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sequenceId', 'Sequence ID', None, None, sequence_id, sequence_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_129809(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129809."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # messageId | Offset: 0, Length: 6, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_AIS_MESSAGE_ID(field.value)
    field_bit_length = 6
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # repeatIndicator | Offset: 6, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 6
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("repeatIndicator")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_REPEAT_INDICATOR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Repeat Indicator' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Repeat Indicator' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Repeat Indicator' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # userId | Offset: 8, Length: 32, Resolution: 1, Field Type: MMSI
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("userId")

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
        raise ValueError("Cant encode this message, 'User ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'User ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'User ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # name | Offset: 40, Length: 160, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 160)
    field_bit_length = 160
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # aisTransceiverInformation | Offset: 200, Length: 5, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 200
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("aisTransceiverInformation")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_AIS_TRANSCEIVER(field.value)
    field_bit_length = 5
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'AIS Transceiver information' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'AIS Transceiver information' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'AIS Transceiver information' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_205 | Offset: 205, Length: 3, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 205
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_205")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 3
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
    # sequenceId | Offset: 208, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 208
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sequenceId")

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
        raise ValueError("Cant encode this message, 'Sequence ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sequence ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sequence ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(27, byteorder="little")
