# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_60416() -> bool:
    """Return True if PGN 60416 is a fast PGN."""
    return False
# Complex PGN. number of matches: 5
def decode_pgn_60416(data_raw: int, data_length_bits: int | None = None) -> NMEA2000Message | None:
    # isoTransportProtocolConnectionManagementRequestToSend | Description: ISO Transport Protocol, Connection Management - Request To Send
    if (
        (((data_raw >> 0) & 0xFF) == 16)
        ):
        return decode_pgn_60416_isoTransportProtocolConnectionManagementRequestToSend(data_raw, data_length_bits)
    
    # isoTransportProtocolConnectionManagementClearToSend | Description: ISO Transport Protocol, Connection Management - Clear To Send
    if (
        (((data_raw >> 0) & 0xFF) == 17)
        ):
        return decode_pgn_60416_isoTransportProtocolConnectionManagementClearToSend(data_raw, data_length_bits)
    
    # isoTransportProtocolConnectionManagementEndOfMessage | Description: ISO Transport Protocol, Connection Management - End Of Message
    if (
        (((data_raw >> 0) & 0xFF) == 19)
        ):
        return decode_pgn_60416_isoTransportProtocolConnectionManagementEndOfMessage(data_raw, data_length_bits)
    
    # isoTransportProtocolConnectionManagementBroadcastAnnounce | Description: ISO Transport Protocol, Connection Management - Broadcast Announce
    if (
        (((data_raw >> 0) & 0xFF) == 32)
        ):
        return decode_pgn_60416_isoTransportProtocolConnectionManagementBroadcastAnnounce(data_raw, data_length_bits)
    
    # isoTransportProtocolConnectionManagementAbort | Description: ISO Transport Protocol, Connection Management - Abort
    if (
        (((data_raw >> 0) & 0xFF) == 255)
        ):
        return decode_pgn_60416_isoTransportProtocolConnectionManagementAbort(data_raw, data_length_bits)
    
    
    return None
    
def decode_pgn_60416_isoTransportProtocolConnectionManagementRequestToSend(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60416."""
    nmea2000Message = NMEA2000Message(PGN=60416, id='isoTransportProtocolConnectionManagementRequestToSend', description='ISO Transport Protocol, Connection Management - Request To Send')
    running_bit_offset = 0
    # 1:group_function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 16, PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_function_code = master_dict['ISO_COMMAND'].get(group_function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupFunctionCode', 'Group Function Code', "RTS", None, group_function_code, group_function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:message_size | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    message_size = message_size_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('messageSize', 'Message size', "bytes", None, message_size, message_size_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:packets | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    packets = packets_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('packets', 'Packets', "packets", None, packets, packets_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:packets_reply | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    packets_reply = packets_reply_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('packetsReply', 'Packets reply', "packets sent in response to CTS", None, packets_reply, packets_reply_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:pgn | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_60416_isoTransportProtocolConnectionManagementRequestToSend(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60416."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # groupFunctionCode | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupFunctionCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ISO_COMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Function Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # messageSize | Offset: 8, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageSize")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message size' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message size' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message size' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # packets | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("packets")

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
        raise ValueError("Cant encode this message, 'Packets' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Packets' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Packets' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # packetsReply | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("packetsReply")

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
        raise ValueError("Cant encode this message, 'Packets reply' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Packets reply' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Packets reply' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pgn | Offset: 40, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 40
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_60416_isoTransportProtocolConnectionManagementClearToSend(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60416."""
    nmea2000Message = NMEA2000Message(PGN=60416, id='isoTransportProtocolConnectionManagementClearToSend', description='ISO Transport Protocol, Connection Management - Clear To Send')
    running_bit_offset = 0
    # 1:group_function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 17, PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_function_code = master_dict['ISO_COMMAND'].get(group_function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupFunctionCode', 'Group Function Code', "CTS", None, group_function_code, group_function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:max_packets | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    max_packets = max_packets_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxPackets', 'Max packets', "Number of frames that can be sent before another CTS is required", None, max_packets, max_packets_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:next_sid | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    next_sid = next_sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('nextSid', 'Next SID', "Number of next frame to be transmitted", None, next_sid, next_sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:reserved_24 | Offset: 24, Length: 16, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    reserved_24 = reserved_24_raw = decode_int(_data_raw_, running_bit_offset, 16)
    nmea2000Message.fields.append(NMEA2000Field('reserved_24', 'Reserved', None, None, reserved_24, reserved_24_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 16

    # 5:pgn | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_60416_isoTransportProtocolConnectionManagementClearToSend(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60416."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # groupFunctionCode | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupFunctionCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ISO_COMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Function Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxPackets | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxPackets")

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
        raise ValueError("Cant encode this message, 'Max packets' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max packets' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max packets' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # nextSid | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("nextSid")

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
        raise ValueError("Cant encode this message, 'Next SID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Next SID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Next SID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_24 | Offset: 24, Length: 16, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_24")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 16
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
    # pgn | Offset: 40, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 40
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_60416_isoTransportProtocolConnectionManagementEndOfMessage(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60416."""
    nmea2000Message = NMEA2000Message(PGN=60416, id='isoTransportProtocolConnectionManagementEndOfMessage', description='ISO Transport Protocol, Connection Management - End Of Message')
    running_bit_offset = 0
    # 1:group_function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 19, PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_function_code = master_dict['ISO_COMMAND'].get(group_function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupFunctionCode', 'Group Function Code', "EOM", None, group_function_code, group_function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:total_message_size | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    total_message_size = total_message_size_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('totalMessageSize', 'Total message size', "bytes", None, total_message_size, total_message_size_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:total_number_of_frames_received | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    total_number_of_frames_received = total_number_of_frames_received_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('totalNumberOfFramesReceived', 'Total number of frames received', "Total number of of frames received", None, total_number_of_frames_received, total_number_of_frames_received_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:reserved_32 | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    reserved_32 = reserved_32_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_32', 'Reserved', None, None, reserved_32, reserved_32_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    # 5:pgn | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_60416_isoTransportProtocolConnectionManagementEndOfMessage(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60416."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # groupFunctionCode | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupFunctionCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ISO_COMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Function Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalMessageSize | Offset: 8, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalMessageSize")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Total message size' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total message size' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total message size' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalNumberOfFramesReceived | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalNumberOfFramesReceived")

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
        raise ValueError("Cant encode this message, 'Total number of frames received' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total number of frames received' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total number of frames received' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_32 | Offset: 32, Length: 8, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_32")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 8
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
    # pgn | Offset: 40, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 40
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_60416_isoTransportProtocolConnectionManagementBroadcastAnnounce(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60416."""
    nmea2000Message = NMEA2000Message(PGN=60416, id='isoTransportProtocolConnectionManagementBroadcastAnnounce', description='ISO Transport Protocol, Connection Management - Broadcast Announce')
    running_bit_offset = 0
    # 1:group_function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32, PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_function_code = master_dict['ISO_COMMAND'].get(group_function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupFunctionCode', 'Group Function Code', "BAM", None, group_function_code, group_function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:message_size | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    message_size = message_size_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('messageSize', 'Message size', "bytes", None, message_size, message_size_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:packets | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    packets = packets_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('packets', 'Packets', "frames", None, packets, packets_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:reserved_32 | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    reserved_32 = reserved_32_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_32', 'Reserved', None, None, reserved_32, reserved_32_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    # 5:pgn | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_60416_isoTransportProtocolConnectionManagementBroadcastAnnounce(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60416."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # groupFunctionCode | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupFunctionCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ISO_COMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Function Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # messageSize | Offset: 8, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageSize")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message size' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message size' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message size' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # packets | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("packets")

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
        raise ValueError("Cant encode this message, 'Packets' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Packets' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Packets' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_32 | Offset: 32, Length: 8, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_32")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 8
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
    # pgn | Offset: 40, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 40
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_60416_isoTransportProtocolConnectionManagementAbort(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60416."""
    nmea2000Message = NMEA2000Message(PGN=60416, id='isoTransportProtocolConnectionManagementAbort', description='ISO Transport Protocol, Connection Management - Abort')
    running_bit_offset = 0
    # 1:group_function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 255, PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    group_function_code = master_dict['ISO_COMMAND'].get(group_function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('groupFunctionCode', 'Group Function Code', "Abort", None, group_function_code, group_function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:reason | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    reason = reason_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 8))
    nmea2000Message.fields.append(NMEA2000Field('reason', 'Reason', None, None, reason, reason_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 8

    # 3:reserved_16 | Offset: 16, Length: 24, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    reserved_16 = reserved_16_raw = decode_int(_data_raw_, running_bit_offset, 24)
    nmea2000Message.fields.append(NMEA2000Field('reserved_16', 'Reserved', None, None, reserved_16, reserved_16_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 24

    # 4:pgn | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', None, None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_60416_isoTransportProtocolConnectionManagementAbort(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60416."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # groupFunctionCode | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupFunctionCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ISO_COMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Function Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Function Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reason | Offset: 8, Length: 8, Resolution: 1, Field Type: BINARY
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reason")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Reason' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Reason' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Reason' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_16 | Offset: 16, Length: 24, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_16")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 24
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
    # pgn | Offset: 40, Length: 24, Resolution: 1, Field Type: PGN
    running_bit_offset = 40
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
    return data_raw.to_bytes(8, byteorder="little")
