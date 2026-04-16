# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129795() -> bool:
    """Return True if PGN 129795 is a fast PGN."""
    return True
def decode_pgn_129795(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129795."""
    nmea2000Message = NMEA2000Message(PGN=129795, id='aisAddressedBinaryMessage', description='AIS Addressed Binary Message')
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

    # 3:source_id | Offset: 8, Length: 32, Signed: False Resolution: 1, Field Type: MMSI, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 8
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 2000000, 999999999)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.MMSI, True))
    running_bit_offset += 32

    # 4:reserved_40 | Offset: 40, Length: 1, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    reserved_40 = reserved_40_raw = decode_int(_data_raw_, running_bit_offset, 1)
    nmea2000Message.fields.append(NMEA2000Field('reserved_40', 'Reserved', None, None, reserved_40, reserved_40_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 1

    # 5:ais_transceiver_information | Offset: 41, Length: 5, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 41
    ais_transceiver_information_raw = decode_int(_data_raw_, running_bit_offset, 5)
    ais_transceiver_information = master_dict['AIS_TRANSCEIVER'].get(ais_transceiver_information_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('aisTransceiverInformation', 'AIS Transceiver information', None, None, ais_transceiver_information, ais_transceiver_information_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 5

    # 6:sequence_number | Offset: 46, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 46
    sequence_number = sequence_number_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 3)
    nmea2000Message.fields.append(NMEA2000Field('sequenceNumber', 'Sequence Number', None, None, sequence_number, sequence_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 7:destination_id | Offset: 48, Length: 32, Signed: False Resolution: 1, Field Type: MMSI, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 48
    destination_id = destination_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 2000000, 999999999)
    nmea2000Message.fields.append(NMEA2000Field('destinationId', 'Destination ID', None, None, destination_id, destination_id_raw, None, FieldTypes.MMSI, True))
    running_bit_offset += 32

    # 8:reserved_80 | Offset: 80, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    reserved_80 = reserved_80_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_80', 'Reserved', None, None, reserved_80, reserved_80_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 9:retransmit_flag | Offset: 86, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 86
    retransmit_flag_raw = decode_int(_data_raw_, running_bit_offset, 1)
    retransmit_flag = master_dict['YES_NO'].get(retransmit_flag_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('retransmitFlag', 'Retransmit flag', None, None, retransmit_flag, retransmit_flag_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 10:reserved_87 | Offset: 87, Length: 1, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 87
    reserved_87 = reserved_87_raw = decode_int(_data_raw_, running_bit_offset, 1)
    nmea2000Message.fields.append(NMEA2000Field('reserved_87', 'Reserved', None, None, reserved_87, reserved_87_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 1

    # 11:number_of_bits_in_binary_data_field | Offset: 88, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    number_of_bits_in_binary_data_field = number_of_bits_in_binary_data_field_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfBitsInBinaryDataField', 'Number of Bits in Binary Data Field', None, None, number_of_bits_in_binary_data_field, number_of_bits_in_binary_data_field_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 12:binary_data | Offset: 104, Length: , Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 104
    assert isinstance(number_of_bits_in_binary_data_field, (int, float))
    binary_data = binary_data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, int(number_of_bits_in_binary_data_field)))
    nmea2000Message.fields.append(NMEA2000Field('binaryData', 'Binary Data', None, None, binary_data, binary_data_raw, None, FieldTypes.BINARY, False))
    

    return nmea2000Message

def encode_pgn_129795(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129795."""
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
    # sourceId | Offset: 8, Length: 32, Resolution: 1, Field Type: MMSI
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_40 | Offset: 40, Length: 1, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_40")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 1
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
    # aisTransceiverInformation | Offset: 41, Length: 5, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 41
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
    # sequenceNumber | Offset: 46, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 46
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sequenceNumber")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 2, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 2, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 2, False, 1)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Sequence Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sequence Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sequence Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # destinationId | Offset: 48, Length: 32, Resolution: 1, Field Type: MMSI
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("destinationId")

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
        raise ValueError("Cant encode this message, 'Destination ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Destination ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Destination ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_80 | Offset: 80, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_80")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 6
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
    # retransmitFlag | Offset: 86, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 86
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("retransmitFlag")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 1
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Retransmit flag' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Retransmit flag' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Retransmit flag' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_87 | Offset: 87, Length: 1, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 87
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_87")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 1
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
    # numberOfBitsInBinaryDataField | Offset: 88, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfBitsInBinaryDataField")

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
        raise ValueError("Cant encode this message, 'Number of Bits in Binary Data Field' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Bits in Binary Data Field' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Bits in Binary Data Field' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # binaryData | Offset: 104, Length: , Resolution: 1, Field Type: BINARY
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("binaryData")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    length_field = nmea2000Message.get_field_by_id("numberOfBitsInBinaryDataField")
    if isinstance(length_field.raw_value, (int, float)):
        field_bit_length = int(length_field.raw_value)
    else:
        assert length_field.value is None or isinstance(length_field.value, (int, float))
        field_bit_length = int(length_field.value or 0)
    advance_running_offset = False
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Binary Data' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Binary Data' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Binary Data' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
