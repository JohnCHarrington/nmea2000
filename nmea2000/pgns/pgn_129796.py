# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129796() -> bool:
    """Return True if PGN 129796 is a fast PGN."""
    return True
def decode_pgn_129796(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129796."""
    nmea2000Message = NMEA2000Message(PGN=129796, id='aisAcknowledge', description='AIS Acknowledge')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
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

    # 6:spare | Offset: 46, Length: 2, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 46
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('spare6', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 2

    # 7:destination_id | Offset: 48, Length: 32, Signed: False Resolution: 1, Field Type: MMSI, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    destination_id = destination_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 2000000, 999999999)
    nmea2000Message.fields.append(NMEA2000Field('destinationId', 'Destination ID', None, None, destination_id, destination_id_raw, None, FieldTypes.MMSI, True))
    running_bit_offset += 32

    # 8:sequence_number | Offset: 80, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    sequence_number = sequence_number_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 3)
    nmea2000Message.fields.append(NMEA2000Field('sequenceNumber', 'Sequence Number', None, None, sequence_number, sequence_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 9:reserved_82 | Offset: 82, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 82
    reserved_82 = reserved_82_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_82', 'Reserved', None, None, reserved_82, reserved_82_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = None
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        destination_id = destination_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 2000000, 999999999)
        running_bit_offset += 32
        repeating_entry["destinationId"] = _repeating_entry_value(destination_id, destination_id_raw)
    
        sequence_number = sequence_number_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 3)
        running_bit_offset += 2
        repeating_entry["sequenceNumber"] = _repeating_entry_value(sequence_number, sequence_number_raw)
    
        reserved_82 = reserved_82_raw = decode_int(_data_raw_, running_bit_offset, 6)
        running_bit_offset += 6
        repeating_entry["reserved_82"] = _repeating_entry_value(reserved_82, reserved_82_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "destinationId",
                "sequenceNumber",
                "reserved_82",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_129796(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129796."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "destinationId",
        "sequenceNumber",
        "reserved_82",
    ))
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
    # spare6 | Offset: 46, Length: 2, Resolution: 1, Field Type: SPARE
    running_bit_offset = 46
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare6")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Spare' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Spare' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Spare' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 48
    for repeating_entry in repeating_field_set_1_entries:
        # destinationId | Offset: 48, Length: 32, Resolution: 1, Field Type: MMSI
        field = repeating_entry.get("destinationId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Destination ID'")
        field_offset = running_bit_offset
    
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
        # sequenceNumber | Offset: 80, Length: 2, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("sequenceNumber")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Sequence Number'")
        field_offset = running_bit_offset
    
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
        # reserved_82 | Offset: 82, Length: 6, Resolution: 1, Field Type: RESERVED
        field = repeating_entry.get("reserved_82")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reserved'")
        field_offset = running_bit_offset
    
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
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
