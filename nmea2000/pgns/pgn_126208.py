# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126208() -> bool:
    """Return True if PGN 126208 is a fast PGN."""
    return True
# Complex PGN. number of matches: 8
def decode_pgn_126208(data_raw: int, data_length_bits: int | None = None) -> NMEA2000Message | None:
    # nmeaRequestGroupFunction | Description: NMEA - Request group function
    if (
        (((data_raw >> 0) & 0xFF) == 0)
        ):
        return decode_pgn_126208_nmeaRequestGroupFunction(data_raw, data_length_bits)
    
    # nmeaCommandGroupFunction | Description: NMEA - Command group function
    if (
        (((data_raw >> 0) & 0xFF) == 1)
        ):
        return decode_pgn_126208_nmeaCommandGroupFunction(data_raw, data_length_bits)
    
    # nmeaAcknowledgeGroupFunction | Description: NMEA - Acknowledge group function
    if (
        (((data_raw >> 0) & 0xFF) == 2)
        ):
        return decode_pgn_126208_nmeaAcknowledgeGroupFunction(data_raw, data_length_bits)
    
    # nmeaReadFieldsGroupFunction | Description: NMEA - Read Fields group function
    if (
        (((data_raw >> 0) & 0xFF) == 3)
        ):
        return decode_pgn_126208_nmeaReadFieldsGroupFunction(data_raw, data_length_bits)
    
    # nmeaReadFieldsReplyGroupFunction | Description: NMEA - Read Fields reply group function
    if (
        (((data_raw >> 0) & 0xFF) == 4)
        ):
        return decode_pgn_126208_nmeaReadFieldsReplyGroupFunction(data_raw, data_length_bits)
    
    # nmeaWriteFieldsGroupFunction | Description: NMEA - Write Fields group function
    if (
        (((data_raw >> 0) & 0xFF) == 5)
        ):
        return decode_pgn_126208_nmeaWriteFieldsGroupFunction(data_raw, data_length_bits)
    
    # nmeaWriteFieldsReplyGroupFunction | Description: NMEA - Write Fields reply group function
    if (
        (((data_raw >> 0) & 0xFF) == 6)
        ):
        return decode_pgn_126208_nmeaWriteFieldsReplyGroupFunction(data_raw, data_length_bits)
    
    return decode_pgn_126208_0x1ed000x1ee00StandardizedFastPacketAddressed(data_raw, data_length_bits)
    
def decode_pgn_126208_0x1ed000x1ee00StandardizedFastPacketAddressed(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='0x1ed000x1ee00StandardizedFastPacketAddressed', description='0x1ED00 - 0x1EE00: Standardized fast-packet addressed')
    running_bit_offset = 0
    # 1:data | Offset: 0, Length: 1784, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    data = data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 1784))
    nmea2000Message.fields.append(NMEA2000Field('data', 'Data', None, None, data, data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 1784

    return nmea2000Message

def encode_pgn_126208_0x1ed000x1ee00StandardizedFastPacketAddressed(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
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

def decode_pgn_126208_nmeaRequestGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaRequestGroupFunction', description='NMEA - Request group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 0, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Request", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Requested PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:transmission_interval | Offset: 32, Length: 32, Signed: False Resolution: 0.001, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    transmission_interval = transmission_interval_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.001, 0, 4294967.292)
    nmea2000Message.fields.append(NMEA2000Field('transmissionInterval', 'Transmission interval', "0x0: Turn off transmission; 0xFFFF FFFE = Restore default interval, 0xFFFF FFFF in this field and 0xFFFF in field 4 = Transmit now without changing timing variables", 's', transmission_interval, transmission_interval_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 4:transmission_interval_offset | Offset: 64, Length: 16, Signed: False Resolution: 0.01, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    transmission_interval_offset = transmission_interval_offset_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('transmissionIntervalOffset', 'Transmission interval offset', None, 's', transmission_interval_offset, transmission_interval_offset_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 5:number_of_parameters | Offset: 80, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', "How many parameter pairs will follow", None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:parameter | Offset: 88, Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    _repeating_field_set_1_offset = running_bit_offset
    parameter = parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', "Parameter index", None, parameter, parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 7:value | Offset: 96, Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        value = value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        value = value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', "Parameter value", None, value, value_raw, None, FieldTypes.VARIABLE, False))
    

    return nmea2000Message

def encode_pgn_126208_nmeaRequestGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "parameter",
        "value",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # transmissionInterval | Offset: 32, Length: 32, Resolution: 0.001, Field Type: DURATION
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("transmissionInterval")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.001)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Transmission interval' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Transmission interval' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Transmission interval' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # transmissionIntervalOffset | Offset: 64, Length: 16, Resolution: 0.01, Field Type: DURATION
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("transmissionIntervalOffset")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.01)):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.01)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 16, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Transmission interval offset' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Transmission interval offset' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Transmission interval offset' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfParameters | Offset: 80, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 88
    for repeating_entry in repeating_field_set_1_entries:
        # parameter | Offset: 88, Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # value | Offset: 96, Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("value")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_126208_nmeaCommandGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaCommandGroupFunction', description='NMEA - Command group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Command", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Commanded PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:priority | Offset: 32, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    priority_raw = decode_int(_data_raw_, running_bit_offset, 4)
    priority = master_dict['PRIORITY'].get(priority_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('priority', 'Priority', None, None, priority, priority_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:reserved_36 | Offset: 36, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    reserved_36 = reserved_36_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_36', 'Reserved', None, None, reserved_36, reserved_36_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 5:number_of_parameters | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', "How many parameter pairs will follow", None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:parameter | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    parameter = parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', "Parameter index", None, parameter, parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 7:value | Offset: 56, Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        value = value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        value = value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', "Parameter value", None, value, value_raw, None, FieldTypes.VARIABLE, False))
    

    return nmea2000Message

def encode_pgn_126208_nmeaCommandGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "parameter",
        "value",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # priority | Offset: 32, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("priority")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_PRIORITY(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Priority' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Priority' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Priority' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_36 | Offset: 36, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_36")

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
    # numberOfParameters | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 48
    for repeating_entry in repeating_field_set_1_entries:
        # parameter | Offset: 48, Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # value | Offset: 56, Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("value")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_126208_nmeaAcknowledgeGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaAcknowledgeGroupFunction', description='NMEA - Acknowledge group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 2, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Acknowledge", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Commanded PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:pgn_error_code | Offset: 32, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    pgn_error_code_raw = decode_int(_data_raw_, running_bit_offset, 4)
    pgn_error_code = master_dict['PGN_ERROR_CODE'].get(pgn_error_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('pgnErrorCode', 'PGN error code', None, None, pgn_error_code, pgn_error_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:transmission_interval_priority_error_code | Offset: 36, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    transmission_interval_priority_error_code_raw = decode_int(_data_raw_, running_bit_offset, 4)
    transmission_interval_priority_error_code = master_dict['TRANSMISSION_INTERVAL'].get(transmission_interval_priority_error_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('transmissionIntervalPriorityErrorCode', 'Transmission interval/Priority error code', None, None, transmission_interval_priority_error_code, transmission_interval_priority_error_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 5:number_of_parameters | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', None, None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:parameter | Offset: 48, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    parameter_raw = decode_int(_data_raw_, running_bit_offset, 4)
    parameter = master_dict['PARAMETER_FIELD'].get(parameter_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', None, None, parameter, parameter_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(number_of_parameters_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        parameter_raw = decode_int(_data_raw_, running_bit_offset, 4)
        parameter = master_dict['PARAMETER_FIELD'].get(parameter_raw, None)
        running_bit_offset += 4
        repeating_entry["parameter"] = _repeating_entry_value(parameter, parameter_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "parameter",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_126208_nmeaAcknowledgeGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "parameter",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # pgnErrorCode | Offset: 32, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pgnErrorCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_PGN_ERROR_CODE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'PGN error code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PGN error code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PGN error code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # transmissionIntervalPriorityErrorCode | Offset: 36, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("transmissionIntervalPriorityErrorCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_TRANSMISSION_INTERVAL(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Transmission interval/Priority error code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Transmission interval/Priority error code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Transmission interval/Priority error code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfParameters | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 48
    for repeating_entry in repeating_field_set_1_entries:
        # parameter | Offset: 48, Length: 4, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_PARAMETER_FIELD(field.value)
        field_bit_length = 4
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_126208_nmeaReadFieldsGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaReadFieldsGroupFunction', description='NMEA - Read Fields group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 3, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Read Fields", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Commanded PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:manufacturer_code | Offset: 32, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Only in PGN when Commanded PGN is proprietary", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 4:reserved_ | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', "Only in PGN when Commanded PGN is proprietary", None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 5:industry_code | Offset: , Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Only in PGN when Commanded PGN is proprietary", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 6:unique_id | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    unique_id = unique_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('uniqueId', 'Unique ID', None, None, unique_id, unique_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:number_of_selection_pairs | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_selection_pairs = number_of_selection_pairs_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfSelectionPairs', 'Number of Selection Pairs', None, None, number_of_selection_pairs, number_of_selection_pairs_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:number_of_parameters | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', None, None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:selection_parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    selection_parameter = selection_parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('selectionParameter', 'Selection Parameter', "Parameter index", None, selection_parameter, selection_parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 10:selection_value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        selection_value = selection_value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        selection_value = selection_value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('selectionValue', 'Selection Value', None, None, selection_value, selection_value_raw, None, FieldTypes.VARIABLE, False))
    

    # 11:parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    parameter = parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', "Parameter index", None, parameter, parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_126208_nmeaReadFieldsGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "selectionParameter",
        "selectionValue",
    ))
    repeating_field_set_2_entries = _get_repeating_entries(nmea2000Message, "list2", (
        "parameter",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # manufacturerCode | Offset: 32, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 2, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 2
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
    # industryCode | Offset: , Length: 3, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # uniqueId | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("uniqueId")

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
        raise ValueError("Cant encode this message, 'Unique ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Unique ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Unique ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfSelectionPairs | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfSelectionPairs")

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
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfParameters | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # selectionParameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("selectionParameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Selection Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # selectionValue | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("selectionValue")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Selection Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    for repeating_entry in repeating_field_set_2_entries:
        # parameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_126208_nmeaReadFieldsReplyGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaReadFieldsReplyGroupFunction', description='NMEA - Read Fields reply group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Read Fields Reply", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Commanded PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:manufacturer_code | Offset: 32, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Only in PGN when Commanded PGN is proprietary", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 4:reserved_ | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', "Only in PGN when Commanded PGN is proprietary", None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 5:industry_code | Offset: , Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Only in PGN when Commanded PGN is proprietary", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 6:unique_id | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    unique_id = unique_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('uniqueId', 'Unique ID', None, None, unique_id, unique_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:number_of_selection_pairs | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_selection_pairs = number_of_selection_pairs_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfSelectionPairs', 'Number of Selection Pairs', None, None, number_of_selection_pairs, number_of_selection_pairs_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:number_of_parameters | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', None, None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:selection_parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    selection_parameter = selection_parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('selectionParameter', 'Selection Parameter', "Parameter index", None, selection_parameter, selection_parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 10:selection_value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        selection_value = selection_value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        selection_value = selection_value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('selectionValue', 'Selection Value', None, None, selection_value, selection_value_raw, None, FieldTypes.VARIABLE, False))
    

    # 11:parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    parameter = parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', "Parameter index", None, parameter, parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 12:value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        value = value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        value = value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', None, None, value, value_raw, None, FieldTypes.VARIABLE, False))
    

    return nmea2000Message

def encode_pgn_126208_nmeaReadFieldsReplyGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "selectionParameter",
        "selectionValue",
    ))
    repeating_field_set_2_entries = _get_repeating_entries(nmea2000Message, "list2", (
        "parameter",
        "value",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # manufacturerCode | Offset: 32, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 2, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 2
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
    # industryCode | Offset: , Length: 3, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # uniqueId | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("uniqueId")

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
        raise ValueError("Cant encode this message, 'Unique ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Unique ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Unique ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfSelectionPairs | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfSelectionPairs")

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
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfParameters | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # selectionParameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("selectionParameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Selection Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # selectionValue | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("selectionValue")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Selection Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    for repeating_entry in repeating_field_set_2_entries:
        # parameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # value | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("value")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_126208_nmeaWriteFieldsGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaWriteFieldsGroupFunction', description='NMEA - Write Fields group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 5, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Write Fields", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Commanded PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:manufacturer_code | Offset: 32, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Only in PGN when Commanded PGN is proprietary", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 4:reserved_ | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', "Only in PGN when Commanded PGN is proprietary", None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 5:industry_code | Offset: , Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Only in PGN when Commanded PGN is proprietary", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 6:unique_id | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    unique_id = unique_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('uniqueId', 'Unique ID', None, None, unique_id, unique_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:number_of_selection_pairs | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_selection_pairs = number_of_selection_pairs_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfSelectionPairs', 'Number of Selection Pairs', None, None, number_of_selection_pairs, number_of_selection_pairs_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:number_of_parameters | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', None, None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:selection_parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    selection_parameter = selection_parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('selectionParameter', 'Selection Parameter', "Parameter index", None, selection_parameter, selection_parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 10:selection_value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        selection_value = selection_value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        selection_value = selection_value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('selectionValue', 'Selection Value', None, None, selection_value, selection_value_raw, None, FieldTypes.VARIABLE, False))
    

    # 11:parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    parameter = parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', "Parameter index", None, parameter, parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 12:value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        value = value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        value = value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', None, None, value, value_raw, None, FieldTypes.VARIABLE, False))
    

    return nmea2000Message

def encode_pgn_126208_nmeaWriteFieldsGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "selectionParameter",
        "selectionValue",
    ))
    repeating_field_set_2_entries = _get_repeating_entries(nmea2000Message, "list2", (
        "parameter",
        "value",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # manufacturerCode | Offset: 32, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 2, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 2
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
    # industryCode | Offset: , Length: 3, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # uniqueId | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("uniqueId")

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
        raise ValueError("Cant encode this message, 'Unique ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Unique ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Unique ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfSelectionPairs | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfSelectionPairs")

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
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfParameters | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # selectionParameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("selectionParameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Selection Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # selectionValue | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("selectionValue")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Selection Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    for repeating_entry in repeating_field_set_2_entries:
        # parameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # value | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("value")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_126208_nmeaWriteFieldsReplyGroupFunction(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126208."""
    nmea2000Message = NMEA2000Message(PGN=126208, id='nmeaWriteFieldsReplyGroupFunction', description='NMEA - Write Fields reply group function')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:function_code | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 6, PartOfPrimaryKey: ,
    running_bit_offset = 0
    function_code_raw = decode_int(_data_raw_, running_bit_offset, 8)
    function_code = master_dict['GROUP_FUNCTION'].get(function_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('functionCode', 'Function Code', "Write Fields Reply", None, function_code, function_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:pgn | Offset: 8, Length: 24, Signed: False Resolution: 1, Field Type: PGN, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    pgn = pgn_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 262143)
    nmea2000Message.fields.append(NMEA2000Field('pgn', 'PGN', "Commanded PGN", None, pgn, pgn_raw, None, FieldTypes.PGN, False))
    running_bit_offset += 24

    # 3:manufacturer_code | Offset: 32, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Only in PGN when Commanded PGN is proprietary", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 4:reserved_ | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', "Only in PGN when Commanded PGN is proprietary", None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 5:industry_code | Offset: , Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Only in PGN when Commanded PGN is proprietary", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 6:unique_id | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    unique_id = unique_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('uniqueId', 'Unique ID', None, None, unique_id, unique_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:number_of_selection_pairs | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_selection_pairs = number_of_selection_pairs_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfSelectionPairs', 'Number of Selection Pairs', None, None, number_of_selection_pairs, number_of_selection_pairs_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:number_of_parameters | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', None, None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:selection_parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    selection_parameter = selection_parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('selectionParameter', 'Selection Parameter', "Parameter index", None, selection_parameter, selection_parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 10:selection_value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        selection_value = selection_value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        selection_value = selection_value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('selectionValue', 'Selection Value', None, None, selection_value, selection_value_raw, None, FieldTypes.VARIABLE, False))
    

    # 11:parameter | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: FIELD_INDEX, Match: , PartOfPrimaryKey: ,
    parameter = parameter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('parameter', 'Parameter', "Parameter index", None, parameter, parameter_raw, None, FieldTypes.FIELD_INDEX, False))
    running_bit_offset += 8

    # 12:value | Offset: , Length: , Signed: False Resolution: , Field Type: VARIABLE, Match: , PartOfPrimaryKey: ,
    _remaining_bits = _data_length_bits_ - running_bit_offset
    if _remaining_bits > 0:
        value = value_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, _remaining_bits))
    else:
        value = value_raw = None
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', None, None, value, value_raw, None, FieldTypes.VARIABLE, False))
    

    return nmea2000Message

def encode_pgn_126208_nmeaWriteFieldsReplyGroupFunction(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126208."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "selectionParameter",
        "selectionValue",
    ))
    repeating_field_set_2_entries = _get_repeating_entries(nmea2000Message, "list2", (
        "parameter",
        "value",
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
        field_value = lookup_encode_GROUP_FUNCTION(field.value)
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
    # manufacturerCode | Offset: 32, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 2, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 2
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
    # industryCode | Offset: , Length: 3, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # uniqueId | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("uniqueId")

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
        raise ValueError("Cant encode this message, 'Unique ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Unique ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Unique ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfSelectionPairs | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfSelectionPairs")

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
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Selection Pairs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfParameters | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfParameters")

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
        raise ValueError("Cant encode this message, 'Number of Parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # selectionParameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("selectionParameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Selection Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # selectionValue | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("selectionValue")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Selection Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Selection Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Selection Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Selection Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    for repeating_entry in repeating_field_set_2_entries:
        # parameter | Offset: , Length: 8, Resolution: 1, Field Type: FIELD_INDEX
        field = repeating_entry.get("parameter")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Parameter'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Parameter' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Parameter' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Parameter' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # value | Offset: , Length: , Resolution: , Field Type: VARIABLE
        field = repeating_entry.get("value")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
        field_value = encode_binary_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        advance_running_offset = False
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
