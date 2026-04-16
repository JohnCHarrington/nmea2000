# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65240() -> bool:
    """Return True if PGN 65240 is a fast PGN."""
    raise ValueError('PGEN type ISO not supported')

def decode_pgn_65240(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65240."""
    nmea2000Message = NMEA2000Message(PGN=65240, id='isoCommandedAddress', description='ISO Commanded Address')
    running_bit_offset = 0
    # 1:unique_number | Offset: 0, Length: 21, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    unique_number = unique_number_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 21))
    nmea2000Message.fields.append(NMEA2000Field('uniqueNumber', 'Unique Number', "ISO Identity Number", None, unique_number, unique_number_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 21

    # 2:manufacturer_code | Offset: 21, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 21
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', None, None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 3:device_instance_lower | Offset: 32, Length: 3, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    device_instance_lower = device_instance_lower_raw = decode_number(_data_raw_, running_bit_offset, 3, False, 1, 0, 7)
    nmea2000Message.fields.append(NMEA2000Field('deviceInstanceLower', 'Device Instance Lower', "ISO ECU Instance", None, device_instance_lower, device_instance_lower_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 3

    # 4:device_instance_upper | Offset: 35, Length: 5, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 35
    device_instance_upper = device_instance_upper_raw = decode_number(_data_raw_, running_bit_offset, 5, False, 1, 0, 31)
    nmea2000Message.fields.append(NMEA2000Field('deviceInstanceUpper', 'Device Instance Upper', "ISO Function Instance", None, device_instance_upper, device_instance_upper_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 5

    # 5:device_function | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: INDIRECT_LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    device_function_raw = decode_int(_data_raw_, running_bit_offset, 8)
    device_function = 'TEMP_VAL'
    nmea2000Message.fields.append(NMEA2000Field('deviceFunction', 'Device Function', "ISO Function", None, device_function, device_function_raw, None, FieldTypes.INDIRECT_LOOKUP, False))
    running_bit_offset += 8

    # 6:reserved_48 | Offset: 48, Length: 1, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    reserved_48 = reserved_48_raw = decode_int(_data_raw_, running_bit_offset, 1)
    nmea2000Message.fields.append(NMEA2000Field('reserved_48', 'Reserved', None, None, reserved_48, reserved_48_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 1

    # 7:device_class | Offset: 49, Length: 7, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 49
    device_class_raw = decode_int(_data_raw_, running_bit_offset, 7)
    device_class = master_dict['DEVICE_CLASS'].get(device_class_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('deviceClass', 'Device Class', None, None, device_class, device_class_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 7
    combined_key = str(device_class_raw) + "_" + str(device_function_raw)
    device_function = master_indirect_lookup_dict['DEVICE_FUNCTION'].get(combined_key, None)
    nmea2000Message.fields[4].value = device_function

    # 8:system_instance | Offset: 56, Length: 4, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    system_instance = system_instance_raw = decode_number(_data_raw_, running_bit_offset, 4, False, 1, 0, 13)
    nmea2000Message.fields.append(NMEA2000Field('systemInstance', 'System Instance', "ISO Device Class Instance", None, system_instance, system_instance_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 4

    # 9:industry_code | Offset: 60, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 60
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', None, None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 10:reserved_63 | Offset: 63, Length: 1, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 63
    reserved_63 = reserved_63_raw = decode_int(_data_raw_, running_bit_offset, 1)
    nmea2000Message.fields.append(NMEA2000Field('reserved_63', 'Reserved', None, None, reserved_63, reserved_63_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 1

    # 11:new_source_address | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    new_source_address = new_source_address_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('newSourceAddress', 'New Source Address', None, None, new_source_address, new_source_address_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_65240(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65240."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # uniqueNumber | Offset: 0, Length: 21, Resolution: 1, Field Type: BINARY
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("uniqueNumber")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 21
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Unique Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Unique Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Unique Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # manufacturerCode | Offset: 21, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 21
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
    # deviceInstanceLower | Offset: 32, Length: 3, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deviceInstanceLower")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 3, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 3, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 3, False, 1)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Device Instance Lower' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Device Instance Lower' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Device Instance Lower' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # deviceInstanceUpper | Offset: 35, Length: 5, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 35
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deviceInstanceUpper")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 5, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 5, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 5, False, 1)
    field_bit_length = 5
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Device Instance Upper' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Device Instance Upper' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Device Instance Upper' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # deviceFunction | Offset: 40, Length: 8, Resolution: 1, Field Type: INDIRECT_LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deviceFunction")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        controller_field = nmea2000Message.get_field_by_id("deviceClass")
        if isinstance(controller_field.raw_value, int):
            controller_raw_value = controller_field.raw_value
        else:
            controller_raw_value = lookup_encode_DEVICE_CLASS(controller_field.value)
        indirect_lookup_values = IndirectLookupEncodeMaps['DEVICE_FUNCTION'].get(controller_raw_value)
        if indirect_lookup_values is None:
            raise ValueError("Cant encode this message, 'Device Function' controller value is missing")
        field_value = indirect_lookup_values.get(field.value)
        if field_value is None:
            raise ValueError("Cant encode this message, 'Device Function' indirect lookup value is missing")
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Device Function' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Device Function' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Device Function' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_48 | Offset: 48, Length: 1, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_48")

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
    # deviceClass | Offset: 49, Length: 7, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 49
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deviceClass")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DEVICE_CLASS(field.value)
    field_bit_length = 7
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Device Class' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Device Class' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Device Class' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # systemInstance | Offset: 56, Length: 4, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("systemInstance")

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
        raise ValueError("Cant encode this message, 'System Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'System Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'System Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # industryCode | Offset: 60, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 60
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
    # reserved_63 | Offset: 63, Length: 1, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 63
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_63")

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
    # newSourceAddress | Offset: 64, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("newSourceAddress")

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
        raise ValueError("Cant encode this message, 'New Source Address' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'New Source Address' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'New Source Address' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(9, byteorder="little")
