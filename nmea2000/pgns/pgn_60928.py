# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_60928() -> bool:
    """Return True if PGN 60928 is a fast PGN."""
    return False
def decode_pgn_60928(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 60928."""
    nmea2000Message = NMEA2000Message(PGN=60928, id='isoAddressClaim', description='ISO Address Claim')
    running_bit_offset = 0
    # 1:unique_number | Offset: 0, Length: 21, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    unique_number = unique_number_raw = decode_number(_data_raw_, running_bit_offset, 21, False, 1, 0, 2097148)
    nmea2000Message.fields.append(NMEA2000Field('uniqueNumber', 'Unique Number', "ISO Identity Number", None, unique_number, unique_number_raw, None, FieldTypes.NUMBER, False))
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

    # 6:spare | Offset: 48, Length: 1, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 1)
    nmea2000Message.fields.append(NMEA2000Field('spare', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
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

    # 9:industry_group | Offset: 60, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 60
    industry_group_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_group = master_dict['INDUSTRY_CODE'].get(industry_group_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryGroup', 'Industry Group', None, None, industry_group, industry_group_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 10:arbitrary_address_capable | Offset: 63, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 63
    arbitrary_address_capable_raw = decode_int(_data_raw_, running_bit_offset, 1)
    arbitrary_address_capable = master_dict['YES_NO'].get(arbitrary_address_capable_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('arbitraryAddressCapable', 'Arbitrary address capable', "Field indicates whether the device is capable to claim arbitrary source address. Value is 1 for NMEA200 devices. Could be 0 for J1939 device claims", None, arbitrary_address_capable, arbitrary_address_capable_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    return nmea2000Message

def encode_pgn_60928(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 60928."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # uniqueNumber | Offset: 0, Length: 21, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("uniqueNumber")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 21, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 21, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 21, False, 1)
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
    # spare | Offset: 48, Length: 1, Resolution: 1, Field Type: SPARE
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 1
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
    # industryGroup | Offset: 60, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 60
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryGroup")

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
        raise ValueError("Cant encode this message, 'Industry Group' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Group' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Group' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # arbitraryAddressCapable | Offset: 63, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 63
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("arbitraryAddressCapable")

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
        raise ValueError("Cant encode this message, 'Arbitrary address capable' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Arbitrary address capable' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Arbitrary address capable' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
