# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126996() -> bool:
    """Return True if PGN 126996 is a fast PGN."""
    return True
def decode_pgn_126996(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126996."""
    nmea2000Message = NMEA2000Message(PGN=126996, id='productInformation', description='Product Information')
    running_bit_offset = 0
    # 1:nmea_2000_version | Offset: 0, Length: 16, Signed: False Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    nmea_2000_version = nmea_2000_version_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.001, 0, 65.532)
    nmea2000Message.fields.append(NMEA2000Field('nmea2000Version', 'NMEA 2000 Version', "Binary number containing a decimal number of format AABBB, where AA is the major and BBB is the minor release. The decimal point position is assumed.", None, nmea_2000_version, nmea_2000_version_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:product_code | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    product_code = product_code_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('productCode', 'Product Code', None, None, product_code, product_code_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:model_id | Offset: 32, Length: 256, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    model_id, model_id_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 256)
    nmea2000Message.fields.append(NMEA2000Field('modelId', 'Model ID', None, None, model_id, model_id_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 256

    # 4:software_version_code | Offset: 288, Length: 256, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 288
    software_version_code, software_version_code_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 256)
    nmea2000Message.fields.append(NMEA2000Field('softwareVersionCode', 'Software Version Code', None, None, software_version_code, software_version_code_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 256

    # 5:model_version | Offset: 544, Length: 256, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 544
    model_version, model_version_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 256)
    nmea2000Message.fields.append(NMEA2000Field('modelVersion', 'Model Version', None, None, model_version, model_version_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 256

    # 6:model_serial_code | Offset: 800, Length: 256, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 800
    model_serial_code, model_serial_code_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 256)
    nmea2000Message.fields.append(NMEA2000Field('modelSerialCode', 'Model Serial Code', None, None, model_serial_code, model_serial_code_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 256

    # 7:certification_level | Offset: 1056, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 1056
    certification_level_raw = decode_int(_data_raw_, running_bit_offset, 8)
    certification_level = master_dict['CERTIFICATION_LEVEL'].get(certification_level_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('certificationLevel', 'Certification Level', None, None, certification_level, certification_level_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 8:load_equivalency | Offset: 1064, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 1064
    load_equivalency = load_equivalency_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('loadEquivalency', 'Load Equivalency', "Garantueed maximum power consumption, 50 mA per LEN", None, load_equivalency, load_equivalency_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_126996(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126996."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # nmea2000Version | Offset: 0, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("nmea2000Version")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'NMEA 2000 Version' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'NMEA 2000 Version' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'NMEA 2000 Version' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # productCode | Offset: 16, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("productCode")

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
        raise ValueError("Cant encode this message, 'Product Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Product Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Product Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # modelId | Offset: 32, Length: 256, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("modelId")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 256)
    field_bit_length = 256
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # softwareVersionCode | Offset: 288, Length: 256, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 288
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("softwareVersionCode")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 256)
    field_bit_length = 256
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Software Version Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Software Version Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Software Version Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # modelVersion | Offset: 544, Length: 256, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 544
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("modelVersion")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 256)
    field_bit_length = 256
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model Version' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model Version' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model Version' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # modelSerialCode | Offset: 800, Length: 256, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 800
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("modelSerialCode")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 256)
    field_bit_length = 256
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model Serial Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model Serial Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model Serial Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # certificationLevel | Offset: 1056, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 1056
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("certificationLevel")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CERTIFICATION_LEVEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Certification Level' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Certification Level' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Certification Level' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # loadEquivalency | Offset: 1064, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 1064
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("loadEquivalency")

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
        raise ValueError("Cant encode this message, 'Load Equivalency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Load Equivalency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Load Equivalency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(134, byteorder="little")
