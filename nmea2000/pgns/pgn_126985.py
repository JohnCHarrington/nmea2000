# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126985() -> bool:
    """Return True if PGN 126985 is a fast PGN."""
    return True
def decode_pgn_126985(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126985."""
    nmea2000Message = NMEA2000Message(PGN=126985, id='alertText', description='Alert Text')
    running_bit_offset = 0
    # 1:alert_type | Offset: 0, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    alert_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    alert_type = master_dict['ALERT_TYPE'].get(alert_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('alertType', 'Alert Type', None, None, alert_type, alert_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 2:alert_category | Offset: 4, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 4
    alert_category_raw = decode_int(_data_raw_, running_bit_offset, 4)
    alert_category = master_dict['ALERT_CATEGORY'].get(alert_category_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('alertCategory', 'Alert Category', None, None, alert_category, alert_category_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 3:alert_system | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    alert_system = alert_system_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('alertSystem', 'Alert System', None, None, alert_system, alert_system_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:alert_sub_system | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    alert_sub_system = alert_sub_system_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('alertSubSystem', 'Alert Sub-System', None, None, alert_sub_system, alert_sub_system_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:alert_id | Offset: 24, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    alert_id = alert_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('alertId', 'Alert ID', None, None, alert_id, alert_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:data_source_network_id_name | Offset: 40, Length: 64, Signed: False Resolution: 1, Field Type: ISO_NAME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    data_source_network_id_name = data_source_network_id_name_raw = decode_int(_data_raw_, running_bit_offset, 64)
    nmea2000Message.fields.append(NMEA2000Field('dataSourceNetworkIdName', 'Data Source Network ID NAME', None, None, data_source_network_id_name, data_source_network_id_name_raw, None, FieldTypes.ISO_NAME, False))
    running_bit_offset += 64

    # 7:data_source_instance | Offset: 104, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 104
    data_source_instance = data_source_instance_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('dataSourceInstance', 'Data Source Instance', None, None, data_source_instance, data_source_instance_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 8:data_source_index_source | Offset: 112, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    data_source_index_source = data_source_index_source_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('dataSourceIndexSource', 'Data Source Index-Source', None, None, data_source_index_source, data_source_index_source_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:alert_occurrence_number | Offset: 120, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    alert_occurrence_number = alert_occurrence_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('alertOccurrenceNumber', 'Alert Occurrence Number', None, None, alert_occurrence_number, alert_occurrence_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 10:language_id | Offset: 128, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    language_id_raw = decode_int(_data_raw_, running_bit_offset, 8)
    language_id = master_dict['ALERT_LANGUAGE_ID'].get(language_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('languageId', 'Language ID', None, None, language_id, language_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 11:alert_text_description | Offset: 136, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    alert_text_description_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    alert_text_description = alert_text_description_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('alertTextDescription', 'Alert Text Description', None, None, alert_text_description, alert_text_description_raw, None, FieldTypes.STRING_LAU, False))
    

    # 12:alert_location_text_description | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    alert_location_text_description_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    alert_location_text_description = alert_location_text_description_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('alertLocationTextDescription', 'Alert Location Text Description', None, None, alert_location_text_description, alert_location_text_description_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_126985(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126985."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # alertType | Offset: 0, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertType")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ALERT_TYPE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Alert Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertCategory | Offset: 4, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 4
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertCategory")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ALERT_CATEGORY(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Alert Category' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Category' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Category' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertSystem | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertSystem")

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
        raise ValueError("Cant encode this message, 'Alert System' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert System' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert System' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertSubSystem | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertSubSystem")

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
        raise ValueError("Cant encode this message, 'Alert Sub-System' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Sub-System' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Sub-System' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertId | Offset: 24, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertId")

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
        raise ValueError("Cant encode this message, 'Alert ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dataSourceNetworkIdName | Offset: 40, Length: 64, Resolution: 1, Field Type: ISO_NAME
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dataSourceNetworkIdName")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_iso_name(field.value)
    field_bit_length = 64
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Data Source Network ID NAME' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data Source Network ID NAME' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data Source Network ID NAME' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dataSourceInstance | Offset: 104, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dataSourceInstance")

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
        raise ValueError("Cant encode this message, 'Data Source Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data Source Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data Source Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dataSourceIndexSource | Offset: 112, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dataSourceIndexSource")

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
        raise ValueError("Cant encode this message, 'Data Source Index-Source' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data Source Index-Source' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data Source Index-Source' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertOccurrenceNumber | Offset: 120, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertOccurrenceNumber")

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
        raise ValueError("Cant encode this message, 'Alert Occurrence Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Occurrence Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Occurrence Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # languageId | Offset: 128, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("languageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ALERT_LANGUAGE_ID(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Language ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Language ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Language ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertTextDescription | Offset: 136, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertTextDescription")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Alert Text Description' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Text Description' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Text Description' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertLocationTextDescription | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertLocationTextDescription")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Alert Location Text Description' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Location Text Description' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Location Text Description' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
