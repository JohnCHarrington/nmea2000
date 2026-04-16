# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126988() -> bool:
    """Return True if PGN 126988 is a fast PGN."""
    return True
def decode_pgn_126988(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126988."""
    nmea2000Message = NMEA2000Message(PGN=126988, id='alertValue', description='Alert Value')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
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

    # 10:number_of_parameters | Offset: 128, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    number_of_parameters = number_of_parameters_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfParameters', 'Number of Parameters', "Total Number of Value Parameters", None, number_of_parameters, number_of_parameters_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 11:value_parameter_number | Offset: 136, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    _repeating_field_set_1_offset = running_bit_offset
    value_parameter_number = value_parameter_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('valueParameterNumber', 'Value Parameter Number', None, None, value_parameter_number, value_parameter_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 12:value_data_format | Offset: 144, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    value_data_format = value_data_format_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('valueDataFormat', 'Value Data Format', None, None, value_data_format, value_data_format_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 13:value_data | Offset: 152, Length: 64, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    value_data = value_data_raw = decode_number(_data_raw_, running_bit_offset, 64, False, 1, 0, 18446744073709551615)
    nmea2000Message.fields.append(NMEA2000Field('valueData', 'Value Data', None, None, value_data, value_data_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 64

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(number_of_parameters_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        value_parameter_number = value_parameter_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["valueParameterNumber"] = _repeating_entry_value(value_parameter_number, value_parameter_number_raw)
    
        value_data_format = value_data_format_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["valueDataFormat"] = _repeating_entry_value(value_data_format, value_data_format_raw)
    
        value_data = value_data_raw = decode_number(_data_raw_, running_bit_offset, 64, False, 1, 0, 18446744073709551615)
        running_bit_offset += 64
        repeating_entry["valueData"] = _repeating_entry_value(value_data, value_data_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "valueParameterNumber",
                "valueDataFormat",
                "valueData",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_126988(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126988."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "valueParameterNumber",
        "valueDataFormat",
        "valueData",
    ))
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
    # numberOfParameters | Offset: 128, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 128
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
    running_bit_offset = 136
    for repeating_entry in repeating_field_set_1_entries:
        # valueParameterNumber | Offset: 136, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("valueParameterNumber")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value Parameter Number'")
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
            raise ValueError("Cant encode this message, 'Value Parameter Number' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value Parameter Number' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value Parameter Number' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # valueDataFormat | Offset: 144, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("valueDataFormat")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value Data Format'")
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
            raise ValueError("Cant encode this message, 'Value Data Format' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value Data Format' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value Data Format' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # valueData | Offset: 152, Length: 64, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("valueData")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value Data'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
            field_value = encode_number_raw(field.raw_value, 64, False)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 64, False, 1)
        else:
            assert field.value is None or isinstance(field.value, (int, float))
            field_value = encode_number(field.value, 64, False, 1)
        field_bit_length = 64
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Value Data' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value Data' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value Data' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
