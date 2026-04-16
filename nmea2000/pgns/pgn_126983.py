# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126983() -> bool:
    """Return True if PGN 126983 is a fast PGN."""
    return True
def decode_pgn_126983(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126983."""
    nmea2000Message = NMEA2000Message(PGN=126983, id='alert', description='Alert')
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

    # 10:temporary_silence_status | Offset: 128, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    temporary_silence_status_raw = decode_int(_data_raw_, running_bit_offset, 1)
    temporary_silence_status = master_dict['YES_NO'].get(temporary_silence_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('temporarySilenceStatus', 'Temporary Silence Status', None, None, temporary_silence_status, temporary_silence_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 11:acknowledge_status | Offset: 129, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 129
    acknowledge_status_raw = decode_int(_data_raw_, running_bit_offset, 1)
    acknowledge_status = master_dict['YES_NO'].get(acknowledge_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('acknowledgeStatus', 'Acknowledge Status', None, None, acknowledge_status, acknowledge_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 12:escalation_status | Offset: 130, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 130
    escalation_status_raw = decode_int(_data_raw_, running_bit_offset, 1)
    escalation_status = master_dict['YES_NO'].get(escalation_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('escalationStatus', 'Escalation Status', None, None, escalation_status, escalation_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 13:temporary_silence_support | Offset: 131, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 131
    temporary_silence_support_raw = decode_int(_data_raw_, running_bit_offset, 1)
    temporary_silence_support = master_dict['YES_NO'].get(temporary_silence_support_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('temporarySilenceSupport', 'Temporary Silence Support', None, None, temporary_silence_support, temporary_silence_support_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 14:acknowledge_support | Offset: 132, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 132
    acknowledge_support_raw = decode_int(_data_raw_, running_bit_offset, 1)
    acknowledge_support = master_dict['YES_NO'].get(acknowledge_support_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('acknowledgeSupport', 'Acknowledge Support', None, None, acknowledge_support, acknowledge_support_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 15:escalation_support | Offset: 133, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 133
    escalation_support_raw = decode_int(_data_raw_, running_bit_offset, 1)
    escalation_support = master_dict['YES_NO'].get(escalation_support_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('escalationSupport', 'Escalation Support', None, None, escalation_support, escalation_support_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 16:reserved_134 | Offset: 134, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 134
    reserved_134 = reserved_134_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_134', 'Reserved', None, None, reserved_134, reserved_134_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 17:acknowledge_source_network_id_name | Offset: 136, Length: 64, Signed: False Resolution: 1, Field Type: ISO_NAME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    acknowledge_source_network_id_name = acknowledge_source_network_id_name_raw = decode_int(_data_raw_, running_bit_offset, 64)
    nmea2000Message.fields.append(NMEA2000Field('acknowledgeSourceNetworkIdName', 'Acknowledge Source Network ID NAME', None, None, acknowledge_source_network_id_name, acknowledge_source_network_id_name_raw, None, FieldTypes.ISO_NAME, False))
    running_bit_offset += 64

    # 18:trigger_condition | Offset: 200, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 200
    trigger_condition_raw = decode_int(_data_raw_, running_bit_offset, 4)
    trigger_condition = master_dict['ALERT_TRIGGER_CONDITION'].get(trigger_condition_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('triggerCondition', 'Trigger Condition', None, None, trigger_condition, trigger_condition_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 19:threshold_status | Offset: 204, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 204
    threshold_status_raw = decode_int(_data_raw_, running_bit_offset, 4)
    threshold_status = master_dict['ALERT_THRESHOLD_STATUS'].get(threshold_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('thresholdStatus', 'Threshold Status', None, None, threshold_status, threshold_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 20:alert_priority | Offset: 208, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 208
    alert_priority = alert_priority_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('alertPriority', 'Alert Priority', None, None, alert_priority, alert_priority_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 21:alert_state | Offset: 216, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 216
    alert_state_raw = decode_int(_data_raw_, running_bit_offset, 8)
    alert_state = master_dict['ALERT_STATE'].get(alert_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('alertState', 'Alert State', None, None, alert_state, alert_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_126983(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126983."""
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
    # temporarySilenceStatus | Offset: 128, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("temporarySilenceStatus")

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
        raise ValueError("Cant encode this message, 'Temporary Silence Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Temporary Silence Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Temporary Silence Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # acknowledgeStatus | Offset: 129, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 129
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("acknowledgeStatus")

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
        raise ValueError("Cant encode this message, 'Acknowledge Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Acknowledge Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Acknowledge Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # escalationStatus | Offset: 130, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 130
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("escalationStatus")

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
        raise ValueError("Cant encode this message, 'Escalation Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Escalation Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Escalation Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # temporarySilenceSupport | Offset: 131, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 131
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("temporarySilenceSupport")

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
        raise ValueError("Cant encode this message, 'Temporary Silence Support' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Temporary Silence Support' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Temporary Silence Support' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # acknowledgeSupport | Offset: 132, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 132
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("acknowledgeSupport")

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
        raise ValueError("Cant encode this message, 'Acknowledge Support' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Acknowledge Support' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Acknowledge Support' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # escalationSupport | Offset: 133, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 133
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("escalationSupport")

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
        raise ValueError("Cant encode this message, 'Escalation Support' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Escalation Support' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Escalation Support' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_134 | Offset: 134, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 134
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_134")

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
    # acknowledgeSourceNetworkIdName | Offset: 136, Length: 64, Resolution: 1, Field Type: ISO_NAME
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("acknowledgeSourceNetworkIdName")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_iso_name(field.value)
    field_bit_length = 64
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Acknowledge Source Network ID NAME' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Acknowledge Source Network ID NAME' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Acknowledge Source Network ID NAME' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # triggerCondition | Offset: 200, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 200
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("triggerCondition")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ALERT_TRIGGER_CONDITION(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Trigger Condition' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Trigger Condition' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Trigger Condition' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # thresholdStatus | Offset: 204, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 204
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("thresholdStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ALERT_THRESHOLD_STATUS(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Threshold Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Threshold Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Threshold Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertPriority | Offset: 208, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 208
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertPriority")

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
        raise ValueError("Cant encode this message, 'Alert Priority' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert Priority' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert Priority' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alertState | Offset: 216, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 216
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alertState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ALERT_STATE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Alert State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(28, byteorder="little")
