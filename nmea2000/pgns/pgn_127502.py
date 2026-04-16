# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127502() -> bool:
    """Return True if PGN 127502 is a fast PGN."""
    return False
def decode_pgn_127502(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127502."""
    nmea2000Message = NMEA2000Message(PGN=127502, id='switchBankControl', description='Switch Bank Control')
    running_bit_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance = instance_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 2:switch1 | Offset: 8, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    switch1_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch1 = master_dict['OFF_ON_CONTROL'].get(switch1_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch1', 'Switch1', None, None, switch1, switch1_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:switch2 | Offset: 10, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 10
    switch2_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch2 = master_dict['OFF_ON_CONTROL'].get(switch2_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch2', 'Switch2', None, None, switch2, switch2_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:switch3 | Offset: 12, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    switch3_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch3 = master_dict['OFF_ON_CONTROL'].get(switch3_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch3', 'Switch3', None, None, switch3, switch3_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:switch4 | Offset: 14, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 14
    switch4_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch4 = master_dict['OFF_ON_CONTROL'].get(switch4_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch4', 'Switch4', None, None, switch4, switch4_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:switch5 | Offset: 16, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    switch5_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch5 = master_dict['OFF_ON_CONTROL'].get(switch5_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch5', 'Switch5', None, None, switch5, switch5_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 7:switch6 | Offset: 18, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 18
    switch6_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch6 = master_dict['OFF_ON_CONTROL'].get(switch6_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch6', 'Switch6', None, None, switch6, switch6_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 8:switch7 | Offset: 20, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    switch7_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch7 = master_dict['OFF_ON_CONTROL'].get(switch7_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch7', 'Switch7', None, None, switch7, switch7_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 9:switch8 | Offset: 22, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 22
    switch8_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch8 = master_dict['OFF_ON_CONTROL'].get(switch8_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch8', 'Switch8', None, None, switch8, switch8_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:switch9 | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    switch9_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch9 = master_dict['OFF_ON_CONTROL'].get(switch9_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch9', 'Switch9', None, None, switch9, switch9_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:switch10 | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    switch10_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch10 = master_dict['OFF_ON_CONTROL'].get(switch10_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch10', 'Switch10', None, None, switch10, switch10_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 12:switch11 | Offset: 28, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    switch11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch11 = master_dict['OFF_ON_CONTROL'].get(switch11_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch11', 'Switch11', None, None, switch11, switch11_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 13:switch12 | Offset: 30, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 30
    switch12_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch12 = master_dict['OFF_ON_CONTROL'].get(switch12_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch12', 'Switch12', None, None, switch12, switch12_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 14:switch13 | Offset: 32, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    switch13_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch13 = master_dict['OFF_ON_CONTROL'].get(switch13_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch13', 'Switch13', None, None, switch13, switch13_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 15:switch14 | Offset: 34, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 34
    switch14_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch14 = master_dict['OFF_ON_CONTROL'].get(switch14_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch14', 'Switch14', None, None, switch14, switch14_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 16:switch15 | Offset: 36, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    switch15_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch15 = master_dict['OFF_ON_CONTROL'].get(switch15_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch15', 'Switch15', None, None, switch15, switch15_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 17:switch16 | Offset: 38, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 38
    switch16_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch16 = master_dict['OFF_ON_CONTROL'].get(switch16_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch16', 'Switch16', None, None, switch16, switch16_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 18:switch17 | Offset: 40, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    switch17_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch17 = master_dict['OFF_ON_CONTROL'].get(switch17_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch17', 'Switch17', None, None, switch17, switch17_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 19:switch18 | Offset: 42, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 42
    switch18_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch18 = master_dict['OFF_ON_CONTROL'].get(switch18_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch18', 'Switch18', None, None, switch18, switch18_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 20:switch19 | Offset: 44, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 44
    switch19_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch19 = master_dict['OFF_ON_CONTROL'].get(switch19_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch19', 'Switch19', None, None, switch19, switch19_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 21:switch20 | Offset: 46, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 46
    switch20_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch20 = master_dict['OFF_ON_CONTROL'].get(switch20_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch20', 'Switch20', None, None, switch20, switch20_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 22:switch21 | Offset: 48, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    switch21_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch21 = master_dict['OFF_ON_CONTROL'].get(switch21_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch21', 'Switch21', None, None, switch21, switch21_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 23:switch22 | Offset: 50, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 50
    switch22_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch22 = master_dict['OFF_ON_CONTROL'].get(switch22_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch22', 'Switch22', None, None, switch22, switch22_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 24:switch23 | Offset: 52, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 52
    switch23_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch23 = master_dict['OFF_ON_CONTROL'].get(switch23_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch23', 'Switch23', None, None, switch23, switch23_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 25:switch24 | Offset: 54, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 54
    switch24_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch24 = master_dict['OFF_ON_CONTROL'].get(switch24_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch24', 'Switch24', None, None, switch24, switch24_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 26:switch25 | Offset: 56, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    switch25_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch25 = master_dict['OFF_ON_CONTROL'].get(switch25_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch25', 'Switch25', None, None, switch25, switch25_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 27:switch26 | Offset: 58, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 58
    switch26_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch26 = master_dict['OFF_ON_CONTROL'].get(switch26_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch26', 'Switch26', None, None, switch26, switch26_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 28:switch27 | Offset: 60, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 60
    switch27_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch27 = master_dict['OFF_ON_CONTROL'].get(switch27_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch27', 'Switch27', None, None, switch27, switch27_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 29:switch28 | Offset: 62, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 62
    switch28_raw = decode_int(_data_raw_, running_bit_offset, 2)
    switch28 = master_dict['OFF_ON_CONTROL'].get(switch28_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('switch28', 'Switch28', None, None, switch28, switch28_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    return nmea2000Message

def encode_pgn_127502(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127502."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # instance | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("instance")

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
        raise ValueError("Cant encode this message, 'Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch1 | Offset: 8, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch1")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch2 | Offset: 10, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 10
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch2")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch3 | Offset: 12, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 12
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch3")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch4 | Offset: 14, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 14
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch4")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch4' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch4' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch4' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch5 | Offset: 16, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch5")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch5' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch5' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch5' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch6 | Offset: 18, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 18
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch6")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch6' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch6' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch6' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch7 | Offset: 20, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch7")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch7' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch7' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch7' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch8 | Offset: 22, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 22
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch8")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch8' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch8' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch8' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch9 | Offset: 24, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch9")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch9' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch9' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch9' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch10 | Offset: 26, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch10")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch10' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch10' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch10' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch11 | Offset: 28, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch11")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch11' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch11' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch11' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch12 | Offset: 30, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 30
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch12")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch12' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch12' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch12' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch13 | Offset: 32, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch13")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch13' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch13' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch13' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch14 | Offset: 34, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 34
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch14")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch14' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch14' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch14' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch15 | Offset: 36, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch15")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch15' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch15' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch15' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch16 | Offset: 38, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 38
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch16")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch16' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch16' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch16' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch17 | Offset: 40, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch17")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch17' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch17' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch17' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch18 | Offset: 42, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 42
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch18")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch18' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch18' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch18' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch19 | Offset: 44, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 44
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch19")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch19' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch19' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch19' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch20 | Offset: 46, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 46
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch20")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch20' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch20' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch20' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch21 | Offset: 48, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch21")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch21' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch21' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch21' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch22 | Offset: 50, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 50
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch22")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch22' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch22' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch22' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch23 | Offset: 52, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 52
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch23")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch23' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch23' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch23' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch24 | Offset: 54, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 54
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch24")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch24' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch24' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch24' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch25 | Offset: 56, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch25")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch25' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch25' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch25' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch26 | Offset: 58, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 58
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch26")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch26' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch26' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch26' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch27 | Offset: 60, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 60
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch27")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch27' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch27' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch27' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # switch28 | Offset: 62, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 62
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("switch28")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Switch28' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Switch28' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Switch28' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
