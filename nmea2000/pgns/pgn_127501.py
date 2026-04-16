# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127501() -> bool:
    """Return True if PGN 127501 is a fast PGN."""
    return False
def decode_pgn_127501(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127501."""
    nmea2000Message = NMEA2000Message(PGN=127501, id='binarySwitchBankStatus', description='Binary Switch Bank Status')
    running_bit_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance = instance_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 2:indicator1 | Offset: 8, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    indicator1_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator1 = master_dict['OFF_ON'].get(indicator1_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator1', 'Indicator1', None, None, indicator1, indicator1_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:indicator2 | Offset: 10, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 10
    indicator2_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator2 = master_dict['OFF_ON'].get(indicator2_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator2', 'Indicator2', None, None, indicator2, indicator2_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:indicator3 | Offset: 12, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    indicator3_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator3 = master_dict['OFF_ON'].get(indicator3_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator3', 'Indicator3', None, None, indicator3, indicator3_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:indicator4 | Offset: 14, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 14
    indicator4_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator4 = master_dict['OFF_ON'].get(indicator4_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator4', 'Indicator4', None, None, indicator4, indicator4_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:indicator5 | Offset: 16, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    indicator5_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator5 = master_dict['OFF_ON'].get(indicator5_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator5', 'Indicator5', None, None, indicator5, indicator5_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 7:indicator6 | Offset: 18, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 18
    indicator6_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator6 = master_dict['OFF_ON'].get(indicator6_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator6', 'Indicator6', None, None, indicator6, indicator6_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 8:indicator7 | Offset: 20, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    indicator7_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator7 = master_dict['OFF_ON'].get(indicator7_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator7', 'Indicator7', None, None, indicator7, indicator7_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 9:indicator8 | Offset: 22, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 22
    indicator8_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator8 = master_dict['OFF_ON'].get(indicator8_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator8', 'Indicator8', None, None, indicator8, indicator8_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:indicator9 | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    indicator9_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator9 = master_dict['OFF_ON'].get(indicator9_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator9', 'Indicator9', None, None, indicator9, indicator9_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:indicator10 | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    indicator10_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator10 = master_dict['OFF_ON'].get(indicator10_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator10', 'Indicator10', None, None, indicator10, indicator10_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 12:indicator11 | Offset: 28, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    indicator11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator11 = master_dict['OFF_ON'].get(indicator11_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator11', 'Indicator11', None, None, indicator11, indicator11_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 13:indicator12 | Offset: 30, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 30
    indicator12_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator12 = master_dict['OFF_ON'].get(indicator12_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator12', 'Indicator12', None, None, indicator12, indicator12_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 14:indicator13 | Offset: 32, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    indicator13_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator13 = master_dict['OFF_ON'].get(indicator13_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator13', 'Indicator13', None, None, indicator13, indicator13_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 15:indicator14 | Offset: 34, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 34
    indicator14_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator14 = master_dict['OFF_ON'].get(indicator14_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator14', 'Indicator14', None, None, indicator14, indicator14_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 16:indicator15 | Offset: 36, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    indicator15_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator15 = master_dict['OFF_ON'].get(indicator15_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator15', 'Indicator15', None, None, indicator15, indicator15_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 17:indicator16 | Offset: 38, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 38
    indicator16_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator16 = master_dict['OFF_ON'].get(indicator16_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator16', 'Indicator16', None, None, indicator16, indicator16_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 18:indicator17 | Offset: 40, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    indicator17_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator17 = master_dict['OFF_ON'].get(indicator17_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator17', 'Indicator17', None, None, indicator17, indicator17_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 19:indicator18 | Offset: 42, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 42
    indicator18_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator18 = master_dict['OFF_ON'].get(indicator18_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator18', 'Indicator18', None, None, indicator18, indicator18_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 20:indicator19 | Offset: 44, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 44
    indicator19_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator19 = master_dict['OFF_ON'].get(indicator19_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator19', 'Indicator19', None, None, indicator19, indicator19_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 21:indicator20 | Offset: 46, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 46
    indicator20_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator20 = master_dict['OFF_ON'].get(indicator20_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator20', 'Indicator20', None, None, indicator20, indicator20_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 22:indicator21 | Offset: 48, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    indicator21_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator21 = master_dict['OFF_ON'].get(indicator21_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator21', 'Indicator21', None, None, indicator21, indicator21_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 23:indicator22 | Offset: 50, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 50
    indicator22_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator22 = master_dict['OFF_ON'].get(indicator22_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator22', 'Indicator22', None, None, indicator22, indicator22_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 24:indicator23 | Offset: 52, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 52
    indicator23_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator23 = master_dict['OFF_ON'].get(indicator23_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator23', 'Indicator23', None, None, indicator23, indicator23_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 25:indicator24 | Offset: 54, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 54
    indicator24_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator24 = master_dict['OFF_ON'].get(indicator24_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator24', 'Indicator24', None, None, indicator24, indicator24_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 26:indicator25 | Offset: 56, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    indicator25_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator25 = master_dict['OFF_ON'].get(indicator25_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator25', 'Indicator25', None, None, indicator25, indicator25_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 27:indicator26 | Offset: 58, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 58
    indicator26_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator26 = master_dict['OFF_ON'].get(indicator26_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator26', 'Indicator26', None, None, indicator26, indicator26_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 28:indicator27 | Offset: 60, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 60
    indicator27_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator27 = master_dict['OFF_ON'].get(indicator27_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator27', 'Indicator27', None, None, indicator27, indicator27_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 29:indicator28 | Offset: 62, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 62
    indicator28_raw = decode_int(_data_raw_, running_bit_offset, 2)
    indicator28 = master_dict['OFF_ON'].get(indicator28_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('indicator28', 'Indicator28', None, None, indicator28, indicator28_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    return nmea2000Message

def encode_pgn_127501(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127501."""
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
    # indicator1 | Offset: 8, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator1")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator2 | Offset: 10, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 10
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator2")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator3 | Offset: 12, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 12
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator3")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator4 | Offset: 14, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 14
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator4")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator4' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator4' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator4' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator5 | Offset: 16, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator5")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator5' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator5' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator5' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator6 | Offset: 18, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 18
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator6")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator6' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator6' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator6' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator7 | Offset: 20, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator7")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator7' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator7' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator7' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator8 | Offset: 22, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 22
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator8")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator8' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator8' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator8' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator9 | Offset: 24, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator9")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator9' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator9' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator9' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator10 | Offset: 26, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator10")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator10' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator10' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator10' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator11 | Offset: 28, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator11")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator11' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator11' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator11' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator12 | Offset: 30, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 30
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator12")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator12' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator12' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator12' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator13 | Offset: 32, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator13")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator13' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator13' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator13' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator14 | Offset: 34, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 34
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator14")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator14' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator14' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator14' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator15 | Offset: 36, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator15")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator15' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator15' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator15' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator16 | Offset: 38, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 38
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator16")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator16' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator16' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator16' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator17 | Offset: 40, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator17")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator17' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator17' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator17' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator18 | Offset: 42, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 42
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator18")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator18' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator18' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator18' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator19 | Offset: 44, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 44
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator19")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator19' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator19' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator19' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator20 | Offset: 46, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 46
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator20")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator20' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator20' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator20' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator21 | Offset: 48, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator21")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator21' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator21' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator21' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator22 | Offset: 50, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 50
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator22")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator22' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator22' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator22' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator23 | Offset: 52, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 52
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator23")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator23' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator23' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator23' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator24 | Offset: 54, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 54
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator24")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator24' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator24' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator24' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator25 | Offset: 56, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator25")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator25' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator25' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator25' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator26 | Offset: 58, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 58
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator26")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator26' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator26' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator26' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator27 | Offset: 60, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 60
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator27")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator27' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator27' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator27' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # indicator28 | Offset: 62, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 62
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indicator28")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OFF_ON(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Indicator28' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Indicator28' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Indicator28' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
