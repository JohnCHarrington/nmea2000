# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130580() -> bool:
    """Return True if PGN 130580 is a fast PGN."""
    return True
def decode_pgn_130580(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130580."""
    nmea2000Message = NMEA2000Message(PGN=130580, id='systemConfigurationDeprecated', description='System Configuration (deprecated)')
    running_bit_offset = 0
    # 1:power | Offset: 0, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    power_raw = decode_int(_data_raw_, running_bit_offset, 2)
    power = master_dict['YES_NO'].get(power_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('power', 'Power', None, None, power, power_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 2:default_settings | Offset: 2, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 2
    default_settings_raw = decode_int(_data_raw_, running_bit_offset, 2)
    default_settings = master_dict['ENTERTAINMENT_DEFAULT_SETTINGS'].get(default_settings_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('defaultSettings', 'Default Settings', None, None, default_settings, default_settings_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:tuner_regions | Offset: 4, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 4
    tuner_regions_raw = decode_int(_data_raw_, running_bit_offset, 4)
    tuner_regions = master_dict['ENTERTAINMENT_REGIONS'].get(tuner_regions_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('tunerRegions', 'Tuner regions', None, None, tuner_regions, tuner_regions_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:max_favorites | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    max_favorites = max_favorites_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxFavorites', 'Max favorites', None, None, max_favorites, max_favorites_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130580(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130580."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # power | Offset: 0, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("power")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # defaultSettings | Offset: 2, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 2
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("defaultSettings")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_DEFAULT_SETTINGS(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Default Settings' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Default Settings' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Default Settings' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tunerRegions | Offset: 4, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 4
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tunerRegions")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_REGIONS(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Tuner regions' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tuner regions' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tuner regions' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxFavorites | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxFavorites")

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
        raise ValueError("Cant encode this message, 'Max favorites' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max favorites' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max favorites' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(2, byteorder="little")
