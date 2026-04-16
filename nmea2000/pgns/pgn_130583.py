# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130583() -> bool:
    """Return True if PGN 130583 is a fast PGN."""
    return True
def decode_pgn_130583(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130583."""
    nmea2000Message = NMEA2000Message(PGN=130583, id='availableAudioEqPresets', description='Available Audio EQ presets')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:first_preset | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    first_preset = first_preset_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('firstPreset', 'First preset', "First preset in this PGN", None, first_preset, first_preset_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:preset_count | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    preset_count = preset_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('presetCount', 'Preset count', None, None, preset_count, preset_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:total_preset_count | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    total_preset_count = total_preset_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('totalPresetCount', 'Total preset count', None, None, total_preset_count, total_preset_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:preset_type | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    _repeating_field_set_1_offset = running_bit_offset
    preset_type_raw = decode_int(_data_raw_, running_bit_offset, 8)
    preset_type = master_dict['ENTERTAINMENT_EQ'].get(preset_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('presetType', 'Preset type', None, None, preset_type, preset_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:preset_name | Offset: 32, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    preset_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    preset_name = preset_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('presetName', 'Preset name', None, None, preset_name, preset_name_raw, None, FieldTypes.STRING_LAU, False))
    

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(preset_count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        preset_type_raw = decode_int(_data_raw_, running_bit_offset, 8)
        preset_type = master_dict['ENTERTAINMENT_EQ'].get(preset_type_raw, None)
        running_bit_offset += 8
        repeating_entry["presetType"] = _repeating_entry_value(preset_type, preset_type_raw)
    
        preset_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        preset_name = preset_name_raw
        running_bit_offset += bits_to_skip
        repeating_entry["presetName"] = _repeating_entry_value(preset_name, preset_name_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "presetType",
                "presetName",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130583(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130583."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "presetType",
        "presetName",
    ))
    # firstPreset | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("firstPreset")

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
        raise ValueError("Cant encode this message, 'First preset' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'First preset' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'First preset' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # presetCount | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("presetCount")

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
        raise ValueError("Cant encode this message, 'Preset count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Preset count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Preset count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalPresetCount | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalPresetCount")

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
        raise ValueError("Cant encode this message, 'Total preset count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total preset count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total preset count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 24
    for repeating_entry in repeating_field_set_1_entries:
        # presetType | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("presetType")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Preset type'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_ENTERTAINMENT_EQ(field.value)
        field_bit_length = 8
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Preset type' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Preset type' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Preset type' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # presetName | Offset: 32, Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("presetName")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Preset name'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Preset name' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Preset name' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Preset name' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
