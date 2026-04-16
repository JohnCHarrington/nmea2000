# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130565() -> bool:
    """Return True if PGN 130565 is a fast PGN."""
    return True
def decode_pgn_130565(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130565."""
    nmea2000Message = NMEA2000Message(PGN=130565, id='lightingColorSequence', description='Lighting Color Sequence')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:sequence_index | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sequence_index = sequence_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sequenceIndex', 'Sequence Index', None, None, sequence_index, sequence_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:color_count | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    color_count = color_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('colorCount', 'Color Count', None, None, color_count, color_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:color_index | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    _repeating_field_set_1_offset = running_bit_offset
    color_index = color_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('colorIndex', 'Color Index', None, None, color_index, color_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:red_component | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    red_component = red_component_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('redComponent', 'Red Component', None, None, red_component, red_component_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:green_component | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    green_component = green_component_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('greenComponent', 'Green Component', None, None, green_component, green_component_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:blue_component | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    blue_component = blue_component_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('blueComponent', 'Blue Component', None, None, blue_component, blue_component_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:color_temperature | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    color_temperature = color_temperature_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('colorTemperature', 'Color Temperature', None, None, color_temperature, color_temperature_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:intensity | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    intensity = intensity_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('intensity', 'Intensity', None, None, intensity, intensity_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(color_count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        color_index = color_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["colorIndex"] = _repeating_entry_value(color_index, color_index_raw)
    
        red_component = red_component_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["redComponent"] = _repeating_entry_value(red_component, red_component_raw)
    
        green_component = green_component_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["greenComponent"] = _repeating_entry_value(green_component, green_component_raw)
    
        blue_component = blue_component_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["blueComponent"] = _repeating_entry_value(blue_component, blue_component_raw)
    
        color_temperature = color_temperature_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        running_bit_offset += 16
        repeating_entry["colorTemperature"] = _repeating_entry_value(color_temperature, color_temperature_raw)
    
        intensity = intensity_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["intensity"] = _repeating_entry_value(intensity, intensity_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "colorIndex",
                "redComponent",
                "greenComponent",
                "blueComponent",
                "colorTemperature",
                "intensity",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130565(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130565."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "colorIndex",
        "redComponent",
        "greenComponent",
        "blueComponent",
        "colorTemperature",
        "intensity",
    ))
    # sequenceIndex | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sequenceIndex")

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
        raise ValueError("Cant encode this message, 'Sequence Index' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sequence Index' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sequence Index' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # colorCount | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("colorCount")

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
        raise ValueError("Cant encode this message, 'Color Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Color Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Color Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 16
    for repeating_entry in repeating_field_set_1_entries:
        # colorIndex | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("colorIndex")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Color Index'")
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
            raise ValueError("Cant encode this message, 'Color Index' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Color Index' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Color Index' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # redComponent | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("redComponent")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Red Component'")
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
            raise ValueError("Cant encode this message, 'Red Component' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Red Component' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Red Component' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # greenComponent | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("greenComponent")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Green Component'")
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
            raise ValueError("Cant encode this message, 'Green Component' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Green Component' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Green Component' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # blueComponent | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("blueComponent")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Blue Component'")
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
            raise ValueError("Cant encode this message, 'Blue Component' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Blue Component' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Blue Component' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # colorTemperature | Offset: 48, Length: 16, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("colorTemperature")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Color Temperature'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Color Temperature' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Color Temperature' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Color Temperature' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # intensity | Offset: 64, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("intensity")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Intensity'")
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
            raise ValueError("Cant encode this message, 'Intensity' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Intensity' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Intensity' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
