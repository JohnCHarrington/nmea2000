# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130562() -> bool:
    """Return True if PGN 130562 is a fast PGN."""
    return True
def decode_pgn_130562(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130562."""
    nmea2000Message = NMEA2000Message(PGN=130562, id='lightingScene', description='Lighting Scene')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:scene_index | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    scene_index = scene_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sceneIndex', 'Scene Index', None, None, scene_index, scene_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:zone_name | Offset: 8, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    zone_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    zone_name = zone_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('zoneName', 'Zone Name', None, None, zone_name, zone_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 3:control | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    control = control_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('control', 'Control', None, None, control, control_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:configuration_count | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    configuration_count = configuration_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('configurationCount', 'Configuration Count', None, None, configuration_count, configuration_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:configuration_index | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    configuration_index = configuration_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('configurationIndex', 'Configuration Index', None, None, configuration_index, configuration_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:zone_index | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    zone_index = zone_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zoneIndex', 'Zone Index', None, None, zone_index, zone_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:devices_id | Offset: , Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    devices_id = devices_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('devicesId', 'Devices ID', None, None, devices_id, devices_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 8:program_index | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    program_index = program_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('programIndex', 'Program Index', None, None, program_index, program_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:program_color_sequence_index | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    program_color_sequence_index = program_color_sequence_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('programColorSequenceIndex', 'Program Color Sequence Index', None, None, program_color_sequence_index, program_color_sequence_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 10:program_intensity | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    program_intensity = program_intensity_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('programIntensity', 'Program Intensity', None, None, program_intensity, program_intensity_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 11:program_rate | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    program_rate = program_rate_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('programRate', 'Program Rate', None, None, program_rate, program_rate_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 12:program_color_sequence_rate | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    program_color_sequence_rate = program_color_sequence_rate_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('programColorSequenceRate', 'Program Color Sequence Rate', None, None, program_color_sequence_rate, program_color_sequence_rate_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(configuration_count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        configuration_index = configuration_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["configurationIndex"] = _repeating_entry_value(configuration_index, configuration_index_raw)
    
        zone_index = zone_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["zoneIndex"] = _repeating_entry_value(zone_index, zone_index_raw)
    
        devices_id = devices_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["devicesId"] = _repeating_entry_value(devices_id, devices_id_raw)
    
        program_index = program_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["programIndex"] = _repeating_entry_value(program_index, program_index_raw)
    
        program_color_sequence_index = program_color_sequence_index_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["programColorSequenceIndex"] = _repeating_entry_value(program_color_sequence_index, program_color_sequence_index_raw)
    
        program_intensity = program_intensity_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["programIntensity"] = _repeating_entry_value(program_intensity, program_intensity_raw)
    
        program_rate = program_rate_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["programRate"] = _repeating_entry_value(program_rate, program_rate_raw)
    
        program_color_sequence_rate = program_color_sequence_rate_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["programColorSequenceRate"] = _repeating_entry_value(program_color_sequence_rate, program_color_sequence_rate_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "configurationIndex",
                "zoneIndex",
                "devicesId",
                "programIndex",
                "programColorSequenceIndex",
                "programIntensity",
                "programRate",
                "programColorSequenceRate",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130562(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130562."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "configurationIndex",
        "zoneIndex",
        "devicesId",
        "programIndex",
        "programColorSequenceIndex",
        "programIntensity",
        "programRate",
        "programColorSequenceRate",
    ))
    # sceneIndex | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sceneIndex")

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
        raise ValueError("Cant encode this message, 'Scene Index' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Scene Index' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Scene Index' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zoneName | Offset: 8, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zoneName")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Zone Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # control | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("control")

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
        raise ValueError("Cant encode this message, 'Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # configurationCount | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("configurationCount")

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
        raise ValueError("Cant encode this message, 'Configuration Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Configuration Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Configuration Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # configurationIndex | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("configurationIndex")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Configuration Index'")
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
            raise ValueError("Cant encode this message, 'Configuration Index' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Configuration Index' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Configuration Index' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # zoneIndex | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("zoneIndex")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Zone Index'")
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
            raise ValueError("Cant encode this message, 'Zone Index' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Zone Index' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Zone Index' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # devicesId | Offset: , Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("devicesId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Devices ID'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
            field_value = encode_number_raw(field.raw_value, 32, False)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 32, False, 1)
        else:
            assert field.value is None or isinstance(field.value, (int, float))
            field_value = encode_number(field.value, 32, False, 1)
        field_bit_length = 32
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Devices ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Devices ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Devices ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # programIndex | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("programIndex")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Program Index'")
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
            raise ValueError("Cant encode this message, 'Program Index' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Program Index' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Program Index' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # programColorSequenceIndex | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("programColorSequenceIndex")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Program Color Sequence Index'")
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
            raise ValueError("Cant encode this message, 'Program Color Sequence Index' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Program Color Sequence Index' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Program Color Sequence Index' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # programIntensity | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("programIntensity")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Program Intensity'")
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
            raise ValueError("Cant encode this message, 'Program Intensity' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Program Intensity' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Program Intensity' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # programRate | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("programRate")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Program Rate'")
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
            raise ValueError("Cant encode this message, 'Program Rate' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Program Rate' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Program Rate' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # programColorSequenceRate | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("programColorSequenceRate")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Program Color Sequence Rate'")
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
            raise ValueError("Cant encode this message, 'Program Color Sequence Rate' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Program Color Sequence Rate' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Program Color Sequence Rate' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
