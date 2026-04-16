# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130330() -> bool:
    """Return True if PGN 130330 is a fast PGN."""
    return True
def decode_pgn_130330(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130330."""
    nmea2000Message = NMEA2000Message(PGN=130330, id='lightingSystemSettings', description='Lighting System Settings')
    running_bit_offset = 0
    # 1:global_enable | Offset: 0, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    global_enable = global_enable_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('globalEnable', 'Global Enable', None, None, global_enable, global_enable_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 2:default_settings_command | Offset: 2, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 2
    default_settings_command_raw = decode_int(_data_raw_, running_bit_offset, 3)
    default_settings_command = master_dict['LIGHTING_COMMAND'].get(default_settings_command_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('defaultSettingsCommand', 'Default Settings/Command', None, None, default_settings_command, default_settings_command_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 3:reserved_5 | Offset: 5, Length: 3, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 5
    reserved_5 = reserved_5_raw = decode_int(_data_raw_, running_bit_offset, 3)
    nmea2000Message.fields.append(NMEA2000Field('reserved_5', 'Reserved', None, None, reserved_5, reserved_5_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 3

    # 4:name_of_the_lighting_controller | Offset: 8, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    name_of_the_lighting_controller_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    name_of_the_lighting_controller = name_of_the_lighting_controller_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('nameOfTheLightingController', 'Name of the lighting controller', None, None, name_of_the_lighting_controller, name_of_the_lighting_controller_raw, None, FieldTypes.STRING_LAU, False))
    

    # 5:max_scenes | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    max_scenes = max_scenes_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxScenes', 'Max Scenes', None, None, max_scenes, max_scenes_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:max_scene_configuration_count | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    max_scene_configuration_count = max_scene_configuration_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxSceneConfigurationCount', 'Max Scene Configuration Count', None, None, max_scene_configuration_count, max_scene_configuration_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:max_zones | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    max_zones = max_zones_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxZones', 'Max Zones', None, None, max_zones, max_zones_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:max_color_sequences | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    max_color_sequences = max_color_sequences_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxColorSequences', 'Max Color Sequences', None, None, max_color_sequences, max_color_sequences_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:max_color_sequence_color_count | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    max_color_sequence_color_count = max_color_sequence_color_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maxColorSequenceColorCount', 'Max Color Sequence Color Count', None, None, max_color_sequence_color_count, max_color_sequence_color_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 10:number_of_programs | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_programs = number_of_programs_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfPrograms', 'Number of Programs', None, None, number_of_programs, number_of_programs_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 11:controller_capabilities | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    controller_capabilities = controller_capabilities_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('controllerCapabilities', 'Controller Capabilities', None, None, controller_capabilities, controller_capabilities_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 12:identify_device | Offset: , Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    identify_device = identify_device_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('identifyDevice', 'Identify Device', None, None, identify_device, identify_device_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130330(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130330."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # globalEnable | Offset: 0, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("globalEnable")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 2, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 2, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 2, False, 1)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Global Enable' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Global Enable' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Global Enable' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # defaultSettingsCommand | Offset: 2, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 2
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("defaultSettingsCommand")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_LIGHTING_COMMAND(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Default Settings/Command' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Default Settings/Command' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Default Settings/Command' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_5 | Offset: 5, Length: 3, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 5
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_5")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 3
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
    # nameOfTheLightingController | Offset: 8, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("nameOfTheLightingController")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name of the lighting controller' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name of the lighting controller' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name of the lighting controller' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxScenes | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxScenes")

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
        raise ValueError("Cant encode this message, 'Max Scenes' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max Scenes' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max Scenes' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxSceneConfigurationCount | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxSceneConfigurationCount")

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
        raise ValueError("Cant encode this message, 'Max Scene Configuration Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max Scene Configuration Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max Scene Configuration Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxZones | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxZones")

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
        raise ValueError("Cant encode this message, 'Max Zones' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max Zones' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max Zones' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxColorSequences | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxColorSequences")

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
        raise ValueError("Cant encode this message, 'Max Color Sequences' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max Color Sequences' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max Color Sequences' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxColorSequenceColorCount | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxColorSequenceColorCount")

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
        raise ValueError("Cant encode this message, 'Max Color Sequence Color Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max Color Sequence Color Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max Color Sequence Color Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfPrograms | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfPrograms")

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
        raise ValueError("Cant encode this message, 'Number of Programs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Programs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Programs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # controllerCapabilities | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("controllerCapabilities")

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
        raise ValueError("Cant encode this message, 'Controller Capabilities' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Controller Capabilities' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Controller Capabilities' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # identifyDevice | Offset: , Length: 32, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("identifyDevice")

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
        raise ValueError("Cant encode this message, 'Identify Device' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Identify Device' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Identify Device' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
