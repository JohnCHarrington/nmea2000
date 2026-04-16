# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130586() -> bool:
    """Return True if PGN 130586 is a fast PGN."""
    return True
def decode_pgn_130586(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130586."""
    nmea2000Message = NMEA2000Message(PGN=130586, id='zoneConfiguration', description='Zone Configuration')
    running_bit_offset = 0
    # 1:zone_id | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    zone_id_raw = decode_int(_data_raw_, running_bit_offset, 8)
    zone_id = master_dict['ENTERTAINMENT_ZONE'].get(zone_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('zoneId', 'Zone ID', None, None, zone_id, zone_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:volume_limit | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    volume_limit = volume_limit_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('volumeLimit', 'Volume limit', None, '%', volume_limit, volume_limit_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:fade | Offset: 16, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    fade = fade_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('fade', 'Fade', None, '%', fade, fade_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:balance | Offset: 24, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    balance = balance_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('balance', 'Balance', None, '%', balance, balance_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:sub_volume | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    sub_volume = sub_volume_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('subVolume', 'Sub volume', None, '%', sub_volume, sub_volume_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:eq___treble | Offset: 40, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    eq___treble = eq___treble_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('eqTreble', 'EQ - Treble', None, '%', eq___treble, eq___treble_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:eq___mid_range | Offset: 48, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    eq___mid_range = eq___mid_range_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('eqMidRange', 'EQ - Mid range', None, '%', eq___mid_range, eq___mid_range_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:eq___bass | Offset: 56, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    eq___bass = eq___bass_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('eqBass', 'EQ - Bass', None, '%', eq___bass, eq___bass_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:preset_type | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    preset_type_raw = decode_int(_data_raw_, running_bit_offset, 8)
    preset_type = master_dict['ENTERTAINMENT_EQ'].get(preset_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('presetType', 'Preset type', None, None, preset_type, preset_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 10:audio_filter | Offset: 72, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    audio_filter_raw = decode_int(_data_raw_, running_bit_offset, 8)
    audio_filter = master_dict['ENTERTAINMENT_FILTER'].get(audio_filter_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('audioFilter', 'Audio filter', None, None, audio_filter, audio_filter_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 11:high_pass_filter_frequency | Offset: 80, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    high_pass_filter_frequency = high_pass_filter_frequency_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('highPassFilterFrequency', 'High pass filter frequency', None, 'Hz', high_pass_filter_frequency, high_pass_filter_frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 12:low_pass_filter_frequency | Offset: 96, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    low_pass_filter_frequency = low_pass_filter_frequency_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('lowPassFilterFrequency', 'Low pass filter frequency', None, 'Hz', low_pass_filter_frequency, low_pass_filter_frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 13:channel | Offset: 112, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    channel_raw = decode_int(_data_raw_, running_bit_offset, 8)
    channel = master_dict['ENTERTAINMENT_CHANNEL'].get(channel_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130586(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130586."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # zoneId | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zoneId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_ZONE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Zone ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # volumeLimit | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("volumeLimit")

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
        raise ValueError("Cant encode this message, 'Volume limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Volume limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Volume limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # fade | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("fade")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Fade' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Fade' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Fade' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # balance | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("balance")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Balance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Balance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Balance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # subVolume | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("subVolume")

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
        raise ValueError("Cant encode this message, 'Sub volume' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sub volume' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sub volume' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # eqTreble | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("eqTreble")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'EQ - Treble' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'EQ - Treble' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'EQ - Treble' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # eqMidRange | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("eqMidRange")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'EQ - Mid range' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'EQ - Mid range' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'EQ - Mid range' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # eqBass | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("eqBass")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'EQ - Bass' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'EQ - Bass' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'EQ - Bass' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # presetType | Offset: 64, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("presetType")

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
    # audioFilter | Offset: 72, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("audioFilter")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_FILTER(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Audio filter' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Audio filter' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Audio filter' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # highPassFilterFrequency | Offset: 80, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("highPassFilterFrequency")

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
        raise ValueError("Cant encode this message, 'High pass filter frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'High pass filter frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'High pass filter frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # lowPassFilterFrequency | Offset: 96, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("lowPassFilterFrequency")

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
        raise ValueError("Cant encode this message, 'Low pass filter frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Low pass filter frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Low pass filter frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 112, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_CHANNEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(15, byteorder="little")
