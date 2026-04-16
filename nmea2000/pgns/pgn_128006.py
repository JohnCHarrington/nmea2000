# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128006() -> bool:
    """Return True if PGN 128006 is a fast PGN."""
    return False
def decode_pgn_128006(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128006."""
    nmea2000Message = NMEA2000Message(PGN=128006, id='thrusterControlStatus', description='Thruster Control Status')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:identifier | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    identifier = identifier_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('identifier', 'Identifier', None, None, identifier, identifier_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:direction_control | Offset: 16, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    direction_control_raw = decode_int(_data_raw_, running_bit_offset, 4)
    direction_control = master_dict['THRUSTER_DIRECTION_CONTROL'].get(direction_control_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('directionControl', 'Direction Control', None, None, direction_control, direction_control_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:power_enabled | Offset: 20, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    power_enabled_raw = decode_int(_data_raw_, running_bit_offset, 2)
    power_enabled = master_dict['OFF_ON'].get(power_enabled_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('powerEnabled', 'Power Enabled', None, None, power_enabled, power_enabled_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:retract_control | Offset: 22, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 22
    retract_control_raw = decode_int(_data_raw_, running_bit_offset, 2)
    retract_control = master_dict['THRUSTER_RETRACT_CONTROL'].get(retract_control_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('retractControl', 'Retract Control', None, None, retract_control, retract_control_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:speed_control | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    speed_control = speed_control_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('speedControl', 'Speed Control', None, '%', speed_control, speed_control_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:control_events | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    control_events_raw = decode_int(_data_raw_, running_bit_offset, 8)
    control_events = decode_bit_lookup(control_events_raw, master_flags_dict['THRUSTER_CONTROL_EVENTS'])
    nmea2000Message.fields.append(NMEA2000Field('controlEvents', 'Control Events', None, None, control_events, control_events_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 8

    # 8:command_timeout | Offset: 40, Length: 8, Signed: False Resolution: 0.005, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    command_timeout = command_timeout_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 0.005, 0, 1.26)
    nmea2000Message.fields.append(NMEA2000Field('commandTimeout', 'Command Timeout', None, 's', command_timeout, command_timeout_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 8

    # 9:azimuth_control | Offset: 48, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    azimuth_control = azimuth_control_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('azimuthControl', 'Azimuth Control', None, 'rad', azimuth_control, azimuth_control_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_128006(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128006."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # sid | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sid")

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
        raise ValueError("Cant encode this message, 'SID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # identifier | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("identifier")

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
        raise ValueError("Cant encode this message, 'Identifier' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Identifier' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Identifier' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # directionControl | Offset: 16, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("directionControl")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_THRUSTER_DIRECTION_CONTROL(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Direction Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Direction Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Direction Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # powerEnabled | Offset: 20, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("powerEnabled")

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
        raise ValueError("Cant encode this message, 'Power Enabled' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Power Enabled' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Power Enabled' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # retractControl | Offset: 22, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 22
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("retractControl")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_THRUSTER_RETRACT_CONTROL(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Retract Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Retract Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Retract Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speedControl | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedControl")

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
        raise ValueError("Cant encode this message, 'Speed Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # controlEvents | Offset: 32, Length: 8, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("controlEvents")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['THRUSTER_CONTROL_EVENTS'])
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Control Events' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Control Events' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Control Events' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # commandTimeout | Offset: 40, Length: 8, Resolution: 0.005, Field Type: DURATION
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("commandTimeout")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.005)):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 0.005)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 8, False, 0.005)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 8)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Command Timeout' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Command Timeout' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Command Timeout' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # azimuthControl | Offset: 48, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("azimuthControl")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Azimuth Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Azimuth Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Azimuth Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
