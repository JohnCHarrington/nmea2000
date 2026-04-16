# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128776() -> bool:
    """Return True if PGN 128776 is a fast PGN."""
    return False
def decode_pgn_128776(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128776."""
    nmea2000Message = NMEA2000Message(PGN=128776, id='windlassControlStatus', description='Windlass Control Status')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:windlass_id | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    windlass_id = windlass_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('windlassId', 'Windlass ID', None, None, windlass_id, windlass_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:windlass_direction_control | Offset: 16, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    windlass_direction_control_raw = decode_int(_data_raw_, running_bit_offset, 2)
    windlass_direction_control = master_dict['WINDLASS_DIRECTION'].get(windlass_direction_control_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('windlassDirectionControl', 'Windlass Direction Control', None, None, windlass_direction_control, windlass_direction_control_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:anchor_docking_control | Offset: 18, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 18
    anchor_docking_control_raw = decode_int(_data_raw_, running_bit_offset, 2)
    anchor_docking_control = master_dict['OFF_ON'].get(anchor_docking_control_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('anchorDockingControl', 'Anchor Docking Control', None, None, anchor_docking_control, anchor_docking_control_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:speed_control_type | Offset: 20, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    speed_control_type_raw = decode_int(_data_raw_, running_bit_offset, 2)
    speed_control_type = master_dict['SPEED_TYPE'].get(speed_control_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('speedControlType', 'Speed Control Type', None, None, speed_control_type, speed_control_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:reserved_22 | Offset: 22, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 22
    reserved_22 = reserved_22_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_22', 'Reserved', None, None, reserved_22, reserved_22_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 7:speed_control | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    speed_control = speed_control_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 8))
    nmea2000Message.fields.append(NMEA2000Field('speedControl', 'Speed Control', "0=Off,Single speed:1-100=On,Dual Speed:1-49=Slow/50-100=Fast,Proportional:10-100", None, speed_control, speed_control_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 8

    # 8:power_enable | Offset: 32, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    power_enable_raw = decode_int(_data_raw_, running_bit_offset, 2)
    power_enable = master_dict['OFF_ON'].get(power_enable_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('powerEnable', 'Power Enable', None, None, power_enable, power_enable_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 9:mechanical_lock | Offset: 34, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 34
    mechanical_lock_raw = decode_int(_data_raw_, running_bit_offset, 2)
    mechanical_lock = master_dict['OFF_ON'].get(mechanical_lock_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('mechanicalLock', 'Mechanical Lock', None, None, mechanical_lock, mechanical_lock_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:deck_and_anchor_wash | Offset: 36, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    deck_and_anchor_wash_raw = decode_int(_data_raw_, running_bit_offset, 2)
    deck_and_anchor_wash = master_dict['OFF_ON'].get(deck_and_anchor_wash_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('deckAndAnchorWash', 'Deck and Anchor Wash', None, None, deck_and_anchor_wash, deck_and_anchor_wash_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:anchor_light | Offset: 38, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 38
    anchor_light_raw = decode_int(_data_raw_, running_bit_offset, 2)
    anchor_light = master_dict['OFF_ON'].get(anchor_light_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('anchorLight', 'Anchor Light', None, None, anchor_light, anchor_light_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 12:command_timeout | Offset: 40, Length: 8, Signed: False Resolution: 0.005, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    command_timeout = command_timeout_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 0.005, 0, 1.26)
    nmea2000Message.fields.append(NMEA2000Field('commandTimeout', 'Command Timeout', "If timeout elapses the thruster stops operating and reverts to static mode", 's', command_timeout, command_timeout_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 8

    # 13:windlass_control_events | Offset: 48, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    windlass_control_events_raw = decode_int(_data_raw_, running_bit_offset, 4)
    windlass_control_events = decode_bit_lookup(windlass_control_events_raw, master_flags_dict['WINDLASS_CONTROL'])
    nmea2000Message.fields.append(NMEA2000Field('windlassControlEvents', 'Windlass Control Events', None, None, windlass_control_events, windlass_control_events_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 14:reserved_52 | Offset: 52, Length: 12, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 52
    reserved_52 = reserved_52_raw = decode_int(_data_raw_, running_bit_offset, 12)
    nmea2000Message.fields.append(NMEA2000Field('reserved_52', 'Reserved', None, None, reserved_52, reserved_52_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 12

    return nmea2000Message

def encode_pgn_128776(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128776."""
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
    # windlassId | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("windlassId")

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
        raise ValueError("Cant encode this message, 'Windlass ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Windlass ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Windlass ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # windlassDirectionControl | Offset: 16, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("windlassDirectionControl")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_WINDLASS_DIRECTION(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Windlass Direction Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Windlass Direction Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Windlass Direction Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # anchorDockingControl | Offset: 18, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 18
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("anchorDockingControl")

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
        raise ValueError("Cant encode this message, 'Anchor Docking Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Anchor Docking Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Anchor Docking Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speedControlType | Offset: 20, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedControlType")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SPEED_TYPE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Speed Control Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed Control Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed Control Type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_22 | Offset: 22, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 22
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_22")

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
    # speedControl | Offset: 24, Length: 8, Resolution: 1, Field Type: BINARY
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedControl")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
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
    # powerEnable | Offset: 32, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("powerEnable")

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
        raise ValueError("Cant encode this message, 'Power Enable' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Power Enable' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Power Enable' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mechanicalLock | Offset: 34, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 34
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mechanicalLock")

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
        raise ValueError("Cant encode this message, 'Mechanical Lock' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Mechanical Lock' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Mechanical Lock' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # deckAndAnchorWash | Offset: 36, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deckAndAnchorWash")

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
        raise ValueError("Cant encode this message, 'Deck and Anchor Wash' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Deck and Anchor Wash' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Deck and Anchor Wash' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # anchorLight | Offset: 38, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 38
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("anchorLight")

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
        raise ValueError("Cant encode this message, 'Anchor Light' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Anchor Light' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Anchor Light' exceeds the encoded bit length")
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
    # windlassControlEvents | Offset: 48, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("windlassControlEvents")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['WINDLASS_CONTROL'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Windlass Control Events' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Windlass Control Events' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Windlass Control Events' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_52 | Offset: 52, Length: 12, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 52
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_52")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 12
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
    return data_raw.to_bytes(8, byteorder="little")
