# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127507() -> bool:
    """Return True if PGN 127507 is a fast PGN."""
    return True
def decode_pgn_127507(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127507."""
    nmea2000Message = NMEA2000Message(PGN=127507, id='chargerStatus', description='Charger Status', ttl=timedelta(milliseconds=1500))
    running_bit_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance = instance_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 2:battery_instance | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 8
    battery_instance = battery_instance_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('batteryInstance', 'Battery Instance', None, None, battery_instance, battery_instance_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 3:operating_state | Offset: 16, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    operating_state_raw = decode_int(_data_raw_, running_bit_offset, 4)
    operating_state = master_dict['CHARGER_STATE'].get(operating_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('operatingState', 'Operating State', None, None, operating_state, operating_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:charge_mode | Offset: 20, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    charge_mode_raw = decode_int(_data_raw_, running_bit_offset, 4)
    charge_mode = master_dict['CHARGER_MODE'].get(charge_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('chargeMode', 'Charge Mode', None, None, charge_mode, charge_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 5:enabled | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    enabled_raw = decode_int(_data_raw_, running_bit_offset, 2)
    enabled = master_dict['OFF_ON'].get(enabled_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('enabled', 'Enabled', None, None, enabled, enabled_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:equalization_pending | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    equalization_pending_raw = decode_int(_data_raw_, running_bit_offset, 2)
    equalization_pending = master_dict['OFF_ON'].get(equalization_pending_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('equalizationPending', 'Equalization Pending', None, None, equalization_pending, equalization_pending_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 7:reserved_28 | Offset: 28, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    reserved_28 = reserved_28_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_28', 'Reserved', None, None, reserved_28, reserved_28_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 8:equalization_time_remaining | Offset: 32, Length: 16, Signed: False Resolution: 60, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    equalization_time_remaining = equalization_time_remaining_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 60, 0, 3931920)
    nmea2000Message.fields.append(NMEA2000Field('equalizationTimeRemaining', 'Equalization Time Remaining', None, 's', equalization_time_remaining, equalization_time_remaining_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_127507(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127507."""
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
    # batteryInstance | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("batteryInstance")

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
        raise ValueError("Cant encode this message, 'Battery Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Battery Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Battery Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # operatingState | Offset: 16, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("operatingState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CHARGER_STATE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Operating State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Operating State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Operating State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # chargeMode | Offset: 20, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("chargeMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CHARGER_MODE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Charge Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Charge Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Charge Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # enabled | Offset: 24, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("enabled")

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
        raise ValueError("Cant encode this message, 'Enabled' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Enabled' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Enabled' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # equalizationPending | Offset: 26, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("equalizationPending")

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
        raise ValueError("Cant encode this message, 'Equalization Pending' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Equalization Pending' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Equalization Pending' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_28 | Offset: 28, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_28")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 4
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
    # equalizationTimeRemaining | Offset: 32, Length: 16, Resolution: 60, Field Type: DURATION
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("equalizationTimeRemaining")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 60)):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 60)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 16, False, 60)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Equalization Time Remaining' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Equalization Time Remaining' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Equalization Time Remaining' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")
