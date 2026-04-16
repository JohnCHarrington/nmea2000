# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128778() -> bool:
    """Return True if PGN 128778 is a fast PGN."""
    return False
def decode_pgn_128778(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128778."""
    nmea2000Message = NMEA2000Message(PGN=128778, id='anchorWindlassMonitoringStatus', description='Anchor Windlass Monitoring Status')
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

    # 3:windlass_monitoring_events | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    windlass_monitoring_events_raw = decode_int(_data_raw_, running_bit_offset, 8)
    windlass_monitoring_events = decode_bit_lookup(windlass_monitoring_events_raw, master_flags_dict['WINDLASS_MONITORING'])
    nmea2000Message.fields.append(NMEA2000Field('windlassMonitoringEvents', 'Windlass Monitoring Events', None, None, windlass_monitoring_events, windlass_monitoring_events_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 8

    # 4:controller_voltage | Offset: 24, Length: 8, Signed: False Resolution: 0.2, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    controller_voltage = controller_voltage_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 0.2, 0, 50.4)
    nmea2000Message.fields.append(NMEA2000Field('controllerVoltage', 'Controller voltage', None, 'V', controller_voltage, controller_voltage_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:motor_current | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    motor_current = motor_current_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('motorCurrent', 'Motor current', None, 'A', motor_current, motor_current_raw, PhysicalQuantities.ELECTRICAL_CURRENT, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:total_motor_time | Offset: 40, Length: 16, Signed: False Resolution: 60, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    total_motor_time = total_motor_time_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 60, 0, 3931920)
    nmea2000Message.fields.append(NMEA2000Field('totalMotorTime', 'Total Motor Time', None, 's', total_motor_time, total_motor_time_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 7:reserved_56 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    reserved_56 = reserved_56_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_56', 'Reserved', None, None, reserved_56, reserved_56_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_128778(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128778."""
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
    # windlassMonitoringEvents | Offset: 16, Length: 8, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("windlassMonitoringEvents")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['WINDLASS_MONITORING'])
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Windlass Monitoring Events' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Windlass Monitoring Events' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Windlass Monitoring Events' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # controllerVoltage | Offset: 24, Length: 8, Resolution: 0.2, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("controllerVoltage")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.2):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 0.2)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, False, 0.2)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Controller voltage' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Controller voltage' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Controller voltage' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorCurrent | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorCurrent")

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
        raise ValueError("Cant encode this message, 'Motor current' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor current' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor current' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalMotorTime | Offset: 40, Length: 16, Resolution: 60, Field Type: DURATION
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalMotorTime")

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
        raise ValueError("Cant encode this message, 'Total Motor Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total Motor Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total Motor Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_56 | Offset: 56, Length: 8, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_56")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 8
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
