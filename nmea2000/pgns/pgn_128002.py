# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128002() -> bool:
    """Return True if PGN 128002 is a fast PGN."""
    return False
def decode_pgn_128002(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128002."""
    nmea2000Message = NMEA2000Message(PGN=128002, id='electricDriveStatusRapidUpdate', description='Electric Drive Status, Rapid Update')
    running_bit_offset = 0
    # 1:inverter_motor_controller | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    inverter_motor_controller = inverter_motor_controller_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('inverterMotorController', 'Inverter/Motor Controller', None, None, inverter_motor_controller, inverter_motor_controller_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:active_motor_mode | Offset: 8, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    active_motor_mode = active_motor_mode_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('activeMotorMode', 'Active Motor Mode', None, None, active_motor_mode, active_motor_mode_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 3:brake_mode | Offset: 10, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 10
    brake_mode = brake_mode_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('brakeMode', 'Brake Mode', None, None, brake_mode, brake_mode_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 4:reserved_12 | Offset: 12, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    reserved_12 = reserved_12_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_12', 'Reserved', None, None, reserved_12, reserved_12_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 5:rotational_shaft_speed | Offset: 16, Length: 16, Signed: False Resolution: 0.25, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    rotational_shaft_speed = rotational_shaft_speed_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.25, 0, 16383)
    nmea2000Message.fields.append(NMEA2000Field('rotationalShaftSpeed', 'Rotational Shaft Speed', None, 'rpm', rotational_shaft_speed, rotational_shaft_speed_raw, PhysicalQuantities.ANGULAR_VELOCITY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:motor_dc_voltage | Offset: 32, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    motor_dc_voltage = motor_dc_voltage_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('motorDcVoltage', 'Motor DC Voltage', None, 'V', motor_dc_voltage, motor_dc_voltage_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:motor_dc_current | Offset: 48, Length: 16, Signed: True Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    motor_dc_current = motor_dc_current_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.1, -3276.7, 3276.4)
    nmea2000Message.fields.append(NMEA2000Field('motorDcCurrent', 'Motor DC Current', None, 'A', motor_dc_current, motor_dc_current_raw, PhysicalQuantities.ELECTRICAL_CURRENT, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_128002(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128002."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # inverterMotorController | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("inverterMotorController")

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
        raise ValueError("Cant encode this message, 'Inverter/Motor Controller' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Inverter/Motor Controller' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Inverter/Motor Controller' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # activeMotorMode | Offset: 8, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("activeMotorMode")

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
        raise ValueError("Cant encode this message, 'Active Motor Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Active Motor Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Active Motor Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # brakeMode | Offset: 10, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 10
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("brakeMode")

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
        raise ValueError("Cant encode this message, 'Brake Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Brake Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Brake Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_12 | Offset: 12, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 12
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_12")

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
    # rotationalShaftSpeed | Offset: 16, Length: 16, Resolution: 0.25, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rotationalShaftSpeed")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.25):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.25)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.25)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rotational Shaft Speed' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rotational Shaft Speed' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rotational Shaft Speed' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorDcVoltage | Offset: 32, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorDcVoltage")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Motor DC Voltage' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor DC Voltage' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor DC Voltage' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorDcCurrent | Offset: 48, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorDcCurrent")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.1):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Motor DC Current' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor DC Current' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor DC Current' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
