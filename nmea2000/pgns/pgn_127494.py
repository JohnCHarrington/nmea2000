# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127494() -> bool:
    """Return True if PGN 127494 is a fast PGN."""
    return True
def decode_pgn_127494(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127494."""
    nmea2000Message = NMEA2000Message(PGN=127494, id='electricDriveInformation', description='Electric Drive Information')
    running_bit_offset = 0
    # 1:inverter_motor_identifier | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    inverter_motor_identifier = inverter_motor_identifier_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('inverterMotorIdentifier', 'Inverter/Motor Identifier', None, None, inverter_motor_identifier, inverter_motor_identifier_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 2:motor_type | Offset: 8, Length: 4, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    motor_type = motor_type_raw = decode_number(_data_raw_, running_bit_offset, 4, False, 1, 0, 13)
    nmea2000Message.fields.append(NMEA2000Field('motorType', 'Motor Type', None, None, motor_type, motor_type_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 4

    # 3:reserved_12 | Offset: 12, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    reserved_12 = reserved_12_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_12', 'Reserved', None, None, reserved_12, reserved_12_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 4:motor_voltage_rating | Offset: 16, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    motor_voltage_rating = motor_voltage_rating_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('motorVoltageRating', 'Motor Voltage Rating', None, 'V', motor_voltage_rating, motor_voltage_rating_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:maximum_continuous_motor_power | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    maximum_continuous_motor_power = maximum_continuous_motor_power_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('maximumContinuousMotorPower', 'Maximum Continuous Motor Power', None, 'W', maximum_continuous_motor_power, maximum_continuous_motor_power_raw, PhysicalQuantities.ELECTRICAL_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 6:maximum_boost_motor_power | Offset: 64, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    maximum_boost_motor_power = maximum_boost_motor_power_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('maximumBoostMotorPower', 'Maximum Boost Motor Power', None, 'W', maximum_boost_motor_power, maximum_boost_motor_power_raw, PhysicalQuantities.ELECTRICAL_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 7:maximum_motor_temperature_rating | Offset: 96, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    maximum_motor_temperature_rating = maximum_motor_temperature_rating_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('maximumMotorTemperatureRating', 'Maximum Motor Temperature Rating', None, 'K', maximum_motor_temperature_rating, maximum_motor_temperature_rating_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:rated_motor_speed | Offset: 112, Length: 16, Signed: False Resolution: 0.25, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    rated_motor_speed = rated_motor_speed_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.25, 0, 16383)
    nmea2000Message.fields.append(NMEA2000Field('ratedMotorSpeed', 'Rated Motor Speed', None, 'rpm', rated_motor_speed, rated_motor_speed_raw, PhysicalQuantities.ANGULAR_VELOCITY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:maximum_controller_temperature_rating | Offset: 128, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    maximum_controller_temperature_rating = maximum_controller_temperature_rating_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('maximumControllerTemperatureRating', 'Maximum Controller Temperature Rating', None, 'K', maximum_controller_temperature_rating, maximum_controller_temperature_rating_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 10:motor_shaft_torque_rating | Offset: 144, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    motor_shaft_torque_rating = motor_shaft_torque_rating_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('motorShaftTorqueRating', 'Motor Shaft Torque Rating', None, None, motor_shaft_torque_rating, motor_shaft_torque_rating_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 11:motor_dc_voltage_derating_threshold | Offset: 160, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    motor_dc_voltage_derating_threshold = motor_dc_voltage_derating_threshold_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('motorDcVoltageDeratingThreshold', 'Motor DC-Voltage Derating Threshold', None, 'V', motor_dc_voltage_derating_threshold, motor_dc_voltage_derating_threshold_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 12:motor_dc_voltage_cut_off_threshold | Offset: 176, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 176
    motor_dc_voltage_cut_off_threshold = motor_dc_voltage_cut_off_threshold_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('motorDcVoltageCutOffThreshold', 'Motor DC-Voltage Cut Off Threshold', None, 'V', motor_dc_voltage_cut_off_threshold, motor_dc_voltage_cut_off_threshold_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 13:drive_motor_hours | Offset: 192, Length: 32, Signed: False Resolution: 1, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 192
    drive_motor_hours = drive_motor_hours_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('driveMotorHours', 'Drive/Motor Hours', None, 's', drive_motor_hours, drive_motor_hours_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_127494(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127494."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # inverterMotorIdentifier | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("inverterMotorIdentifier")

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
        raise ValueError("Cant encode this message, 'Inverter/Motor Identifier' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Inverter/Motor Identifier' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Inverter/Motor Identifier' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorType | Offset: 8, Length: 4, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorType")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 4, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 4, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 4, False, 1)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Motor Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor Type' exceeds the encoded bit length")
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
    # motorVoltageRating | Offset: 16, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorVoltageRating")

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
        raise ValueError("Cant encode this message, 'Motor Voltage Rating' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor Voltage Rating' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor Voltage Rating' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maximumContinuousMotorPower | Offset: 32, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumContinuousMotorPower")

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
        raise ValueError("Cant encode this message, 'Maximum Continuous Motor Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Continuous Motor Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Continuous Motor Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maximumBoostMotorPower | Offset: 64, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumBoostMotorPower")

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
        raise ValueError("Cant encode this message, 'Maximum Boost Motor Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Boost Motor Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Boost Motor Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maximumMotorTemperatureRating | Offset: 96, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumMotorTemperatureRating")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Maximum Motor Temperature Rating' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Motor Temperature Rating' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Motor Temperature Rating' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # ratedMotorSpeed | Offset: 112, Length: 16, Resolution: 0.25, Field Type: NUMBER
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("ratedMotorSpeed")

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
        raise ValueError("Cant encode this message, 'Rated Motor Speed' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rated Motor Speed' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rated Motor Speed' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maximumControllerTemperatureRating | Offset: 128, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumControllerTemperatureRating")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Maximum Controller Temperature Rating' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Controller Temperature Rating' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Controller Temperature Rating' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorShaftTorqueRating | Offset: 144, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorShaftTorqueRating")

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
        raise ValueError("Cant encode this message, 'Motor Shaft Torque Rating' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor Shaft Torque Rating' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor Shaft Torque Rating' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorDcVoltageDeratingThreshold | Offset: 160, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorDcVoltageDeratingThreshold")

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
        raise ValueError("Cant encode this message, 'Motor DC-Voltage Derating Threshold' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor DC-Voltage Derating Threshold' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor DC-Voltage Derating Threshold' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # motorDcVoltageCutOffThreshold | Offset: 176, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 176
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("motorDcVoltageCutOffThreshold")

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
        raise ValueError("Cant encode this message, 'Motor DC-Voltage Cut Off Threshold' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Motor DC-Voltage Cut Off Threshold' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Motor DC-Voltage Cut Off Threshold' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # driveMotorHours | Offset: 192, Length: 32, Resolution: 1, Field Type: DURATION
    running_bit_offset = 192
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("driveMotorHours")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 1)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 1)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 1)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Drive/Motor Hours' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Drive/Motor Hours' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Drive/Motor Hours' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(28, byteorder="little")
