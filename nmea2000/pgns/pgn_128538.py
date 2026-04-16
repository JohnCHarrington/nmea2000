# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_128538() -> bool:
    """Return True if PGN 128538 is a fast PGN."""
    return True
def decode_pgn_128538(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 128538."""
    nmea2000Message = NMEA2000Message(PGN=128538, id='elevatorCarStatus', description='Elevator Car Status')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:elevator_car_id | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    elevator_car_id = elevator_car_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('elevatorCarId', 'Elevator Car ID', None, None, elevator_car_id, elevator_car_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:elevator_car_usage | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    elevator_car_usage = elevator_car_usage_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('elevatorCarUsage', 'Elevator Car Usage', None, None, elevator_car_usage, elevator_car_usage_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:smoke_sensor_status | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    smoke_sensor_status = smoke_sensor_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('smokeSensorStatus', 'Smoke Sensor Status', None, None, smoke_sensor_status, smoke_sensor_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 5:limit_switch_sensor_status | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    limit_switch_sensor_status = limit_switch_sensor_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('limitSwitchSensorStatus', 'Limit Switch Sensor Status', None, None, limit_switch_sensor_status, limit_switch_sensor_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 6:proximity_switch_sensor_status | Offset: 28, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    proximity_switch_sensor_status = proximity_switch_sensor_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('proximitySwitchSensorStatus', 'Proximity Switch Sensor Status', None, None, proximity_switch_sensor_status, proximity_switch_sensor_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 7:inertial_measurement_unit__imu__sensor_status | Offset: 30, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 30
    inertial_measurement_unit__imu__sensor_status = inertial_measurement_unit__imu__sensor_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('inertialMeasurementUnitImuSensorStatus', 'Inertial Measurement Unit (IMU) Sensor Status', None, None, inertial_measurement_unit__imu__sensor_status, inertial_measurement_unit__imu__sensor_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 8:elevator_load_limit_status | Offset: 32, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    elevator_load_limit_status = elevator_load_limit_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorLoadLimitStatus', 'Elevator Load Limit Status', None, None, elevator_load_limit_status, elevator_load_limit_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 9:elevator_load_balance_status | Offset: 34, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 34
    elevator_load_balance_status = elevator_load_balance_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorLoadBalanceStatus', 'Elevator Load Balance Status', None, None, elevator_load_balance_status, elevator_load_balance_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 10:elevator_load_sensor_1_status | Offset: 36, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    elevator_load_sensor_1_status = elevator_load_sensor_1_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorLoadSensor1Status', 'Elevator Load Sensor 1 Status', None, None, elevator_load_sensor_1_status, elevator_load_sensor_1_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 11:elevator_load_sensor_2_status | Offset: 38, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 38
    elevator_load_sensor_2_status = elevator_load_sensor_2_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorLoadSensor2Status', 'Elevator Load Sensor 2 Status', None, None, elevator_load_sensor_2_status, elevator_load_sensor_2_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 12:elevator_load_sensor_3_status | Offset: 40, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    elevator_load_sensor_3_status = elevator_load_sensor_3_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorLoadSensor3Status', 'Elevator Load Sensor 3 Status', None, None, elevator_load_sensor_3_status, elevator_load_sensor_3_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 13:elevator_load_sensor_4_status | Offset: 42, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 42
    elevator_load_sensor_4_status = elevator_load_sensor_4_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorLoadSensor4Status', 'Elevator Load Sensor 4 Status', None, None, elevator_load_sensor_4_status, elevator_load_sensor_4_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 14:reserved_44 | Offset: 44, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 44
    reserved_44 = reserved_44_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_44', 'Reserved', None, None, reserved_44, reserved_44_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 15:elevator_car_motion_status | Offset: 48, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    elevator_car_motion_status = elevator_car_motion_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorCarMotionStatus', 'Elevator Car Motion Status', None, None, elevator_car_motion_status, elevator_car_motion_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 16:elevator_car_door_status | Offset: 50, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 50
    elevator_car_door_status = elevator_car_door_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorCarDoorStatus', 'Elevator Car Door Status', None, None, elevator_car_door_status, elevator_car_door_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 17:elevator_car_emergency_button_status | Offset: 52, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 52
    elevator_car_emergency_button_status = elevator_car_emergency_button_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorCarEmergencyButtonStatus', 'Elevator Car Emergency Button Status', None, None, elevator_car_emergency_button_status, elevator_car_emergency_button_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 18:elevator_car_buzzer_status | Offset: 54, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 54
    elevator_car_buzzer_status = elevator_car_buzzer_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorCarBuzzerStatus', 'Elevator Car Buzzer Status', None, None, elevator_car_buzzer_status, elevator_car_buzzer_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 19:open_door_button_status | Offset: 56, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    open_door_button_status = open_door_button_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('openDoorButtonStatus', 'Open Door Button Status', None, None, open_door_button_status, open_door_button_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 20:close_door_button_status | Offset: 58, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 58
    close_door_button_status = close_door_button_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('closeDoorButtonStatus', 'Close Door Button Status', None, None, close_door_button_status, close_door_button_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 21:reserved_60 | Offset: 60, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 60
    reserved_60 = reserved_60_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_60', 'Reserved', None, None, reserved_60, reserved_60_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 22:current_deck | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    current_deck = current_deck_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('currentDeck', 'Current Deck', None, None, current_deck, current_deck_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 23:destination_deck | Offset: 72, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    destination_deck = destination_deck_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('destinationDeck', 'Destination Deck', None, None, destination_deck, destination_deck_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 24:total_number_of_decks | Offset: 80, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    total_number_of_decks = total_number_of_decks_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('totalNumberOfDecks', 'Total Number of Decks', None, None, total_number_of_decks, total_number_of_decks_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 25:weight_of_load_cell_1 | Offset: 88, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    weight_of_load_cell_1 = weight_of_load_cell_1_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('weightOfLoadCell1', 'Weight of Load Cell 1', None, None, weight_of_load_cell_1, weight_of_load_cell_1_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 26:weight_of_load_cell_2 | Offset: 104, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 104
    weight_of_load_cell_2 = weight_of_load_cell_2_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('weightOfLoadCell2', 'Weight of Load Cell 2', None, None, weight_of_load_cell_2, weight_of_load_cell_2_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 27:weight_of_load_cell_3 | Offset: 120, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    weight_of_load_cell_3 = weight_of_load_cell_3_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('weightOfLoadCell3', 'Weight of Load Cell 3', None, None, weight_of_load_cell_3, weight_of_load_cell_3_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 28:weight_of_load_cell_4 | Offset: 136, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    weight_of_load_cell_4 = weight_of_load_cell_4_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('weightOfLoadCell4', 'Weight of Load Cell 4', None, None, weight_of_load_cell_4, weight_of_load_cell_4_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 29:speed_of_elevator_car | Offset: 152, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    speed_of_elevator_car = speed_of_elevator_car_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('speedOfElevatorCar', 'Speed of Elevator Car', None, 'm/s', speed_of_elevator_car, speed_of_elevator_car_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 30:elevator_brake_status | Offset: 168, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 168
    elevator_brake_status = elevator_brake_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorBrakeStatus', 'Elevator Brake Status', None, None, elevator_brake_status, elevator_brake_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 31:elevator_motor_rotation_control_status | Offset: 170, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 170
    elevator_motor_rotation_control_status = elevator_motor_rotation_control_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('elevatorMotorRotationControlStatus', 'Elevator Motor rotation control Status', None, None, elevator_motor_rotation_control_status, elevator_motor_rotation_control_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 32:reserved_172 | Offset: 172, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 172
    reserved_172 = reserved_172_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_172', 'Reserved', None, None, reserved_172, reserved_172_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    return nmea2000Message

def encode_pgn_128538(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 128538."""
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
    # elevatorCarId | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorCarId")

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
        raise ValueError("Cant encode this message, 'Elevator Car ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Car ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Car ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorCarUsage | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorCarUsage")

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
        raise ValueError("Cant encode this message, 'Elevator Car Usage' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Car Usage' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Car Usage' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # smokeSensorStatus | Offset: 24, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("smokeSensorStatus")

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
        raise ValueError("Cant encode this message, 'Smoke Sensor Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Smoke Sensor Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Smoke Sensor Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # limitSwitchSensorStatus | Offset: 26, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("limitSwitchSensorStatus")

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
        raise ValueError("Cant encode this message, 'Limit Switch Sensor Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Limit Switch Sensor Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Limit Switch Sensor Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # proximitySwitchSensorStatus | Offset: 28, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("proximitySwitchSensorStatus")

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
        raise ValueError("Cant encode this message, 'Proximity Switch Sensor Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Proximity Switch Sensor Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Proximity Switch Sensor Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # inertialMeasurementUnitImuSensorStatus | Offset: 30, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 30
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("inertialMeasurementUnitImuSensorStatus")

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
        raise ValueError("Cant encode this message, 'Inertial Measurement Unit (IMU) Sensor Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Inertial Measurement Unit (IMU) Sensor Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Inertial Measurement Unit (IMU) Sensor Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorLoadLimitStatus | Offset: 32, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorLoadLimitStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Load Limit Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Load Limit Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Load Limit Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorLoadBalanceStatus | Offset: 34, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 34
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorLoadBalanceStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Load Balance Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Load Balance Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Load Balance Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorLoadSensor1Status | Offset: 36, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorLoadSensor1Status")

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
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 1 Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 1 Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 1 Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorLoadSensor2Status | Offset: 38, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 38
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorLoadSensor2Status")

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
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 2 Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 2 Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 2 Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorLoadSensor3Status | Offset: 40, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorLoadSensor3Status")

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
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 3 Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 3 Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 3 Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorLoadSensor4Status | Offset: 42, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 42
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorLoadSensor4Status")

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
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 4 Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 4 Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Load Sensor 4 Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_44 | Offset: 44, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 44
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_44")

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
    # elevatorCarMotionStatus | Offset: 48, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorCarMotionStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Car Motion Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Car Motion Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Car Motion Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorCarDoorStatus | Offset: 50, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 50
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorCarDoorStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Car Door Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Car Door Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Car Door Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorCarEmergencyButtonStatus | Offset: 52, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 52
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorCarEmergencyButtonStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Car Emergency Button Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Car Emergency Button Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Car Emergency Button Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorCarBuzzerStatus | Offset: 54, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 54
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorCarBuzzerStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Car Buzzer Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Car Buzzer Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Car Buzzer Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # openDoorButtonStatus | Offset: 56, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("openDoorButtonStatus")

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
        raise ValueError("Cant encode this message, 'Open Door Button Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Open Door Button Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Open Door Button Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # closeDoorButtonStatus | Offset: 58, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 58
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("closeDoorButtonStatus")

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
        raise ValueError("Cant encode this message, 'Close Door Button Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Close Door Button Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Close Door Button Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_60 | Offset: 60, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 60
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_60")

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
    # currentDeck | Offset: 64, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("currentDeck")

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
        raise ValueError("Cant encode this message, 'Current Deck' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Current Deck' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Current Deck' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # destinationDeck | Offset: 72, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("destinationDeck")

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
        raise ValueError("Cant encode this message, 'Destination Deck' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Destination Deck' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Destination Deck' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalNumberOfDecks | Offset: 80, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalNumberOfDecks")

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
        raise ValueError("Cant encode this message, 'Total Number of Decks' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total Number of Decks' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total Number of Decks' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # weightOfLoadCell1 | Offset: 88, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("weightOfLoadCell1")

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
        raise ValueError("Cant encode this message, 'Weight of Load Cell 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # weightOfLoadCell2 | Offset: 104, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("weightOfLoadCell2")

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
        raise ValueError("Cant encode this message, 'Weight of Load Cell 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # weightOfLoadCell3 | Offset: 120, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("weightOfLoadCell3")

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
        raise ValueError("Cant encode this message, 'Weight of Load Cell 3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # weightOfLoadCell4 | Offset: 136, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("weightOfLoadCell4")

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
        raise ValueError("Cant encode this message, 'Weight of Load Cell 4' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 4' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Weight of Load Cell 4' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # speedOfElevatorCar | Offset: 152, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("speedOfElevatorCar")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Speed of Elevator Car' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Speed of Elevator Car' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Speed of Elevator Car' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorBrakeStatus | Offset: 168, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 168
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorBrakeStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Brake Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Brake Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Brake Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # elevatorMotorRotationControlStatus | Offset: 170, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 170
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("elevatorMotorRotationControlStatus")

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
        raise ValueError("Cant encode this message, 'Elevator Motor rotation control Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Elevator Motor rotation control Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Elevator Motor rotation control Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_172 | Offset: 172, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 172
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_172")

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
    return data_raw.to_bytes(22, byteorder="little")
