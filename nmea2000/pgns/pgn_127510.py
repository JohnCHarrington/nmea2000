# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127510() -> bool:
    """Return True if PGN 127510 is a fast PGN."""
    return True
def decode_pgn_127510(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127510."""
    nmea2000Message = NMEA2000Message(PGN=127510, id='chargerConfigurationStatus', description='Charger Configuration Status')
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

    # 3:charger_enable_disable | Offset: 16, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    charger_enable_disable_raw = decode_int(_data_raw_, running_bit_offset, 2)
    charger_enable_disable = master_dict['OFF_ON'].get(charger_enable_disable_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('chargerEnableDisable', 'Charger Enable/Disable', None, None, charger_enable_disable, charger_enable_disable_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:reserved_18 | Offset: 18, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 18
    reserved_18 = reserved_18_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_18', 'Reserved', None, None, reserved_18, reserved_18_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 5:charge_current_limit | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    charge_current_limit = charge_current_limit_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('chargeCurrentLimit', 'Charge Current Limit', None, '%', charge_current_limit, charge_current_limit_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:charging_algorithm | Offset: 32, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    charging_algorithm_raw = decode_int(_data_raw_, running_bit_offset, 4)
    charging_algorithm = master_dict['CHARGING_ALGORITHM'].get(charging_algorithm_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('chargingAlgorithm', 'Charging Algorithm', None, None, charging_algorithm, charging_algorithm_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 7:charger_mode | Offset: 36, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 36
    charger_mode_raw = decode_int(_data_raw_, running_bit_offset, 4)
    charger_mode = master_dict['CHARGER_MODE'].get(charger_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('chargerMode', 'Charger Mode', None, None, charger_mode, charger_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 8:estimated_temperature | Offset: 40, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    estimated_temperature_raw = decode_int(_data_raw_, running_bit_offset, 4)
    estimated_temperature = master_dict['DEVICE_TEMP_STATE'].get(estimated_temperature_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('estimatedTemperature', 'Estimated Temperature', "If there is no battery temperature sensor the charger will use this field to steer the charging algorithm", None, estimated_temperature, estimated_temperature_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 9:equalize_one_time_enable_disable | Offset: 44, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 44
    equalize_one_time_enable_disable_raw = decode_int(_data_raw_, running_bit_offset, 2)
    equalize_one_time_enable_disable = master_dict['OFF_ON'].get(equalize_one_time_enable_disable_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('equalizeOneTimeEnableDisable', 'Equalize One Time Enable/Disable', None, None, equalize_one_time_enable_disable, equalize_one_time_enable_disable_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:over_charge_enable_disable | Offset: 46, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 46
    over_charge_enable_disable_raw = decode_int(_data_raw_, running_bit_offset, 2)
    over_charge_enable_disable = master_dict['OFF_ON'].get(over_charge_enable_disable_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('overChargeEnableDisable', 'Over Charge Enable/Disable', None, None, over_charge_enable_disable, over_charge_enable_disable_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:equalize_time | Offset: 48, Length: 16, Signed: False Resolution: 60, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    equalize_time = equalize_time_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 60, 0, 3931920)
    nmea2000Message.fields.append(NMEA2000Field('equalizeTime', 'Equalize Time', None, 's', equalize_time, equalize_time_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_127510(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127510."""
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
    # chargerEnableDisable | Offset: 16, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("chargerEnableDisable")

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
        raise ValueError("Cant encode this message, 'Charger Enable/Disable' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Charger Enable/Disable' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Charger Enable/Disable' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_18 | Offset: 18, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 18
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_18")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 6
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
    # chargeCurrentLimit | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("chargeCurrentLimit")

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
        raise ValueError("Cant encode this message, 'Charge Current Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Charge Current Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Charge Current Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # chargingAlgorithm | Offset: 32, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("chargingAlgorithm")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CHARGING_ALGORITHM(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Charging Algorithm' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Charging Algorithm' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Charging Algorithm' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # chargerMode | Offset: 36, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 36
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("chargerMode")

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
        raise ValueError("Cant encode this message, 'Charger Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Charger Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Charger Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # estimatedTemperature | Offset: 40, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("estimatedTemperature")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DEVICE_TEMP_STATE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Estimated Temperature' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Estimated Temperature' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Estimated Temperature' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # equalizeOneTimeEnableDisable | Offset: 44, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 44
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("equalizeOneTimeEnableDisable")

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
        raise ValueError("Cant encode this message, 'Equalize One Time Enable/Disable' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Equalize One Time Enable/Disable' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Equalize One Time Enable/Disable' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # overChargeEnableDisable | Offset: 46, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 46
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("overChargeEnableDisable")

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
        raise ValueError("Cant encode this message, 'Over Charge Enable/Disable' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Over Charge Enable/Disable' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Over Charge Enable/Disable' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # equalizeTime | Offset: 48, Length: 16, Resolution: 60, Field Type: DURATION
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("equalizeTime")

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
        raise ValueError("Cant encode this message, 'Equalize Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Equalize Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Equalize Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
