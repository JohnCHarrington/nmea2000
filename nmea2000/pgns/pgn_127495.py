# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127495() -> bool:
    """Return True if PGN 127495 is a fast PGN."""
    return True
def decode_pgn_127495(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127495."""
    nmea2000Message = NMEA2000Message(PGN=127495, id='electricEnergyStorageInformation', description='Electric Energy Storage Information')
    running_bit_offset = 0
    # 1:energy_storage_identifier | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    energy_storage_identifier = energy_storage_identifier_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('energyStorageIdentifier', 'Energy Storage Identifier', None, None, energy_storage_identifier, energy_storage_identifier_raw, None, FieldTypes.NUMBER, True))
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

    # 4:storage_chemistry_conversion | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    storage_chemistry_conversion = storage_chemistry_conversion_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('storageChemistryConversion', 'Storage Chemistry/Conversion', None, None, storage_chemistry_conversion, storage_chemistry_conversion_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:maximum_temperature_derating | Offset: 24, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    maximum_temperature_derating = maximum_temperature_derating_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('maximumTemperatureDerating', 'Maximum Temperature Derating', None, 'K', maximum_temperature_derating, maximum_temperature_derating_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:maximum_temperature_shut_off | Offset: 40, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    maximum_temperature_shut_off = maximum_temperature_shut_off_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('maximumTemperatureShutOff', 'Maximum Temperature Shut Off', None, 'K', maximum_temperature_shut_off, maximum_temperature_shut_off_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:minimum_temperature_derating | Offset: 56, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    minimum_temperature_derating = minimum_temperature_derating_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('minimumTemperatureDerating', 'Minimum Temperature Derating', None, 'K', minimum_temperature_derating, minimum_temperature_derating_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:minimum_temperature_shut_off | Offset: 72, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    minimum_temperature_shut_off = minimum_temperature_shut_off_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('minimumTemperatureShutOff', 'Minimum Temperature Shut Off', None, 'K', minimum_temperature_shut_off, minimum_temperature_shut_off_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:usable_battery_energy | Offset: 88, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    usable_battery_energy = usable_battery_energy_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('usableBatteryEnergy', 'Usable Battery Energy', None, 'kWh', usable_battery_energy, usable_battery_energy_raw, PhysicalQuantities.ELECTRICAL_ENERGY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 10:state_of_health | Offset: 120, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    state_of_health = state_of_health_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('stateOfHealth', 'State of Health', None, None, state_of_health, state_of_health_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 11:battery_cycle_counter | Offset: 128, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    battery_cycle_counter = battery_cycle_counter_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('batteryCycleCounter', 'Battery Cycle Counter', None, None, battery_cycle_counter, battery_cycle_counter_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 12:battery_full_status | Offset: 144, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    battery_full_status = battery_full_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('batteryFullStatus', 'Battery Full Status', None, None, battery_full_status, battery_full_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 13:battery_empty_status | Offset: 146, Length: 2, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 146
    battery_empty_status = battery_empty_status_raw = decode_number(_data_raw_, running_bit_offset, 2, False, 1, 0, 2)
    nmea2000Message.fields.append(NMEA2000Field('batteryEmptyStatus', 'Battery Empty Status', None, None, battery_empty_status, battery_empty_status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 2

    # 14:reserved_148 | Offset: 148, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 148
    reserved_148 = reserved_148_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_148', 'Reserved', None, None, reserved_148, reserved_148_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 15:maximum_charge__soc_ | Offset: 152, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    maximum_charge__soc_ = maximum_charge__soc__raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('maximumChargeSoc', 'Maximum Charge (SOC)', None, None, maximum_charge__soc_, maximum_charge__soc__raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 16:minimum_charge__soc_ | Offset: 160, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    minimum_charge__soc_ = minimum_charge__soc__raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('minimumChargeSoc', 'Minimum Charge (SOC)', None, None, minimum_charge__soc_, minimum_charge__soc__raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_127495(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127495."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # energyStorageIdentifier | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("energyStorageIdentifier")

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
        raise ValueError("Cant encode this message, 'Energy Storage Identifier' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Energy Storage Identifier' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Energy Storage Identifier' exceeds the encoded bit length")
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
    # storageChemistryConversion | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("storageChemistryConversion")

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
        raise ValueError("Cant encode this message, 'Storage Chemistry/Conversion' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Storage Chemistry/Conversion' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Storage Chemistry/Conversion' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maximumTemperatureDerating | Offset: 24, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumTemperatureDerating")

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
        raise ValueError("Cant encode this message, 'Maximum Temperature Derating' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Temperature Derating' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Temperature Derating' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maximumTemperatureShutOff | Offset: 40, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumTemperatureShutOff")

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
        raise ValueError("Cant encode this message, 'Maximum Temperature Shut Off' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Temperature Shut Off' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Temperature Shut Off' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # minimumTemperatureDerating | Offset: 56, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("minimumTemperatureDerating")

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
        raise ValueError("Cant encode this message, 'Minimum Temperature Derating' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Minimum Temperature Derating' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Minimum Temperature Derating' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # minimumTemperatureShutOff | Offset: 72, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("minimumTemperatureShutOff")

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
        raise ValueError("Cant encode this message, 'Minimum Temperature Shut Off' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Minimum Temperature Shut Off' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Minimum Temperature Shut Off' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # usableBatteryEnergy | Offset: 88, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("usableBatteryEnergy")

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
        raise ValueError("Cant encode this message, 'Usable Battery Energy' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Usable Battery Energy' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Usable Battery Energy' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stateOfHealth | Offset: 120, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stateOfHealth")

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
        raise ValueError("Cant encode this message, 'State of Health' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'State of Health' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'State of Health' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # batteryCycleCounter | Offset: 128, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("batteryCycleCounter")

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
        raise ValueError("Cant encode this message, 'Battery Cycle Counter' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Battery Cycle Counter' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Battery Cycle Counter' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # batteryFullStatus | Offset: 144, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("batteryFullStatus")

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
        raise ValueError("Cant encode this message, 'Battery Full Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Battery Full Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Battery Full Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # batteryEmptyStatus | Offset: 146, Length: 2, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 146
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("batteryEmptyStatus")

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
        raise ValueError("Cant encode this message, 'Battery Empty Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Battery Empty Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Battery Empty Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_148 | Offset: 148, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 148
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_148")

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
    # maximumChargeSoc | Offset: 152, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maximumChargeSoc")

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
        raise ValueError("Cant encode this message, 'Maximum Charge (SOC)' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Maximum Charge (SOC)' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Maximum Charge (SOC)' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # minimumChargeSoc | Offset: 160, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("minimumChargeSoc")

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
        raise ValueError("Cant encode this message, 'Minimum Charge (SOC)' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Minimum Charge (SOC)' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Minimum Charge (SOC)' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(21, byteorder="little")
