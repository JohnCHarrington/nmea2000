# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130567() -> bool:
    """Return True if PGN 130567 is a fast PGN."""
    return True
def decode_pgn_130567(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130567."""
    nmea2000Message = NMEA2000Message(PGN=130567, id='watermakerInputSettingAndStatus', description='Watermaker Input Setting and Status')
    running_bit_offset = 0
    # 1:watermaker_operating_state | Offset: 0, Length: 6, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    watermaker_operating_state_raw = decode_int(_data_raw_, running_bit_offset, 6)
    watermaker_operating_state = master_dict['WATERMAKER_STATE'].get(watermaker_operating_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('watermakerOperatingState', 'Watermaker Operating State', None, None, watermaker_operating_state, watermaker_operating_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 6

    # 2:production_start_stop | Offset: 6, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 6
    production_start_stop_raw = decode_int(_data_raw_, running_bit_offset, 2)
    production_start_stop = master_dict['YES_NO'].get(production_start_stop_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('productionStartStop', 'Production Start/Stop', None, None, production_start_stop, production_start_stop_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:rinse_start_stop | Offset: 8, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    rinse_start_stop_raw = decode_int(_data_raw_, running_bit_offset, 2)
    rinse_start_stop = master_dict['YES_NO'].get(rinse_start_stop_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('rinseStartStop', 'Rinse Start/Stop', None, None, rinse_start_stop, rinse_start_stop_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:low_pressure_pump_status | Offset: 10, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 10
    low_pressure_pump_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    low_pressure_pump_status = master_dict['YES_NO'].get(low_pressure_pump_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('lowPressurePumpStatus', 'Low Pressure Pump Status', None, None, low_pressure_pump_status, low_pressure_pump_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:high_pressure_pump_status | Offset: 12, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    high_pressure_pump_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    high_pressure_pump_status = master_dict['YES_NO'].get(high_pressure_pump_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('highPressurePumpStatus', 'High Pressure Pump Status', None, None, high_pressure_pump_status, high_pressure_pump_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:emergency_stop | Offset: 14, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 14
    emergency_stop_raw = decode_int(_data_raw_, running_bit_offset, 2)
    emergency_stop = master_dict['YES_NO'].get(emergency_stop_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('emergencyStop', 'Emergency Stop', None, None, emergency_stop, emergency_stop_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 7:product_solenoid_valve_status | Offset: 16, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    product_solenoid_valve_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    product_solenoid_valve_status = master_dict['OK_WARNING'].get(product_solenoid_valve_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('productSolenoidValveStatus', 'Product Solenoid Valve Status', None, None, product_solenoid_valve_status, product_solenoid_valve_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 8:flush_mode_status | Offset: 18, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 18
    flush_mode_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    flush_mode_status = master_dict['YES_NO'].get(flush_mode_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('flushModeStatus', 'Flush Mode Status', None, None, flush_mode_status, flush_mode_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 9:salinity_status | Offset: 20, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    salinity_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    salinity_status = master_dict['OK_WARNING'].get(salinity_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('salinityStatus', 'Salinity Status', None, None, salinity_status, salinity_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:sensor_status | Offset: 22, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 22
    sensor_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    sensor_status = master_dict['OK_WARNING'].get(sensor_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('sensorStatus', 'Sensor Status', None, None, sensor_status, sensor_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:oil_change_indicator_status | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    oil_change_indicator_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    oil_change_indicator_status = master_dict['OK_WARNING'].get(oil_change_indicator_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('oilChangeIndicatorStatus', 'Oil Change Indicator Status', None, None, oil_change_indicator_status, oil_change_indicator_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 12:filter_status | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    filter_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    filter_status = master_dict['OK_WARNING'].get(filter_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('filterStatus', 'Filter Status', None, None, filter_status, filter_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 13:system_status | Offset: 28, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    system_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    system_status = master_dict['OK_WARNING'].get(system_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('systemStatus', 'System Status', None, None, system_status, system_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 14:reserved_30 | Offset: 30, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 30
    reserved_30 = reserved_30_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_30', 'Reserved', None, None, reserved_30, reserved_30_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 15:salinity | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    salinity = salinity_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('salinity', 'Salinity', None, 'ppm', salinity, salinity_raw, PhysicalQuantities.CONCENTRATION, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 16:product_water_temperature | Offset: 48, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    product_water_temperature = product_water_temperature_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('productWaterTemperature', 'Product Water Temperature', None, 'K', product_water_temperature, product_water_temperature_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 17:pre_filter_pressure | Offset: 64, Length: 16, Signed: False Resolution: 100, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    pre_filter_pressure = pre_filter_pressure_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 100, 0, 6553200)
    nmea2000Message.fields.append(NMEA2000Field('preFilterPressure', 'Pre-filter Pressure', None, 'Pa', pre_filter_pressure, pre_filter_pressure_raw, PhysicalQuantities.PRESSURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 18:post_filter_pressure | Offset: 80, Length: 16, Signed: False Resolution: 100, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    post_filter_pressure = post_filter_pressure_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 100, 0, 6553200)
    nmea2000Message.fields.append(NMEA2000Field('postFilterPressure', 'Post-filter Pressure', None, 'Pa', post_filter_pressure, post_filter_pressure_raw, PhysicalQuantities.PRESSURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 19:feed_pressure | Offset: 96, Length: 16, Signed: True Resolution: 1000, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    feed_pressure = feed_pressure_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1000, -32767000, 32764000)
    nmea2000Message.fields.append(NMEA2000Field('feedPressure', 'Feed Pressure', None, 'Pa', feed_pressure, feed_pressure_raw, PhysicalQuantities.PRESSURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 20:system_high_pressure | Offset: 112, Length: 16, Signed: False Resolution: 1000, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    system_high_pressure = system_high_pressure_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1000, 0, 65532000)
    nmea2000Message.fields.append(NMEA2000Field('systemHighPressure', 'System High Pressure', None, 'Pa', system_high_pressure, system_high_pressure_raw, PhysicalQuantities.PRESSURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 21:product_water_flow | Offset: 128, Length: 16, Signed: True Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    product_water_flow = product_water_flow_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.1, -3276.7, 3276.4)
    nmea2000Message.fields.append(NMEA2000Field('productWaterFlow', 'Product Water Flow', None, 'L/h', product_water_flow, product_water_flow_raw, PhysicalQuantities.VOLUMETRIC_FLOW, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 22:brine_water_flow | Offset: 144, Length: 16, Signed: True Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    brine_water_flow = brine_water_flow_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.1, -3276.7, 3276.4)
    nmea2000Message.fields.append(NMEA2000Field('brineWaterFlow', 'Brine Water Flow', None, 'L/h', brine_water_flow, brine_water_flow_raw, PhysicalQuantities.VOLUMETRIC_FLOW, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 23:run_time | Offset: 160, Length: 32, Signed: False Resolution: 1, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    run_time = run_time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('runTime', 'Run Time', None, 's', run_time, run_time_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130567(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130567."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # watermakerOperatingState | Offset: 0, Length: 6, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("watermakerOperatingState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_WATERMAKER_STATE(field.value)
    field_bit_length = 6
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Watermaker Operating State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Watermaker Operating State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Watermaker Operating State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # productionStartStop | Offset: 6, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 6
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("productionStartStop")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Production Start/Stop' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Production Start/Stop' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Production Start/Stop' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rinseStartStop | Offset: 8, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rinseStartStop")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rinse Start/Stop' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rinse Start/Stop' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rinse Start/Stop' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # lowPressurePumpStatus | Offset: 10, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 10
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("lowPressurePumpStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Low Pressure Pump Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Low Pressure Pump Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Low Pressure Pump Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # highPressurePumpStatus | Offset: 12, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 12
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("highPressurePumpStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'High Pressure Pump Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'High Pressure Pump Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'High Pressure Pump Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # emergencyStop | Offset: 14, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 14
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("emergencyStop")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Emergency Stop' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Emergency Stop' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Emergency Stop' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # productSolenoidValveStatus | Offset: 16, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("productSolenoidValveStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OK_WARNING(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Product Solenoid Valve Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Product Solenoid Valve Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Product Solenoid Valve Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # flushModeStatus | Offset: 18, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 18
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("flushModeStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Flush Mode Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Flush Mode Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Flush Mode Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # salinityStatus | Offset: 20, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("salinityStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OK_WARNING(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Salinity Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Salinity Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Salinity Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sensorStatus | Offset: 22, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 22
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sensorStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OK_WARNING(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Sensor Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sensor Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sensor Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # oilChangeIndicatorStatus | Offset: 24, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("oilChangeIndicatorStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OK_WARNING(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Oil Change Indicator Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Oil Change Indicator Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Oil Change Indicator Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # filterStatus | Offset: 26, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("filterStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OK_WARNING(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Filter Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Filter Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Filter Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # systemStatus | Offset: 28, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("systemStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_OK_WARNING(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'System Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'System Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'System Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_30 | Offset: 30, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 30
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_30")

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
    # salinity | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("salinity")

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
        raise ValueError("Cant encode this message, 'Salinity' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Salinity' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Salinity' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # productWaterTemperature | Offset: 48, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("productWaterTemperature")

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
        raise ValueError("Cant encode this message, 'Product Water Temperature' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Product Water Temperature' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Product Water Temperature' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # preFilterPressure | Offset: 64, Length: 16, Resolution: 100, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("preFilterPressure")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 100):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 100)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 100)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Pre-filter Pressure' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Pre-filter Pressure' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Pre-filter Pressure' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # postFilterPressure | Offset: 80, Length: 16, Resolution: 100, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("postFilterPressure")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 100):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 100)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 100)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Post-filter Pressure' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Post-filter Pressure' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Post-filter Pressure' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # feedPressure | Offset: 96, Length: 16, Resolution: 1000, Field Type: NUMBER
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("feedPressure")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1000):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 1000)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 1000)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Feed Pressure' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Feed Pressure' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Feed Pressure' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # systemHighPressure | Offset: 112, Length: 16, Resolution: 1000, Field Type: NUMBER
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("systemHighPressure")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1000):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 1000)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 1000)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'System High Pressure' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'System High Pressure' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'System High Pressure' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # productWaterFlow | Offset: 128, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("productWaterFlow")

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
        raise ValueError("Cant encode this message, 'Product Water Flow' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Product Water Flow' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Product Water Flow' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # brineWaterFlow | Offset: 144, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("brineWaterFlow")

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
        raise ValueError("Cant encode this message, 'Brine Water Flow' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Brine Water Flow' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Brine Water Flow' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # runTime | Offset: 160, Length: 32, Resolution: 1, Field Type: DURATION
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("runTime")

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
        raise ValueError("Cant encode this message, 'Run Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Run Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Run Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(24, byteorder="little")
