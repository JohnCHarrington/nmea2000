# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127493() -> bool:
    """Return True if PGN 127493 is a fast PGN."""
    return False
def decode_pgn_127493(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127493."""
    nmea2000Message = NMEA2000Message(PGN=127493, id='transmissionParametersDynamic', description='Transmission Parameters, Dynamic', ttl=timedelta(milliseconds=100))
    running_bit_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance_raw = decode_int(_data_raw_, running_bit_offset, 8)
    instance = master_dict['ENGINE_INSTANCE'].get(instance_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 8

    # 2:transmission_gear | Offset: 8, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    transmission_gear_raw = decode_int(_data_raw_, running_bit_offset, 2)
    transmission_gear = master_dict['GEAR_STATUS'].get(transmission_gear_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('transmissionGear', 'Transmission Gear', None, None, transmission_gear, transmission_gear_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:reserved_10 | Offset: 10, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 10
    reserved_10 = reserved_10_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_10', 'Reserved', None, None, reserved_10, reserved_10_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 4:oil_pressure | Offset: 16, Length: 16, Signed: False Resolution: 100, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    oil_pressure = oil_pressure_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 100, 0, 6553200)
    nmea2000Message.fields.append(NMEA2000Field('oilPressure', 'Oil pressure', None, 'Pa', oil_pressure, oil_pressure_raw, PhysicalQuantities.PRESSURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:oil_temperature | Offset: 32, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    oil_temperature = oil_temperature_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('oilTemperature', 'Oil temperature', None, 'K', oil_temperature, oil_temperature_raw, PhysicalQuantities.TEMPERATURE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:discrete_status_1 | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    discrete_status_1 = discrete_status_1_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('discreteStatus1', 'Discrete Status 1', None, None, discrete_status_1, discrete_status_1_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:reserved_56 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    reserved_56 = reserved_56_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_56', 'Reserved', None, None, reserved_56, reserved_56_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_127493(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127493."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # instance | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("instance")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENGINE_INSTANCE(field.value)
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
    # transmissionGear | Offset: 8, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("transmissionGear")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GEAR_STATUS(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Transmission Gear' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Transmission Gear' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Transmission Gear' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_10 | Offset: 10, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 10
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_10")

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
    # oilPressure | Offset: 16, Length: 16, Resolution: 100, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("oilPressure")

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
        raise ValueError("Cant encode this message, 'Oil pressure' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Oil pressure' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Oil pressure' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # oilTemperature | Offset: 32, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("oilTemperature")

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
        raise ValueError("Cant encode this message, 'Oil temperature' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Oil temperature' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Oil temperature' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # discreteStatus1 | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("discreteStatus1")

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
        raise ValueError("Cant encode this message, 'Discrete Status 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Discrete Status 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Discrete Status 1' exceeds the encoded bit length")
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
