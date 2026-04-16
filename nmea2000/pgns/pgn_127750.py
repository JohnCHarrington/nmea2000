# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127750() -> bool:
    """Return True if PGN 127750 is a fast PGN."""
    return False
def decode_pgn_127750(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127750."""
    nmea2000Message = NMEA2000Message(PGN=127750, id='converterStatus', description='Converter Status')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 8))
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 8

    # 2:connection_number | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    connection_number = connection_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('connectionNumber', 'Connection Number', None, None, connection_number, connection_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:operating_state | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    operating_state_raw = decode_int(_data_raw_, running_bit_offset, 8)
    operating_state = master_dict['CONVERTER_STATE'].get(operating_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('operatingState', 'Operating State', None, None, operating_state, operating_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 4:temperature_state | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    temperature_state_raw = decode_int(_data_raw_, running_bit_offset, 2)
    temperature_state = master_dict['GOOD_WARNING_ERROR'].get(temperature_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('temperatureState', 'Temperature State', None, None, temperature_state, temperature_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:overload_state | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    overload_state_raw = decode_int(_data_raw_, running_bit_offset, 2)
    overload_state = master_dict['GOOD_WARNING_ERROR'].get(overload_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('overloadState', 'Overload State', None, None, overload_state, overload_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:low_dc_voltage_state | Offset: 28, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    low_dc_voltage_state_raw = decode_int(_data_raw_, running_bit_offset, 2)
    low_dc_voltage_state = master_dict['GOOD_WARNING_ERROR'].get(low_dc_voltage_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('lowDcVoltageState', 'Low DC Voltage State', None, None, low_dc_voltage_state, low_dc_voltage_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 7:ripple_state | Offset: 30, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 30
    ripple_state_raw = decode_int(_data_raw_, running_bit_offset, 2)
    ripple_state = master_dict['GOOD_WARNING_ERROR'].get(ripple_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('rippleState', 'Ripple State', None, None, ripple_state, ripple_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 8:reserved_32 | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    reserved_32 = reserved_32_raw = decode_int(_data_raw_, running_bit_offset, 32)
    nmea2000Message.fields.append(NMEA2000Field('reserved_32', 'Reserved', None, None, reserved_32, reserved_32_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_127750(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127750."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # sid | Offset: 0, Length: 8, Resolution: 1, Field Type: BINARY
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sid")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
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
    # connectionNumber | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("connectionNumber")

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
        raise ValueError("Cant encode this message, 'Connection Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Connection Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Connection Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # operatingState | Offset: 16, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("operatingState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CONVERTER_STATE(field.value)
    field_bit_length = 8
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
    # temperatureState | Offset: 24, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("temperatureState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GOOD_WARNING_ERROR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Temperature State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Temperature State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Temperature State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # overloadState | Offset: 26, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("overloadState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GOOD_WARNING_ERROR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Overload State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Overload State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Overload State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # lowDcVoltageState | Offset: 28, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("lowDcVoltageState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GOOD_WARNING_ERROR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Low DC Voltage State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Low DC Voltage State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Low DC Voltage State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rippleState | Offset: 30, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 30
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rippleState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GOOD_WARNING_ERROR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Ripple State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Ripple State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Ripple State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_32 | Offset: 32, Length: 32, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_32")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 32
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
