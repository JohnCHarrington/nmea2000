# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130585() -> bool:
    """Return True if PGN 130585 is a fast PGN."""
    return False
def decode_pgn_130585(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130585."""
    nmea2000Message = NMEA2000Message(PGN=130585, id='bluetoothSourceStatus', description='Bluetooth source status')
    running_bit_offset = 0
    # 1:source_number | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    source_number = source_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceNumber', 'Source number', None, None, source_number, source_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:status | Offset: 8, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    status_raw = decode_int(_data_raw_, running_bit_offset, 4)
    status = master_dict['BLUETOOTH_SOURCE_STATUS'].get(status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 3:forget_device | Offset: 12, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    forget_device_raw = decode_int(_data_raw_, running_bit_offset, 2)
    forget_device = master_dict['YES_NO'].get(forget_device_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('forgetDevice', 'Forget device', None, None, forget_device, forget_device_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:discovering | Offset: 14, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 14
    discovering_raw = decode_int(_data_raw_, running_bit_offset, 2)
    discovering = master_dict['YES_NO'].get(discovering_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('discovering', 'Discovering', None, None, discovering, discovering_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:bluetooth_address | Offset: 16, Length: 48, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    bluetooth_address = bluetooth_address_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 48))
    nmea2000Message.fields.append(NMEA2000Field('bluetoothAddress', 'Bluetooth address', None, None, bluetooth_address, bluetooth_address_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 48

    return nmea2000Message

def encode_pgn_130585(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130585."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # sourceNumber | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceNumber")

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
        raise ValueError("Cant encode this message, 'Source number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # status | Offset: 8, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("status")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_BLUETOOTH_SOURCE_STATUS(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # forgetDevice | Offset: 12, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 12
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("forgetDevice")

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
        raise ValueError("Cant encode this message, 'Forget device' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Forget device' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Forget device' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # discovering | Offset: 14, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 14
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("discovering")

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
        raise ValueError("Cant encode this message, 'Discovering' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Discovering' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Discovering' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # bluetoothAddress | Offset: 16, Length: 48, Resolution: 1, Field Type: BINARY
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("bluetoothAddress")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Bluetooth address' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Bluetooth address' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Bluetooth address' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
