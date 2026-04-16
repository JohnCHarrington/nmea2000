# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130564() -> bool:
    """Return True if PGN 130564 is a fast PGN."""
    return True
def decode_pgn_130564(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130564."""
    nmea2000Message = NMEA2000Message(PGN=130564, id='lightingDeviceEnumeration', description='Lighting Device Enumeration')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:index_of_first_device | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    index_of_first_device = index_of_first_device_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('indexOfFirstDevice', 'Index of First Device', None, None, index_of_first_device, index_of_first_device_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:total_number_of_devices | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    total_number_of_devices = total_number_of_devices_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('totalNumberOfDevices', 'Total Number of Devices', None, None, total_number_of_devices, total_number_of_devices_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:number_of_devices | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    number_of_devices = number_of_devices_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfDevices', 'Number of Devices', None, None, number_of_devices, number_of_devices_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:device_id | Offset: 48, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    device_id = device_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('deviceId', 'Device ID', None, None, device_id, device_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 5:status | Offset: 80, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    status = status_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(number_of_devices_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        device_id = device_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["deviceId"] = _repeating_entry_value(device_id, device_id_raw)
    
        status = status_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["status"] = _repeating_entry_value(status, status_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "deviceId",
                "status",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130564(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130564."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "deviceId",
        "status",
    ))
    # indexOfFirstDevice | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("indexOfFirstDevice")

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
        raise ValueError("Cant encode this message, 'Index of First Device' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Index of First Device' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Index of First Device' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalNumberOfDevices | Offset: 16, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalNumberOfDevices")

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
        raise ValueError("Cant encode this message, 'Total Number of Devices' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total Number of Devices' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total Number of Devices' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfDevices | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfDevices")

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
        raise ValueError("Cant encode this message, 'Number of Devices' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Devices' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Devices' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 48
    for repeating_entry in repeating_field_set_1_entries:
        # deviceId | Offset: 48, Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("deviceId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Device ID'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Device ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Device ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Device ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # status | Offset: 80, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("status")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Status'")
        field_offset = running_bit_offset
    
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
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
