# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130584() -> bool:
    """Return True if PGN 130584 is a fast PGN."""
    return True
def decode_pgn_130584(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130584."""
    nmea2000Message = NMEA2000Message(PGN=130584, id='availableBluetoothAddresses', description='Available Bluetooth addresses')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:first_address | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    first_address = first_address_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('firstAddress', 'First address', "First address in this PGN", None, first_address, first_address_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:address_count | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    address_count = address_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('addressCount', 'Address count', None, None, address_count, address_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:total_address_count | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    total_address_count = total_address_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('totalAddressCount', 'Total address count', None, None, total_address_count, total_address_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:bluetooth_address | Offset: 24, Length: 48, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    _repeating_field_set_1_offset = running_bit_offset
    bluetooth_address = bluetooth_address_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 48))
    nmea2000Message.fields.append(NMEA2000Field('bluetoothAddress', 'Bluetooth address', None, None, bluetooth_address, bluetooth_address_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 48

    # 5:status | Offset: 72, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    status_raw = decode_int(_data_raw_, running_bit_offset, 8)
    status = master_dict['BLUETOOTH_STATUS'].get(status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:device_name | Offset: 80, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    device_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    device_name = device_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('deviceName', 'Device name', None, None, device_name, device_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 7:signal_strength | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    signal_strength = signal_strength_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('signalStrength', 'Signal strength', None, '%', signal_strength, signal_strength_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(address_count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        raise ValueError("PGN unknown repeating FieldType (BINARY) not supported")
        repeating_entry["bluetoothAddress"] = _repeating_entry_value(bluetooth_address, bluetooth_address_raw)
    
        status_raw = decode_int(_data_raw_, running_bit_offset, 8)
        status = master_dict['BLUETOOTH_STATUS'].get(status_raw, None)
        running_bit_offset += 8
        repeating_entry["status"] = _repeating_entry_value(status, status_raw)
    
        device_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        device_name = device_name_raw
        running_bit_offset += bits_to_skip
        repeating_entry["deviceName"] = _repeating_entry_value(device_name, device_name_raw)
    
        signal_strength = signal_strength_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["signalStrength"] = _repeating_entry_value(signal_strength, signal_strength_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "bluetoothAddress",
                "status",
                "deviceName",
                "signalStrength",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130584(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130584."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "bluetoothAddress",
        "status",
        "deviceName",
        "signalStrength",
    ))
    # firstAddress | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("firstAddress")

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
        raise ValueError("Cant encode this message, 'First address' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'First address' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'First address' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # addressCount | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("addressCount")

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
        raise ValueError("Cant encode this message, 'Address count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Address count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Address count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalAddressCount | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalAddressCount")

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
        raise ValueError("Cant encode this message, 'Total address count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total address count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total address count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 24
    for repeating_entry in repeating_field_set_1_entries:
        # bluetoothAddress | Offset: 24, Length: 48, Resolution: 1, Field Type: BINARY
        field = repeating_entry.get("bluetoothAddress")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Bluetooth address'")
        field_offset = running_bit_offset
    
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
        # status | Offset: 72, Length: 8, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("status")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Status'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_BLUETOOTH_STATUS(field.value)
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
        # deviceName | Offset: 80, Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("deviceName")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Device name'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Device name' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Device name' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Device name' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # signalStrength | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("signalStrength")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Signal strength'")
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
            raise ValueError("Cant encode this message, 'Signal strength' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Signal strength' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Signal strength' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
