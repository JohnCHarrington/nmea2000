# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130581() -> bool:
    """Return True if PGN 130581 is a fast PGN."""
    return True
def decode_pgn_130581(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130581."""
    nmea2000Message = NMEA2000Message(PGN=130581, id='zoneConfigurationDeprecated', description='Zone Configuration (deprecated)')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:first_zone_id | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    first_zone_id = first_zone_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('firstZoneId', 'First zone ID', "First Zone in this PGN", None, first_zone_id, first_zone_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:zone_count | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    zone_count = zone_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zoneCount', 'Zone count', "Number of Zones in this PGN", None, zone_count, zone_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:total_zone_count | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    total_zone_count = total_zone_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('totalZoneCount', 'Total zone count', "Total Zones supported by this device", None, total_zone_count, total_zone_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:zone_id | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    _repeating_field_set_1_offset = running_bit_offset
    zone_id_raw = decode_int(_data_raw_, running_bit_offset, 8)
    zone_id = master_dict['ENTERTAINMENT_ZONE'].get(zone_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('zoneId', 'Zone ID', None, None, zone_id, zone_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:zone_name | Offset: 32, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    zone_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    zone_name = zone_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('zoneName', 'Zone name', None, None, zone_name, zone_name_raw, None, FieldTypes.STRING_LAU, False))
    

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(zone_count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        zone_id_raw = decode_int(_data_raw_, running_bit_offset, 8)
        zone_id = master_dict['ENTERTAINMENT_ZONE'].get(zone_id_raw, None)
        running_bit_offset += 8
        repeating_entry["zoneId"] = _repeating_entry_value(zone_id, zone_id_raw)
    
        zone_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        zone_name = zone_name_raw
        running_bit_offset += bits_to_skip
        repeating_entry["zoneName"] = _repeating_entry_value(zone_name, zone_name_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "zoneId",
                "zoneName",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130581(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130581."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "zoneId",
        "zoneName",
    ))
    # firstZoneId | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("firstZoneId")

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
        raise ValueError("Cant encode this message, 'First zone ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'First zone ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'First zone ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zoneCount | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zoneCount")

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
        raise ValueError("Cant encode this message, 'Zone count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalZoneCount | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalZoneCount")

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
        raise ValueError("Cant encode this message, 'Total zone count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total zone count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total zone count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 24
    for repeating_entry in repeating_field_set_1_entries:
        # zoneId | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("zoneId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Zone ID'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_ENTERTAINMENT_ZONE(field.value)
        field_bit_length = 8
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Zone ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Zone ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Zone ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # zoneName | Offset: 32, Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("zoneName")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Zone name'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Zone name' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Zone name' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Zone name' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
