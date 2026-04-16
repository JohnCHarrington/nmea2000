# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130065() -> bool:
    """Return True if PGN 130065 is a fast PGN."""
    return True
def decode_pgn_130065(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130065."""
    nmea2000Message = NMEA2000Message(PGN=130065, id='routeAndWpServiceRouteList', description='Route and WP Service - Route List')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:start_route_id | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    start_route_id = start_route_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('startRouteId', 'Start Route ID', None, None, start_route_id, start_route_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:nitems | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    nitems = nitems_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('nitems', 'nItems', None, None, nitems, nitems_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:number_of_routes_in_database | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    number_of_routes_in_database = number_of_routes_in_database_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfRoutesInDatabase', 'Number of Routes in Database', None, None, number_of_routes_in_database, number_of_routes_in_database_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:database_id | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    database_id = database_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('databaseId', 'Database ID', None, None, database_id, database_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:route_id | Offset: 64, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    _repeating_field_set_1_offset = running_bit_offset
    route_id = route_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('routeId', 'Route ID', None, None, route_id, route_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:route_name | Offset: 80, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    route_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    route_name = route_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('routeName', 'Route Name', None, None, route_name, route_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 7:reserved_ | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', None, None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 8:wp_identification_method | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    wp_identification_method_raw = decode_int(_data_raw_, running_bit_offset, 2)
    wp_identification_method = master_dict['WP_IDENTIFICATION_METHOD'].get(wp_identification_method_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('wpIdentificationMethod', 'WP Identification Method', None, None, wp_identification_method, wp_identification_method_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 9:route_status | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    route_status_raw = decode_int(_data_raw_, running_bit_offset, 4)
    route_status = master_dict['WP_ROUTE_STATUS'].get(route_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('routeStatus', 'Route Status', None, None, route_status, route_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(nitems_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        route_id = route_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        running_bit_offset += 16
        repeating_entry["routeId"] = _repeating_entry_value(route_id, route_id_raw)
    
        route_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        route_name = route_name_raw
        running_bit_offset += bits_to_skip
        repeating_entry["routeName"] = _repeating_entry_value(route_name, route_name_raw)
    
        reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 2)
        running_bit_offset += 2
        repeating_entry["reserved_"] = _repeating_entry_value(reserved_, reserved__raw)
    
        wp_identification_method_raw = decode_int(_data_raw_, running_bit_offset, 2)
        wp_identification_method = master_dict['WP_IDENTIFICATION_METHOD'].get(wp_identification_method_raw, None)
        running_bit_offset += 2
        repeating_entry["wpIdentificationMethod"] = _repeating_entry_value(wp_identification_method, wp_identification_method_raw)
    
        route_status_raw = decode_int(_data_raw_, running_bit_offset, 4)
        route_status = master_dict['WP_ROUTE_STATUS'].get(route_status_raw, None)
        running_bit_offset += 4
        repeating_entry["routeStatus"] = _repeating_entry_value(route_status, route_status_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "routeId",
                "routeName",
                "reserved_",
                "wpIdentificationMethod",
                "routeStatus",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130065(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130065."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "routeId",
        "routeName",
        "reserved_",
        "wpIdentificationMethod",
        "routeStatus",
    ))
    # startRouteId | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("startRouteId")

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
        raise ValueError("Cant encode this message, 'Start Route ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Start Route ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Start Route ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # nitems | Offset: 16, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("nitems")

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
        raise ValueError("Cant encode this message, 'nItems' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'nItems' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'nItems' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfRoutesInDatabase | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfRoutesInDatabase")

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
        raise ValueError("Cant encode this message, 'Number of Routes in Database' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Routes in Database' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Routes in Database' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # databaseId | Offset: 48, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("databaseId")

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
        raise ValueError("Cant encode this message, 'Database ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Database ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Database ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 64
    for repeating_entry in repeating_field_set_1_entries:
        # routeId | Offset: 64, Length: 16, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("routeId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Route ID'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Route ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Route ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Route ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # routeName | Offset: 80, Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("routeName")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Route Name'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Route Name' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Route Name' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Route Name' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # reserved_ | Offset: , Length: 2, Resolution: 1, Field Type: RESERVED
        field = repeating_entry.get("reserved_")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reserved'")
        field_offset = running_bit_offset
    
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
        # wpIdentificationMethod | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("wpIdentificationMethod")
        if field is None:
            raise ValueError("Cant encode this message, missing 'WP Identification Method'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_WP_IDENTIFICATION_METHOD(field.value)
        field_bit_length = 2
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'WP Identification Method' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'WP Identification Method' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'WP Identification Method' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # routeStatus | Offset: , Length: 4, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("routeStatus")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Route Status'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_WP_ROUTE_STATUS(field.value)
        field_bit_length = 4
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Route Status' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Route Status' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Route Status' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
