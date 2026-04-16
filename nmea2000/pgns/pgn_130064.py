# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130064() -> bool:
    """Return True if PGN 130064 is a fast PGN."""
    return True
def decode_pgn_130064(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130064."""
    nmea2000Message = NMEA2000Message(PGN=130064, id='routeAndWpServiceDatabaseList', description='Route and WP Service - Database List')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:start_database_id | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    start_database_id = start_database_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('startDatabaseId', 'Start Database ID', None, None, start_database_id, start_database_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:nitems | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    nitems = nitems_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('nitems', 'nItems', None, None, nitems, nitems_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:number_of_databases_available | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    number_of_databases_available = number_of_databases_available_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfDatabasesAvailable', 'Number of Databases Available', None, None, number_of_databases_available, number_of_databases_available_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:database_id | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    database_id = database_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('databaseId', 'Database ID', None, None, database_id, database_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:database_name | Offset: 64, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    database_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    database_name = database_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('databaseName', 'Database Name', None, None, database_name, database_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 6:database_timestamp | Offset: , Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    database_timestamp_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    database_timestamp = decode_time(database_timestamp_raw)
    nmea2000Message.fields.append(NMEA2000Field('databaseTimestamp', 'Database Timestamp', "Seconds since midnight", 's', database_timestamp, database_timestamp_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 7:database_datestamp | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    database_datestamp_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    database_datestamp = decode_date(database_datestamp_raw)
    nmea2000Message.fields.append(NMEA2000Field('databaseDatestamp', 'Database Datestamp', None, 'd', database_datestamp, database_datestamp_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 8:wp_position_resolution | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    wp_position_resolution_raw = decode_int(_data_raw_, running_bit_offset, 4)
    wp_position_resolution = master_dict['WP_POSITION_RESOLUTION'].get(wp_position_resolution_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('wpPositionResolution', 'WP Position Resolution', None, None, wp_position_resolution, wp_position_resolution_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 9:reserved_ | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', None, None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 10:number_of_routes_in_database | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_routes_in_database = number_of_routes_in_database_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfRoutesInDatabase', 'Number of Routes in Database', None, None, number_of_routes_in_database, number_of_routes_in_database_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 11:number_of_wps_in_database | Offset: , Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_wps_in_database = number_of_wps_in_database_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('numberOfWpsInDatabase', 'Number of WPs in Database', None, None, number_of_wps_in_database, number_of_wps_in_database_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 12:number_of_bytes_in_database | Offset: , Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_bytes_in_database = number_of_bytes_in_database_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('numberOfBytesInDatabase', 'Number of Bytes in Database', None, None, number_of_bytes_in_database, number_of_bytes_in_database_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(nitems_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        database_id = database_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        running_bit_offset += 16
        repeating_entry["databaseId"] = _repeating_entry_value(database_id, database_id_raw)
    
        database_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        database_name = database_name_raw
        running_bit_offset += bits_to_skip
        repeating_entry["databaseName"] = _repeating_entry_value(database_name, database_name_raw)
    
        database_timestamp_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
        database_timestamp = decode_time(database_timestamp_raw)
        running_bit_offset += 32
        repeating_entry["databaseTimestamp"] = _repeating_entry_value(database_timestamp, database_timestamp_raw)
    
        database_datestamp_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        database_datestamp = decode_date(database_datestamp_raw)
        running_bit_offset += 16
        repeating_entry["databaseDatestamp"] = _repeating_entry_value(database_datestamp, database_datestamp_raw)
    
        wp_position_resolution_raw = decode_int(_data_raw_, running_bit_offset, 4)
        wp_position_resolution = master_dict['WP_POSITION_RESOLUTION'].get(wp_position_resolution_raw, None)
        running_bit_offset += 4
        repeating_entry["wpPositionResolution"] = _repeating_entry_value(wp_position_resolution, wp_position_resolution_raw)
    
        reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 4)
        running_bit_offset += 4
        repeating_entry["reserved_"] = _repeating_entry_value(reserved_, reserved__raw)
    
        number_of_routes_in_database = number_of_routes_in_database_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        running_bit_offset += 16
        repeating_entry["numberOfRoutesInDatabase"] = _repeating_entry_value(number_of_routes_in_database, number_of_routes_in_database_raw)
    
        number_of_wps_in_database = number_of_wps_in_database_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["numberOfWpsInDatabase"] = _repeating_entry_value(number_of_wps_in_database, number_of_wps_in_database_raw)
    
        number_of_bytes_in_database = number_of_bytes_in_database_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["numberOfBytesInDatabase"] = _repeating_entry_value(number_of_bytes_in_database, number_of_bytes_in_database_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "databaseId",
                "databaseName",
                "databaseTimestamp",
                "databaseDatestamp",
                "wpPositionResolution",
                "reserved_",
                "numberOfRoutesInDatabase",
                "numberOfWpsInDatabase",
                "numberOfBytesInDatabase",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130064(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130064."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "databaseId",
        "databaseName",
        "databaseTimestamp",
        "databaseDatestamp",
        "wpPositionResolution",
        "reserved_",
        "numberOfRoutesInDatabase",
        "numberOfWpsInDatabase",
        "numberOfBytesInDatabase",
    ))
    # startDatabaseId | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("startDatabaseId")

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
        raise ValueError("Cant encode this message, 'Start Database ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Start Database ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Start Database ID' exceeds the encoded bit length")
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
    # numberOfDatabasesAvailable | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfDatabasesAvailable")

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
        raise ValueError("Cant encode this message, 'Number of Databases Available' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Databases Available' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Databases Available' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 48
    for repeating_entry in repeating_field_set_1_entries:
        # databaseId | Offset: 48, Length: 16, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("databaseId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Database ID'")
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
        # databaseName | Offset: 64, Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("databaseName")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Database Name'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Database Name' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Database Name' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Database Name' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # databaseTimestamp | Offset: , Length: 32, Resolution: 0.0001, Field Type: TIME
        field = repeating_entry.get("databaseTimestamp")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Database Timestamp'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
            field_value = encode_number_raw(field.raw_value, 32, False)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 32, False, 0.0001)
        elif isinstance(field.value, (int, float)):
            field_value = encode_number(field.value, 32, False, 0.0001)
        else:
            assert field.value is None or isinstance(field.value, time)
            field_value = encode_time(field.value, 32)
        field_bit_length = 32
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Database Timestamp' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Database Timestamp' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Database Timestamp' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # databaseDatestamp | Offset: , Length: 16, Resolution: 1, Field Type: DATE
        field = repeating_entry.get("databaseDatestamp")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Database Datestamp'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        else:
            assert field.value is None or isinstance(field.value, date)
            field_value = encode_date(field.value, 16)
        field_bit_length = 16
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Database Datestamp' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Database Datestamp' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Database Datestamp' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # wpPositionResolution | Offset: , Length: 4, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("wpPositionResolution")
        if field is None:
            raise ValueError("Cant encode this message, missing 'WP Position Resolution'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_WP_POSITION_RESOLUTION(field.value)
        field_bit_length = 4
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'WP Position Resolution' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'WP Position Resolution' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'WP Position Resolution' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # reserved_ | Offset: , Length: 4, Resolution: 1, Field Type: RESERVED
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
        # numberOfRoutesInDatabase | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("numberOfRoutesInDatabase")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Number of Routes in Database'")
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
        # numberOfWpsInDatabase | Offset: , Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("numberOfWpsInDatabase")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Number of WPs in Database'")
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
            raise ValueError("Cant encode this message, 'Number of WPs in Database' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Number of WPs in Database' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Number of WPs in Database' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # numberOfBytesInDatabase | Offset: , Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("numberOfBytesInDatabase")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Number of Bytes in Database'")
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
            raise ValueError("Cant encode this message, 'Number of Bytes in Database' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Number of Bytes in Database' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Number of Bytes in Database' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
