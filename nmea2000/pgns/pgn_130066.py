# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130066() -> bool:
    """Return True if PGN 130066 is a fast PGN."""
    return True
def decode_pgn_130066(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130066."""
    nmea2000Message = NMEA2000Message(PGN=130066, id='routeAndWpServiceRouteWpListAttributes', description='Route and WP Service - Route/WP-List Attributes')
    running_bit_offset = 0
    # 1:database_id | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    database_id = database_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('databaseId', 'Database ID', None, None, database_id, database_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:route_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    route_id = route_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('routeId', 'Route ID', None, None, route_id, route_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:route_wp_list_name | Offset: 32, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    route_wp_list_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    route_wp_list_name = route_wp_list_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('routeWpListName', 'Route/WP-List Name', None, None, route_wp_list_name, route_wp_list_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 4:route_wp_list_timestamp | Offset: , Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    route_wp_list_timestamp_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    route_wp_list_timestamp = decode_time(route_wp_list_timestamp_raw)
    nmea2000Message.fields.append(NMEA2000Field('routeWpListTimestamp', 'Route/WP-List Timestamp', "Seconds since midnight", 's', route_wp_list_timestamp, route_wp_list_timestamp_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 5:route_wp_list_datestamp | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    route_wp_list_datestamp_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    route_wp_list_datestamp = decode_date(route_wp_list_datestamp_raw)
    nmea2000Message.fields.append(NMEA2000Field('routeWpListDatestamp', 'Route/WP-List Datestamp', None, 'd', route_wp_list_datestamp, route_wp_list_datestamp_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 6:change_at_last_timestamp | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    change_at_last_timestamp_raw = decode_int(_data_raw_, running_bit_offset, 8)
    change_at_last_timestamp = decode_bit_lookup(change_at_last_timestamp_raw, master_flags_dict['WP_CHANGE'])
    nmea2000Message.fields.append(NMEA2000Field('changeAtLastTimestamp', 'Change at Last Timestamp', None, None, change_at_last_timestamp, change_at_last_timestamp_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 8

    # 7:number_of_wps_in_the_route_wp_list | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    number_of_wps_in_the_route_wp_list = number_of_wps_in_the_route_wp_list_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfWpsInTheRouteWpList', 'Number of WPs in the Route/WP-List', None, None, number_of_wps_in_the_route_wp_list, number_of_wps_in_the_route_wp_list_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:critical_supplementary_parameters | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    critical_supplementary_parameters_raw = decode_int(_data_raw_, running_bit_offset, 8)
    critical_supplementary_parameters = decode_bit_lookup(critical_supplementary_parameters_raw, master_flags_dict['WP_CRITICAL_PARAMETERS'])
    nmea2000Message.fields.append(NMEA2000Field('criticalSupplementaryParameters', 'Critical supplementary parameters', None, None, critical_supplementary_parameters, critical_supplementary_parameters_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 8

    # 9:navigation_method | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    navigation_method_raw = decode_int(_data_raw_, running_bit_offset, 2)
    navigation_method = master_dict['WP_NAVIGATION_METHOD'].get(navigation_method_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('navigationMethod', 'Navigation Method', None, None, navigation_method, navigation_method_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:wp_identification_method | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    wp_identification_method_raw = decode_int(_data_raw_, running_bit_offset, 2)
    wp_identification_method = master_dict['WP_IDENTIFICATION_METHOD'].get(wp_identification_method_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('wpIdentificationMethod', 'WP Identification Method', None, None, wp_identification_method, wp_identification_method_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:route_status | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    route_status_raw = decode_int(_data_raw_, running_bit_offset, 4)
    route_status = master_dict['WP_ROUTE_STATUS'].get(route_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('routeStatus', 'Route Status', None, None, route_status, route_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 12:xte_limit_for_the_route | Offset: , Length: 16, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    xte_limit_for_the_route = xte_limit_for_the_route_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1, -32767, 32764)
    nmea2000Message.fields.append(NMEA2000Field('xteLimitForTheRoute', 'XTE Limit for the Route', None, 'm', xte_limit_for_the_route, xte_limit_for_the_route_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_130066(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130066."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # databaseId | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
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
    # routeId | Offset: 16, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("routeId")

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
    # routeWpListName | Offset: 32, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("routeWpListName")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Route/WP-List Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Route/WP-List Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Route/WP-List Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # routeWpListTimestamp | Offset: , Length: 32, Resolution: 0.0001, Field Type: TIME
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("routeWpListTimestamp")

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
        raise ValueError("Cant encode this message, 'Route/WP-List Timestamp' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Route/WP-List Timestamp' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Route/WP-List Timestamp' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # routeWpListDatestamp | Offset: , Length: 16, Resolution: 1, Field Type: DATE
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("routeWpListDatestamp")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Route/WP-List Datestamp' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Route/WP-List Datestamp' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Route/WP-List Datestamp' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # changeAtLastTimestamp | Offset: , Length: 8, Resolution: 1, Field Type: BITLOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("changeAtLastTimestamp")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['WP_CHANGE'])
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Change at Last Timestamp' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Change at Last Timestamp' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Change at Last Timestamp' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfWpsInTheRouteWpList | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfWpsInTheRouteWpList")

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
        raise ValueError("Cant encode this message, 'Number of WPs in the Route/WP-List' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of WPs in the Route/WP-List' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of WPs in the Route/WP-List' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # criticalSupplementaryParameters | Offset: , Length: 8, Resolution: 1, Field Type: BITLOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("criticalSupplementaryParameters")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['WP_CRITICAL_PARAMETERS'])
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Critical supplementary parameters' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Critical supplementary parameters' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Critical supplementary parameters' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # navigationMethod | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("navigationMethod")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_WP_NAVIGATION_METHOD(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Navigation Method' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Navigation Method' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Navigation Method' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # wpIdentificationMethod | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("wpIdentificationMethod")

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
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("routeStatus")

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
    # xteLimitForTheRoute | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("xteLimitForTheRoute")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'XTE Limit for the Route' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'XTE Limit for the Route' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'XTE Limit for the Route' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
