# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130069() -> bool:
    """Return True if PGN 130069 is a fast PGN."""
    return True
def decode_pgn_130069(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130069."""
    nmea2000Message = NMEA2000Message(PGN=130069, id='routeAndWpServiceXteLimitNavigationMethod', description='Route and WP Service - XTE Limit & Navigation Method')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:start_rps_ | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    start_rps_ = start_rps__raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('startRps', 'Start RPS#', None, None, start_rps_, start_rps__raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:nitems | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    nitems = nitems_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('nitems', 'nItems', None, None, nitems, nitems_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:number_of_wps_with_a_specific_xte_limit_or_nav__method | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    number_of_wps_with_a_specific_xte_limit_or_nav__method = number_of_wps_with_a_specific_xte_limit_or_nav__method_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('numberOfWpsWithASpecificXteLimitOrNavMethod', 'Number of WPs with a specific XTE Limit or Nav. Method', None, None, number_of_wps_with_a_specific_xte_limit_or_nav__method, number_of_wps_with_a_specific_xte_limit_or_nav__method_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:database_id | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    database_id = database_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('databaseId', 'Database ID', None, None, database_id, database_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:route_id | Offset: 64, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    route_id = route_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('routeId', 'Route ID', None, None, route_id, route_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:rps_ | Offset: 80, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    rps_ = rps__raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('rps', 'RPS#', None, None, rps_, rps__raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:xte_limit_in_the_leg_after_wp | Offset: 96, Length: 16, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    xte_limit_in_the_leg_after_wp = xte_limit_in_the_leg_after_wp_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1, -32767, 32764)
    nmea2000Message.fields.append(NMEA2000Field('xteLimitInTheLegAfterWp', 'XTE Limit in the leg after WP', None, 'm', xte_limit_in_the_leg_after_wp, xte_limit_in_the_leg_after_wp_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:nav__method_in_the_leg_after_wp | Offset: 112, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    nav__method_in_the_leg_after_wp_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nav__method_in_the_leg_after_wp = master_dict['WP_NAVIGATION_METHOD'].get(nav__method_in_the_leg_after_wp_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('navMethodInTheLegAfterWp', 'Nav. Method in the leg after WP', None, None, nav__method_in_the_leg_after_wp, nav__method_in_the_leg_after_wp_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 9:reserved_114 | Offset: 114, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 114
    reserved_114 = reserved_114_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_114', 'Reserved', None, None, reserved_114, reserved_114_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

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
    
        route_id = route_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        running_bit_offset += 16
        repeating_entry["routeId"] = _repeating_entry_value(route_id, route_id_raw)
    
        rps_ = rps__raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
        running_bit_offset += 16
        repeating_entry["rps"] = _repeating_entry_value(rps_, rps__raw)
    
        xte_limit_in_the_leg_after_wp = xte_limit_in_the_leg_after_wp_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1, -32767, 32764)
        running_bit_offset += 16
        repeating_entry["xteLimitInTheLegAfterWp"] = _repeating_entry_value(xte_limit_in_the_leg_after_wp, xte_limit_in_the_leg_after_wp_raw)
    
        nav__method_in_the_leg_after_wp_raw = decode_int(_data_raw_, running_bit_offset, 2)
        nav__method_in_the_leg_after_wp = master_dict['WP_NAVIGATION_METHOD'].get(nav__method_in_the_leg_after_wp_raw, None)
        running_bit_offset += 2
        repeating_entry["navMethodInTheLegAfterWp"] = _repeating_entry_value(nav__method_in_the_leg_after_wp, nav__method_in_the_leg_after_wp_raw)
    
        reserved_114 = reserved_114_raw = decode_int(_data_raw_, running_bit_offset, 6)
        running_bit_offset += 6
        repeating_entry["reserved_114"] = _repeating_entry_value(reserved_114, reserved_114_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "databaseId",
                "routeId",
                "rps",
                "xteLimitInTheLegAfterWp",
                "navMethodInTheLegAfterWp",
                "reserved_114",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130069(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130069."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "databaseId",
        "routeId",
        "rps",
        "xteLimitInTheLegAfterWp",
        "navMethodInTheLegAfterWp",
        "reserved_114",
    ))
    # startRps | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("startRps")

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
        raise ValueError("Cant encode this message, 'Start RPS#' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Start RPS#' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Start RPS#' exceeds the encoded bit length")
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
    # numberOfWpsWithASpecificXteLimitOrNavMethod | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfWpsWithASpecificXteLimitOrNavMethod")

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
        raise ValueError("Cant encode this message, 'Number of WPs with a specific XTE Limit or Nav. Method' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of WPs with a specific XTE Limit or Nav. Method' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of WPs with a specific XTE Limit or Nav. Method' exceeds the encoded bit length")
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
        # rps | Offset: 80, Length: 16, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("rps")
        if field is None:
            raise ValueError("Cant encode this message, missing 'RPS#'")
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
            raise ValueError("Cant encode this message, 'RPS#' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'RPS#' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'RPS#' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # xteLimitInTheLegAfterWp | Offset: 96, Length: 16, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("xteLimitInTheLegAfterWp")
        if field is None:
            raise ValueError("Cant encode this message, missing 'XTE Limit in the leg after WP'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'XTE Limit in the leg after WP' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'XTE Limit in the leg after WP' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'XTE Limit in the leg after WP' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # navMethodInTheLegAfterWp | Offset: 112, Length: 2, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("navMethodInTheLegAfterWp")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Nav. Method in the leg after WP'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Nav. Method in the leg after WP' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Nav. Method in the leg after WP' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Nav. Method in the leg after WP' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # reserved_114 | Offset: 114, Length: 6, Resolution: 1, Field Type: RESERVED
        field = repeating_entry.get("reserved_114")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reserved'")
        field_offset = running_bit_offset
    
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
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
