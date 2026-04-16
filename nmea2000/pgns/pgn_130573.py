# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130573() -> bool:
    """Return True if PGN 130573 is a fast PGN."""
    return True
def decode_pgn_130573(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130573."""
    nmea2000Message = NMEA2000Message(PGN=130573, id='supportedSourceData', description='Supported Source Data')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:id_offset | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    id_offset = id_offset_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('idOffset', 'ID offset', "First ID in this PGN", None, id_offset, id_offset_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:id_count | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    id_count = id_count_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('idCount', 'ID count', "Number of IDs in this PGN", None, id_count, id_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:total_id_count | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    total_id_count = total_id_count_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('totalIdCount', 'Total ID count', "Total IDs in group", None, total_id_count, total_id_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:id | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _repeating_field_set_1_offset = running_bit_offset
    id = id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('id', 'ID', "Source ID", None, id, id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:source | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    source_raw = decode_int(_data_raw_, running_bit_offset, 8)
    source = master_dict['ENTERTAINMENT_SOURCE'].get(source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('source', 'Source', None, None, source, source_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:number | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    number = number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('number', 'Number', "Source number per type", None, number, number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:name | Offset: 72, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    name = name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 8:play_support | Offset: , Length: 32, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    play_support_raw = decode_int(_data_raw_, running_bit_offset, 32)
    play_support = decode_bit_lookup(play_support_raw, master_flags_dict['ENTERTAINMENT_PLAY_STATUS_BITFIELD'])
    nmea2000Message.fields.append(NMEA2000Field('playSupport', 'Play support', None, None, play_support, play_support_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 32

    # 9:browse_support | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    browse_support_raw = decode_int(_data_raw_, running_bit_offset, 16)
    browse_support = decode_bit_lookup(browse_support_raw, master_flags_dict['ENTERTAINMENT_GROUP_BITFIELD'])
    nmea2000Message.fields.append(NMEA2000Field('browseSupport', 'Browse support', None, None, browse_support, browse_support_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 16

    # 10:thumbs_support | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    thumbs_support_raw = decode_int(_data_raw_, running_bit_offset, 2)
    thumbs_support = master_dict['YES_NO'].get(thumbs_support_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('thumbsSupport', 'Thumbs support', None, None, thumbs_support, thumbs_support_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:connected | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    connected_raw = decode_int(_data_raw_, running_bit_offset, 2)
    connected = master_dict['YES_NO'].get(connected_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('connected', 'Connected', None, None, connected, connected_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 12:repeat_support | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    repeat_support_raw = decode_int(_data_raw_, running_bit_offset, 2)
    repeat_support = decode_bit_lookup(repeat_support_raw, master_flags_dict['ENTERTAINMENT_REPEAT_BITFIELD'])
    nmea2000Message.fields.append(NMEA2000Field('repeatSupport', 'Repeat support', None, None, repeat_support, repeat_support_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 2

    # 13:shuffle_support | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    shuffle_support_raw = decode_int(_data_raw_, running_bit_offset, 2)
    shuffle_support = decode_bit_lookup(shuffle_support_raw, master_flags_dict['ENTERTAINMENT_SHUFFLE_BITFIELD'])
    nmea2000Message.fields.append(NMEA2000Field('shuffleSupport', 'Shuffle support', None, None, shuffle_support, shuffle_support_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 2

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(id_count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        id = id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["id"] = _repeating_entry_value(id, id_raw)
    
        source_raw = decode_int(_data_raw_, running_bit_offset, 8)
        source = master_dict['ENTERTAINMENT_SOURCE'].get(source_raw, None)
        running_bit_offset += 8
        repeating_entry["source"] = _repeating_entry_value(source, source_raw)
    
        number = number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["number"] = _repeating_entry_value(number, number_raw)
    
        name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        name = name_raw
        running_bit_offset += bits_to_skip
        repeating_entry["name"] = _repeating_entry_value(name, name_raw)
    
        play_support_raw = decode_int(_data_raw_, running_bit_offset, 32)
        play_support = decode_bit_lookup(play_support_raw, master_flags_dict['ENTERTAINMENT_PLAY_STATUS_BITFIELD'])
        running_bit_offset += 32
        repeating_entry["playSupport"] = _repeating_entry_value(play_support, play_support_raw)
    
        browse_support_raw = decode_int(_data_raw_, running_bit_offset, 16)
        browse_support = decode_bit_lookup(browse_support_raw, master_flags_dict['ENTERTAINMENT_GROUP_BITFIELD'])
        running_bit_offset += 16
        repeating_entry["browseSupport"] = _repeating_entry_value(browse_support, browse_support_raw)
    
        thumbs_support_raw = decode_int(_data_raw_, running_bit_offset, 2)
        thumbs_support = master_dict['YES_NO'].get(thumbs_support_raw, None)
        running_bit_offset += 2
        repeating_entry["thumbsSupport"] = _repeating_entry_value(thumbs_support, thumbs_support_raw)
    
        connected_raw = decode_int(_data_raw_, running_bit_offset, 2)
        connected = master_dict['YES_NO'].get(connected_raw, None)
        running_bit_offset += 2
        repeating_entry["connected"] = _repeating_entry_value(connected, connected_raw)
    
        repeat_support_raw = decode_int(_data_raw_, running_bit_offset, 2)
        repeat_support = decode_bit_lookup(repeat_support_raw, master_flags_dict['ENTERTAINMENT_REPEAT_BITFIELD'])
        running_bit_offset += 2
        repeating_entry["repeatSupport"] = _repeating_entry_value(repeat_support, repeat_support_raw)
    
        shuffle_support_raw = decode_int(_data_raw_, running_bit_offset, 2)
        shuffle_support = decode_bit_lookup(shuffle_support_raw, master_flags_dict['ENTERTAINMENT_SHUFFLE_BITFIELD'])
        running_bit_offset += 2
        repeating_entry["shuffleSupport"] = _repeating_entry_value(shuffle_support, shuffle_support_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "id",
                "source",
                "number",
                "name",
                "playSupport",
                "browseSupport",
                "thumbsSupport",
                "connected",
                "repeatSupport",
                "shuffleSupport",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130573(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130573."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "id",
        "source",
        "number",
        "name",
        "playSupport",
        "browseSupport",
        "thumbsSupport",
        "connected",
        "repeatSupport",
        "shuffleSupport",
    ))
    # idOffset | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("idOffset")

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
        raise ValueError("Cant encode this message, 'ID offset' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'ID offset' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'ID offset' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # idCount | Offset: 16, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("idCount")

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
        raise ValueError("Cant encode this message, 'ID count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'ID count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'ID count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalIdCount | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalIdCount")

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
        raise ValueError("Cant encode this message, 'Total ID count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total ID count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total ID count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 48
    for repeating_entry in repeating_field_set_1_entries:
        # id | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("id")
        if field is None:
            raise ValueError("Cant encode this message, missing 'ID'")
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
            raise ValueError("Cant encode this message, 'ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # source | Offset: 56, Length: 8, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("source")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Source'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_ENTERTAINMENT_SOURCE(field.value)
        field_bit_length = 8
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Source' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Source' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Source' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # number | Offset: 64, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("number")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Number'")
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
            raise ValueError("Cant encode this message, 'Number' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Number' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Number' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # name | Offset: 72, Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("name")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Name'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Name' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Name' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # playSupport | Offset: , Length: 32, Resolution: 1, Field Type: BITLOOKUP
        field = repeating_entry.get("playSupport")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Play support'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['ENTERTAINMENT_PLAY_STATUS_BITFIELD'])
        field_bit_length = 32
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Play support' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Play support' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Play support' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # browseSupport | Offset: , Length: 16, Resolution: 1, Field Type: BITLOOKUP
        field = repeating_entry.get("browseSupport")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Browse support'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['ENTERTAINMENT_GROUP_BITFIELD'])
        field_bit_length = 16
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Browse support' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Browse support' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Browse support' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # thumbsSupport | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("thumbsSupport")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Thumbs support'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Thumbs support' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Thumbs support' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Thumbs support' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # connected | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("connected")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Connected'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Connected' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Connected' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Connected' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # repeatSupport | Offset: , Length: 2, Resolution: 1, Field Type: BITLOOKUP
        field = repeating_entry.get("repeatSupport")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Repeat support'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['ENTERTAINMENT_REPEAT_BITFIELD'])
        field_bit_length = 2
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Repeat support' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Repeat support' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Repeat support' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # shuffleSupport | Offset: , Length: 2, Resolution: 1, Field Type: BITLOOKUP
        field = repeating_entry.get("shuffleSupport")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Shuffle support'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['ENTERTAINMENT_SHUFFLE_BITFIELD'])
        field_bit_length = 2
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Shuffle support' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Shuffle support' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Shuffle support' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    
    
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
