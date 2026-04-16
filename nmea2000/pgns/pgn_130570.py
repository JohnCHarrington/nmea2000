# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130570() -> bool:
    """Return True if PGN 130570 is a fast PGN."""
    return True
def decode_pgn_130570(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130570."""
    nmea2000Message = NMEA2000Message(PGN=130570, id='libraryDataFile', description='Library Data File')
    running_bit_offset = 0
    # 1:source | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    source_raw = decode_int(_data_raw_, running_bit_offset, 8)
    source = master_dict['ENTERTAINMENT_SOURCE'].get(source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('source', 'Source', None, None, source, source_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:number | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    number = number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('number', 'Number', "Source number per type", None, number, number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:id | Offset: 16, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    id = id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('id', 'ID', "Unique file ID", None, id, id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 4:type | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    type_raw = decode_int(_data_raw_, running_bit_offset, 8)
    type = master_dict['ENTERTAINMENT_TYPE'].get(type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('type', 'Type', None, None, type, type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:name | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    name = name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 6:track | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    track = track_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('track', 'Track', None, None, track, track_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:station | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    station = station_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('station', 'Station', None, None, station, station_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:favorite | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    favorite = favorite_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('favorite', 'Favorite', None, None, favorite, favorite_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:radio_frequency | Offset: , Length: 32, Signed: False Resolution: 10, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    radio_frequency = radio_frequency_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 10, 0, 42949672920)
    nmea2000Message.fields.append(NMEA2000Field('radioFrequency', 'Radio Frequency', None, 'Hz', radio_frequency, radio_frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 10:hd_frequency | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    hd_frequency = hd_frequency_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('hdFrequency', 'HD Frequency', None, None, hd_frequency, hd_frequency_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 11:zone | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    zone_raw = decode_int(_data_raw_, running_bit_offset, 8)
    zone = master_dict['ENTERTAINMENT_ZONE'].get(zone_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('zone', 'Zone', None, None, zone, zone_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 12:in_play_queue | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    in_play_queue_raw = decode_int(_data_raw_, running_bit_offset, 2)
    in_play_queue = master_dict['YES_NO'].get(in_play_queue_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('inPlayQueue', 'In play queue', None, None, in_play_queue, in_play_queue_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 13:locked | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    locked_raw = decode_int(_data_raw_, running_bit_offset, 2)
    locked = master_dict['YES_NO'].get(locked_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('locked', 'Locked', None, None, locked, locked_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 14:reserved_ | Offset: , Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', None, None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 15:artist_name | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    artist_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    artist_name = artist_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('artistName', 'Artist Name', None, None, artist_name, artist_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 16:album_name | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    album_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    album_name = album_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('albumName', 'Album Name', None, None, album_name, album_name_raw, None, FieldTypes.STRING_LAU, False))
    

    # 17:station_name | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    station_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    station_name = station_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('stationName', 'Station Name', None, None, station_name, station_name_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_130570(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130570."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # source | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("source")

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
    # number | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("number")

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
    # id | Offset: 16, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("id")

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
    # type | Offset: 48, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("type")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_ENTERTAINMENT_TYPE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # name | Offset: 56, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

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
    # track | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("track")

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
        raise ValueError("Cant encode this message, 'Track' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Track' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Track' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # station | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("station")

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
        raise ValueError("Cant encode this message, 'Station' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # favorite | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("favorite")

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
        raise ValueError("Cant encode this message, 'Favorite' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Favorite' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Favorite' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # radioFrequency | Offset: , Length: 32, Resolution: 10, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("radioFrequency")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 10):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 10)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, False, 10)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Radio Frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Radio Frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Radio Frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # hdFrequency | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("hdFrequency")

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
        raise ValueError("Cant encode this message, 'HD Frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'HD Frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'HD Frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone | Offset: , Length: 8, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone")

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
        raise ValueError("Cant encode this message, 'Zone' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # inPlayQueue | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("inPlayQueue")

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
        raise ValueError("Cant encode this message, 'In play queue' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'In play queue' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'In play queue' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # locked | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("locked")

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
        raise ValueError("Cant encode this message, 'Locked' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Locked' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Locked' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 4, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

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
    # artistName | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("artistName")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Artist Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Artist Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Artist Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # albumName | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("albumName")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Album Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Album Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Album Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationName | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationName")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
