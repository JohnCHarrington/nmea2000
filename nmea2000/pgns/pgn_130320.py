# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130320() -> bool:
    """Return True if PGN 130320 is a fast PGN."""
    return True
def decode_pgn_130320(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130320."""
    nmea2000Message = NMEA2000Message(PGN=130320, id='tideStationData', description='Tide Station Data', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:mode | Offset: 0, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    mode_raw = decode_int(_data_raw_, running_bit_offset, 4)
    mode = master_dict['RESIDUAL_MODE'].get(mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('mode', 'Mode', None, None, mode, mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 2:tide_tendency | Offset: 4, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 4
    tide_tendency_raw = decode_int(_data_raw_, running_bit_offset, 2)
    tide_tendency = master_dict['TIDE'].get(tide_tendency_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('tideTendency', 'Tide Tendency', None, None, tide_tendency, tide_tendency_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:reserved_6 | Offset: 6, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 6
    reserved_6 = reserved_6_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_6', 'Reserved', None, None, reserved_6, reserved_6_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 4:measurement_date | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    measurement_date_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    measurement_date = decode_date(measurement_date_raw)
    nmea2000Message.fields.append(NMEA2000Field('measurementDate', 'Measurement Date', None, 'd', measurement_date, measurement_date_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 5:measurement_time | Offset: 24, Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    measurement_time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    measurement_time = decode_time(measurement_time_raw)
    nmea2000Message.fields.append(NMEA2000Field('measurementTime', 'Measurement Time', "Seconds since midnight", 's', measurement_time, measurement_time_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 6:station_latitude | Offset: 56, Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    station_latitude = station_latitude_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -90, 90)
    nmea2000Message.fields.append(NMEA2000Field('stationLatitude', 'Station Latitude', None, 'deg', station_latitude, station_latitude_raw, PhysicalQuantities.GEOGRAPHICAL_LATITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 7:station_longitude | Offset: 88, Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    station_longitude = station_longitude_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -180, 180)
    nmea2000Message.fields.append(NMEA2000Field('stationLongitude', 'Station Longitude', None, 'deg', station_longitude, station_longitude_raw, PhysicalQuantities.GEOGRAPHICAL_LONGITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 8:tide_level | Offset: 120, Length: 16, Signed: True Resolution: 0.001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    tide_level = tide_level_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.001, -32.767, 32.764)
    nmea2000Message.fields.append(NMEA2000Field('tideLevel', 'Tide Level', "Relative to MLLW", 'm', tide_level, tide_level_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:tide_level_standard_deviation | Offset: 136, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    tide_level_standard_deviation = tide_level_standard_deviation_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('tideLevelStandardDeviation', 'Tide Level standard deviation', None, 'm', tide_level_standard_deviation, tide_level_standard_deviation_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 10:station_id | Offset: 152, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 152
    station_id_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    station_id = station_id_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('stationId', 'Station ID', None, None, station_id, station_id_raw, None, FieldTypes.STRING_LAU, True))
    

    # 11:station_name | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    station_name_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    station_name = station_name_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('stationName', 'Station Name', None, None, station_name, station_name_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_130320(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130320."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # mode | Offset: 0, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_RESIDUAL_MODE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tideTendency | Offset: 4, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 4
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tideTendency")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_TIDE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Tide Tendency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tide Tendency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tide Tendency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_6 | Offset: 6, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 6
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_6")

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
    # measurementDate | Offset: 8, Length: 16, Resolution: 1, Field Type: DATE
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("measurementDate")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Measurement Date' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Measurement Date' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Measurement Date' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # measurementTime | Offset: 24, Length: 32, Resolution: 0.0001, Field Type: TIME
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("measurementTime")

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
        raise ValueError("Cant encode this message, 'Measurement Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Measurement Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Measurement Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationLatitude | Offset: 56, Length: 32, Resolution: 1e-07, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationLatitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station Latitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station Latitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station Latitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationLongitude | Offset: 88, Length: 32, Resolution: 1e-07, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationLongitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station Longitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station Longitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station Longitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tideLevel | Offset: 120, Length: 16, Resolution: 0.001, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tideLevel")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Tide Level' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tide Level' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tide Level' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tideLevelStandardDeviation | Offset: 136, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tideLevelStandardDeviation")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Tide Level standard deviation' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tide Level standard deviation' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tide Level standard deviation' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationId | Offset: 152, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationId")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station ID' exceeds the encoded bit length")
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
