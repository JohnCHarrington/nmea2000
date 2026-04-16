# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129029() -> bool:
    """Return True if PGN 129029 is a fast PGN."""
    return True
def decode_pgn_129029(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129029."""
    nmea2000Message = NMEA2000Message(PGN=129029, id='gnssPositionData', description='GNSS Position Data', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:date | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    date_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    date = decode_date(date_raw)
    nmea2000Message.fields.append(NMEA2000Field('date', 'Date', None, 'd', date, date_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 3:time | Offset: 24, Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time = decode_time(time_raw)
    nmea2000Message.fields.append(NMEA2000Field('time', 'Time', "Seconds since midnight", 's', time, time_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 4:latitude | Offset: 56, Length: 64, Signed: True Resolution: 1e-16, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    latitude = latitude_raw = decode_number(_data_raw_, running_bit_offset, 64, True, 1e-16, -90, 90)
    nmea2000Message.fields.append(NMEA2000Field('latitude', 'Latitude', None, 'deg', latitude, latitude_raw, PhysicalQuantities.GEOGRAPHICAL_LATITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 64

    # 5:longitude | Offset: 120, Length: 64, Signed: True Resolution: 1e-16, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    longitude = longitude_raw = decode_number(_data_raw_, running_bit_offset, 64, True, 1e-16, -180, 180)
    nmea2000Message.fields.append(NMEA2000Field('longitude', 'Longitude', None, 'deg', longitude, longitude_raw, PhysicalQuantities.GEOGRAPHICAL_LONGITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 64

    # 6:altitude | Offset: 184, Length: 64, Signed: True Resolution: 1e-06, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 184
    altitude = altitude_raw = decode_number(_data_raw_, running_bit_offset, 64, True, 1e-06, -9223372036854.78, 9223372036854.78)
    nmea2000Message.fields.append(NMEA2000Field('altitude', 'Altitude', "Altitude referenced to WGS-84", 'm', altitude, altitude_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 64

    # 7:gnss_type | Offset: 248, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 248
    gnss_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    gnss_type = master_dict['GNS'].get(gnss_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('gnssType', 'GNSS type', None, None, gnss_type, gnss_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 8:method | Offset: 252, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 252
    method_raw = decode_int(_data_raw_, running_bit_offset, 4)
    method = master_dict['GNS_METHOD'].get(method_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('method', 'Method', None, None, method, method_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 9:integrity | Offset: 256, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 256
    integrity_raw = decode_int(_data_raw_, running_bit_offset, 2)
    integrity = master_dict['GNS_INTEGRITY'].get(integrity_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('integrity', 'Integrity', None, None, integrity, integrity_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 10:reserved_258 | Offset: 258, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 258
    reserved_258 = reserved_258_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_258', 'Reserved', None, None, reserved_258, reserved_258_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 11:number_of_svs | Offset: 264, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 264
    number_of_svs = number_of_svs_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfSvs', 'Number of SVs', "Number of satellites used in solution", None, number_of_svs, number_of_svs_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 12:hdop | Offset: 272, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 272
    hdop = hdop_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('hdop', 'HDOP', "Horizontal dilution of precision", None, hdop, hdop_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 13:pdop | Offset: 288, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 288
    pdop = pdop_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('pdop', 'PDOP', "Positional dilution of precision", None, pdop, pdop_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 14:geoidal_separation | Offset: 304, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 304
    geoidal_separation = geoidal_separation_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('geoidalSeparation', 'Geoidal Separation', "Geoidal Separation", 'm', geoidal_separation, geoidal_separation_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 15:reference_stations | Offset: 336, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 336
    reference_stations = reference_stations_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('referenceStations', 'Reference Stations', "Number of reference stations", None, reference_stations, reference_stations_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 16:reference_station_type | Offset: 344, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 344
    _repeating_field_set_1_offset = running_bit_offset
    reference_station_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    reference_station_type = master_dict['GNS'].get(reference_station_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('referenceStationType', 'Reference Station Type', None, None, reference_station_type, reference_station_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 17:reference_station_id | Offset: 348, Length: 12, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 348
    reference_station_id = reference_station_id_raw = decode_number(_data_raw_, running_bit_offset, 12, False, 1, 0, 4092)
    nmea2000Message.fields.append(NMEA2000Field('referenceStationId', 'Reference Station ID', None, None, reference_station_id, reference_station_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 12

    # 18:age_of_dgnss_corrections | Offset: 360, Length: 16, Signed: False Resolution: 0.01, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 360
    age_of_dgnss_corrections = age_of_dgnss_corrections_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('ageOfDgnssCorrections', 'Age of DGNSS Corrections', None, 's', age_of_dgnss_corrections, age_of_dgnss_corrections_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(reference_stations_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        reference_station_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
        reference_station_type = master_dict['GNS'].get(reference_station_type_raw, None)
        running_bit_offset += 4
        repeating_entry["referenceStationType"] = _repeating_entry_value(reference_station_type, reference_station_type_raw)
    
        reference_station_id = reference_station_id_raw = decode_number(_data_raw_, running_bit_offset, 12, False, 1, 0, 4092)
        running_bit_offset += 12
        repeating_entry["referenceStationId"] = _repeating_entry_value(reference_station_id, reference_station_id_raw)
    
        age_of_dgnss_corrections = age_of_dgnss_corrections_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
        running_bit_offset += 16
        repeating_entry["ageOfDgnssCorrections"] = _repeating_entry_value(age_of_dgnss_corrections, age_of_dgnss_corrections_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "referenceStationType",
                "referenceStationId",
                "ageOfDgnssCorrections",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_129029(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129029."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "referenceStationType",
        "referenceStationId",
        "ageOfDgnssCorrections",
    ))
    # sid | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sid")

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
        raise ValueError("Cant encode this message, 'SID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # date | Offset: 8, Length: 16, Resolution: 1, Field Type: DATE
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("date")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Date' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Date' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Date' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # time | Offset: 24, Length: 32, Resolution: 0.0001, Field Type: TIME
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("time")

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
        raise ValueError("Cant encode this message, 'Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # latitude | Offset: 56, Length: 64, Resolution: 1e-16, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("latitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-16):
        field_value = encode_number_raw(field.raw_value, 64, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 64, True, 1e-16)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 64, True, 1e-16)
    field_bit_length = 64
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Latitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Latitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Latitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitude | Offset: 120, Length: 64, Resolution: 1e-16, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-16):
        field_value = encode_number_raw(field.raw_value, 64, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 64, True, 1e-16)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 64, True, 1e-16)
    field_bit_length = 64
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # altitude | Offset: 184, Length: 64, Resolution: 1e-06, Field Type: NUMBER
    running_bit_offset = 184
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("altitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-06):
        field_value = encode_number_raw(field.raw_value, 64, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 64, True, 1e-06)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 64, True, 1e-06)
    field_bit_length = 64
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Altitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Altitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Altitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # gnssType | Offset: 248, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 248
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("gnssType")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GNS(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'GNSS type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'GNSS type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'GNSS type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # method | Offset: 252, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 252
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("method")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GNS_METHOD(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Method' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Method' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Method' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # integrity | Offset: 256, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 256
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("integrity")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GNS_INTEGRITY(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Integrity' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Integrity' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Integrity' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_258 | Offset: 258, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 258
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_258")

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
    # numberOfSvs | Offset: 264, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 264
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfSvs")

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
        raise ValueError("Cant encode this message, 'Number of SVs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of SVs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of SVs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # hdop | Offset: 272, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 272
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("hdop")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'HDOP' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'HDOP' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'HDOP' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pdop | Offset: 288, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 288
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pdop")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.01)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'PDOP' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PDOP' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PDOP' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # geoidalSeparation | Offset: 304, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 304
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("geoidalSeparation")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 0.01)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 0.01)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Geoidal Separation' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Geoidal Separation' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Geoidal Separation' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # referenceStations | Offset: 336, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 336
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("referenceStations")

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
        raise ValueError("Cant encode this message, 'Reference Stations' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Reference Stations' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Reference Stations' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 344
    for repeating_entry in repeating_field_set_1_entries:
        # referenceStationType | Offset: 344, Length: 4, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("referenceStationType")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reference Station Type'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_GNS(field.value)
        field_bit_length = 4
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Reference Station Type' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Reference Station Type' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Reference Station Type' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # referenceStationId | Offset: 348, Length: 12, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("referenceStationId")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reference Station ID'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
            field_value = encode_number_raw(field.raw_value, 12, False)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 12, False, 1)
        else:
            assert field.value is None or isinstance(field.value, (int, float))
            field_value = encode_number(field.value, 12, False, 1)
        field_bit_length = 12
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Reference Station ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Reference Station ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Reference Station ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # ageOfDgnssCorrections | Offset: 360, Length: 16, Resolution: 0.01, Field Type: DURATION
        field = repeating_entry.get("ageOfDgnssCorrections")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Age of DGNSS Corrections'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.01)):
            field_value = encode_number_raw(field.raw_value, 16, False)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 16, False, 0.01)
        elif isinstance(field.value, (int, float)):
            field_value = encode_number(field.value, 16, False, 0.01)
        else:
            assert field.value is None or isinstance(field.value, time)
            field_value = encode_time(field.value, 16)
        field_bit_length = 16
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Age of DGNSS Corrections' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Age of DGNSS Corrections' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Age of DGNSS Corrections' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
