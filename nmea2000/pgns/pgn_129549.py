# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129549() -> bool:
    """Return True if PGN 129549 is a fast PGN."""
    return True
def decode_pgn_129549(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129549."""
    nmea2000Message = NMEA2000Message(PGN=129549, id='dgnssCorrections', description='DGNSS Corrections')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:reference_station_id | Offset: 8, Length: 12, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    reference_station_id = reference_station_id_raw = decode_number(_data_raw_, running_bit_offset, 12, False, 1, 0, 4092)
    nmea2000Message.fields.append(NMEA2000Field('referenceStationId', 'Reference Station ID', None, None, reference_station_id, reference_station_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 12

    # 3:reference_station_type | Offset: 20, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    reference_station_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    reference_station_type = master_dict['GNS'].get(reference_station_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('referenceStationType', 'Reference Station Type', None, None, reference_station_type, reference_station_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 4:time_of_corrections | Offset: 24, Length: 16, Signed: False Resolution: 0.1, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    time_of_corrections = time_of_corrections_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('timeOfCorrections', 'Time of corrections', None, 's', time_of_corrections, time_of_corrections_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 5:station_health | Offset: 40, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    station_health_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_health = master_dict['STATION_HEALTH'].get(station_health_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('stationHealth', 'Station Health', None, None, station_health, station_health_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 6:reserved_44 | Offset: 44, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 44
    reserved_44 = reserved_44_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_44', 'Reserved', None, None, reserved_44, reserved_44_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 7:satellite_id | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    satellite_id = satellite_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('satelliteId', 'Satellite ID', None, None, satellite_id, satellite_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:prc | Offset: 56, Length: 32, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    prc = prc_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.0001, -214748.3647, 214748.3644)
    nmea2000Message.fields.append(NMEA2000Field('prc', 'PRC', None, 'm', prc, prc_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 9:rrc | Offset: 88, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    rrc = rrc_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.2767, 3.2764)
    nmea2000Message.fields.append(NMEA2000Field('rrc', 'RRC', None, 'm/s', rrc, rrc_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 10:udre | Offset: 104, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 104
    udre = udre_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('udre', 'UDRE', None, 'm', udre, udre_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 11:iod | Offset: 120, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    iod = iod_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('iod', 'IOD', None, None, iod, iod_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_129549(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129549."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
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
    # referenceStationId | Offset: 8, Length: 12, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("referenceStationId")

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
    # referenceStationType | Offset: 20, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 20
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("referenceStationType")

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
    # timeOfCorrections | Offset: 24, Length: 16, Resolution: 0.1, Field Type: DURATION
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeOfCorrections")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.1)):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.1)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 16, False, 0.1)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time of corrections' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time of corrections' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time of corrections' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationHealth | Offset: 40, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationHealth")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_STATION_HEALTH(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station Health' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station Health' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station Health' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_44 | Offset: 44, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 44
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_44")

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
    # satelliteId | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("satelliteId")

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
        raise ValueError("Cant encode this message, 'Satellite ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Satellite ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Satellite ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # prc | Offset: 56, Length: 32, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("prc")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 0.0001)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'PRC' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PRC' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PRC' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rrc | Offset: 88, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rrc")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'RRC' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'RRC' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'RRC' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # udre | Offset: 104, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("udre")

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
        raise ValueError("Cant encode this message, 'UDRE' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'UDRE' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'UDRE' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # iod | Offset: 120, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("iod")

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
        raise ValueError("Cant encode this message, 'IOD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'IOD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'IOD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(16, byteorder="little")
