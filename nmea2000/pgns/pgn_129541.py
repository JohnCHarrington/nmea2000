# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129541() -> bool:
    """Return True if PGN 129541 is a fast PGN."""
    return True
def decode_pgn_129541(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129541."""
    nmea2000Message = NMEA2000Message(PGN=129541, id='gpsAlmanacData', description='GPS Almanac Data')
    running_bit_offset = 0
    # 1:prn | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    prn = prn_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('prn', 'PRN', None, None, prn, prn_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:gps_week_number | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    gps_week_number = gps_week_number_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('gpsWeekNumber', 'GPS Week number', None, None, gps_week_number, gps_week_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:sv_health_bits | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    sv_health_bits = sv_health_bits_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 8))
    nmea2000Message.fields.append(NMEA2000Field('svHealthBits', 'SV Health Bits', None, None, sv_health_bits, sv_health_bits_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 8

    # 4:eccentricity | Offset: 32, Length: 16, Signed: False Resolution: 4.76837e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    eccentricity = eccentricity_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 4.76837e-07, 0, 0.0312480926513672)
    nmea2000Message.fields.append(NMEA2000Field('eccentricity', 'Eccentricity', "'e' in table 20-VI in ICD-GPS-200", 'm/m', eccentricity, eccentricity_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:almanac_reference_time | Offset: 48, Length: 8, Signed: False Resolution: 4096, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    almanac_reference_time = almanac_reference_time_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 4096, 0, 1032192)
    nmea2000Message.fields.append(NMEA2000Field('almanacReferenceTime', 'Almanac Reference Time', "'t oa' in table 20-VI in ICD-GPS-200", 's', almanac_reference_time, almanac_reference_time_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:inclination_angle | Offset: 56, Length: 16, Signed: True Resolution: 1.90735e-06, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    inclination_angle = inclination_angle_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1.90735e-06, -0.0624980926513672, 0.0624923706054688)
    nmea2000Message.fields.append(NMEA2000Field('inclinationAngle', 'Inclination Angle', "'delta i' in table 20-VI in ICD-GPS-200", 'semi-circle', inclination_angle, inclination_angle_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:rate_of_right_ascension | Offset: 72, Length: 16, Signed: True Resolution: 3.63798e-12, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    rate_of_right_ascension = rate_of_right_ascension_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 3.63798e-12, -1.19205651571974e-07, 1.19194737635553e-07)
    nmea2000Message.fields.append(NMEA2000Field('rateOfRightAscension', 'Rate of Right Ascension', "'OMEGADOT' in table 20-VI in ICD-GPS-200", 'semi-circle/s', rate_of_right_ascension, rate_of_right_ascension_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:root_of_semi_major_axis | Offset: 88, Length: 24, Signed: False Resolution: 0.000488281, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    root_of_semi_major_axis = root_of_semi_major_axis_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 0.000488281, 0, 8191.998046875)
    nmea2000Message.fields.append(NMEA2000Field('rootOfSemiMajorAxis', 'Root of Semi-major Axis', "'(A)^0.5' in table 20-VI in ICD-GPS-200", 'sqrt(m)', root_of_semi_major_axis, root_of_semi_major_axis_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 9:argument_of_perigee | Offset: 112, Length: 24, Signed: True Resolution: 1.19209e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    argument_of_perigee = argument_of_perigee_raw = decode_number(_data_raw_, running_bit_offset, 24, True, 1.19209e-07, -0.99999988079071, 0.999999523162842)
    nmea2000Message.fields.append(NMEA2000Field('argumentOfPerigee', 'Argument of Perigee', "'(OMEGA)0' in table 20-VI in ICD-GPS-200", 'semi-circle', argument_of_perigee, argument_of_perigee_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 10:longitude_of_ascension_node | Offset: 136, Length: 24, Signed: True Resolution: 1.19209e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    longitude_of_ascension_node = longitude_of_ascension_node_raw = decode_number(_data_raw_, running_bit_offset, 24, True, 1.19209e-07, -0.99999988079071, 0.999999523162842)
    nmea2000Message.fields.append(NMEA2000Field('longitudeOfAscensionNode', 'Longitude of Ascension Node', "'small-omega' in table 20-VI in ICD-GPS-200", 'semi-circle', longitude_of_ascension_node, longitude_of_ascension_node_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 11:mean_anomaly | Offset: 160, Length: 24, Signed: True Resolution: 1.19209e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    mean_anomaly = mean_anomaly_raw = decode_number(_data_raw_, running_bit_offset, 24, True, 1.19209e-07, -0.99999988079071, 0.999999523162842)
    nmea2000Message.fields.append(NMEA2000Field('meanAnomaly', 'Mean Anomaly', "'M 0' in table 20-VI in ICD-GPS-200", 'semi-circle', mean_anomaly, mean_anomaly_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 12:clock_parameter_1 | Offset: 184, Length: 11, Signed: True Resolution: 9.53674e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 184
    clock_parameter_1 = clock_parameter_1_raw = decode_number(_data_raw_, running_bit_offset, 11, True, 9.53674e-07, -0.000975608825683594, 0.000972747802734375)
    nmea2000Message.fields.append(NMEA2000Field('clockParameter1', 'Clock Parameter 1', "'a f0' in table 20-VI in ICD-GPS-200", 's', clock_parameter_1, clock_parameter_1_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 11

    # 13:clock_parameter_2 | Offset: 195, Length: 11, Signed: True Resolution: 3.63798e-12, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 195
    clock_parameter_2 = clock_parameter_2_raw = decode_number(_data_raw_, running_bit_offset, 11, True, 3.63798e-12, -3.72165231965482e-09, 3.71073838323355e-09)
    nmea2000Message.fields.append(NMEA2000Field('clockParameter2', 'Clock Parameter 2', "'a f1' in table 20-VI in ICD-GPS-200", 's/s', clock_parameter_2, clock_parameter_2_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 11

    # 14:reserved_206 | Offset: 206, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 206
    reserved_206 = reserved_206_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_206', 'Reserved', None, None, reserved_206, reserved_206_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    return nmea2000Message

def encode_pgn_129541(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129541."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # prn | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("prn")

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
        raise ValueError("Cant encode this message, 'PRN' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PRN' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PRN' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # gpsWeekNumber | Offset: 8, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("gpsWeekNumber")

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
        raise ValueError("Cant encode this message, 'GPS Week number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'GPS Week number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'GPS Week number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # svHealthBits | Offset: 24, Length: 8, Resolution: 1, Field Type: BINARY
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("svHealthBits")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'SV Health Bits' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SV Health Bits' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SV Health Bits' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # eccentricity | Offset: 32, Length: 16, Resolution: 4.76837e-07, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("eccentricity")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 4.76837e-07):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 4.76837e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 4.76837e-07)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Eccentricity' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Eccentricity' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Eccentricity' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # almanacReferenceTime | Offset: 48, Length: 8, Resolution: 4096, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("almanacReferenceTime")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 4096):
        field_value = encode_number_raw(field.raw_value, 8, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, False, 4096)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, False, 4096)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Almanac Reference Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Almanac Reference Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Almanac Reference Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # inclinationAngle | Offset: 56, Length: 16, Resolution: 1.90735e-06, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("inclinationAngle")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1.90735e-06):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 1.90735e-06)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 1.90735e-06)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Inclination Angle' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Inclination Angle' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Inclination Angle' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rateOfRightAscension | Offset: 72, Length: 16, Resolution: 3.63798e-12, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rateOfRightAscension")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 3.63798e-12):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 3.63798e-12)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 3.63798e-12)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rate of Right Ascension' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rate of Right Ascension' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rate of Right Ascension' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rootOfSemiMajorAxis | Offset: 88, Length: 24, Resolution: 0.000488281, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rootOfSemiMajorAxis")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.000488281):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 0.000488281)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 0.000488281)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Root of Semi-major Axis' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Root of Semi-major Axis' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Root of Semi-major Axis' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # argumentOfPerigee | Offset: 112, Length: 24, Resolution: 1.19209e-07, Field Type: NUMBER
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("argumentOfPerigee")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1.19209e-07):
        field_value = encode_number_raw(field.raw_value, 24, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, True, 1.19209e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, True, 1.19209e-07)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Argument of Perigee' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Argument of Perigee' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Argument of Perigee' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitudeOfAscensionNode | Offset: 136, Length: 24, Resolution: 1.19209e-07, Field Type: NUMBER
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitudeOfAscensionNode")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1.19209e-07):
        field_value = encode_number_raw(field.raw_value, 24, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, True, 1.19209e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, True, 1.19209e-07)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitude of Ascension Node' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitude of Ascension Node' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitude of Ascension Node' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # meanAnomaly | Offset: 160, Length: 24, Resolution: 1.19209e-07, Field Type: NUMBER
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("meanAnomaly")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1.19209e-07):
        field_value = encode_number_raw(field.raw_value, 24, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, True, 1.19209e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, True, 1.19209e-07)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Mean Anomaly' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Mean Anomaly' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Mean Anomaly' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # clockParameter1 | Offset: 184, Length: 11, Resolution: 9.53674e-07, Field Type: NUMBER
    running_bit_offset = 184
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("clockParameter1")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 9.53674e-07):
        field_value = encode_number_raw(field.raw_value, 11, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 11, True, 9.53674e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 11, True, 9.53674e-07)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Clock Parameter 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Clock Parameter 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Clock Parameter 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # clockParameter2 | Offset: 195, Length: 11, Resolution: 3.63798e-12, Field Type: NUMBER
    running_bit_offset = 195
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("clockParameter2")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 3.63798e-12):
        field_value = encode_number_raw(field.raw_value, 11, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 11, True, 3.63798e-12)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 11, True, 3.63798e-12)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Clock Parameter 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Clock Parameter 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Clock Parameter 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_206 | Offset: 206, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 206
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_206")

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
    return data_raw.to_bytes(26, byteorder="little")
