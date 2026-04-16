# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129538() -> bool:
    """Return True if PGN 129538 is a fast PGN."""
    return True
def decode_pgn_129538(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129538."""
    nmea2000Message = NMEA2000Message(PGN=129538, id='gnssControlStatus', description='GNSS Control Status')
    running_bit_offset = 0
    # 1:sv_elevation_mask | Offset: 0, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sv_elevation_mask = sv_elevation_mask_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('svElevationMask', 'SV Elevation Mask', "Will not use SV below this elevation", 'rad', sv_elevation_mask, sv_elevation_mask_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:pdop_mask | Offset: 16, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    pdop_mask = pdop_mask_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('pdopMask', 'PDOP Mask', "Will not report position above this PDOP", None, pdop_mask, pdop_mask_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:pdop_switch | Offset: 32, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    pdop_switch = pdop_switch_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('pdopSwitch', 'PDOP Switch', "Will report 2D position above this PDOP", None, pdop_switch, pdop_switch_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:snr_mask | Offset: 48, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    snr_mask = snr_mask_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('snrMask', 'SNR Mask', "Will not use SV below this SNR", 'dB', snr_mask, snr_mask_raw, PhysicalQuantities.SIGNAL_TO_NOISE_RATIO, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:gnss_mode__desired_ | Offset: 64, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    gnss_mode__desired__raw = decode_int(_data_raw_, running_bit_offset, 3)
    gnss_mode__desired_ = master_dict['GNSS_MODE'].get(gnss_mode__desired__raw, None)
    nmea2000Message.fields.append(NMEA2000Field('gnssModeDesired', 'GNSS Mode (desired)', None, None, gnss_mode__desired_, gnss_mode__desired__raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 6:dgnss_mode__desired_ | Offset: 67, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 67
    dgnss_mode__desired__raw = decode_int(_data_raw_, running_bit_offset, 3)
    dgnss_mode__desired_ = master_dict['DGNSS_MODE'].get(dgnss_mode__desired__raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dgnssModeDesired', 'DGNSS Mode (desired)', None, None, dgnss_mode__desired_, dgnss_mode__desired__raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 7:position_velocity_filter | Offset: 70, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 70
    position_velocity_filter_raw = decode_int(_data_raw_, running_bit_offset, 2)
    position_velocity_filter = master_dict['YES_NO'].get(position_velocity_filter_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('positionVelocityFilter', 'Position/Velocity Filter', None, None, position_velocity_filter, position_velocity_filter_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 8:max_correction_age | Offset: 72, Length: 16, Signed: False Resolution: 0.01, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    max_correction_age = max_correction_age_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('maxCorrectionAge', 'Max Correction Age', None, 's', max_correction_age, max_correction_age_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 9:antenna_altitude_for_2d_mode | Offset: 88, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    antenna_altitude_for_2d_mode = antenna_altitude_for_2d_mode_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('antennaAltitudeFor2dMode', 'Antenna Altitude for 2D Mode', None, 'm', antenna_altitude_for_2d_mode, antenna_altitude_for_2d_mode_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 10:use_antenna_altitude_for_2d_mode | Offset: 120, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    use_antenna_altitude_for_2d_mode_raw = decode_int(_data_raw_, running_bit_offset, 2)
    use_antenna_altitude_for_2d_mode = master_dict['YES_NO'].get(use_antenna_altitude_for_2d_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('useAntennaAltitudeFor2dMode', 'Use Antenna Altitude for 2D Mode', None, None, use_antenna_altitude_for_2d_mode, use_antenna_altitude_for_2d_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:reserved_122 | Offset: 122, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 122
    reserved_122 = reserved_122_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_122', 'Reserved', None, None, reserved_122, reserved_122_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    return nmea2000Message

def encode_pgn_129538(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129538."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # svElevationMask | Offset: 0, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("svElevationMask")

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
        raise ValueError("Cant encode this message, 'SV Elevation Mask' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SV Elevation Mask' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SV Elevation Mask' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pdopMask | Offset: 16, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pdopMask")

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
        raise ValueError("Cant encode this message, 'PDOP Mask' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PDOP Mask' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PDOP Mask' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pdopSwitch | Offset: 32, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pdopSwitch")

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
        raise ValueError("Cant encode this message, 'PDOP Switch' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'PDOP Switch' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'PDOP Switch' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # snrMask | Offset: 48, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("snrMask")

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
        raise ValueError("Cant encode this message, 'SNR Mask' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SNR Mask' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SNR Mask' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # gnssModeDesired | Offset: 64, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("gnssModeDesired")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_GNSS_MODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'GNSS Mode (desired)' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'GNSS Mode (desired)' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'GNSS Mode (desired)' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dgnssModeDesired | Offset: 67, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 67
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dgnssModeDesired")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DGNSS_MODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DGNSS Mode (desired)' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DGNSS Mode (desired)' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DGNSS Mode (desired)' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # positionVelocityFilter | Offset: 70, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 70
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionVelocityFilter")

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
        raise ValueError("Cant encode this message, 'Position/Velocity Filter' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position/Velocity Filter' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position/Velocity Filter' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # maxCorrectionAge | Offset: 72, Length: 16, Resolution: 0.01, Field Type: DURATION
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("maxCorrectionAge")

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
        raise ValueError("Cant encode this message, 'Max Correction Age' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Max Correction Age' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Max Correction Age' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # antennaAltitudeFor2dMode | Offset: 88, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("antennaAltitudeFor2dMode")

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
        raise ValueError("Cant encode this message, 'Antenna Altitude for 2D Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Antenna Altitude for 2D Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Antenna Altitude for 2D Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # useAntennaAltitudeFor2dMode | Offset: 120, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("useAntennaAltitudeFor2dMode")

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
        raise ValueError("Cant encode this message, 'Use Antenna Altitude for 2D Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Use Antenna Altitude for 2D Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Use Antenna Altitude for 2D Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_122 | Offset: 122, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 122
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_122")

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
    return data_raw.to_bytes(16, byteorder="little")
