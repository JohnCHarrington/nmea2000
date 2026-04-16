# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127237() -> bool:
    """Return True if PGN 127237 is a fast PGN."""
    return True
def decode_pgn_127237(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127237."""
    nmea2000Message = NMEA2000Message(PGN=127237, id='headingTrackControl', description='Heading/Track control', ttl=timedelta(milliseconds=250))
    running_bit_offset = 0
    # 1:rudder_limit_exceeded | Offset: 0, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    rudder_limit_exceeded_raw = decode_int(_data_raw_, running_bit_offset, 2)
    rudder_limit_exceeded = master_dict['YES_NO'].get(rudder_limit_exceeded_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('rudderLimitExceeded', 'Rudder Limit Exceeded', None, None, rudder_limit_exceeded, rudder_limit_exceeded_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 2:off_heading_limit_exceeded | Offset: 2, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 2
    off_heading_limit_exceeded_raw = decode_int(_data_raw_, running_bit_offset, 2)
    off_heading_limit_exceeded = master_dict['YES_NO'].get(off_heading_limit_exceeded_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('offHeadingLimitExceeded', 'Off-Heading Limit Exceeded', None, None, off_heading_limit_exceeded, off_heading_limit_exceeded_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:off_track_limit_exceeded | Offset: 4, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 4
    off_track_limit_exceeded_raw = decode_int(_data_raw_, running_bit_offset, 2)
    off_track_limit_exceeded = master_dict['YES_NO'].get(off_track_limit_exceeded_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('offTrackLimitExceeded', 'Off-Track Limit Exceeded', None, None, off_track_limit_exceeded, off_track_limit_exceeded_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:override | Offset: 6, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 6
    override_raw = decode_int(_data_raw_, running_bit_offset, 2)
    override = master_dict['YES_NO'].get(override_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('override', 'Override', None, None, override, override_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:steering_mode | Offset: 8, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    steering_mode_raw = decode_int(_data_raw_, running_bit_offset, 3)
    steering_mode = master_dict['STEERING_MODE'].get(steering_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('steeringMode', 'Steering Mode', None, None, steering_mode, steering_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 6:turn_mode | Offset: 11, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    turn_mode_raw = decode_int(_data_raw_, running_bit_offset, 3)
    turn_mode = master_dict['TURN_MODE'].get(turn_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('turnMode', 'Turn Mode', None, None, turn_mode, turn_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 7:heading_reference | Offset: 14, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 14
    heading_reference_raw = decode_int(_data_raw_, running_bit_offset, 2)
    heading_reference = master_dict['DIRECTION_REFERENCE'].get(heading_reference_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('headingReference', 'Heading Reference', None, None, heading_reference, heading_reference_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 8:reserved_16 | Offset: 16, Length: 5, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    reserved_16 = reserved_16_raw = decode_int(_data_raw_, running_bit_offset, 5)
    nmea2000Message.fields.append(NMEA2000Field('reserved_16', 'Reserved', None, None, reserved_16, reserved_16_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 5

    # 9:commanded_rudder_direction | Offset: 21, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 21
    commanded_rudder_direction_raw = decode_int(_data_raw_, running_bit_offset, 3)
    commanded_rudder_direction = master_dict['DIRECTION_RUDDER'].get(commanded_rudder_direction_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('commandedRudderDirection', 'Commanded Rudder Direction', None, None, commanded_rudder_direction, commanded_rudder_direction_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 10:commanded_rudder_angle | Offset: 24, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    commanded_rudder_angle = commanded_rudder_angle_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('commandedRudderAngle', 'Commanded Rudder Angle', None, 'rad', commanded_rudder_angle, commanded_rudder_angle_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 11:heading_to_steer__course_ | Offset: 40, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    heading_to_steer__course_ = heading_to_steer__course__raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('headingToSteerCourse', 'Heading-To-Steer (Course)', None, 'rad', heading_to_steer__course_, heading_to_steer__course__raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 12:track | Offset: 56, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    track = track_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('track', 'Track', None, 'rad', track, track_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 13:rudder_limit | Offset: 72, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    rudder_limit = rudder_limit_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('rudderLimit', 'Rudder Limit', None, 'rad', rudder_limit, rudder_limit_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 14:off_heading_limit | Offset: 88, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    off_heading_limit = off_heading_limit_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('offHeadingLimit', 'Off-Heading Limit', None, 'rad', off_heading_limit, off_heading_limit_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 15:radius_of_turn_order | Offset: 104, Length: 16, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 104
    radius_of_turn_order = radius_of_turn_order_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1, -32767, 32764)
    nmea2000Message.fields.append(NMEA2000Field('radiusOfTurnOrder', 'Radius of Turn Order', None, 'm', radius_of_turn_order, radius_of_turn_order_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 16:rate_of_turn_order | Offset: 120, Length: 16, Signed: True Resolution: 3.125e-05, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    rate_of_turn_order = rate_of_turn_order_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 3.125e-05, -1.02396875, 1.023875)
    nmea2000Message.fields.append(NMEA2000Field('rateOfTurnOrder', 'Rate of Turn Order', None, 'rad/s', rate_of_turn_order, rate_of_turn_order_raw, PhysicalQuantities.ANGULAR_VELOCITY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 17:off_track_limit | Offset: 136, Length: 16, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    off_track_limit = off_track_limit_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 1, -32767, 32764)
    nmea2000Message.fields.append(NMEA2000Field('offTrackLimit', 'Off-Track Limit', None, 'm', off_track_limit, off_track_limit_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 18:vessel_heading | Offset: 152, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    vessel_heading = vessel_heading_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('vesselHeading', 'Vessel Heading', None, 'rad', vessel_heading, vessel_heading_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_127237(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127237."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # rudderLimitExceeded | Offset: 0, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rudderLimitExceeded")

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
        raise ValueError("Cant encode this message, 'Rudder Limit Exceeded' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rudder Limit Exceeded' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rudder Limit Exceeded' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # offHeadingLimitExceeded | Offset: 2, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 2
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("offHeadingLimitExceeded")

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
        raise ValueError("Cant encode this message, 'Off-Heading Limit Exceeded' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Off-Heading Limit Exceeded' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Off-Heading Limit Exceeded' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # offTrackLimitExceeded | Offset: 4, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 4
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("offTrackLimitExceeded")

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
        raise ValueError("Cant encode this message, 'Off-Track Limit Exceeded' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Off-Track Limit Exceeded' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Off-Track Limit Exceeded' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # override | Offset: 6, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 6
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("override")

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
        raise ValueError("Cant encode this message, 'Override' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Override' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Override' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # steeringMode | Offset: 8, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("steeringMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_STEERING_MODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Steering Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Steering Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Steering Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # turnMode | Offset: 11, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("turnMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_TURN_MODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Turn Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Turn Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Turn Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # headingReference | Offset: 14, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 14
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("headingReference")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DIRECTION_REFERENCE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Heading Reference' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Heading Reference' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Heading Reference' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_16 | Offset: 16, Length: 5, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_16")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 5
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
    # commandedRudderDirection | Offset: 21, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 21
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("commandedRudderDirection")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DIRECTION_RUDDER(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Commanded Rudder Direction' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Commanded Rudder Direction' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Commanded Rudder Direction' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # commandedRudderAngle | Offset: 24, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("commandedRudderAngle")

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
        raise ValueError("Cant encode this message, 'Commanded Rudder Angle' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Commanded Rudder Angle' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Commanded Rudder Angle' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # headingToSteerCourse | Offset: 40, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("headingToSteerCourse")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Heading-To-Steer (Course)' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Heading-To-Steer (Course)' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Heading-To-Steer (Course)' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # track | Offset: 56, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("track")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
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
    # rudderLimit | Offset: 72, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rudderLimit")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rudder Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rudder Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rudder Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # offHeadingLimit | Offset: 88, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("offHeadingLimit")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Off-Heading Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Off-Heading Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Off-Heading Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # radiusOfTurnOrder | Offset: 104, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("radiusOfTurnOrder")

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
        raise ValueError("Cant encode this message, 'Radius of Turn Order' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Radius of Turn Order' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Radius of Turn Order' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rateOfTurnOrder | Offset: 120, Length: 16, Resolution: 3.125e-05, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rateOfTurnOrder")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 3.125e-05):
        field_value = encode_number_raw(field.raw_value, 16, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, True, 3.125e-05)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, True, 3.125e-05)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rate of Turn Order' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rate of Turn Order' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rate of Turn Order' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # offTrackLimit | Offset: 136, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("offTrackLimit")

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
        raise ValueError("Cant encode this message, 'Off-Track Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Off-Track Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Off-Track Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # vesselHeading | Offset: 152, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("vesselHeading")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0001):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0001)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Vessel Heading' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Vessel Heading' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Vessel Heading' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(21, byteorder="little")
