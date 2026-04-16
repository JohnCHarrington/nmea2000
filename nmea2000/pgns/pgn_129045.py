# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129045() -> bool:
    """Return True if PGN 129045 is a fast PGN."""
    return True
def decode_pgn_129045(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129045."""
    nmea2000Message = NMEA2000Message(PGN=129045, id='userDatum', description='User Datum')
    running_bit_offset = 0
    # 1:delta_x | Offset: 0, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    delta_x = delta_x_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('deltaX', 'Delta X', "Delta shift in X axis from WGS 84", 'm', delta_x, delta_x_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 2:delta_y | Offset: 32, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    delta_y = delta_y_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('deltaY', 'Delta Y', "Delta shift in Y axis from WGS 84", 'm', delta_y, delta_y_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 3:delta_z | Offset: 64, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    delta_z = delta_z_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('deltaZ', 'Delta Z', "Delta shift in Z axis from WGS 84", 'm', delta_z, delta_z_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 4:rotation_in_x | Offset: 96, Length: 32, Signed: True Resolution: 1, Field Type: FLOAT, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    rotation_in_x = rotation_in_x_raw = decode_float(_data_raw_, running_bit_offset, 32, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('rotationInX', 'Rotation in X', "Rotational shift in X axis from WGS 84. Rotations presented use the geodetic sign convention.  When looking along the positive axis towards the origin, counter-clockwise rotations are positive.", 'rad', rotation_in_x, rotation_in_x_raw, None, FieldTypes.FLOAT, False))
    running_bit_offset += 32

    # 5:rotation_in_y | Offset: 128, Length: 32, Signed: True Resolution: 1, Field Type: FLOAT, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    rotation_in_y = rotation_in_y_raw = decode_float(_data_raw_, running_bit_offset, 32, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('rotationInY', 'Rotation in Y', "Rotational shift in Y axis from WGS 84. Rotations presented use the geodetic sign convention.  When looking along the positive axis towards the origin, counter-clockwise rotations are positive.", 'rad', rotation_in_y, rotation_in_y_raw, None, FieldTypes.FLOAT, False))
    running_bit_offset += 32

    # 6:rotation_in_z | Offset: 160, Length: 32, Signed: True Resolution: 1, Field Type: FLOAT, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    rotation_in_z = rotation_in_z_raw = decode_float(_data_raw_, running_bit_offset, 32, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('rotationInZ', 'Rotation in Z', "Rotational shift in Z axis from WGS 84. Rotations presented use the geodetic sign convention.  When looking along the positive axis towards the origin, counter-clockwise rotations are positive.", 'rad', rotation_in_z, rotation_in_z_raw, None, FieldTypes.FLOAT, False))
    running_bit_offset += 32

    # 7:scale | Offset: 192, Length: 32, Signed: True Resolution: 1, Field Type: FLOAT, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 192
    scale = scale_raw = decode_float(_data_raw_, running_bit_offset, 32, -3.40282346638529e+38, 3.40282346638529e+38)
    nmea2000Message.fields.append(NMEA2000Field('scale', 'Scale', None, 'ppm', scale, scale_raw, None, FieldTypes.FLOAT, False))
    running_bit_offset += 32

    # 8:ellipsoid_semi_major_axis | Offset: 224, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 224
    ellipsoid_semi_major_axis = ellipsoid_semi_major_axis_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('ellipsoidSemiMajorAxis', 'Ellipsoid Semi-major Axis', "Semi-major axis (a) of the User Datum ellipsoid", 'm', ellipsoid_semi_major_axis, ellipsoid_semi_major_axis_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 9:ellipsoid_flattening_inverse | Offset: 256, Length: 32, Signed: True Resolution: 1, Field Type: FLOAT, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 256
    ellipsoid_flattening_inverse = ellipsoid_flattening_inverse_raw = decode_float(_data_raw_, running_bit_offset, 32, -3.40282346638529e+38, 3.40282346638529e+38)
    nmea2000Message.fields.append(NMEA2000Field('ellipsoidFlatteningInverse', 'Ellipsoid Flattening Inverse', "Flattening (1/f) of the User Datum ellipsoid", None, ellipsoid_flattening_inverse, ellipsoid_flattening_inverse_raw, None, FieldTypes.FLOAT, False))
    running_bit_offset += 32

    # 10:datum_name | Offset: 288, Length: 32, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 288
    datum_name, datum_name_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 32)
    nmea2000Message.fields.append(NMEA2000Field('datumName', 'Datum Name', "4 character code from IHO Publication S-60,Appendices B and C. First three chars are datum ID as per IHO tables. Fourth char is local datum subdivision code.", None, datum_name, datum_name_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_129045(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129045."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # deltaX | Offset: 0, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deltaX")

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
        raise ValueError("Cant encode this message, 'Delta X' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Delta X' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Delta X' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # deltaY | Offset: 32, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deltaY")

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
        raise ValueError("Cant encode this message, 'Delta Y' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Delta Y' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Delta Y' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # deltaZ | Offset: 64, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("deltaZ")

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
        raise ValueError("Cant encode this message, 'Delta Z' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Delta Z' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Delta Z' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rotationInX | Offset: 96, Length: 32, Resolution: 1, Field Type: FLOAT
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rotationInX")

    advance_running_offset = True
    assert field.value is None or isinstance(field.value, (int, float))
    field_value = encode_float(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rotation in X' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rotation in X' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rotation in X' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rotationInY | Offset: 128, Length: 32, Resolution: 1, Field Type: FLOAT
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rotationInY")

    advance_running_offset = True
    assert field.value is None or isinstance(field.value, (int, float))
    field_value = encode_float(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rotation in Y' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rotation in Y' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rotation in Y' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # rotationInZ | Offset: 160, Length: 32, Resolution: 1, Field Type: FLOAT
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rotationInZ")

    advance_running_offset = True
    assert field.value is None or isinstance(field.value, (int, float))
    field_value = encode_float(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Rotation in Z' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rotation in Z' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rotation in Z' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # scale | Offset: 192, Length: 32, Resolution: 1, Field Type: FLOAT
    running_bit_offset = 192
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("scale")

    advance_running_offset = True
    assert field.value is None or isinstance(field.value, (int, float))
    field_value = encode_float(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Scale' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Scale' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Scale' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # ellipsoidSemiMajorAxis | Offset: 224, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 224
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("ellipsoidSemiMajorAxis")

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
        raise ValueError("Cant encode this message, 'Ellipsoid Semi-major Axis' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Ellipsoid Semi-major Axis' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Ellipsoid Semi-major Axis' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # ellipsoidFlatteningInverse | Offset: 256, Length: 32, Resolution: 1, Field Type: FLOAT
    running_bit_offset = 256
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("ellipsoidFlatteningInverse")

    advance_running_offset = True
    assert field.value is None or isinstance(field.value, (int, float))
    field_value = encode_float(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Ellipsoid Flattening Inverse' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Ellipsoid Flattening Inverse' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Ellipsoid Flattening Inverse' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # datumName | Offset: 288, Length: 32, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 288
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("datumName")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Datum Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Datum Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Datum Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(40, byteorder="little")
