# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129547() -> bool:
    """Return True if PGN 129547 is a fast PGN."""
    return True
def decode_pgn_129547(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129547."""
    nmea2000Message = NMEA2000Message(PGN=129547, id='gnssPseudorangeErrorStatistics', description='GNSS Pseudorange Error Statistics')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:rms_std_dev_of_range_inputs | Offset: 8, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    rms_std_dev_of_range_inputs = rms_std_dev_of_range_inputs_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('rmsStdDevOfRangeInputs', 'RMS Std Dev of Range Inputs', None, 'm', rms_std_dev_of_range_inputs, rms_std_dev_of_range_inputs_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:std_dev_of_major_error_ellipse | Offset: 24, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    std_dev_of_major_error_ellipse = std_dev_of_major_error_ellipse_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('stdDevOfMajorErrorEllipse', 'Std Dev of Major error ellipse', None, 'm', std_dev_of_major_error_ellipse, std_dev_of_major_error_ellipse_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:std_dev_of_minor_error_ellipse | Offset: 40, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    std_dev_of_minor_error_ellipse = std_dev_of_minor_error_ellipse_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('stdDevOfMinorErrorEllipse', 'Std Dev of Minor error ellipse', None, 'm', std_dev_of_minor_error_ellipse, std_dev_of_minor_error_ellipse_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:orientation_of_error_ellipse | Offset: 56, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    orientation_of_error_ellipse = orientation_of_error_ellipse_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('orientationOfErrorEllipse', 'Orientation of error ellipse', None, 'rad', orientation_of_error_ellipse, orientation_of_error_ellipse_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:std_dev_lat_error | Offset: 72, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    std_dev_lat_error = std_dev_lat_error_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('stdDevLatError', 'Std Dev Lat Error', None, 'm', std_dev_lat_error, std_dev_lat_error_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:std_dev_lon_error | Offset: 88, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    std_dev_lon_error = std_dev_lon_error_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('stdDevLonError', 'Std Dev Lon Error', None, 'm', std_dev_lon_error, std_dev_lon_error_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:std_dev_alt_error | Offset: 104, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 104
    std_dev_alt_error = std_dev_alt_error_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('stdDevAltError', 'Std Dev Alt Error', None, 'm', std_dev_alt_error, std_dev_alt_error_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_129547(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129547."""
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
    # rmsStdDevOfRangeInputs | Offset: 8, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rmsStdDevOfRangeInputs")

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
        raise ValueError("Cant encode this message, 'RMS Std Dev of Range Inputs' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'RMS Std Dev of Range Inputs' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'RMS Std Dev of Range Inputs' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stdDevOfMajorErrorEllipse | Offset: 24, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stdDevOfMajorErrorEllipse")

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
        raise ValueError("Cant encode this message, 'Std Dev of Major error ellipse' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Std Dev of Major error ellipse' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Std Dev of Major error ellipse' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stdDevOfMinorErrorEllipse | Offset: 40, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stdDevOfMinorErrorEllipse")

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
        raise ValueError("Cant encode this message, 'Std Dev of Minor error ellipse' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Std Dev of Minor error ellipse' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Std Dev of Minor error ellipse' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # orientationOfErrorEllipse | Offset: 56, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("orientationOfErrorEllipse")

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
        raise ValueError("Cant encode this message, 'Orientation of error ellipse' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Orientation of error ellipse' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Orientation of error ellipse' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stdDevLatError | Offset: 72, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stdDevLatError")

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
        raise ValueError("Cant encode this message, 'Std Dev Lat Error' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Std Dev Lat Error' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Std Dev Lat Error' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stdDevLonError | Offset: 88, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stdDevLonError")

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
        raise ValueError("Cant encode this message, 'Std Dev Lon Error' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Std Dev Lon Error' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Std Dev Lon Error' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stdDevAltError | Offset: 104, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stdDevAltError")

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
        raise ValueError("Cant encode this message, 'Std Dev Alt Error' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Std Dev Alt Error' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Std Dev Alt Error' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(15, byteorder="little")
