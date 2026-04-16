# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129546() -> bool:
    """Return True if PGN 129546 is a fast PGN."""
    return False
def decode_pgn_129546(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129546."""
    nmea2000Message = NMEA2000Message(PGN=129546, id='gnssRaimSettings', description='GNSS RAIM Settings')
    running_bit_offset = 0
    # 1:radial_position_error_maximum_threshold | Offset: 0, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    radial_position_error_maximum_threshold = radial_position_error_maximum_threshold_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('radialPositionErrorMaximumThreshold', 'Radial Position Error Maximum Threshold', None, 'm', radial_position_error_maximum_threshold, radial_position_error_maximum_threshold_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:probability_of_false_alarm | Offset: 16, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    probability_of_false_alarm = probability_of_false_alarm_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('probabilityOfFalseAlarm', 'Probability of False Alarm', None, '%', probability_of_false_alarm, probability_of_false_alarm_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:probability_of_missed_detection | Offset: 24, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    probability_of_missed_detection = probability_of_missed_detection_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('probabilityOfMissedDetection', 'Probability of Missed Detection', None, '%', probability_of_missed_detection, probability_of_missed_detection_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 4:pseudorange_residual_filtering_time_constant | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    pseudorange_residual_filtering_time_constant = pseudorange_residual_filtering_time_constant_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('pseudorangeResidualFilteringTimeConstant', 'Pseudorange Residual Filtering Time Constant', None, 's', pseudorange_residual_filtering_time_constant, pseudorange_residual_filtering_time_constant_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 5:reserved_48 | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    reserved_48 = reserved_48_raw = decode_int(_data_raw_, running_bit_offset, 16)
    nmea2000Message.fields.append(NMEA2000Field('reserved_48', 'Reserved', None, None, reserved_48, reserved_48_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_129546(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129546."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # radialPositionErrorMaximumThreshold | Offset: 0, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("radialPositionErrorMaximumThreshold")

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
        raise ValueError("Cant encode this message, 'Radial Position Error Maximum Threshold' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Radial Position Error Maximum Threshold' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Radial Position Error Maximum Threshold' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # probabilityOfFalseAlarm | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("probabilityOfFalseAlarm")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Probability of False Alarm' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Probability of False Alarm' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Probability of False Alarm' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # probabilityOfMissedDetection | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("probabilityOfMissedDetection")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Probability of Missed Detection' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Probability of Missed Detection' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Probability of Missed Detection' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # pseudorangeResidualFilteringTimeConstant | Offset: 32, Length: 16, Resolution: 1, Field Type: DURATION
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("pseudorangeResidualFilteringTimeConstant")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 1)):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 1)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 16, False, 1)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Pseudorange Residual Filtering Time Constant' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Pseudorange Residual Filtering Time Constant' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Pseudorange Residual Filtering Time Constant' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_48 | Offset: 48, Length: 16, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_48")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 16
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
    return data_raw.to_bytes(8, byteorder="little")
