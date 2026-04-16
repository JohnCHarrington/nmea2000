# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129551() -> bool:
    """Return True if PGN 129551 is a fast PGN."""
    return True
def decode_pgn_129551(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129551."""
    nmea2000Message = NMEA2000Message(PGN=129551, id='gnssDifferentialCorrectionReceiverSignal', description='GNSS Differential Correction Receiver Signal')
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:channel | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:signal_strength | Offset: 16, Length: 32, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    signal_strength = signal_strength_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 0.01, -21474836.47, 21474836.44)
    nmea2000Message.fields.append(NMEA2000Field('signalStrength', 'Signal Strength', "Signal strength in dB relative to 1 uV/m", 'dB', signal_strength, signal_strength_raw, PhysicalQuantities.SIGNAL_STRENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 4:signal_snr | Offset: 48, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    signal_snr = signal_snr_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('signalSnr', 'Signal SNR', None, 'dB', signal_snr, signal_snr_raw, PhysicalQuantities.SIGNAL_TO_NOISE_RATIO, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:frequency | Offset: 64, Length: 32, Signed: False Resolution: 10, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    frequency = frequency_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 10, 0, 42949672920)
    nmea2000Message.fields.append(NMEA2000Field('frequency', 'Frequency', None, 'Hz', frequency, frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 6:station_type | Offset: 96, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    station_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_type = master_dict['GNS'].get(station_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('stationType', 'Station Type', None, None, station_type, station_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 7:reference_station_id | Offset: 100, Length: 12, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 100
    reference_station_id = reference_station_id_raw = decode_number(_data_raw_, running_bit_offset, 12, False, 1, 0, 4092)
    nmea2000Message.fields.append(NMEA2000Field('referenceStationId', 'Reference Station ID', None, None, reference_station_id, reference_station_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 12

    # 8:differential_signal_bit_rate | Offset: 112, Length: 5, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    differential_signal_bit_rate_raw = decode_int(_data_raw_, running_bit_offset, 5)
    differential_signal_bit_rate = master_dict['SERIAL_BIT_RATE'].get(differential_signal_bit_rate_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('differentialSignalBitRate', 'Differential Signal Bit Rate', None, None, differential_signal_bit_rate, differential_signal_bit_rate_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 5

    # 9:differential_signal_detection_mode | Offset: 117, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 117
    differential_signal_detection_mode_raw = decode_int(_data_raw_, running_bit_offset, 3)
    differential_signal_detection_mode = master_dict['SERIAL_DETECTION_MODE'].get(differential_signal_detection_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('differentialSignalDetectionMode', 'Differential Signal Detection Mode', None, None, differential_signal_detection_mode, differential_signal_detection_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 10:used_as_correction_source | Offset: 120, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    used_as_correction_source_raw = decode_int(_data_raw_, running_bit_offset, 2)
    used_as_correction_source = master_dict['YES_NO'].get(used_as_correction_source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('usedAsCorrectionSource', 'Used as Correction Source', None, None, used_as_correction_source, used_as_correction_source_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 11:reserved_122 | Offset: 122, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 122
    reserved_122 = reserved_122_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_122', 'Reserved', None, None, reserved_122, reserved_122_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 12:differential_source | Offset: 124, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 124
    differential_source_raw = decode_int(_data_raw_, running_bit_offset, 4)
    differential_source = master_dict['DIFFERENTIAL_SOURCE'].get(differential_source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('differentialSource', 'Differential Source', None, None, differential_source, differential_source_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 13:time_since_last_sat_differential_sync | Offset: 128, Length: 16, Signed: False Resolution: 0.01, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    time_since_last_sat_differential_sync = time_since_last_sat_differential_sync_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('timeSinceLastSatDifferentialSync', 'Time since Last Sat Differential Sync', "Age of differential corrections", 's', time_since_last_sat_differential_sync, time_since_last_sat_differential_sync_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 14:satellite_service_id_no_ | Offset: 144, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    satellite_service_id_no_ = satellite_service_id_no__raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('satelliteServiceIdNo', 'Satellite Service ID No.', None, None, satellite_service_id_no_, satellite_service_id_no__raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_129551(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129551."""
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
    # channel | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

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
        raise ValueError("Cant encode this message, 'Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # signalStrength | Offset: 16, Length: 32, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("signalStrength")

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
        raise ValueError("Cant encode this message, 'Signal Strength' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Signal Strength' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Signal Strength' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # signalSnr | Offset: 48, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("signalSnr")

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
        raise ValueError("Cant encode this message, 'Signal SNR' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Signal SNR' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Signal SNR' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # frequency | Offset: 64, Length: 32, Resolution: 10, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("frequency")

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
        raise ValueError("Cant encode this message, 'Frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationType | Offset: 96, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationType")

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
        raise ValueError("Cant encode this message, 'Station Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station Type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # referenceStationId | Offset: 100, Length: 12, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 100
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
    # differentialSignalBitRate | Offset: 112, Length: 5, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("differentialSignalBitRate")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SERIAL_BIT_RATE(field.value)
    field_bit_length = 5
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Differential Signal Bit Rate' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Differential Signal Bit Rate' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Differential Signal Bit Rate' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # differentialSignalDetectionMode | Offset: 117, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 117
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("differentialSignalDetectionMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SERIAL_DETECTION_MODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Differential Signal Detection Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Differential Signal Detection Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Differential Signal Detection Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # usedAsCorrectionSource | Offset: 120, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("usedAsCorrectionSource")

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
        raise ValueError("Cant encode this message, 'Used as Correction Source' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Used as Correction Source' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Used as Correction Source' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_122 | Offset: 122, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 122
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_122")

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
    # differentialSource | Offset: 124, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 124
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("differentialSource")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DIFFERENTIAL_SOURCE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Differential Source' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Differential Source' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Differential Source' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # timeSinceLastSatDifferentialSync | Offset: 128, Length: 16, Resolution: 0.01, Field Type: DURATION
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeSinceLastSatDifferentialSync")

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
        raise ValueError("Cant encode this message, 'Time since Last Sat Differential Sync' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time since Last Sat Differential Sync' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time since Last Sat Differential Sync' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # satelliteServiceIdNo | Offset: 144, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("satelliteServiceIdNo")

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
        raise ValueError("Cant encode this message, 'Satellite Service ID No.' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Satellite Service ID No.' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Satellite Service ID No.' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(20, byteorder="little")
