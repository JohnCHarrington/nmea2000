# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130054() -> bool:
    """Return True if PGN 130054 is a fast PGN."""
    return True
def decode_pgn_130054(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130054."""
    nmea2000Message = NMEA2000Message(PGN=130054, id='loranCSignalData', description='Loran-C Signal Data')
    running_bit_offset = 0
    # 1:group_repetition_interval__gri_ | Offset: 0, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_repetition_interval__gri_ = group_repetition_interval__gri__raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('groupRepetitionIntervalGri', 'Group Repetition Interval (GRI)', None, 's', group_repetition_interval__gri_, group_repetition_interval__gri__raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 2:station_identifier | Offset: 32, Length: 8, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    station_identifier, station_identifier_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('stationIdentifier', 'Station identifier', None, None, station_identifier, station_identifier_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 8

    # 3:station_snr | Offset: 40, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    station_snr = station_snr_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('stationSnr', 'Station SNR', None, 'dB', station_snr, station_snr_raw, PhysicalQuantities.SIGNAL_TO_NOISE_RATIO, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:station_ecd | Offset: 56, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    station_ecd = station_ecd_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('stationEcd', 'Station ECD', None, 's', station_ecd, station_ecd_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 5:station_asf | Offset: 88, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    station_asf = station_asf_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('stationAsf', 'Station ASF', None, 's', station_asf, station_asf_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130054(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130054."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # groupRepetitionIntervalGri | Offset: 0, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("groupRepetitionIntervalGri")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 1e-09)):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-09)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, True, 1e-09)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Group Repetition Interval (GRI)' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Group Repetition Interval (GRI)' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Group Repetition Interval (GRI)' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationIdentifier | Offset: 32, Length: 8, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationIdentifier")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 8)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station identifier' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station identifier' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station identifier' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationSnr | Offset: 40, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationSnr")

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
        raise ValueError("Cant encode this message, 'Station SNR' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station SNR' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station SNR' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationEcd | Offset: 56, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationEcd")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 1e-09)):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-09)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, True, 1e-09)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station ECD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station ECD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station ECD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationAsf | Offset: 88, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationAsf")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 1e-09)):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-09)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, True, 1e-09)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station ASF' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station ASF' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station ASF' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(15, byteorder="little")
