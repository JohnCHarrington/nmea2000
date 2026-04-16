# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129540() -> bool:
    """Return True if PGN 129540 is a fast PGN."""
    return True
def decode_pgn_129540(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129540."""
    nmea2000Message = NMEA2000Message(PGN=129540, id='gnssSatsInView', description='GNSS Sats in View', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', None, None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:range_residual_mode | Offset: 8, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    range_residual_mode_raw = decode_int(_data_raw_, running_bit_offset, 2)
    range_residual_mode = master_dict['RANGE_RESIDUAL_MODE'].get(range_residual_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('rangeResidualMode', 'Range Residual Mode', None, None, range_residual_mode, range_residual_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:reserved_10 | Offset: 10, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 10
    reserved_10 = reserved_10_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_10', 'Reserved', None, None, reserved_10, reserved_10_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 4:sats_in_view | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    sats_in_view = sats_in_view_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('satsInView', 'Sats in View', None, None, sats_in_view, sats_in_view_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:prn | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    _repeating_field_set_1_offset = running_bit_offset
    prn = prn_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('prn', 'PRN', None, None, prn, prn_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:elevation | Offset: 32, Length: 16, Signed: True Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    elevation = elevation_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
    nmea2000Message.fields.append(NMEA2000Field('elevation', 'Elevation', None, 'rad', elevation, elevation_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:azimuth | Offset: 48, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    azimuth = azimuth_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('azimuth', 'Azimuth', None, 'rad', azimuth, azimuth_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:snr | Offset: 64, Length: 16, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    snr = snr_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
    nmea2000Message.fields.append(NMEA2000Field('snr', 'SNR', None, 'dB', snr, snr_raw, PhysicalQuantities.SIGNAL_TO_NOISE_RATIO, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:range_residuals | Offset: 80, Length: 32, Signed: True Resolution: 1e-05, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    range_residuals = range_residuals_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-05, -21474.83647, 21474.83644)
    nmea2000Message.fields.append(NMEA2000Field('rangeResiduals', 'Range residuals', None, 'm', range_residuals, range_residuals_raw, PhysicalQuantities.DISTANCE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 10:status | Offset: 112, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    status_raw = decode_int(_data_raw_, running_bit_offset, 4)
    status = master_dict['SATELLITE_STATUS'].get(status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 11:reserved_116 | Offset: 116, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 116
    reserved_116 = reserved_116_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_116', 'Reserved', None, None, reserved_116, reserved_116_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(sats_in_view_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        prn = prn_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
        running_bit_offset += 8
        repeating_entry["prn"] = _repeating_entry_value(prn, prn_raw)
    
        elevation = elevation_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.0001, -3.1415926, 3.1415926)
        running_bit_offset += 16
        repeating_entry["elevation"] = _repeating_entry_value(elevation, elevation_raw)
    
        azimuth = azimuth_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
        running_bit_offset += 16
        repeating_entry["azimuth"] = _repeating_entry_value(azimuth, azimuth_raw)
    
        snr = snr_raw = decode_number(_data_raw_, running_bit_offset, 16, True, 0.01, -327.67, 327.64)
        running_bit_offset += 16
        repeating_entry["snr"] = _repeating_entry_value(snr, snr_raw)
    
        range_residuals = range_residuals_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-05, -21474.83647, 21474.83644)
        running_bit_offset += 32
        repeating_entry["rangeResiduals"] = _repeating_entry_value(range_residuals, range_residuals_raw)
    
        status_raw = decode_int(_data_raw_, running_bit_offset, 4)
        status = master_dict['SATELLITE_STATUS'].get(status_raw, None)
        running_bit_offset += 4
        repeating_entry["status"] = _repeating_entry_value(status, status_raw)
    
        reserved_116 = reserved_116_raw = decode_int(_data_raw_, running_bit_offset, 4)
        running_bit_offset += 4
        repeating_entry["reserved_116"] = _repeating_entry_value(reserved_116, reserved_116_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "prn",
                "elevation",
                "azimuth",
                "snr",
                "rangeResiduals",
                "status",
                "reserved_116",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_129540(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129540."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "prn",
        "elevation",
        "azimuth",
        "snr",
        "rangeResiduals",
        "status",
        "reserved_116",
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
    # rangeResidualMode | Offset: 8, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rangeResidualMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_RANGE_RESIDUAL_MODE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Range Residual Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Range Residual Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Range Residual Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_10 | Offset: 10, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 10
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_10")

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
    # satsInView | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("satsInView")

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
        raise ValueError("Cant encode this message, 'Sats in View' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sats in View' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sats in View' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 24
    for repeating_entry in repeating_field_set_1_entries:
        # prn | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("prn")
        if field is None:
            raise ValueError("Cant encode this message, missing 'PRN'")
        field_offset = running_bit_offset
    
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
        # elevation | Offset: 32, Length: 16, Resolution: 0.0001, Field Type: NUMBER
        field = repeating_entry.get("elevation")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Elevation'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Elevation' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Elevation' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Elevation' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # azimuth | Offset: 48, Length: 16, Resolution: 0.0001, Field Type: NUMBER
        field = repeating_entry.get("azimuth")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Azimuth'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Azimuth' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Azimuth' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Azimuth' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # snr | Offset: 64, Length: 16, Resolution: 0.01, Field Type: NUMBER
        field = repeating_entry.get("snr")
        if field is None:
            raise ValueError("Cant encode this message, missing 'SNR'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'SNR' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'SNR' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'SNR' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # rangeResiduals | Offset: 80, Length: 32, Resolution: 1e-05, Field Type: NUMBER
        field = repeating_entry.get("rangeResiduals")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Range residuals'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-05):
            field_value = encode_number_raw(field.raw_value, 32, True)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 32, True, 1e-05)
        else:
            assert field.value is None or isinstance(field.value, (int, float))
            field_value = encode_number(field.value, 32, True, 1e-05)
        field_bit_length = 32
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Range residuals' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Range residuals' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Range residuals' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # status | Offset: 112, Length: 4, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("status")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Status'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_SATELLITE_STATUS(field.value)
        field_bit_length = 4
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Status' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Status' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Status' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # reserved_116 | Offset: 116, Length: 4, Resolution: 1, Field Type: RESERVED
        field = repeating_entry.get("reserved_116")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reserved'")
        field_offset = running_bit_offset
    
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
    
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
