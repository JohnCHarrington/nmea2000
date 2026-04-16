# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130052() -> bool:
    """Return True if PGN 130052 is a fast PGN."""
    return True
def decode_pgn_130052(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130052."""
    nmea2000Message = NMEA2000Message(PGN=130052, id='loranCTdData', description='Loran-C TD Data', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:group_repetition_interval__gri_ | Offset: 0, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    group_repetition_interval__gri_ = group_repetition_interval__gri__raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('groupRepetitionIntervalGri', 'Group Repetition Interval (GRI)', None, 's', group_repetition_interval__gri_, group_repetition_interval__gri__raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 2:master_range | Offset: 32, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    master_range = master_range_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('masterRange', 'Master Range', None, 's', master_range, master_range_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 3:v_secondary_td | Offset: 64, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    v_secondary_td = v_secondary_td_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('vSecondaryTd', 'V Secondary TD', None, 's', v_secondary_td, v_secondary_td_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 4:w_secondary_td | Offset: 96, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    w_secondary_td = w_secondary_td_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('wSecondaryTd', 'W Secondary TD', None, 's', w_secondary_td, w_secondary_td_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 5:x_secondary_td | Offset: 128, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    x_secondary_td = x_secondary_td_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('xSecondaryTd', 'X Secondary TD', None, 's', x_secondary_td, x_secondary_td_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 6:y_secondary_td | Offset: 160, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    y_secondary_td = y_secondary_td_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('ySecondaryTd', 'Y Secondary TD', None, 's', y_secondary_td, y_secondary_td_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 7:z_secondary_td | Offset: 192, Length: 32, Signed: True Resolution: 1e-09, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 192
    z_secondary_td = z_secondary_td_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-09, -2.147483647, 2.147483644)
    nmea2000Message.fields.append(NMEA2000Field('zSecondaryTd', 'Z Secondary TD', None, 's', z_secondary_td, z_secondary_td_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 8:station_status__master | Offset: 224, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 224
    station_status__master_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_status__master = decode_bit_lookup(station_status__master_raw, master_flags_dict['STATION_STATUS'])
    nmea2000Message.fields.append(NMEA2000Field('stationStatusMaster', 'Station status: Master', None, None, station_status__master, station_status__master_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 9:station_status__v | Offset: 228, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 228
    station_status__v_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_status__v = decode_bit_lookup(station_status__v_raw, master_flags_dict['STATION_STATUS'])
    nmea2000Message.fields.append(NMEA2000Field('stationStatusV', 'Station status: V', None, None, station_status__v, station_status__v_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 10:station_status__w | Offset: 232, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 232
    station_status__w_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_status__w = decode_bit_lookup(station_status__w_raw, master_flags_dict['STATION_STATUS'])
    nmea2000Message.fields.append(NMEA2000Field('stationStatusW', 'Station status: W', None, None, station_status__w, station_status__w_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 11:station_status__x | Offset: 236, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 236
    station_status__x_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_status__x = decode_bit_lookup(station_status__x_raw, master_flags_dict['STATION_STATUS'])
    nmea2000Message.fields.append(NMEA2000Field('stationStatusX', 'Station status: X', None, None, station_status__x, station_status__x_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 12:station_status__y | Offset: 240, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 240
    station_status__y_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_status__y = decode_bit_lookup(station_status__y_raw, master_flags_dict['STATION_STATUS'])
    nmea2000Message.fields.append(NMEA2000Field('stationStatusY', 'Station status: Y', None, None, station_status__y, station_status__y_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 13:station_status__z | Offset: 244, Length: 4, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 244
    station_status__z_raw = decode_int(_data_raw_, running_bit_offset, 4)
    station_status__z = decode_bit_lookup(station_status__z_raw, master_flags_dict['STATION_STATUS'])
    nmea2000Message.fields.append(NMEA2000Field('stationStatusZ', 'Station status: Z', None, None, station_status__z, station_status__z_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 4

    # 14:mode | Offset: 248, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 248
    mode_raw = decode_int(_data_raw_, running_bit_offset, 4)
    mode = master_dict['RESIDUAL_MODE'].get(mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('mode', 'Mode', None, None, mode, mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 15:reserved_252 | Offset: 252, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 252
    reserved_252 = reserved_252_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_252', 'Reserved', None, None, reserved_252, reserved_252_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    return nmea2000Message

def encode_pgn_130052(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130052."""
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
    # masterRange | Offset: 32, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("masterRange")

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
        raise ValueError("Cant encode this message, 'Master Range' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Master Range' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Master Range' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # vSecondaryTd | Offset: 64, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("vSecondaryTd")

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
        raise ValueError("Cant encode this message, 'V Secondary TD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'V Secondary TD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'V Secondary TD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # wSecondaryTd | Offset: 96, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("wSecondaryTd")

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
        raise ValueError("Cant encode this message, 'W Secondary TD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'W Secondary TD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'W Secondary TD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # xSecondaryTd | Offset: 128, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("xSecondaryTd")

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
        raise ValueError("Cant encode this message, 'X Secondary TD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'X Secondary TD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'X Secondary TD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # ySecondaryTd | Offset: 160, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("ySecondaryTd")

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
        raise ValueError("Cant encode this message, 'Y Secondary TD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Y Secondary TD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Y Secondary TD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zSecondaryTd | Offset: 192, Length: 32, Resolution: 1e-09, Field Type: DURATION
    running_bit_offset = 192
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zSecondaryTd")

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
        raise ValueError("Cant encode this message, 'Z Secondary TD' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Z Secondary TD' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Z Secondary TD' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationStatusMaster | Offset: 224, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 224
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationStatusMaster")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['STATION_STATUS'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station status: Master' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station status: Master' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station status: Master' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationStatusV | Offset: 228, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 228
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationStatusV")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['STATION_STATUS'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station status: V' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station status: V' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station status: V' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationStatusW | Offset: 232, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 232
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationStatusW")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['STATION_STATUS'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station status: W' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station status: W' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station status: W' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationStatusX | Offset: 236, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 236
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationStatusX")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['STATION_STATUS'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station status: X' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station status: X' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station status: X' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationStatusY | Offset: 240, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 240
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationStatusY")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['STATION_STATUS'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station status: Y' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station status: Y' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station status: Y' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # stationStatusZ | Offset: 244, Length: 4, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 244
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("stationStatusZ")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['STATION_STATUS'])
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Station status: Z' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Station status: Z' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Station status: Z' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mode | Offset: 248, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 248
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_RESIDUAL_MODE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_252 | Offset: 252, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 252
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_252")

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
    return data_raw.to_bytes(32, byteorder="little")
