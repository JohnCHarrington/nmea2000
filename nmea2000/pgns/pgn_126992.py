# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126992() -> bool:
    """Return True if PGN 126992 is a fast PGN."""
    return False
def decode_pgn_126992(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126992."""
    nmea2000Message = NMEA2000Message(PGN=126992, id='systemTime', description='System Time', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:sid | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    sid = sid_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sid', 'SID', "An upward counting number used to tie releated information together between different PGNs. For example the SID would be used to tie together the COG, SOG and RAIM values to a given position.", None, sid, sid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:source | Offset: 8, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 8
    source_raw = decode_int(_data_raw_, running_bit_offset, 4)
    source = master_dict['SYSTEM_TIME'].get(source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('source', 'Source', None, None, source, source_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 4

    # 3:reserved_12 | Offset: 12, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 12
    reserved_12 = reserved_12_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_12', 'Reserved', None, None, reserved_12, reserved_12_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 4:date | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    date_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    date = decode_date(date_raw)
    nmea2000Message.fields.append(NMEA2000Field('date', 'Date', None, 'd', date, date_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 5:time | Offset: 32, Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time = decode_time(time_raw)
    nmea2000Message.fields.append(NMEA2000Field('time', 'Time', "Seconds since midnight", 's', time, time_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_126992(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126992."""
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
    # source | Offset: 8, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("source")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SYSTEM_TIME(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Source' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_12 | Offset: 12, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 12
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_12")

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
    # date | Offset: 16, Length: 16, Resolution: 1, Field Type: DATE
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("date")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Date' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Date' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Date' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # time | Offset: 32, Length: 32, Resolution: 0.0001, Field Type: TIME
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("time")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.0001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
