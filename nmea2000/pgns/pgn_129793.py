# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129793() -> bool:
    """Return True if PGN 129793 is a fast PGN."""
    return True
def decode_pgn_129793(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129793."""
    nmea2000Message = NMEA2000Message(PGN=129793, id='aisUtcAndDateReport', description='AIS UTC and Date Report')
    running_bit_offset = 0
    # 1:message_id | Offset: 0, Length: 6, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 6)
    message_id = master_dict['AIS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', None, None, message_id, message_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 6

    # 2:repeat_indicator | Offset: 6, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 6
    repeat_indicator_raw = decode_int(_data_raw_, running_bit_offset, 2)
    repeat_indicator = master_dict['REPEAT_INDICATOR'].get(repeat_indicator_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('repeatIndicator', 'Repeat Indicator', None, None, repeat_indicator, repeat_indicator_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 3:user_id | Offset: 8, Length: 32, Signed: False Resolution: 1, Field Type: MMSI, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 8
    user_id = user_id_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 2000000, 999999999)
    nmea2000Message.fields.append(NMEA2000Field('userId', 'User ID', None, None, user_id, user_id_raw, None, FieldTypes.MMSI, True))
    running_bit_offset += 32

    # 4:longitude | Offset: 40, Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    longitude = longitude_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -180, 180)
    nmea2000Message.fields.append(NMEA2000Field('longitude', 'Longitude', None, 'deg', longitude, longitude_raw, PhysicalQuantities.GEOGRAPHICAL_LONGITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 5:latitude | Offset: 72, Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    latitude = latitude_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -90, 90)
    nmea2000Message.fields.append(NMEA2000Field('latitude', 'Latitude', None, 'deg', latitude, latitude_raw, PhysicalQuantities.GEOGRAPHICAL_LATITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 6:position_accuracy | Offset: 104, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 104
    position_accuracy_raw = decode_int(_data_raw_, running_bit_offset, 1)
    position_accuracy = master_dict['POSITION_ACCURACY'].get(position_accuracy_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('positionAccuracy', 'Position Accuracy', None, None, position_accuracy, position_accuracy_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 7:raim | Offset: 105, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 105
    raim_raw = decode_int(_data_raw_, running_bit_offset, 1)
    raim = master_dict['RAIM_FLAG'].get(raim_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('raim', 'RAIM', None, None, raim, raim_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 8:reserved_106 | Offset: 106, Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 106
    reserved_106 = reserved_106_raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_106', 'Reserved', None, None, reserved_106, reserved_106_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 9:position_time | Offset: 112, Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    position_time_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    position_time = decode_time(position_time_raw)
    nmea2000Message.fields.append(NMEA2000Field('positionTime', 'Position Time', "Seconds since midnight", 's', position_time, position_time_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 10:communication_state | Offset: 144, Length: 19, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    communication_state = communication_state_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 19))
    nmea2000Message.fields.append(NMEA2000Field('communicationState', 'Communication State', "Information used by the TDMA slot allocation algorithm and synchronization information", None, communication_state, communication_state_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 19

    # 11:ais_transceiver_information | Offset: 163, Length: 5, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 163
    ais_transceiver_information_raw = decode_int(_data_raw_, running_bit_offset, 5)
    ais_transceiver_information = master_dict['AIS_TRANSCEIVER'].get(ais_transceiver_information_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('aisTransceiverInformation', 'AIS Transceiver information', None, None, ais_transceiver_information, ais_transceiver_information_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 5

    # 12:position_date | Offset: 168, Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 168
    position_date_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    position_date = decode_date(position_date_raw)
    nmea2000Message.fields.append(NMEA2000Field('positionDate', 'Position Date', None, 'd', position_date, position_date_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 13:reserved_184 | Offset: 184, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 184
    reserved_184 = reserved_184_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_184', 'Reserved', None, None, reserved_184, reserved_184_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 14:gnss_type | Offset: 188, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 188
    gnss_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    gnss_type = master_dict['POSITION_FIX_DEVICE'].get(gnss_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('gnssType', 'GNSS type', None, None, gnss_type, gnss_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    return nmea2000Message

def encode_pgn_129793(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129793."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # messageId | Offset: 0, Length: 6, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_AIS_MESSAGE_ID(field.value)
    field_bit_length = 6
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # repeatIndicator | Offset: 6, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 6
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("repeatIndicator")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_REPEAT_INDICATOR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Repeat Indicator' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Repeat Indicator' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Repeat Indicator' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # userId | Offset: 8, Length: 32, Resolution: 1, Field Type: MMSI
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("userId")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, False, 1)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'User ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'User ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'User ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitude | Offset: 40, Length: 32, Resolution: 1e-07, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # latitude | Offset: 72, Length: 32, Resolution: 1e-07, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("latitude")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Latitude' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Latitude' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Latitude' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # positionAccuracy | Offset: 104, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 104
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionAccuracy")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_POSITION_ACCURACY(field.value)
    field_bit_length = 1
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Position Accuracy' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position Accuracy' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position Accuracy' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # raim | Offset: 105, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 105
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("raim")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_RAIM_FLAG(field.value)
    field_bit_length = 1
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'RAIM' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'RAIM' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'RAIM' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_106 | Offset: 106, Length: 6, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 106
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_106")

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
    # positionTime | Offset: 112, Length: 32, Resolution: 0.0001, Field Type: TIME
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionTime")

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
        raise ValueError("Cant encode this message, 'Position Time' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position Time' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position Time' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # communicationState | Offset: 144, Length: 19, Resolution: 1, Field Type: BINARY
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("communicationState")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 19
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Communication State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Communication State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Communication State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # aisTransceiverInformation | Offset: 163, Length: 5, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 163
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("aisTransceiverInformation")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_AIS_TRANSCEIVER(field.value)
    field_bit_length = 5
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'AIS Transceiver information' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'AIS Transceiver information' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'AIS Transceiver information' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # positionDate | Offset: 168, Length: 16, Resolution: 1, Field Type: DATE
    running_bit_offset = 168
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionDate")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Position Date' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position Date' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position Date' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_184 | Offset: 184, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 184
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_184")

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
    # gnssType | Offset: 188, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 188
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("gnssType")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_POSITION_FIX_DEVICE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'GNSS type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'GNSS type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'GNSS type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(24, byteorder="little")
