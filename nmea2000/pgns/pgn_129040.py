# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129040() -> bool:
    """Return True if PGN 129040 is a fast PGN."""
    return True
def decode_pgn_129040(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129040."""
    nmea2000Message = NMEA2000Message(PGN=129040, id='aisClassBExtendedPositionReport', description='AIS Class B Extended Position Report')
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

    # 8:time_stamp | Offset: 106, Length: 6, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 106
    time_stamp_raw = decode_int(_data_raw_, running_bit_offset, 6)
    time_stamp = master_dict['TIME_STAMP'].get(time_stamp_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('timeStamp', 'Time Stamp', None, None, time_stamp, time_stamp_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 6

    # 9:cog | Offset: 112, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    cog = cog_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('cog', 'COG', None, 'rad', cog, cog_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 10:sog | Offset: 128, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    sog = sog_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('sog', 'SOG', None, 'm/s', sog, sog_raw, PhysicalQuantities.SPEED, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 11:regional_application | Offset: 144, Length: 8, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    regional_application = regional_application_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('regionalApplication', 'Regional Application', None, None, regional_application, regional_application_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 8

    # 12:regional_application_b | Offset: 152, Length: 4, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    regional_application_b = regional_application_b_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('regionalApplicationB', 'Regional Application B', None, None, regional_application_b, regional_application_b_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 4

    # 13:reserved_156 | Offset: 156, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 156
    reserved_156 = reserved_156_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_156', 'Reserved', None, None, reserved_156, reserved_156_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 14:type_of_ship | Offset: 160, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 160
    type_of_ship_raw = decode_int(_data_raw_, running_bit_offset, 8)
    type_of_ship = master_dict['SHIP_TYPE'].get(type_of_ship_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('typeOfShip', 'Type of ship', None, None, type_of_ship, type_of_ship_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 15:true_heading | Offset: 168, Length: 16, Signed: False Resolution: 0.0001, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 168
    true_heading = true_heading_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0001, 0, 6.2831852)
    nmea2000Message.fields.append(NMEA2000Field('trueHeading', 'True Heading', None, 'rad', true_heading, true_heading_raw, PhysicalQuantities.ANGLE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 16:reserved_184 | Offset: 184, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 184
    reserved_184 = reserved_184_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_184', 'Reserved', None, None, reserved_184, reserved_184_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 17:gnss_type | Offset: 188, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 188
    gnss_type_raw = decode_int(_data_raw_, running_bit_offset, 4)
    gnss_type = master_dict['POSITION_FIX_DEVICE'].get(gnss_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('gnssType', 'GNSS type', None, None, gnss_type, gnss_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 18:length | Offset: 192, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 192
    length = length_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('length', 'Length', None, 'm', length, length_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 19:beam | Offset: 208, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 208
    beam = beam_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('beam', 'Beam', None, 'm', beam, beam_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 20:position_reference_from_starboard | Offset: 224, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 224
    position_reference_from_starboard = position_reference_from_starboard_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('positionReferenceFromStarboard', 'Position reference from Starboard', None, 'm', position_reference_from_starboard, position_reference_from_starboard_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 21:position_reference_from_bow | Offset: 240, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 240
    position_reference_from_bow = position_reference_from_bow_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('positionReferenceFromBow', 'Position reference from Bow', None, 'm', position_reference_from_bow, position_reference_from_bow_raw, PhysicalQuantities.LENGTH, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 22:name | Offset: 256, Length: 160, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 256
    name, name_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 160)
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 160

    # 23:dte | Offset: 416, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 416
    dte_raw = decode_int(_data_raw_, running_bit_offset, 1)
    dte = master_dict['AVAILABLE'].get(dte_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dte', 'DTE', None, None, dte, dte_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 24:ais_mode | Offset: 417, Length: 1, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 417
    ais_mode_raw = decode_int(_data_raw_, running_bit_offset, 1)
    ais_mode = master_dict['AIS_MODE'].get(ais_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('aisMode', 'AIS mode', None, None, ais_mode, ais_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 1

    # 25:spare | Offset: 418, Length: 4, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 418
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('spare25', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 4

    # 26:ais_transceiver_information | Offset: 422, Length: 5, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 422
    ais_transceiver_information_raw = decode_int(_data_raw_, running_bit_offset, 5)
    ais_transceiver_information = master_dict['AIS_TRANSCEIVER'].get(ais_transceiver_information_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('aisTransceiverInformation', 'AIS Transceiver information', None, None, ais_transceiver_information, ais_transceiver_information_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 5

    # 27:reserved_427 | Offset: 427, Length: 5, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 427
    reserved_427 = reserved_427_raw = decode_int(_data_raw_, running_bit_offset, 5)
    nmea2000Message.fields.append(NMEA2000Field('reserved_427', 'Reserved', None, None, reserved_427, reserved_427_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 5

    return nmea2000Message

def encode_pgn_129040(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129040."""
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
    # timeStamp | Offset: 106, Length: 6, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 106
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeStamp")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_TIME_STAMP(field.value)
    field_bit_length = 6
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time Stamp' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time Stamp' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time Stamp' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # cog | Offset: 112, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("cog")

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
        raise ValueError("Cant encode this message, 'COG' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'COG' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'COG' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sog | Offset: 128, Length: 16, Resolution: 0.01, Field Type: NUMBER
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sog")

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
        raise ValueError("Cant encode this message, 'SOG' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SOG' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SOG' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # regionalApplication | Offset: 144, Length: 8, Resolution: 1, Field Type: SPARE
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("regionalApplication")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Regional Application' must be an integer")
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Regional Application' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Regional Application' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Regional Application' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # regionalApplicationB | Offset: 152, Length: 4, Resolution: 1, Field Type: SPARE
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("regionalApplicationB")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Regional Application B' must be an integer")
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Regional Application B' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Regional Application B' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Regional Application B' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_156 | Offset: 156, Length: 4, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 156
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_156")

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
    # typeOfShip | Offset: 160, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 160
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("typeOfShip")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SHIP_TYPE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Type of ship' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Type of ship' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Type of ship' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # trueHeading | Offset: 168, Length: 16, Resolution: 0.0001, Field Type: NUMBER
    running_bit_offset = 168
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("trueHeading")

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
        raise ValueError("Cant encode this message, 'True Heading' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'True Heading' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'True Heading' exceeds the encoded bit length")
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
    # length | Offset: 192, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 192
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("length")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Length' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Length' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Length' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # beam | Offset: 208, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 208
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("beam")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Beam' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Beam' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Beam' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # positionReferenceFromStarboard | Offset: 224, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 224
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionReferenceFromStarboard")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Position reference from Starboard' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position reference from Starboard' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position reference from Starboard' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # positionReferenceFromBow | Offset: 240, Length: 16, Resolution: 0.1, Field Type: NUMBER
    running_bit_offset = 240
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionReferenceFromBow")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.1):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.1)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Position reference from Bow' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position reference from Bow' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position reference from Bow' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # name | Offset: 256, Length: 160, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 256
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 160)
    field_bit_length = 160
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dte | Offset: 416, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 416
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dte")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_AVAILABLE(field.value)
    field_bit_length = 1
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DTE' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DTE' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DTE' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # aisMode | Offset: 417, Length: 1, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 417
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("aisMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_AIS_MODE(field.value)
    field_bit_length = 1
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'AIS mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'AIS mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'AIS mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # spare25 | Offset: 418, Length: 4, Resolution: 1, Field Type: SPARE
    running_bit_offset = 418
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare25")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Spare' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Spare' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Spare' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # aisTransceiverInformation | Offset: 422, Length: 5, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 422
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
    # reserved_427 | Offset: 427, Length: 5, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 427
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_427")

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
    return data_raw.to_bytes(54, byteorder="little")
