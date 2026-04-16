# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126993() -> bool:
    """Return True if PGN 126993 is a fast PGN."""
    return False
def decode_pgn_126993(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126993."""
    nmea2000Message = NMEA2000Message(PGN=126993, id='heartbeat', description='Heartbeat', ttl=timedelta(milliseconds=60000))
    running_bit_offset = 0
    # 1:data_transmit_offset | Offset: 0, Length: 16, Signed: False Resolution: 0.01, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    data_transmit_offset = data_transmit_offset_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('dataTransmitOffset', 'Data transmit offset', "Offset in transmit time from time of request command: 0x0 = transmit immediately, 0xFFFF = Do not change offset.", 's', data_transmit_offset, data_transmit_offset_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 16

    # 2:sequence_counter | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    sequence_counter = sequence_counter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sequenceCounter', 'Sequence Counter', None, None, sequence_counter, sequence_counter_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:controller_1_state | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    controller_1_state_raw = decode_int(_data_raw_, running_bit_offset, 2)
    controller_1_state = master_dict['CONTROLLER_STATE'].get(controller_1_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('controller1State', 'Controller 1 State', None, None, controller_1_state, controller_1_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:controller_2_state | Offset: 26, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    controller_2_state_raw = decode_int(_data_raw_, running_bit_offset, 2)
    controller_2_state = master_dict['CONTROLLER_STATE'].get(controller_2_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('controller2State', 'Controller 2 State', None, None, controller_2_state, controller_2_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:equipment_status | Offset: 28, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 28
    equipment_status_raw = decode_int(_data_raw_, running_bit_offset, 2)
    equipment_status = master_dict['EQUIPMENT_STATUS'].get(equipment_status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('equipmentStatus', 'Equipment Status', None, None, equipment_status, equipment_status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 6:reserved_30 | Offset: 30, Length: 34, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 30
    reserved_30 = reserved_30_raw = decode_int(_data_raw_, running_bit_offset, 34)
    nmea2000Message.fields.append(NMEA2000Field('reserved_30', 'Reserved', None, None, reserved_30, reserved_30_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 34

    return nmea2000Message

def encode_pgn_126993(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126993."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # dataTransmitOffset | Offset: 0, Length: 16, Resolution: 0.01, Field Type: DURATION
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dataTransmitOffset")

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
        raise ValueError("Cant encode this message, 'Data transmit offset' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data transmit offset' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data transmit offset' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sequenceCounter | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sequenceCounter")

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
        raise ValueError("Cant encode this message, 'Sequence Counter' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Sequence Counter' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Sequence Counter' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # controller1State | Offset: 24, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("controller1State")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CONTROLLER_STATE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Controller 1 State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Controller 1 State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Controller 1 State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # controller2State | Offset: 26, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("controller2State")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_CONTROLLER_STATE(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Controller 2 State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Controller 2 State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Controller 2 State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # equipmentStatus | Offset: 28, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 28
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("equipmentStatus")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_EQUIPMENT_STATUS(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Equipment Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Equipment Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Equipment Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_30 | Offset: 30, Length: 34, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 30
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_30")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 34
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
