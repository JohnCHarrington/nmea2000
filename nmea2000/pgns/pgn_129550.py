# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129550() -> bool:
    """Return True if PGN 129550 is a fast PGN."""
    return False
def decode_pgn_129550(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129550."""
    nmea2000Message = NMEA2000Message(PGN=129550, id='gnssDifferentialCorrectionReceiverInterface', description='GNSS Differential Correction Receiver Interface')
    running_bit_offset = 0
    # 1:channel | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:frequency | Offset: 8, Length: 32, Signed: False Resolution: 10, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    frequency = frequency_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 10, 0, 42949672920)
    nmea2000Message.fields.append(NMEA2000Field('frequency', 'Frequency', None, 'Hz', frequency, frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 3:serial_interface_bit_rate | Offset: 40, Length: 5, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    serial_interface_bit_rate_raw = decode_int(_data_raw_, running_bit_offset, 5)
    serial_interface_bit_rate = master_dict['SERIAL_BIT_RATE'].get(serial_interface_bit_rate_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('serialInterfaceBitRate', 'Serial Interface Bit Rate', None, None, serial_interface_bit_rate, serial_interface_bit_rate_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 5

    # 4:serial_interface_detection_mode | Offset: 45, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 45
    serial_interface_detection_mode_raw = decode_int(_data_raw_, running_bit_offset, 3)
    serial_interface_detection_mode = master_dict['SERIAL_DETECTION_MODE'].get(serial_interface_detection_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('serialInterfaceDetectionMode', 'Serial Interface Detection Mode', None, None, serial_interface_detection_mode, serial_interface_detection_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 5:differential_source | Offset: 48, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    differential_source_raw = decode_int(_data_raw_, running_bit_offset, 4)
    differential_source = master_dict['DIFFERENTIAL_SOURCE'].get(differential_source_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('differentialSource', 'Differential Source', None, None, differential_source, differential_source_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 6:differential_operation_mode | Offset: 52, Length: 4, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 52
    differential_operation_mode_raw = decode_int(_data_raw_, running_bit_offset, 4)
    differential_operation_mode = master_dict['DIFFERENTIAL_MODE'].get(differential_operation_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('differentialOperationMode', 'Differential Operation Mode', None, None, differential_operation_mode, differential_operation_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 4

    # 7:reserved_56 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    reserved_56 = reserved_56_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_56', 'Reserved', None, None, reserved_56, reserved_56_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_129550(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129550."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # channel | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
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
    # frequency | Offset: 8, Length: 32, Resolution: 10, Field Type: NUMBER
    running_bit_offset = 8
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
    # serialInterfaceBitRate | Offset: 40, Length: 5, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("serialInterfaceBitRate")

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
        raise ValueError("Cant encode this message, 'Serial Interface Bit Rate' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Serial Interface Bit Rate' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Serial Interface Bit Rate' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # serialInterfaceDetectionMode | Offset: 45, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 45
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("serialInterfaceDetectionMode")

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
        raise ValueError("Cant encode this message, 'Serial Interface Detection Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Serial Interface Detection Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Serial Interface Detection Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # differentialSource | Offset: 48, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
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
    # differentialOperationMode | Offset: 52, Length: 4, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 52
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("differentialOperationMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DIFFERENTIAL_MODE(field.value)
    field_bit_length = 4
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Differential Operation Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Differential Operation Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Differential Operation Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_56 | Offset: 56, Length: 8, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_56")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 8
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
