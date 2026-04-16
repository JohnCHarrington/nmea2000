# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65011() -> bool:
    """Return True if PGN 65011 is a fast PGN."""
    return False
def decode_pgn_65011(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65011."""
    nmea2000Message = NMEA2000Message(PGN=65011, id='utilityPhaseBBasicAcQuantities', description='Utility Phase B Basic AC Quantities')
    running_bit_offset = 0
    # 1:line_line_ac_rms_voltage | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    line_line_ac_rms_voltage = line_line_ac_rms_voltage_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('lineLineAcRmsVoltage', 'Line-Line AC RMS Voltage', None, 'V', line_line_ac_rms_voltage, line_line_ac_rms_voltage_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:line_neutral_ac_rms_voltage | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    line_neutral_ac_rms_voltage = line_neutral_ac_rms_voltage_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('lineNeutralAcRmsVoltage', 'Line-Neutral AC RMS Voltage', None, 'V', line_neutral_ac_rms_voltage, line_neutral_ac_rms_voltage_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:ac_frequency | Offset: 32, Length: 16, Signed: False Resolution: 0.0078125, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    ac_frequency = ac_frequency_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.0078125, 0, 511.96875)
    nmea2000Message.fields.append(NMEA2000Field('acFrequency', 'AC Frequency', None, 'Hz', ac_frequency, ac_frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 4:ac_rms_current | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    ac_rms_current = ac_rms_current_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('acRmsCurrent', 'AC RMS Current', None, 'A', ac_rms_current, ac_rms_current_raw, PhysicalQuantities.ELECTRICAL_CURRENT, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_65011(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65011."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # lineLineAcRmsVoltage | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("lineLineAcRmsVoltage")

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
        raise ValueError("Cant encode this message, 'Line-Line AC RMS Voltage' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Line-Line AC RMS Voltage' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Line-Line AC RMS Voltage' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # lineNeutralAcRmsVoltage | Offset: 16, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("lineNeutralAcRmsVoltage")

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
        raise ValueError("Cant encode this message, 'Line-Neutral AC RMS Voltage' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Line-Neutral AC RMS Voltage' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Line-Neutral AC RMS Voltage' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # acFrequency | Offset: 32, Length: 16, Resolution: 0.0078125, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("acFrequency")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.0078125):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 0.0078125)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 0.0078125)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'AC Frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'AC Frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'AC Frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # acRmsCurrent | Offset: 48, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("acRmsCurrent")

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
        raise ValueError("Cant encode this message, 'AC RMS Current' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'AC RMS Current' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'AC RMS Current' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
