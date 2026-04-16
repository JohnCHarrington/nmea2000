# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65009() -> bool:
    """Return True if PGN 65009 is a fast PGN."""
    return False
def decode_pgn_65009(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65009."""
    nmea2000Message = NMEA2000Message(PGN=65009, id='utilityPhaseBAcReactivePower', description='Utility Phase B AC Reactive Power')
    running_bit_offset = 0
    # 1:reactive_power | Offset: 0, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    reactive_power = reactive_power_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('reactivePower', 'Reactive Power', None, 'VAR', reactive_power, reactive_power_raw, PhysicalQuantities.ELECTRICAL_REACTIVE_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 2:power_factor | Offset: 16, Length: 16, Signed: False Resolution: 6.10352e-05, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    power_factor = power_factor_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 6.10352e-05, 0, 3.999755859375)
    nmea2000Message.fields.append(NMEA2000Field('powerFactor', 'Power factor', None, 'Cos Phi', power_factor, power_factor_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:power_factor_lagging | Offset: 32, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    power_factor_lagging_raw = decode_int(_data_raw_, running_bit_offset, 2)
    power_factor_lagging = master_dict['POWER_FACTOR'].get(power_factor_lagging_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('powerFactorLagging', 'Power Factor Lagging', None, None, power_factor_lagging, power_factor_lagging_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:reserved_34 | Offset: 34, Length: 30, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 34
    reserved_34 = reserved_34_raw = decode_int(_data_raw_, running_bit_offset, 30)
    nmea2000Message.fields.append(NMEA2000Field('reserved_34', 'Reserved', None, None, reserved_34, reserved_34_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 30

    return nmea2000Message

def encode_pgn_65009(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65009."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # reactivePower | Offset: 0, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reactivePower")

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
        raise ValueError("Cant encode this message, 'Reactive Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Reactive Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Reactive Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # powerFactor | Offset: 16, Length: 16, Resolution: 6.10352e-05, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("powerFactor")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 6.10352e-05):
        field_value = encode_number_raw(field.raw_value, 16, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 16, False, 6.10352e-05)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 16, False, 6.10352e-05)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Power factor' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Power factor' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Power factor' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # powerFactorLagging | Offset: 32, Length: 2, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("powerFactorLagging")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_POWER_FACTOR(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Power Factor Lagging' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Power Factor Lagging' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Power Factor Lagging' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_34 | Offset: 34, Length: 30, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 34
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_34")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 30
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
