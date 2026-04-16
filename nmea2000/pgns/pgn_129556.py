# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129556() -> bool:
    """Return True if PGN 129556 is a fast PGN."""
    return True
def decode_pgn_129556(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129556."""
    nmea2000Message = NMEA2000Message(PGN=129556, id='glonassAlmanacData', description='GLONASS Almanac Data')
    running_bit_offset = 0
    # 1:prn | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    prn = prn_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('prn', 'PRN', "Satellite ID number", None, prn, prn_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 2:na | Offset: 8, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    na = na_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('na', 'NA', "Calendar day count within the four year period beginning with the previous leap year", None, na, na_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 3:reserved_24 | Offset: 24, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    reserved_24 = reserved_24_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_24', 'Reserved', None, None, reserved_24, reserved_24_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 4:cna | Offset: 26, Length: 1, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 26
    cna = cna_raw = decode_number(_data_raw_, running_bit_offset, 1, False, 1, 0, 1)
    nmea2000Message.fields.append(NMEA2000Field('cna', 'CnA', "Generalized health of the satellite", None, cna, cna_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 1

    # 5:hna | Offset: 27, Length: 5, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 27
    hna = hna_raw = decode_number(_data_raw_, running_bit_offset, 5, False, 1, 0, 29)
    nmea2000Message.fields.append(NMEA2000Field('hna', 'HnA', "Carrier frequency number", None, hna, hna_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 5

    # 6:_epsilon_na | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    _epsilon_na = _epsilon_na_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('EpsilonNa', '(epsilon)nA', "Eccentricity", None, _epsilon_na, _epsilon_na_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:_deltatna_dot | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    _deltatna_dot = _deltatna_dot_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('DeltatnaDot', '(deltaTnA)DOT', "Rate of change of the draconitic circling time", None, _deltatna_dot, _deltatna_dot_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:_omega_na | Offset: 56, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    _omega_na = _omega_na_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('OmegaNa', '(omega)nA', "Rate of change of the draconitic circling time", None, _omega_na, _omega_na_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:_delta_tna | Offset: 72, Length: 24, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    _delta_tna = _delta_tna_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 16777212)
    nmea2000Message.fields.append(NMEA2000Field('DeltaTna', '(delta)TnA', "Correction to the average value of the draconitic circling time", None, _delta_tna, _delta_tna_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 10:tna | Offset: 96, Length: 24, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    tna = tna_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 16777212)
    nmea2000Message.fields.append(NMEA2000Field('tna', 'tnA', "Time of the ascension node", None, tna, tna_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 11:_lambda_na | Offset: 120, Length: 24, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    _lambda_na = _lambda_na_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 16777212)
    nmea2000Message.fields.append(NMEA2000Field('LambdaNa', '(lambda)nA', "Greenwich longitude of the ascension node", None, _lambda_na, _lambda_na_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 12:_delta_ina | Offset: 144, Length: 24, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 144
    _delta_ina = _delta_ina_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 1, 0, 16777212)
    nmea2000Message.fields.append(NMEA2000Field('DeltaIna', '(delta)inA', "Correction to the average value of the inclination angle", None, _delta_ina, _delta_ina_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 24

    # 13:_tau_ca | Offset: 168, Length: 28, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 168
    _tau_ca = _tau_ca_raw = decode_number(_data_raw_, running_bit_offset, 28, False, 1, 0, 268435452)
    nmea2000Message.fields.append(NMEA2000Field('TauCa', '(tau)cA', "System time scale correction", None, _tau_ca, _tau_ca_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 28

    # 14:_tau_na | Offset: 196, Length: 12, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 196
    _tau_na = _tau_na_raw = decode_number(_data_raw_, running_bit_offset, 12, False, 1, 0, 4092)
    nmea2000Message.fields.append(NMEA2000Field('TauNa', '(tau)nA', "Course value of the time scale shift", None, _tau_na, _tau_na_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 12

    return nmea2000Message

def encode_pgn_129556(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129556."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # prn | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("prn")

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
    # na | Offset: 8, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("na")

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
        raise ValueError("Cant encode this message, 'NA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'NA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'NA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_24 | Offset: 24, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_24")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 2
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
    # cna | Offset: 26, Length: 1, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 26
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("cna")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 1, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 1, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 1, False, 1)
    field_bit_length = 1
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'CnA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'CnA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'CnA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # hna | Offset: 27, Length: 5, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 27
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("hna")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 5, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 5, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 5, False, 1)
    field_bit_length = 5
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'HnA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'HnA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'HnA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # EpsilonNa | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("EpsilonNa")

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
        raise ValueError("Cant encode this message, '(epsilon)nA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(epsilon)nA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(epsilon)nA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # DeltatnaDot | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("DeltatnaDot")

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
        raise ValueError("Cant encode this message, '(deltaTnA)DOT' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(deltaTnA)DOT' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(deltaTnA)DOT' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # OmegaNa | Offset: 56, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("OmegaNa")

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
        raise ValueError("Cant encode this message, '(omega)nA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(omega)nA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(omega)nA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # DeltaTna | Offset: 72, Length: 24, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("DeltaTna")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 1)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, '(delta)TnA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(delta)TnA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(delta)TnA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tna | Offset: 96, Length: 24, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tna")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 1)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'tnA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'tnA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'tnA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # LambdaNa | Offset: 120, Length: 24, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("LambdaNa")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 1)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, '(lambda)nA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(lambda)nA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(lambda)nA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # DeltaIna | Offset: 144, Length: 24, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 144
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("DeltaIna")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 24, False, 1)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, '(delta)inA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(delta)inA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(delta)inA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # TauCa | Offset: 168, Length: 28, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 168
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("TauCa")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 28, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 28, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 28, False, 1)
    field_bit_length = 28
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, '(tau)cA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(tau)cA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(tau)cA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # TauNa | Offset: 196, Length: 12, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 196
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("TauNa")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 12, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 12, False, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 12, False, 1)
    field_bit_length = 12
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, '(tau)nA' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '(tau)nA' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '(tau)nA' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(26, byteorder="little")
