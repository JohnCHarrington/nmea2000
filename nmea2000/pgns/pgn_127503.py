# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_127503() -> bool:
    """Return True if PGN 127503 is a fast PGN."""
    return True
def decode_pgn_127503(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 127503."""
    nmea2000Message = NMEA2000Message(PGN=127503, id='acInputStatus', description='AC Input Status', ttl=timedelta(milliseconds=1500))
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:instance | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 0
    instance = instance_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('instance', 'Instance', None, None, instance, instance_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 2:number_of_lines | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    number_of_lines = number_of_lines_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('numberOfLines', 'Number of Lines', None, None, number_of_lines, number_of_lines_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 3:line | Offset: 16, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    _repeating_field_set_1_offset = running_bit_offset
    line_raw = decode_int(_data_raw_, running_bit_offset, 2)
    line = master_dict['AC_LINE'].get(line_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('line', 'Line', None, None, line, line_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 4:acceptability | Offset: 18, Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 18
    acceptability_raw = decode_int(_data_raw_, running_bit_offset, 2)
    acceptability = master_dict['ACCEPTABILITY'].get(acceptability_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('acceptability', 'Acceptability', None, None, acceptability, acceptability_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 5:reserved_20 | Offset: 20, Length: 4, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 20
    reserved_20 = reserved_20_raw = decode_int(_data_raw_, running_bit_offset, 4)
    nmea2000Message.fields.append(NMEA2000Field('reserved_20', 'Reserved', None, None, reserved_20, reserved_20_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 4

    # 6:voltage | Offset: 24, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    voltage = voltage_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('voltage', 'Voltage', None, 'V', voltage, voltage_raw, PhysicalQuantities.POTENTIAL_DIFFERENCE, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:current | Offset: 40, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    current = current_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('current', 'Current', None, 'A', current, current_raw, PhysicalQuantities.ELECTRICAL_CURRENT, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:frequency | Offset: 56, Length: 16, Signed: False Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    frequency = frequency_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
    nmea2000Message.fields.append(NMEA2000Field('frequency', 'Frequency', None, 'Hz', frequency, frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:breaker_size | Offset: 72, Length: 16, Signed: False Resolution: 0.1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    breaker_size = breaker_size_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
    nmea2000Message.fields.append(NMEA2000Field('breakerSize', 'Breaker Size', None, 'A', breaker_size, breaker_size_raw, PhysicalQuantities.ELECTRICAL_CURRENT, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 10:real_power | Offset: 88, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    real_power = real_power_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('realPower', 'Real Power', None, 'W', real_power, real_power_raw, PhysicalQuantities.ELECTRICAL_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 11:reactive_power | Offset: 120, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    reactive_power = reactive_power_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('reactivePower', 'Reactive Power', None, 'VAR', reactive_power, reactive_power_raw, PhysicalQuantities.ELECTRICAL_REACTIVE_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 12:power_factor | Offset: 152, Length: 8, Signed: True Resolution: 0.01, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    power_factor = power_factor_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 0.01, -1, 1)
    nmea2000Message.fields.append(NMEA2000Field('powerFactor', 'Power factor', None, 'Cos Phi', power_factor, power_factor_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(number_of_lines_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        line_raw = decode_int(_data_raw_, running_bit_offset, 2)
        line = master_dict['AC_LINE'].get(line_raw, None)
        running_bit_offset += 2
        repeating_entry["line"] = _repeating_entry_value(line, line_raw)
    
        acceptability_raw = decode_int(_data_raw_, running_bit_offset, 2)
        acceptability = master_dict['ACCEPTABILITY'].get(acceptability_raw, None)
        running_bit_offset += 2
        repeating_entry["acceptability"] = _repeating_entry_value(acceptability, acceptability_raw)
    
        reserved_20 = reserved_20_raw = decode_int(_data_raw_, running_bit_offset, 4)
        running_bit_offset += 4
        repeating_entry["reserved_20"] = _repeating_entry_value(reserved_20, reserved_20_raw)
    
        voltage = voltage_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
        running_bit_offset += 16
        repeating_entry["voltage"] = _repeating_entry_value(voltage, voltage_raw)
    
        current = current_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
        running_bit_offset += 16
        repeating_entry["current"] = _repeating_entry_value(current, current_raw)
    
        frequency = frequency_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.01, 0, 655.32)
        running_bit_offset += 16
        repeating_entry["frequency"] = _repeating_entry_value(frequency, frequency_raw)
    
        breaker_size = breaker_size_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 0.1, 0, 6553.2)
        running_bit_offset += 16
        repeating_entry["breakerSize"] = _repeating_entry_value(breaker_size, breaker_size_raw)
    
        real_power = real_power_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["realPower"] = _repeating_entry_value(real_power, real_power_raw)
    
        reactive_power = reactive_power_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["reactivePower"] = _repeating_entry_value(reactive_power, reactive_power_raw)
    
        power_factor = power_factor_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 0.01, -1, 1)
        running_bit_offset += 8
        repeating_entry["powerFactor"] = _repeating_entry_value(power_factor, power_factor_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "line",
                "acceptability",
                "reserved_20",
                "voltage",
                "current",
                "frequency",
                "breakerSize",
                "realPower",
                "reactivePower",
                "powerFactor",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_127503(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 127503."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "line",
        "acceptability",
        "reserved_20",
        "voltage",
        "current",
        "frequency",
        "breakerSize",
        "realPower",
        "reactivePower",
        "powerFactor",
    ))
    # instance | Offset: 0, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("instance")

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
        raise ValueError("Cant encode this message, 'Instance' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Instance' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Instance' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # numberOfLines | Offset: 8, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("numberOfLines")

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
        raise ValueError("Cant encode this message, 'Number of Lines' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number of Lines' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number of Lines' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 16
    for repeating_entry in repeating_field_set_1_entries:
        # line | Offset: 16, Length: 2, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("line")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Line'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_AC_LINE(field.value)
        field_bit_length = 2
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Line' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Line' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Line' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # acceptability | Offset: 18, Length: 2, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("acceptability")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Acceptability'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_ACCEPTABILITY(field.value)
        field_bit_length = 2
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'Acceptability' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Acceptability' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Acceptability' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # reserved_20 | Offset: 20, Length: 4, Resolution: 1, Field Type: RESERVED
        field = repeating_entry.get("reserved_20")
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
        # voltage | Offset: 24, Length: 16, Resolution: 0.01, Field Type: NUMBER
        field = repeating_entry.get("voltage")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Voltage'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Voltage' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Voltage' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Voltage' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # current | Offset: 40, Length: 16, Resolution: 0.1, Field Type: NUMBER
        field = repeating_entry.get("current")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Current'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Current' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Current' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Current' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # frequency | Offset: 56, Length: 16, Resolution: 0.01, Field Type: NUMBER
        field = repeating_entry.get("frequency")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Frequency'")
        field_offset = running_bit_offset
    
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
        # breakerSize | Offset: 72, Length: 16, Resolution: 0.1, Field Type: NUMBER
        field = repeating_entry.get("breakerSize")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Breaker Size'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Breaker Size' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Breaker Size' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Breaker Size' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # realPower | Offset: 88, Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("realPower")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Real Power'")
        field_offset = running_bit_offset
    
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
            raise ValueError("Cant encode this message, 'Real Power' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Real Power' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Real Power' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # reactivePower | Offset: 120, Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("reactivePower")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Reactive Power'")
        field_offset = running_bit_offset
    
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
        # powerFactor | Offset: 152, Length: 8, Resolution: 0.01, Field Type: NUMBER
        field = repeating_entry.get("powerFactor")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Power factor'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 0.01):
            field_value = encode_number_raw(field.raw_value, 8, True)
        elif isinstance(field.raw_value, (int, float)):
            field_value = encode_number(field.raw_value, 8, True, 0.01)
        else:
            assert field.value is None or isinstance(field.value, (int, float))
            field_value = encode_number(field.value, 8, True, 0.01)
        field_bit_length = 8
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
    
    
    
    
    
    
    
    
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
