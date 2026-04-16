# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129799() -> bool:
    """Return True if PGN 129799 is a fast PGN."""
    return True
def decode_pgn_129799(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129799."""
    nmea2000Message = NMEA2000Message(PGN=129799, id='radioFrequencyModePower', description='Radio Frequency/Mode/Power')
    running_bit_offset = 0
    # 1:rx_frequency | Offset: 0, Length: 32, Signed: False Resolution: 10, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    rx_frequency = rx_frequency_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 10, 0, 42949672920)
    nmea2000Message.fields.append(NMEA2000Field('rxFrequency', 'Rx Frequency', None, 'Hz', rx_frequency, rx_frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 2:tx_frequency | Offset: 32, Length: 32, Signed: False Resolution: 10, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    tx_frequency = tx_frequency_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 10, 0, 42949672920)
    nmea2000Message.fields.append(NMEA2000Field('txFrequency', 'Tx Frequency', None, 'Hz', tx_frequency, tx_frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 3:radio_channel | Offset: 64, Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    radio_channel, radio_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('radioChannel', 'Radio Channel', "MF/HF telephone channels to have first digit 3 followed by ITU channel numbers with leading zeros as required. MF/HF teletype channels to have first digit 4; the send and third digit give the frequency bads; and the fourth to sixth digits ITU channel numbers; each with leading zeros as required. VHF channels to have the first digit 9 followed by zero. The next digit is 1 indicating the ship stations transmit frequency is being used as a simplex channel frequency, or 2 indicating the cost stations transmit frequency is being used as a simplex channel frequency, 0 otherwise. THe remaining three numbers are the VHF channel numbers with leading zeros as required.", None, radio_channel, radio_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 4:tx_power | Offset: 112, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 112
    tx_power = tx_power_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('txPower', 'Tx Power', None, 'W', tx_power, tx_power_raw, PhysicalQuantities.ELECTRICAL_POWER, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 5:mode | Offset: 128, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 128
    mode_raw = decode_int(_data_raw_, running_bit_offset, 8)
    mode = master_dict['TELEPHONE_MODE'].get(mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('mode', 'Mode', None, None, mode, mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:channel_bandwidth | Offset: 136, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 136
    channel_bandwidth = channel_bandwidth_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('channelBandwidth', 'Channel Bandwidth', None, 'Hz', channel_bandwidth, channel_bandwidth_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_129799(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129799."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # rxFrequency | Offset: 0, Length: 32, Resolution: 10, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("rxFrequency")

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
        raise ValueError("Cant encode this message, 'Rx Frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Rx Frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Rx Frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # txFrequency | Offset: 32, Length: 32, Resolution: 10, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("txFrequency")

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
        raise ValueError("Cant encode this message, 'Tx Frequency' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tx Frequency' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tx Frequency' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # radioChannel | Offset: 64, Length: 48, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("radioChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Radio Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Radio Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Radio Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # txPower | Offset: 112, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 112
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("txPower")

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
        raise ValueError("Cant encode this message, 'Tx Power' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tx Power' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tx Power' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mode | Offset: 128, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 128
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_TELEPHONE_MODE(field.value)
    field_bit_length = 8
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
    # channelBandwidth | Offset: 136, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 136
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channelBandwidth")

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
        raise ValueError("Cant encode this message, 'Channel Bandwidth' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Channel Bandwidth' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Channel Bandwidth' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(19, byteorder="little")
