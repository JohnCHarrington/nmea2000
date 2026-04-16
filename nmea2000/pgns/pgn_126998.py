# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_126998() -> bool:
    """Return True if PGN 126998 is a fast PGN."""
    return True
def decode_pgn_126998(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 126998."""
    nmea2000Message = NMEA2000Message(PGN=126998, id='configurationInformation', description='Configuration Information')
    running_bit_offset = 0
    # 1:installation_description__1 | Offset: 0, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    installation_description__1_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    installation_description__1 = installation_description__1_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('installationDescription1', 'Installation Description #1', None, None, installation_description__1, installation_description__1_raw, None, FieldTypes.STRING_LAU, False))
    

    # 2:installation_description__2 | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    installation_description__2_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    installation_description__2 = installation_description__2_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('installationDescription2', 'Installation Description #2', None, None, installation_description__2, installation_description__2_raw, None, FieldTypes.STRING_LAU, False))
    

    # 3:manufacturer_information | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    manufacturer_information_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    manufacturer_information = manufacturer_information_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('manufacturerInformation', 'Manufacturer Information', None, None, manufacturer_information, manufacturer_information_raw, None, FieldTypes.STRING_LAU, False))
    

    return nmea2000Message

def encode_pgn_126998(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 126998."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # installationDescription1 | Offset: 0, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("installationDescription1")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Installation Description #1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Installation Description #1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Installation Description #1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # installationDescription2 | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("installationDescription2")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Installation Description #2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Installation Description #2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Installation Description #2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # manufacturerInformation | Offset: , Length: , Resolution: , Field Type: STRING_LAU
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerInformation")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Information' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Information' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Information' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
