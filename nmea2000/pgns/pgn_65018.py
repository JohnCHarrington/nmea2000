# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65018() -> bool:
    """Return True if PGN 65018 is a fast PGN."""
    return False
def decode_pgn_65018(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65018."""
    nmea2000Message = NMEA2000Message(PGN=65018, id='generatorTotalAcEnergy', description='Generator Total AC Energy')
    running_bit_offset = 0
    # 1:total_energy_export | Offset: 0, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    total_energy_export = total_energy_export_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('totalEnergyExport', 'Total Energy Export', None, 'kWh', total_energy_export, total_energy_export_raw, PhysicalQuantities.ELECTRICAL_ENERGY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 2:total_energy_import | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    total_energy_import = total_energy_import_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('totalEnergyImport', 'Total Energy Import', None, 'kWh', total_energy_import, total_energy_import_raw, PhysicalQuantities.ELECTRICAL_ENERGY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_65018(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65018."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # totalEnergyExport | Offset: 0, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalEnergyExport")

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
        raise ValueError("Cant encode this message, 'Total Energy Export' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total Energy Export' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total Energy Export' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # totalEnergyImport | Offset: 32, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("totalEnergyImport")

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
        raise ValueError("Cant encode this message, 'Total Energy Import' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Total Energy Import' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Total Energy Import' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
