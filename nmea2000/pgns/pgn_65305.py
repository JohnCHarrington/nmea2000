# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_65305() -> bool:
    """Return True if PGN 65305 is a fast PGN."""
    return False
# Complex PGN. number of matches: 5
def decode_pgn_65305(data_raw: int, data_length_bits: int | None = None) -> NMEA2000Message | None:
    # simnetDeviceStatus | Description: Simnet: Device Status
    if (
        (((data_raw >> 0) & 0x7FF) == 1857) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 24) & 0xFF) == 2)
        ):
        return decode_pgn_65305_simnetDeviceStatus(data_raw, data_length_bits)
    
    # simnetDeviceStatusRequest | Description: Simnet: Device Status Request
    if (
        (((data_raw >> 0) & 0x7FF) == 1857) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 24) & 0xFF) == 3)
        ):
        return decode_pgn_65305_simnetDeviceStatusRequest(data_raw, data_length_bits)
    
    # simnetPilotMode | Description: Simnet: Pilot Mode
    if (
        (((data_raw >> 0) & 0x7FF) == 1857) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 24) & 0xFF) == 10)
        ):
        return decode_pgn_65305_simnetPilotMode(data_raw, data_length_bits)
    
    # simnetDeviceModeRequest | Description: Simnet: Device Mode Request
    if (
        (((data_raw >> 0) & 0x7FF) == 1857) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 24) & 0xFF) == 11)
        ):
        return decode_pgn_65305_simnetDeviceModeRequest(data_raw, data_length_bits)
    
    # simnetSailingProcessorStatus | Description: Simnet: Sailing Processor Status
    if (
        (((data_raw >> 0) & 0x7FF) == 1857) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 24) & 0xFF) == 23)
        ):
        return decode_pgn_65305_simnetSailingProcessorStatus(data_raw, data_length_bits)
    
    
    return None
    
def decode_pgn_65305_simnetDeviceStatus(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65305."""
    nmea2000Message = NMEA2000Message(PGN=65305, id='simnetDeviceStatus', description='Simnet: Device Status', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1857, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Simrad", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 2:reserved_11 | Offset: 11, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    reserved_11 = reserved_11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_11', 'Reserved', None, None, reserved_11, reserved_11_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 3:industry_code | Offset: 13, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 13
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Marine Industry", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 4:model | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    model_raw = decode_int(_data_raw_, running_bit_offset, 8)
    model = master_dict['SIMNET_DEVICE_MODEL'].get(model_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('model', 'Model', None, None, model, model_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:report | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 2, PartOfPrimaryKey: ,
    running_bit_offset = 24
    report_raw = decode_int(_data_raw_, running_bit_offset, 8)
    report = master_dict['SIMNET_DEVICE_REPORT'].get(report_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('report', 'Report', "Status", None, report, report_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:status | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    status_raw = decode_int(_data_raw_, running_bit_offset, 8)
    status = master_dict['SIMNET_AP_STATUS'].get(status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 7:spare | Offset: 40, Length: 24, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 24)
    nmea2000Message.fields.append(NMEA2000Field('spare7', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_65305_simnetDeviceStatus(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65305."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # manufacturerCode | Offset: 0, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_11 | Offset: 11, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_11")

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
    # industryCode | Offset: 13, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 13
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # model | Offset: 16, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("model")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_MODEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # report | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("report")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_REPORT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Report' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Report' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Report' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # status | Offset: 32, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("status")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_AP_STATUS(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Status' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Status' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Status' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # spare7 | Offset: 40, Length: 24, Resolution: 1, Field Type: SPARE
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare7")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 24
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_65305_simnetDeviceStatusRequest(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65305."""
    nmea2000Message = NMEA2000Message(PGN=65305, id='simnetDeviceStatusRequest', description='Simnet: Device Status Request', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1857, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Simrad", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 2:reserved_11 | Offset: 11, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    reserved_11 = reserved_11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_11', 'Reserved', None, None, reserved_11, reserved_11_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 3:industry_code | Offset: 13, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 13
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Marine Industry", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 4:model | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    model_raw = decode_int(_data_raw_, running_bit_offset, 8)
    model = master_dict['SIMNET_DEVICE_MODEL'].get(model_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('model', 'Model', None, None, model, model_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:report | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 3, PartOfPrimaryKey: ,
    running_bit_offset = 24
    report_raw = decode_int(_data_raw_, running_bit_offset, 8)
    report = master_dict['SIMNET_DEVICE_REPORT'].get(report_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('report', 'Report', "Send Status", None, report, report_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:spare | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 32)
    nmea2000Message.fields.append(NMEA2000Field('spare6', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_65305_simnetDeviceStatusRequest(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65305."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # manufacturerCode | Offset: 0, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_11 | Offset: 11, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_11")

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
    # industryCode | Offset: 13, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 13
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # model | Offset: 16, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("model")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_MODEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # report | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("report")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_REPORT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Report' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Report' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Report' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # spare6 | Offset: 32, Length: 32, Resolution: 1, Field Type: SPARE
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare6")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 32
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_65305_simnetPilotMode(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65305."""
    nmea2000Message = NMEA2000Message(PGN=65305, id='simnetPilotMode', description='Simnet: Pilot Mode', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1857, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Simrad", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 2:reserved_11 | Offset: 11, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    reserved_11 = reserved_11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_11', 'Reserved', None, None, reserved_11, reserved_11_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 3:industry_code | Offset: 13, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 13
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Marine Industry", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 4:model | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    model_raw = decode_int(_data_raw_, running_bit_offset, 8)
    model = master_dict['SIMNET_DEVICE_MODEL'].get(model_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('model', 'Model', None, None, model, model_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:report | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 10, PartOfPrimaryKey: ,
    running_bit_offset = 24
    report_raw = decode_int(_data_raw_, running_bit_offset, 8)
    report = master_dict['SIMNET_DEVICE_REPORT'].get(report_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('report', 'Report', "Mode", None, report, report_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:mode | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: BITLOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    mode_raw = decode_int(_data_raw_, running_bit_offset, 16)
    mode = decode_bit_lookup(mode_raw, master_flags_dict['SIMNET_AP_MODE_BITFIELD'])
    nmea2000Message.fields.append(NMEA2000Field('mode', 'Mode', None, None, mode, mode_raw, None, FieldTypes.BITLOOKUP, False))
    running_bit_offset += 16

    # 7:spare | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 16)
    nmea2000Message.fields.append(NMEA2000Field('spare7', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_65305_simnetPilotMode(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65305."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # manufacturerCode | Offset: 0, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_11 | Offset: 11, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_11")

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
    # industryCode | Offset: 13, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 13
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # model | Offset: 16, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("model")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_MODEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # report | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("report")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_REPORT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Report' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Report' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Report' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mode | Offset: 32, Length: 16, Resolution: 1, Field Type: BITLOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mode")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else encode_bit_lookup(field.value, master_flags_dict['SIMNET_AP_MODE_BITFIELD'])
    field_bit_length = 16
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
    # spare7 | Offset: 48, Length: 16, Resolution: 1, Field Type: SPARE
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare7")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 16
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_65305_simnetDeviceModeRequest(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65305."""
    nmea2000Message = NMEA2000Message(PGN=65305, id='simnetDeviceModeRequest', description='Simnet: Device Mode Request', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1857, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Simrad", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 2:reserved_11 | Offset: 11, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    reserved_11 = reserved_11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_11', 'Reserved', None, None, reserved_11, reserved_11_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 3:industry_code | Offset: 13, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 13
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Marine Industry", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 4:model | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    model_raw = decode_int(_data_raw_, running_bit_offset, 8)
    model = master_dict['SIMNET_DEVICE_MODEL'].get(model_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('model', 'Model', None, None, model, model_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:report | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 11, PartOfPrimaryKey: ,
    running_bit_offset = 24
    report_raw = decode_int(_data_raw_, running_bit_offset, 8)
    report = master_dict['SIMNET_DEVICE_REPORT'].get(report_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('report', 'Report', "Send Mode", None, report, report_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:spare | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: SPARE, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    spare = spare_raw = decode_int(_data_raw_, running_bit_offset, 32)
    nmea2000Message.fields.append(NMEA2000Field('spare6', 'Spare', None, None, spare, spare_raw, None, FieldTypes.SPARE, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_65305_simnetDeviceModeRequest(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65305."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # manufacturerCode | Offset: 0, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_11 | Offset: 11, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_11")

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
    # industryCode | Offset: 13, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 13
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # model | Offset: 16, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("model")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_MODEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # report | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("report")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_REPORT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Report' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Report' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Report' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # spare6 | Offset: 32, Length: 32, Resolution: 1, Field Type: SPARE
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("spare6")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Spare' must be an integer")
    field_bit_length = 32
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
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_65305_simnetSailingProcessorStatus(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 65305."""
    nmea2000Message = NMEA2000Message(PGN=65305, id='simnetSailingProcessorStatus', description='Simnet: Sailing Processor Status', ttl=timedelta(milliseconds=1000))
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1857, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Simrad", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 11

    # 2:reserved_11 | Offset: 11, Length: 2, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 11
    reserved_11 = reserved_11_raw = decode_int(_data_raw_, running_bit_offset, 2)
    nmea2000Message.fields.append(NMEA2000Field('reserved_11', 'Reserved', None, None, reserved_11, reserved_11_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 2

    # 3:industry_code | Offset: 13, Length: 3, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 4, PartOfPrimaryKey: ,
    running_bit_offset = 13
    industry_code_raw = decode_int(_data_raw_, running_bit_offset, 3)
    industry_code = master_dict['INDUSTRY_CODE'].get(industry_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('industryCode', 'Industry Code', "Marine Industry", None, industry_code, industry_code_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 3

    # 4:model | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    model_raw = decode_int(_data_raw_, running_bit_offset, 8)
    model = master_dict['SIMNET_DEVICE_MODEL'].get(model_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('model', 'Model', None, None, model, model_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:report | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 23, PartOfPrimaryKey: ,
    running_bit_offset = 24
    report_raw = decode_int(_data_raw_, running_bit_offset, 8)
    report = master_dict['SIMNET_DEVICE_REPORT'].get(report_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('report', 'Report', "Sailing Processor Status", None, report, report_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:data | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    data = data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 32))
    nmea2000Message.fields.append(NMEA2000Field('data', 'Data', None, None, data, data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_65305_simnetSailingProcessorStatus(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 65305."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    # manufacturerCode | Offset: 0, Length: 11, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("manufacturerCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_MANUFACTURER_CODE(field.value)
    field_bit_length = 11
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Manufacturer Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Manufacturer Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_11 | Offset: 11, Length: 2, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 11
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_11")

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
    # industryCode | Offset: 13, Length: 3, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 13
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("industryCode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_INDUSTRY_CODE(field.value)
    field_bit_length = 3
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Industry Code' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Industry Code' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Industry Code' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # model | Offset: 16, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("model")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_MODEL(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Model' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Model' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Model' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # report | Offset: 24, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("report")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_SIMNET_DEVICE_REPORT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Report' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Report' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Report' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # data | Offset: 32, Length: 32, Resolution: 1, Field Type: BINARY
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("data")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Data' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Data' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Data' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")
