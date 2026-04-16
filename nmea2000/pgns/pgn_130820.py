# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_130820() -> bool:
    """Return True if PGN 130820 is a fast PGN."""
    return True
# Complex PGN. number of matches: 40
def decode_pgn_130820(data_raw: int, data_length_bits: int | None = None) -> NMEA2000Message | None:
    # simnetReprogramStatus | Description: Simnet: Reprogram Status
    if (
        (((data_raw >> 0) & 0x7FF) == 1857) and
        (((data_raw >> 13) & 0x7) == 4)
        ):
        return decode_pgn_130820_simnetReprogramStatus(data_raw, data_length_bits)
    
    # furunoUnknown130820 | Description: Furuno: Unknown 130820
    if (
        (((data_raw >> 0) & 0x7FF) == 1855) and
        (((data_raw >> 13) & 0x7) == 4)
        ):
        return decode_pgn_130820_furunoUnknown130820(data_raw, data_length_bits)
    
    # fusionVersions | Description: Fusion: Versions
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32769)
        ):
        return decode_pgn_130820_fusionVersions(data_raw, data_length_bits)
    
    # fusionSource | Description: Fusion: Source
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32770)
        ):
        return decode_pgn_130820_fusionSource(data_raw, data_length_bits)
    
    # fusionSourceCount | Description: Fusion: Source Count
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32771)
        ):
        return decode_pgn_130820_fusionSourceCount(data_raw, data_length_bits)
    
    # fusionMedia | Description: Fusion: Media
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32772)
        ):
        return decode_pgn_130820_fusionMedia(data_raw, data_length_bits)
    
    # fusionTrackName | Description: Fusion: Track Name
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32773)
        ):
        return decode_pgn_130820_fusionTrackName(data_raw, data_length_bits)
    
    # fusionArtistName | Description: Fusion: Artist Name
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32774)
        ):
        return decode_pgn_130820_fusionArtistName(data_raw, data_length_bits)
    
    # fusionAlbumName | Description: Fusion: Album Name
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32775)
        ):
        return decode_pgn_130820_fusionAlbumName(data_raw, data_length_bits)
    
    # fusionDeviceName | Description: Fusion: Device Name
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32801)
        ):
        return decode_pgn_130820_fusionDeviceName(data_raw, data_length_bits)
    
    # fusionZoneName | Description: Fusion: Zone Name
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32813)
        ):
        return decode_pgn_130820_fusionZoneName(data_raw, data_length_bits)
    
    # fusionTrackPosition | Description: Fusion: Track Position
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32777)
        ):
        return decode_pgn_130820_fusionTrackPosition(data_raw, data_length_bits)
    
    # fusionTuner | Description: Fusion: Tuner
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32779)
        ):
        return decode_pgn_130820_fusionTuner(data_raw, data_length_bits)
    
    # fusionMarineTuner | Description: Fusion: MARINE_TUNER
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32780)
        ):
        return decode_pgn_130820_fusionMarineTuner(data_raw, data_length_bits)
    
    # fusionMarineSquelch | Description: Fusion: Marine Squelch
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32781)
        ):
        return decode_pgn_130820_fusionMarineSquelch(data_raw, data_length_bits)
    
    # fusionMarineScanMode | Description: Fusion: Marine Scan Mode
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32782)
        ):
        return decode_pgn_130820_fusionMarineScanMode(data_raw, data_length_bits)
    
    # fusionMenuItem | Description: Fusion: Menu Item
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32785)
        ):
        return decode_pgn_130820_fusionMenuItem(data_raw, data_length_bits)
    
    # fusionAuxGain | Description: Fusion: Aux Gain
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32787)
        ):
        return decode_pgn_130820_fusionAuxGain(data_raw, data_length_bits)
    
    # fusionUsbRepeatStatus | Description: Fusion: USB Repeat Status
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32788) and
        (((data_raw >> 32) & 0xFFFFFFFF) == 9)
        ):
        return decode_pgn_130820_fusionUsbRepeatStatus(data_raw, data_length_bits)
    
    # fusionSetting | Description: Fusion: Setting
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32788)
        ):
        return decode_pgn_130820_fusionSetting(data_raw, data_length_bits)
    
    # fusionSettings | Description: Fusion: Settings
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32789)
        ):
        return decode_pgn_130820_fusionSettings(data_raw, data_length_bits)
    
    # fusionMute | Description: Fusion: Mute
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32791)
        ):
        return decode_pgn_130820_fusionMute(data_raw, data_length_bits)
    
    # fusionBalance | Description: Fusion: Balance
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32792)
        ):
        return decode_pgn_130820_fusionBalance(data_raw, data_length_bits)
    
    # fusionLowPassFilter | Description: Fusion: Low Pass Filter
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32793)
        ):
        return decode_pgn_130820_fusionLowPassFilter(data_raw, data_length_bits)
    
    # fusionSublevels | Description: Fusion: Sublevels
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32794)
        ):
        return decode_pgn_130820_fusionSublevels(data_raw, data_length_bits)
    
    # fusionEq | Description: Fusion: EQ
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32795)
        ):
        return decode_pgn_130820_fusionEq(data_raw, data_length_bits)
    
    # fusionVolumeLimits | Description: Fusion: Volume Limits
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32796)
        ):
        return decode_pgn_130820_fusionVolumeLimits(data_raw, data_length_bits)
    
    # fusionVolumes | Description: Fusion: Volumes
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32797)
        ):
        return decode_pgn_130820_fusionVolumes(data_raw, data_length_bits)
    
    # fusionCapabilities | Description: Fusion: Capabilities
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32798)
        ):
        return decode_pgn_130820_fusionCapabilities(data_raw, data_length_bits)
    
    # fusionLineLevelControl | Description: Fusion: Line Level Control
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32799)
        ):
        return decode_pgn_130820_fusionLineLevelControl(data_raw, data_length_bits)
    
    # fusionPowerState | Description: Fusion: Power State
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32800)
        ):
        return decode_pgn_130820_fusionPowerState(data_raw, data_length_bits)
    
    # fusionSiriusxm | Description: Fusion: SiriusXM
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32802)
        ):
        return decode_pgn_130820_fusionSiriusxm(data_raw, data_length_bits)
    
    # fusionSiriusxmChannel | Description: Fusion: SiriusXM Channel
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32804)
        ):
        return decode_pgn_130820_fusionSiriusxmChannel(data_raw, data_length_bits)
    
    # fusionSiriusxmTitle | Description: Fusion: SiriusXM Title
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32805)
        ):
        return decode_pgn_130820_fusionSiriusxmTitle(data_raw, data_length_bits)
    
    # fusionSiriusxmArtist | Description: Fusion: SiriusXM Artist
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32806)
        ):
        return decode_pgn_130820_fusionSiriusxmArtist(data_raw, data_length_bits)
    
    # fusionSiriusxmContentInfo | Description: Fusion: SiriusXM Content Info
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32807)
        ):
        return decode_pgn_130820_fusionSiriusxmContentInfo(data_raw, data_length_bits)
    
    # fusionSiriusxmCategory | Description: Fusion: SiriusXM Category
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32808)
        ):
        return decode_pgn_130820_fusionSiriusxmCategory(data_raw, data_length_bits)
    
    # fusionSiriusxmSignal | Description: Fusion: SiriusXM Signal
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32809)
        ):
        return decode_pgn_130820_fusionSiriusxmSignal(data_raw, data_length_bits)
    
    # fusionSiriusxmPresets | Description: Fusion: SiriusXM Presets
    if (
        (((data_raw >> 0) & 0x7FF) == 419) and
        (((data_raw >> 13) & 0x7) == 4) and
        (((data_raw >> 16) & 0xFFFF) == 32812)
        ):
        return decode_pgn_130820_fusionSiriusxmPresets(data_raw, data_length_bits)
    
    # maretronAlertResponse | Description: Maretron: Alert Response
    if (
        (((data_raw >> 0) & 0x7FF) == 137) and
        (((data_raw >> 13) & 0x7) == 4)
        ):
        return decode_pgn_130820_maretronAlertResponse(data_raw, data_length_bits)
    
    
    return None
    
def decode_pgn_130820_simnetReprogramStatus(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='simnetReprogramStatus', description='Simnet: Reprogram Status')
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

    # 4:reserved_16 | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    reserved_16 = reserved_16_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nmea2000Message.fields.append(NMEA2000Field('reserved_16', 'Reserved', None, None, reserved_16, reserved_16_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 8

    # 5:status | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    status = status_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:reserved_32 | Offset: 32, Length: 24, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    reserved_32 = reserved_32_raw = decode_int(_data_raw_, running_bit_offset, 24)
    nmea2000Message.fields.append(NMEA2000Field('reserved_32', 'Reserved', None, None, reserved_32, reserved_32_raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_130820_simnetReprogramStatus(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # reserved_16 | Offset: 16, Length: 8, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_16")

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
    # status | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("status")

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
    # reserved_32 | Offset: 32, Length: 24, Resolution: 1, Field Type: RESERVED
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_32")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 24
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
    return data_raw.to_bytes(7, byteorder="little")

def decode_pgn_130820_furunoUnknown130820(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='furunoUnknown130820', description='Furuno: Unknown 130820')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 1855, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Furuno", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:a | Offset: 16, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    a = a_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('a', 'A', None, None, a, a_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 5:b | Offset: 24, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 24
    b = b_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('b', 'B', None, None, b, b_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:c | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    c = c_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('c', 'C', None, None, c, c_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:d | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    d = d_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('d', 'D', None, None, d, d_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:e | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    e = e_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('e', 'E', None, None, e, e_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_furunoUnknown130820(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # a | Offset: 16, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("a")

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
        raise ValueError("Cant encode this message, 'A' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'A' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'A' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # b | Offset: 24, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 24
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("b")

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
        raise ValueError("Cant encode this message, 'B' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'B' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'B' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # c | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("c")

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
        raise ValueError("Cant encode this message, 'C' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'C' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'C' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # d | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("d")

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
        raise ValueError("Cant encode this message, 'D' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'D' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'D' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # e | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("e")

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
        raise ValueError("Cant encode this message, 'E' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'E' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'E' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(7, byteorder="little")

def decode_pgn_130820_fusionVersions(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionVersions', description='Fusion: Versions')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32769, PartOfPrimaryKey: ,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "API Version", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 16

    # 5:hw_version_major | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    hw_version_major = hw_version_major_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('hwVersionMajor', 'HW Version Major', None, None, hw_version_major, hw_version_major_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:hw_version_minor | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    hw_version_minor = hw_version_minor_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('hwVersionMinor', 'HW Version Minor', None, None, hw_version_minor, hw_version_minor_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:sw_version_major | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    sw_version_major = sw_version_major_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('swVersionMajor', 'SW Version Major', None, None, sw_version_major, sw_version_major_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:sw_version_minor | Offset: 72, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    sw_version_minor = sw_version_minor_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('swVersionMinor', 'SW Version Minor', None, None, sw_version_minor, sw_version_minor_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:build_number | Offset: 80, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    build_number = build_number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('buildNumber', 'Build Number', None, None, build_number, build_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionVersions(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # hwVersionMajor | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("hwVersionMajor")

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
        raise ValueError("Cant encode this message, 'HW Version Major' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'HW Version Major' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'HW Version Major' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # hwVersionMinor | Offset: 48, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("hwVersionMinor")

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
        raise ValueError("Cant encode this message, 'HW Version Minor' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'HW Version Minor' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'HW Version Minor' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # swVersionMajor | Offset: 64, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("swVersionMajor")

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
        raise ValueError("Cant encode this message, 'SW Version Major' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SW Version Major' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SW Version Major' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # swVersionMinor | Offset: 72, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("swVersionMinor")

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
        raise ValueError("Cant encode this message, 'SW Version Minor' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'SW Version Minor' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'SW Version Minor' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # buildNumber | Offset: 80, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("buildNumber")

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
        raise ValueError("Cant encode this message, 'Build Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Build Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Build Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(11, byteorder="little")

def decode_pgn_130820_fusionSource(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSource', description='Fusion: Source')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32770, PartOfPrimaryKey: ,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Source", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:current_source_id | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    current_source_id = current_source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('currentSourceId', 'Current Source ID', None, None, current_source_id, current_source_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:source_type | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    source_type_raw = decode_int(_data_raw_, running_bit_offset, 8)
    source_type = master_dict['FUSION_SOURCE_TYPE'].get(source_type_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('sourceType', 'Source Type', None, None, source_type, source_type_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 8:flags | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    flags = flags_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('flags', 'Flags', None, None, flags, flags_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:source | Offset: 64, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    source = source_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('source', 'Source', None, None, source, source_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSource(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # currentSourceId | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("currentSourceId")

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
        raise ValueError("Cant encode this message, 'Current Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Current Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Current Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceType | Offset: 48, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceType")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_SOURCE_TYPE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Source Type' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source Type' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source Type' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # flags | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("flags")

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
        raise ValueError("Cant encode this message, 'Flags' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Flags' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Flags' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # source | Offset: 64, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("source")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Source' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionSourceCount(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSourceCount', description='Fusion: Source Count')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32771, PartOfPrimaryKey: ,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Source Count", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 16

    # 5:source_count | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    source_count = source_count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceCount', 'Source Count', None, None, source_count, source_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionSourceCount(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceCount | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceCount")

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
        raise ValueError("Cant encode this message, 'Source Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(5, byteorder="little")

def decode_pgn_130820_fusionMedia(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionMedia', description='Fusion: Media')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32772, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Track Info", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:flags | Offset: 40, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    flags_raw = decode_int(_data_raw_, running_bit_offset, 16)
    flags = master_dict['FUSION_PLAY_STATUS'].get(flags_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('flags', 'Flags', None, None, flags, flags_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 16

    # 7:track__ | Offset: 56, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    track__ = track___raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('track', 'Track #', None, None, track__, track___raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 8:track_count | Offset: 88, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    track_count = track_count_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('trackCount', 'Track Count', None, None, track_count, track_count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 9:length | Offset: 120, Length: 32, Signed: False Resolution: 0.001, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    length = length_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.001, 0, 4294967.292)
    nmea2000Message.fields.append(NMEA2000Field('length', 'Length', None, 's', length, length_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    # 10:position_in_track | Offset: 152, Length: 32, Signed: False Resolution: 0.001, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 152
    position_in_track = position_in_track_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.001, 0, 4294967.292)
    nmea2000Message.fields.append(NMEA2000Field('positionInTrack', 'Position in track', None, 's', position_in_track, position_in_track_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130820_fusionMedia(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # flags | Offset: 40, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("flags")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_PLAY_STATUS(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Flags' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Flags' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Flags' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # track | Offset: 56, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("track")

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
        raise ValueError("Cant encode this message, 'Track #' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Track #' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Track #' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # trackCount | Offset: 88, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("trackCount")

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
        raise ValueError("Cant encode this message, 'Track Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Track Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Track Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # length | Offset: 120, Length: 32, Resolution: 0.001, Field Type: DURATION
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("length")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.001)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Length' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Length' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Length' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # positionInTrack | Offset: 152, Length: 32, Resolution: 0.001, Field Type: DURATION
    running_bit_offset = 152
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("positionInTrack")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.001)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Position in track' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Position in track' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Position in track' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(23, byteorder="little")

def decode_pgn_130820_fusionTrackName(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionTrackName', description='Fusion: Track Name')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32773, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Track Title", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:index | Offset: 40, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    index = index_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('index', 'Index', None, None, index, index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 7:track | Offset: 72, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    track = track_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('track', 'Track', None, None, track, track_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionTrackName(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # index | Offset: 40, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("index")

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
        raise ValueError("Cant encode this message, 'Index' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Index' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Index' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # track | Offset: 72, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("track")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Track' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Track' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Track' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionArtistName(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionArtistName', description='Fusion: Artist Name')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32774, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Track Artist", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:index | Offset: 40, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    index = index_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('index', 'Index', None, None, index, index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 7:artist | Offset: 72, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    artist = artist_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('artist', 'Artist', None, None, artist, artist_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionArtistName(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # index | Offset: 40, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("index")

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
        raise ValueError("Cant encode this message, 'Index' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Index' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Index' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # artist | Offset: 72, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("artist")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Artist' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Artist' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Artist' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionAlbumName(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionAlbumName', description='Fusion: Album Name')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32775, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Track Album", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:index | Offset: 40, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    index = index_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('index', 'Index', None, None, index, index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 7:album | Offset: 72, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    album = album_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('album', 'Album', None, None, album, album_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionAlbumName(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # index | Offset: 40, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("index")

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
        raise ValueError("Cant encode this message, 'Index' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Index' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Index' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # album | Offset: 72, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("album")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Album' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Album' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Album' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionDeviceName(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionDeviceName', description='Fusion: Device Name')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32801, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Unit Name", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:name | Offset: 32, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    name = name_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionDeviceName(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # name | Offset: 32, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionZoneName(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionZoneName', description='Fusion: Zone Name')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32813, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Zone Name", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:number | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    number = number_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('number', 'Number', None, None, number, number_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:name | Offset: 40, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    name = name_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionZoneName(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # number | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("number")

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
        raise ValueError("Cant encode this message, 'Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # name | Offset: 40, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionTrackPosition(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionTrackPosition', description='Fusion: Track Position')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32777, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Track Progress", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:progress | Offset: 40, Length: 24, Signed: False Resolution: 0.001, Field Type: DURATION, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    progress = progress_raw = decode_number(_data_raw_, running_bit_offset, 24, False, 0.001, 0, 16777.212)
    nmea2000Message.fields.append(NMEA2000Field('progress', 'Progress', None, 's', progress, progress_raw, PhysicalQuantities.DURATION, FieldTypes.DURATION, False))
    running_bit_offset += 24

    return nmea2000Message

def encode_pgn_130820_fusionTrackPosition(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # progress | Offset: 40, Length: 24, Resolution: 0.001, Field Type: DURATION
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("progress")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (raw_number_matches_value(field.raw_value, field.value, 0.001)):
        field_value = encode_number_raw(field.raw_value, 24, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 24, False, 0.001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 24, False, 0.001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 24)
    field_bit_length = 24
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Progress' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Progress' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Progress' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_130820_fusionTuner(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionTuner', description='Fusion: Tuner')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32779, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Tuner", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id_raw = decode_int(_data_raw_, running_bit_offset, 8)
    source_id = master_dict['FUSION_RADIO_SOURCE'].get(source_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 8

    # 6:scanning | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    scanning = scanning_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('scanning', 'Scanning', None, None, scanning, scanning_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:frequency | Offset: 48, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    frequency = frequency_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('frequency', 'Frequency', None, 'Hz', frequency, frequency_raw, PhysicalQuantities.FREQUENCY, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 8:signal_strength | Offset: 80, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    signal_strength = signal_strength_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('signalStrength', 'Signal Strength', None, None, signal_strength, signal_strength_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:track | Offset: 88, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    track = track_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('track', 'Track', None, None, track, track_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionTuner(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_RADIO_SOURCE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # scanning | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("scanning")

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
        raise ValueError("Cant encode this message, 'Scanning' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Scanning' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Scanning' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # frequency | Offset: 48, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("frequency")

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
    # signalStrength | Offset: 80, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("signalStrength")

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
        raise ValueError("Cant encode this message, 'Signal Strength' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Signal Strength' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Signal Strength' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # track | Offset: 88, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("track")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Track' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Track' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Track' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionMarineTuner(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionMarineTuner', description='Fusion: MARINE_TUNER')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32780, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Marine Tuner", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:channel | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:signal_strength | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    signal_strength = signal_strength_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('signalStrength', 'Signal Strength', None, None, signal_strength, signal_strength_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:name | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    name = name_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionMarineTuner(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
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
    # signalStrength | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("signalStrength")

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
        raise ValueError("Cant encode this message, 'Signal Strength' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Signal Strength' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Signal Strength' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # name | Offset: 56, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionMarineSquelch(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionMarineSquelch', description='Fusion: Marine Squelch')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32781, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Marine Squelch", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:squelch | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    squelch = squelch_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('squelch', 'Squelch', None, None, squelch, squelch_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionMarineSquelch(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # squelch | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("squelch")

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
        raise ValueError("Cant encode this message, 'Squelch' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Squelch' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Squelch' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionMarineScanMode(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionMarineScanMode', description='Fusion: Marine Scan Mode')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32782, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Marine Scan Mode", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:scan | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    scan_raw = decode_int(_data_raw_, running_bit_offset, 8)
    scan = master_dict['YES_NO'].get(scan_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('scan', 'Scan', None, None, scan, scan_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionMarineScanMode(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # scan | Offset: 40, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("scan")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Scan' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Scan' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Scan' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionMenuItem(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionMenuItem', description='Fusion: Menu Item')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32785, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Menu Item", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:item_index | Offset: 40, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    item_index = item_index_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('itemIndex', 'Item Index', None, None, item_index, item_index_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 7:flags | Offset: 72, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    flags = flags_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('flags', 'Flags', None, None, flags, flags_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:lock_id | Offset: 80, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    lock_id = lock_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('lockId', 'Lock ID', None, None, lock_id, lock_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 9:text | Offset: 88, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 88
    text = text_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('text', 'Text', None, None, text, text_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionMenuItem(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # itemIndex | Offset: 40, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("itemIndex")

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
        raise ValueError("Cant encode this message, 'Item Index' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Item Index' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Item Index' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # flags | Offset: 72, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("flags")

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
        raise ValueError("Cant encode this message, 'Flags' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Flags' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Flags' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # lockId | Offset: 80, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("lockId")

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
        raise ValueError("Cant encode this message, 'Lock ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Lock ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Lock ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # text | Offset: 88, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 88
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("text")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Text' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Text' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Text' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionAuxGain(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionAuxGain', description='Fusion: Aux Gain')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32787, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Aux Gain", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:gain | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    gain = gain_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('gain', 'Gain', None, None, gain, gain_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionAuxGain(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # gain | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("gain")

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
        raise ValueError("Cant encode this message, 'Gain' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Gain' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Gain' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionUsbRepeatStatus(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionUsbRepeatStatus', description='Fusion: USB Repeat Status')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32788, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Setting", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:id | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 9, PartOfPrimaryKey: ,
    running_bit_offset = 32
    id_raw = decode_int(_data_raw_, running_bit_offset, 32)
    id = master_dict['FUSION_SETTING'].get(id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('id', 'ID', "USB repeat", None, id, id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 32

    # 6:status | Offset: 64, Length: 32, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    status_raw = decode_int(_data_raw_, running_bit_offset, 32)
    status = master_dict['FUSION_REPEAT_STATUS'].get(status_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('status', 'Status', None, None, status, status_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130820_fusionUsbRepeatStatus(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # id | Offset: 32, Length: 32, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("id")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_SETTING(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # status | Offset: 64, Length: 32, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("status")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_REPEAT_STATUS(field.value)
    field_bit_length = 32
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
    return data_raw.to_bytes(12, byteorder="little")

def decode_pgn_130820_fusionSetting(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSetting', description='Fusion: Setting')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32788, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Setting", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:id | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    id_raw = decode_int(_data_raw_, running_bit_offset, 32)
    id = master_dict['FUSION_SETTING'].get(id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('id', 'ID', None, None, id, id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 32

    # 6:value | Offset: 64, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    value = value_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', None, None, value, value_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    return nmea2000Message

def encode_pgn_130820_fusionSetting(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # id | Offset: 32, Length: 32, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("id")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_SETTING(field.value)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # value | Offset: 64, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("value")

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
        raise ValueError("Cant encode this message, 'Value' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Value' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(12, byteorder="little")

def decode_pgn_130820_fusionSettings(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSettings', description='Fusion: Settings')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32789, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Settings", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:count | Offset: 32, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    count = count_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('count', 'Count', None, None, count, count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 6:id | Offset: 64, Length: 32, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    _repeating_field_set_1_offset = running_bit_offset
    id_raw = decode_int(_data_raw_, running_bit_offset, 32)
    id = master_dict['FUSION_SETTING'].get(id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('id', 'ID', None, None, id, id_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 32

    # 7:value | Offset: 96, Length: 32, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    value = value_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
    nmea2000Message.fields.append(NMEA2000Field('value', 'Value', None, None, value, value_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = int(count_raw or 0)
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        id_raw = decode_int(_data_raw_, running_bit_offset, 32)
        id = master_dict['FUSION_SETTING'].get(id_raw, None)
        running_bit_offset += 32
        repeating_entry["id"] = _repeating_entry_value(id, id_raw)
    
        value = value_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 1, 0, 4294967292)
        running_bit_offset += 32
        repeating_entry["value"] = _repeating_entry_value(value, value_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "id",
                "value",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_130820_fusionSettings(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "id",
        "value",
    ))
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # count | Offset: 32, Length: 32, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("count")

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
        raise ValueError("Cant encode this message, 'Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    running_bit_offset = 64
    for repeating_entry in repeating_field_set_1_entries:
        # id | Offset: 64, Length: 32, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("id")
        if field is None:
            raise ValueError("Cant encode this message, missing 'ID'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_FUSION_SETTING(field.value)
        field_bit_length = 32
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'ID' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'ID' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'ID' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # value | Offset: 96, Length: 32, Resolution: 1, Field Type: NUMBER
        field = repeating_entry.get("value")
        if field is None:
            raise ValueError("Cant encode this message, missing 'Value'")
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
            raise ValueError("Cant encode this message, 'Value' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'Value' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'Value' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionMute(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionMute', description='Fusion: Mute')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32791, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Mute", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:mute | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    mute_raw = decode_int(_data_raw_, running_bit_offset, 8)
    mute = master_dict['FUSION_MUTE_COMMAND'].get(mute_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('mute', 'Mute', None, None, mute, mute_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionMute(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mute | Offset: 32, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mute")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_MUTE_COMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Mute' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Mute' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Mute' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(5, byteorder="little")

def decode_pgn_130820_fusionBalance(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionBalance', description='Fusion: Balance')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32792, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Balance", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    zone = zone_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone', 'Zone', None, None, zone, zone_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:value | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    value = value_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('value', 'VALUE', None, None, value, value_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionBalance(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone")

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
        raise ValueError("Cant encode this message, 'Zone' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # value | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("value")

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
        raise ValueError("Cant encode this message, 'VALUE' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'VALUE' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'VALUE' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionLowPassFilter(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionLowPassFilter', description='Fusion: Low Pass Filter')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32793, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Low Pass Filter", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    zone = zone_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone', 'Zone', None, None, zone, zone_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:filter | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    filter = filter_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('filter', 'Filter', None, None, filter, filter_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionLowPassFilter(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone")

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
        raise ValueError("Cant encode this message, 'Zone' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # filter | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("filter")

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
        raise ValueError("Cant encode this message, 'Filter' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Filter' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Filter' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionSublevels(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSublevels', description='Fusion: Sublevels')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32794, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Sublevels", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone_1 | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    zone_1 = zone_1_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone1', 'Zone 1', None, None, zone_1, zone_1_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:zone_2 | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    zone_2 = zone_2_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone2', 'Zone 2', None, None, zone_2, zone_2_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:zone_3 | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    zone_3 = zone_3_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone3', 'Zone 3', None, None, zone_3, zone_3_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:zone_4 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    zone_4 = zone_4_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone4', 'Zone 4', None, None, zone_4, zone_4_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionSublevels(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone1 | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone1")

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
        raise ValueError("Cant encode this message, 'Zone 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone2 | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone2")

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
        raise ValueError("Cant encode this message, 'Zone 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone3 | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone3")

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
        raise ValueError("Cant encode this message, 'Zone 3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone4 | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone4")

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
        raise ValueError("Cant encode this message, 'Zone 4' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 4' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 4' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_130820_fusionEq(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionEq', description='Fusion: EQ')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32795, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Tone", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    zone = zone_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone', 'Zone', None, None, zone, zone_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:bass | Offset: 40, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    bass = bass_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('bass', 'Bass', None, None, bass, bass_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:mid | Offset: 48, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    mid = mid_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('mid', 'Mid', None, None, mid, mid_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:treble | Offset: 56, Length: 8, Signed: True Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    treble = treble_raw = decode_number(_data_raw_, running_bit_offset, 8, True, 1, -127, 124)
    nmea2000Message.fields.append(NMEA2000Field('treble', 'Treble', None, None, treble, treble_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionEq(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone")

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
        raise ValueError("Cant encode this message, 'Zone' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # bass | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("bass")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Bass' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Bass' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Bass' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mid | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mid")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Mid' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Mid' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Mid' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # treble | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("treble")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1):
        field_value = encode_number_raw(field.raw_value, 8, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 8, True, 1)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 8, True, 1)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Treble' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Treble' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Treble' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_130820_fusionVolumeLimits(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionVolumeLimits', description='Fusion: Volume Limits')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32796, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Volume Limits", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone_1_volume_limit | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    zone_1_volume_limit = zone_1_volume_limit_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone1VolumeLimit', 'Zone 1 Volume Limit', None, None, zone_1_volume_limit, zone_1_volume_limit_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:zone_2_volume_limit | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    zone_2_volume_limit = zone_2_volume_limit_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone2VolumeLimit', 'Zone 2 Volume Limit', None, None, zone_2_volume_limit, zone_2_volume_limit_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:zone_3_volume_limit | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    zone_3_volume_limit = zone_3_volume_limit_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone3VolumeLimit', 'Zone 3 Volume Limit', None, None, zone_3_volume_limit, zone_3_volume_limit_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:zone_4_volume_limit | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    zone_4_volume_limit = zone_4_volume_limit_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone4VolumeLimit', 'Zone 4 Volume Limit', None, None, zone_4_volume_limit, zone_4_volume_limit_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionVolumeLimits(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone1VolumeLimit | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone1VolumeLimit")

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
        raise ValueError("Cant encode this message, 'Zone 1 Volume Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 1 Volume Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 1 Volume Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone2VolumeLimit | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone2VolumeLimit")

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
        raise ValueError("Cant encode this message, 'Zone 2 Volume Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 2 Volume Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 2 Volume Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone3VolumeLimit | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone3VolumeLimit")

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
        raise ValueError("Cant encode this message, 'Zone 3 Volume Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 3 Volume Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 3 Volume Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone4VolumeLimit | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone4VolumeLimit")

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
        raise ValueError("Cant encode this message, 'Zone 4 Volume Limit' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 4 Volume Limit' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 4 Volume Limit' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_130820_fusionVolumes(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionVolumes', description='Fusion: Volumes')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32797, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Volume", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone_1 | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    zone_1 = zone_1_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone1', 'Zone 1', None, None, zone_1, zone_1_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 6:zone_2 | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    zone_2 = zone_2_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone2', 'Zone 2', None, None, zone_2, zone_2_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:zone_3 | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    zone_3 = zone_3_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone3', 'Zone 3', None, None, zone_3, zone_3_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 8:zone_4 | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    zone_4 = zone_4_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone4', 'Zone 4', None, None, zone_4, zone_4_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionVolumes(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone1 | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone1")

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
        raise ValueError("Cant encode this message, 'Zone 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone2 | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone2")

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
        raise ValueError("Cant encode this message, 'Zone 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone3 | Offset: 48, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone3")

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
        raise ValueError("Cant encode this message, 'Zone 3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone4 | Offset: 56, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone4")

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
        raise ValueError("Cant encode this message, 'Zone 4' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 4' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 4' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(8, byteorder="little")

def decode_pgn_130820_fusionCapabilities(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionCapabilities', description='Fusion: Capabilities')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32798, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Capabilities", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone_1 | Offset: 32, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    zone_1 = zone_1_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('zone1', 'Zone 1', None, None, zone_1, zone_1_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 6:zone_2 | Offset: 48, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    zone_2 = zone_2_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('zone2', 'Zone 2', None, None, zone_2, zone_2_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:zone_3 | Offset: 64, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    zone_3 = zone_3_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('zone3', 'Zone 3', None, None, zone_3, zone_3_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 8:zone_4 | Offset: 80, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 80
    zone_4 = zone_4_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('zone4', 'Zone 4', None, None, zone_4, zone_4_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:__global | Offset: 96, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 96
    __global = __global_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('global', 'Global', None, None, __global, __global_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    return nmea2000Message

def encode_pgn_130820_fusionCapabilities(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone1 | Offset: 32, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone1")

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
        raise ValueError("Cant encode this message, 'Zone 1' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 1' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 1' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone2 | Offset: 48, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone2")

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
        raise ValueError("Cant encode this message, 'Zone 2' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 2' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 2' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone3 | Offset: 64, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone3")

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
        raise ValueError("Cant encode this message, 'Zone 3' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 3' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 3' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone4 | Offset: 80, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 80
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone4")

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
        raise ValueError("Cant encode this message, 'Zone 4' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone 4' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone 4' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # global | Offset: 96, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 96
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("global")

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
        raise ValueError("Cant encode this message, 'Global' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Global' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Global' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(14, byteorder="little")

def decode_pgn_130820_fusionLineLevelControl(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionLineLevelControl', description='Fusion: Line Level Control')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32799, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Line Level Control", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:zone | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    zone = zone_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('zone', 'Zone', None, None, zone, zone_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:control | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    control = control_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('control', 'Control', None, None, control, control_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionLineLevelControl(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # zone | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("zone")

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
        raise ValueError("Cant encode this message, 'Zone' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Zone' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Zone' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # control | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("control")

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
        raise ValueError("Cant encode this message, 'Control' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Control' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Control' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionPowerState(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionPowerState', description='Fusion: Power State')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32800, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Power", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:state | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 32
    state_raw = decode_int(_data_raw_, running_bit_offset, 8)
    state = master_dict['FUSION_POWER_STATE'].get(state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('state', 'State', None, None, state, state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionPowerState(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # state | Offset: 32, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("state")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_POWER_STATE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(5, byteorder="little")

def decode_pgn_130820_fusionSiriusxm(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxm', description='Fusion: SiriusXM')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32802, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "Sirius", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:com_state | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    com_state_raw = decode_int(_data_raw_, running_bit_offset, 8)
    com_state = master_dict['FUSION_SIRIUS_COM_STATE'].get(com_state_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('comState', 'Com State', None, None, com_state, com_state_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 7:alert | Offset: 48, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    alert_raw = decode_int(_data_raw_, running_bit_offset, 8)
    alert = master_dict['FUSION_SIRIUS_COM_STATE'].get(alert_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('alert', 'Alert', None, None, alert, alert_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 8:advisory_channel | Offset: 56, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    advisory_channel = advisory_channel_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('advisoryChannel', 'Advisory Channel', None, None, advisory_channel, advisory_channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 9:tuning_mode | Offset: 72, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    tuning_mode_raw = decode_int(_data_raw_, running_bit_offset, 8)
    tuning_mode = master_dict['FUSION_SIRIUS_TUNING_MODE'].get(tuning_mode_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('tuningMode', 'Tuning Mode', None, None, tuning_mode, tuning_mode_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxm(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # comState | Offset: 40, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("comState")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_SIRIUS_COM_STATE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Com State' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Com State' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Com State' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # alert | Offset: 48, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("alert")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_SIRIUS_COM_STATE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Alert' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Alert' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Alert' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # advisoryChannel | Offset: 56, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("advisoryChannel")

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
        raise ValueError("Cant encode this message, 'Advisory Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Advisory Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Advisory Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # tuningMode | Offset: 72, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("tuningMode")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_SIRIUS_TUNING_MODE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Tuning Mode' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Tuning Mode' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Tuning Mode' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(10, byteorder="little")

def decode_pgn_130820_fusionSiriusxmChannel(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmChannel', description='Fusion: SiriusXM Channel')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32804, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXM Channel", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:channel_number | Offset: 40, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    channel_number = channel_number_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('channelNumber', 'Channel Number', None, None, channel_number, channel_number_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:channel | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    channel = channel_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmChannel(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channelNumber | Offset: 40, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channelNumber")

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
        raise ValueError("Cant encode this message, 'Channel Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Channel Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Channel Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 56, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
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
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionSiriusxmTitle(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmTitle', description='Fusion: SiriusXM Title')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32805, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXM Title", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:channel | Offset: 40, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:title | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    title = title_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('title', 'Title', None, None, title, title_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmTitle(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 40, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

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
    # title | Offset: 56, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("title")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Title' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Title' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Title' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionSiriusxmArtist(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmArtist', description='Fusion: SiriusXM Artist')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32806, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXM Artist", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:channel | Offset: 40, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:artist | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    artist = artist_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('artist', 'Artist', None, None, artist, artist_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmArtist(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 40, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

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
    # artist | Offset: 56, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("artist")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Artist' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Artist' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Artist' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionSiriusxmContentInfo(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmContentInfo', description='Fusion: SiriusXM Content Info')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32807, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXM Genre", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:channel | Offset: 40, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:genre | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    genre = genre_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('genre', 'Genre', None, None, genre, genre_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmContentInfo(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 40, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

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
    # genre | Offset: 56, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("genre")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Genre' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Genre' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Genre' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionSiriusxmCategory(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmCategory', description='Fusion: SiriusXM Category')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32808, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXM Category", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:channel | Offset: 40, Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    channel = channel_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('channel', 'Channel', None, None, channel, channel_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 7:name | Offset: 56, Length: , Signed: False Resolution: , Field Type: STRING_LZ, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    name = name_raw = decode_string_lz(_data_raw_, running_bit_offset)
    nmea2000Message.fields.append(NMEA2000Field('name', 'Name', None, None, name, name_raw, None, FieldTypes.STRING_LZ, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmCategory(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # channel | Offset: 40, Length: 16, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("channel")

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
    # name | Offset: 56, Length: , Resolution: , Field Type: STRING_LZ
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("name")

    advance_running_offset = True
    field_bytes = encode_string_lz(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Name' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Name' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Name' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_fusionSiriusxmSignal(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmSignal', description='Fusion: SiriusXM Signal')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32809, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXm Signal", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:signal | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    signal = signal_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('signal', 'Signal', None, None, signal, signal_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmSignal(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # signal | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("signal")

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
        raise ValueError("Cant encode this message, 'Signal' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Signal' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Signal' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes(6, byteorder="little")

def decode_pgn_130820_fusionSiriusxmPresets(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='fusionSiriusxmPresets', description='Fusion: SiriusXM Presets')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 419, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Fusion Electronics", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:message_id | Offset: 16, Length: 16, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 32812, PartOfPrimaryKey: True,
    running_bit_offset = 16
    message_id_raw = decode_int(_data_raw_, running_bit_offset, 16)
    message_id = master_dict['FUSION_STATUS_MESSAGE_ID'].get(message_id_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('messageId', 'Message ID', "SiriusXM Presets", None, message_id, message_id_raw, None, FieldTypes.LOOKUP, True))
    running_bit_offset += 16

    # 5:source_id | Offset: 32, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: True,
    running_bit_offset = 32
    source_id = source_id_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('sourceId', 'Source ID', None, None, source_id, source_id_raw, None, FieldTypes.NUMBER, True))
    running_bit_offset += 8

    # 6:count | Offset: 40, Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 40
    count = count_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('count', 'Count', None, None, count, count_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 7:values | Offset: 48, Length: , Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 48
    assert isinstance(count, (int, float))
    values = values_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, int(count)))
    nmea2000Message.fields.append(NMEA2000Field('values', 'Values', "Preset Values. There will be Count / 4 presets stored", None, values, values_raw, None, FieldTypes.BINARY, False))
    

    return nmea2000Message

def encode_pgn_130820_fusionSiriusxmPresets(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # messageId | Offset: 16, Length: 16, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("messageId")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_FUSION_STATUS_MESSAGE_ID(field.value)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # sourceId | Offset: 32, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 32
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("sourceId")

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
        raise ValueError("Cant encode this message, 'Source ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Source ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Source ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # count | Offset: 40, Length: 8, Resolution: 1, Field Type: NUMBER
    running_bit_offset = 40
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("count")

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
        raise ValueError("Cant encode this message, 'Count' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Count' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Count' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # values | Offset: 48, Length: , Resolution: 1, Field Type: BINARY
    running_bit_offset = 48
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("values")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    length_field = nmea2000Message.get_field_by_id("count")
    if isinstance(length_field.raw_value, (int, float)):
        field_bit_length = int(length_field.raw_value)
    else:
        assert length_field.value is None or isinstance(length_field.value, (int, float))
        field_bit_length = int(length_field.value or 0)
    advance_running_offset = False
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Values' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Values' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Values' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_130820_maretronAlertResponse(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 130820."""
    nmea2000Message = NMEA2000Message(PGN=130820, id='maretronAlertResponse', description='Maretron: Alert Response')
    running_bit_offset = 0
    # 1:manufacturer_code | Offset: 0, Length: 11, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 137, PartOfPrimaryKey: ,
    running_bit_offset = 0
    manufacturer_code_raw = decode_int(_data_raw_, running_bit_offset, 11)
    manufacturer_code = master_dict['MANUFACTURER_CODE'].get(manufacturer_code_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('manufacturerCode', 'Manufacturer Code', "Maretron", None, manufacturer_code, manufacturer_code_raw, None, FieldTypes.LOOKUP, False))
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

    # 4:data | Offset: 16, Length: 1768, Signed: False Resolution: 1, Field Type: BINARY, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    data = data_raw = int_to_bytes(decode_int(_data_raw_, running_bit_offset, 1768))
    nmea2000Message.fields.append(NMEA2000Field('data', 'Data', None, None, data, data_raw, None, FieldTypes.BINARY, False))
    running_bit_offset += 1768

    return nmea2000Message

def encode_pgn_130820_maretronAlertResponse(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 130820."""
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
    # data | Offset: 16, Length: 1768, Resolution: 1, Field Type: BINARY
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("data")

    advance_running_offset = True
    field_bytes = normalize_binary_data(field.raw_value) if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else normalize_binary_data(field.value)
    field_value = encode_binary_data(field_bytes)
    field_bit_length = 1768
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
    return data_raw.to_bytes(223, byteorder="little")
