# pylint: skip-file
from __future__ import annotations

from .lookups import *

def is_fast_pgn_129808() -> bool:
    """Return True if PGN 129808 is a fast PGN."""
    return True
# Complex PGN. number of matches: 2
def decode_pgn_129808(data_raw: int, data_length_bits: int | None = None) -> NMEA2000Message | None:
    # dscDistressCallInformation | Description: DSC Distress Call Information
    if (
        (((data_raw >> 8) & 0xFF) == 112)
        ):
        return decode_pgn_129808_dscDistressCallInformation(data_raw, data_length_bits)
    
    # dscCallInformation | Description: DSC Call Information
    if (
        ):
        return decode_pgn_129808_dscCallInformation(data_raw, data_length_bits)
    
    
    return None
    
def decode_pgn_129808_dscDistressCallInformation(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129808."""
    nmea2000Message = NMEA2000Message(PGN=129808, id='dscDistressCallInformation', description='DSC Distress Call Information')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:dsc_format | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    dsc_format_raw = decode_int(_data_raw_, running_bit_offset, 8)
    dsc_format = master_dict['DSC_FORMAT'].get(dsc_format_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dscFormat', 'DSC Format', None, None, dsc_format, dsc_format_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:dsc_category | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: 112, PartOfPrimaryKey: ,
    running_bit_offset = 8
    dsc_category_raw = decode_int(_data_raw_, running_bit_offset, 8)
    dsc_category = master_dict['DSC_CATEGORY'].get(dsc_category_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dscCategory', 'DSC Category', "Distress", None, dsc_category, dsc_category_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 3:dsc_message_address | Offset: 16, Length: 40, Signed: False Resolution: 1, Field Type: DECIMAL, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    dsc_message_address_raw = decode_int(_data_raw_, running_bit_offset, 40)
    dsc_message_address = decode_decimal(dsc_message_address_raw)
    nmea2000Message.fields.append(NMEA2000Field('dscMessageAddress', 'DSC Message Address', "MMSI, Geographic Area or blank", None, dsc_message_address, dsc_message_address_raw, None, FieldTypes.DECIMAL, False))
    running_bit_offset += 40

    # 4:nature_of_distress | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    nature_of_distress_raw = decode_int(_data_raw_, running_bit_offset, 8)
    nature_of_distress = master_dict['DSC_NATURE'].get(nature_of_distress_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('natureOfDistress', 'Nature of Distress', None, None, nature_of_distress, nature_of_distress_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:subsequent_communication_mode_or_2nd_telecommand | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    subsequent_communication_mode_or_2nd_telecommand_raw = decode_int(_data_raw_, running_bit_offset, 8)
    subsequent_communication_mode_or_2nd_telecommand = master_dict['DSC_SECOND_TELECOMMAND'].get(subsequent_communication_mode_or_2nd_telecommand_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('subsequentCommunicationModeOr2ndTelecommand', 'Subsequent Communication Mode or 2nd Telecommand', None, None, subsequent_communication_mode_or_2nd_telecommand, subsequent_communication_mode_or_2nd_telecommand_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:proposed_rx_frequency_channel | Offset: 72, Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    proposed_rx_frequency_channel, proposed_rx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('proposedRxFrequencyChannel', 'Proposed Rx Frequency/Channel', None, None, proposed_rx_frequency_channel, proposed_rx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 7:proposed_tx_frequency_channel | Offset: 120, Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    proposed_tx_frequency_channel, proposed_tx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('proposedTxFrequencyChannel', 'Proposed Tx Frequency/Channel', None, None, proposed_tx_frequency_channel, proposed_tx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 8:telephone_number | Offset: 168, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 168
    telephone_number_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    telephone_number = telephone_number_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('telephoneNumber', 'Telephone Number', None, None, telephone_number, telephone_number_raw, None, FieldTypes.STRING_LAU, False))
    

    # 9:latitude_of_vessel_reported | Offset: , Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    latitude_of_vessel_reported = latitude_of_vessel_reported_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -90, 90)
    nmea2000Message.fields.append(NMEA2000Field('latitudeOfVesselReported', 'Latitude of Vessel Reported', None, 'deg', latitude_of_vessel_reported, latitude_of_vessel_reported_raw, PhysicalQuantities.GEOGRAPHICAL_LATITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 10:longitude_of_vessel_reported | Offset: , Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    longitude_of_vessel_reported = longitude_of_vessel_reported_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -180, 180)
    nmea2000Message.fields.append(NMEA2000Field('longitudeOfVesselReported', 'Longitude of Vessel Reported', None, 'deg', longitude_of_vessel_reported, longitude_of_vessel_reported_raw, PhysicalQuantities.GEOGRAPHICAL_LONGITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 11:time_of_position | Offset: , Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    time_of_position_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time_of_position = decode_time(time_of_position_raw)
    nmea2000Message.fields.append(NMEA2000Field('timeOfPosition', 'Time of Position', "Seconds since midnight", 's', time_of_position, time_of_position_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 12:mmsi_of_ship_in_distress | Offset: , Length: 40, Signed: False Resolution: 1, Field Type: DECIMAL, Match: , PartOfPrimaryKey: ,
    mmsi_of_ship_in_distress_raw = decode_int(_data_raw_, running_bit_offset, 40)
    mmsi_of_ship_in_distress = decode_decimal(mmsi_of_ship_in_distress_raw)
    nmea2000Message.fields.append(NMEA2000Field('mmsiOfShipInDistress', 'MMSI of Ship In Distress', None, None, mmsi_of_ship_in_distress, mmsi_of_ship_in_distress_raw, None, FieldTypes.DECIMAL, False))
    running_bit_offset += 40

    # 13:dsc_eos_symbol | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    dsc_eos_symbol = dsc_eos_symbol_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('dscEosSymbol', 'DSC EOS Symbol', None, None, dsc_eos_symbol, dsc_eos_symbol_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 14:expansion_enabled | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    expansion_enabled_raw = decode_int(_data_raw_, running_bit_offset, 2)
    expansion_enabled = master_dict['YES_NO'].get(expansion_enabled_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('expansionEnabled', 'Expansion Enabled', None, None, expansion_enabled, expansion_enabled_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 15:reserved_ | Offset: , Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', None, None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 16:calling_rx_frequency_channel | Offset: , Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    calling_rx_frequency_channel, calling_rx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('callingRxFrequencyChannel', 'Calling Rx Frequency/Channel', None, None, calling_rx_frequency_channel, calling_rx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 17:calling_tx_frequency_channel | Offset: , Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    calling_tx_frequency_channel, calling_tx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('callingTxFrequencyChannel', 'Calling Tx Frequency/Channel', None, None, calling_tx_frequency_channel, calling_tx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 18:time_of_receipt | Offset: , Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    time_of_receipt_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time_of_receipt = decode_time(time_of_receipt_raw)
    nmea2000Message.fields.append(NMEA2000Field('timeOfReceipt', 'Time of Receipt', "Seconds since midnight", 's', time_of_receipt, time_of_receipt_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 19:date_of_receipt | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    date_of_receipt_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    date_of_receipt = decode_date(date_of_receipt_raw)
    nmea2000Message.fields.append(NMEA2000Field('dateOfReceipt', 'Date of Receipt', None, 'd', date_of_receipt, date_of_receipt_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 20:dsc_equipment_assigned_message_id | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    dsc_equipment_assigned_message_id = dsc_equipment_assigned_message_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('dscEquipmentAssignedMessageId', 'DSC Equipment Assigned Message ID', None, None, dsc_equipment_assigned_message_id, dsc_equipment_assigned_message_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 21:dsc_expansion_field_symbol | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    dsc_expansion_field_symbol_raw = decode_int(_data_raw_, running_bit_offset, 8)
    dsc_expansion_field_symbol = master_dict['DSC_EXPANSION_DATA'].get(dsc_expansion_field_symbol_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dscExpansionFieldSymbol', 'DSC Expansion Field Symbol', None, None, dsc_expansion_field_symbol, dsc_expansion_field_symbol_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 22:dsc_expansion_field_data | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    dsc_expansion_field_data_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    dsc_expansion_field_data = dsc_expansion_field_data_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('dscExpansionFieldData', 'DSC Expansion Field Data', None, None, dsc_expansion_field_data, dsc_expansion_field_data_raw, None, FieldTypes.STRING_LAU, False))
    

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = None
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        dsc_expansion_field_symbol_raw = decode_int(_data_raw_, running_bit_offset, 8)
        dsc_expansion_field_symbol = master_dict['DSC_EXPANSION_DATA'].get(dsc_expansion_field_symbol_raw, None)
        running_bit_offset += 8
        repeating_entry["dscExpansionFieldSymbol"] = _repeating_entry_value(dsc_expansion_field_symbol, dsc_expansion_field_symbol_raw)
    
        dsc_expansion_field_data_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        dsc_expansion_field_data = dsc_expansion_field_data_raw
        running_bit_offset += bits_to_skip
        repeating_entry["dscExpansionFieldData"] = _repeating_entry_value(dsc_expansion_field_data, dsc_expansion_field_data_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "dscExpansionFieldSymbol",
                "dscExpansionFieldData",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_129808_dscDistressCallInformation(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129808."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "dscExpansionFieldSymbol",
        "dscExpansionFieldData",
    ))
    # dscFormat | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscFormat")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_FORMAT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DSC Format' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Format' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Format' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscCategory | Offset: 8, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscCategory")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_CATEGORY(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DSC Category' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Category' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Category' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscMessageAddress | Offset: 16, Length: 40, Resolution: 1, Field Type: DECIMAL
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscMessageAddress")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        field_value = encode_decimal(field.value)
    if field_value is None:
        field_value = 0
    field_bit_length = 40
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DSC Message Address' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Message Address' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Message Address' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # natureOfDistress | Offset: 56, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("natureOfDistress")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_NATURE(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Nature of Distress' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Nature of Distress' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Nature of Distress' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # subsequentCommunicationModeOr2ndTelecommand | Offset: 64, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("subsequentCommunicationModeOr2ndTelecommand")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_SECOND_TELECOMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Subsequent Communication Mode or 2nd Telecommand' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Subsequent Communication Mode or 2nd Telecommand' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Subsequent Communication Mode or 2nd Telecommand' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # proposedRxFrequencyChannel | Offset: 72, Length: 48, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("proposedRxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Proposed Rx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Proposed Rx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Proposed Rx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # proposedTxFrequencyChannel | Offset: 120, Length: 48, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("proposedTxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Proposed Tx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Proposed Tx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Proposed Tx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # telephoneNumber | Offset: 168, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 168
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("telephoneNumber")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Telephone Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Telephone Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Telephone Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # latitudeOfVesselReported | Offset: , Length: 32, Resolution: 1e-07, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("latitudeOfVesselReported")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Latitude of Vessel Reported' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Latitude of Vessel Reported' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Latitude of Vessel Reported' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitudeOfVesselReported | Offset: , Length: 32, Resolution: 1e-07, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitudeOfVesselReported")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitude of Vessel Reported' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitude of Vessel Reported' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitude of Vessel Reported' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # timeOfPosition | Offset: , Length: 32, Resolution: 0.0001, Field Type: TIME
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeOfPosition")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.0001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time of Position' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time of Position' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time of Position' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mmsiOfShipInDistress | Offset: , Length: 40, Resolution: 1, Field Type: DECIMAL
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mmsiOfShipInDistress")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        field_value = encode_decimal(field.value)
    if field_value is None:
        field_value = 0
    field_bit_length = 40
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'MMSI of Ship In Distress' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'MMSI of Ship In Distress' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'MMSI of Ship In Distress' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscEosSymbol | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscEosSymbol")

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
        raise ValueError("Cant encode this message, 'DSC EOS Symbol' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC EOS Symbol' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC EOS Symbol' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # expansionEnabled | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("expansionEnabled")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Expansion Enabled' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Expansion Enabled' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Expansion Enabled' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 6, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 6
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
    # callingRxFrequencyChannel | Offset: , Length: 48, Resolution: , Field Type: STRING_FIX
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("callingRxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Calling Rx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Calling Rx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Calling Rx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # callingTxFrequencyChannel | Offset: , Length: 48, Resolution: , Field Type: STRING_FIX
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("callingTxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Calling Tx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Calling Tx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Calling Tx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # timeOfReceipt | Offset: , Length: 32, Resolution: 0.0001, Field Type: TIME
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeOfReceipt")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.0001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time of Receipt' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time of Receipt' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time of Receipt' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dateOfReceipt | Offset: , Length: 16, Resolution: 1, Field Type: DATE
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dateOfReceipt")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Date of Receipt' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Date of Receipt' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Date of Receipt' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscEquipmentAssignedMessageId | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscEquipmentAssignedMessageId")

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
        raise ValueError("Cant encode this message, 'DSC Equipment Assigned Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Equipment Assigned Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Equipment Assigned Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # dscExpansionFieldSymbol | Offset: , Length: 8, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("dscExpansionFieldSymbol")
        if field is None:
            raise ValueError("Cant encode this message, missing 'DSC Expansion Field Symbol'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_DSC_EXPANSION_DATA(field.value)
        field_bit_length = 8
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Symbol' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Symbol' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Symbol' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # dscExpansionFieldData | Offset: , Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("dscExpansionFieldData")
        if field is None:
            raise ValueError("Cant encode this message, missing 'DSC Expansion Field Data'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Data' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Data' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Data' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")

def decode_pgn_129808_dscCallInformation(_data_raw_: int, _data_length_bits_: int) -> NMEA2000Message:
    """Decode PGN 129808."""
    nmea2000Message = NMEA2000Message(PGN=129808, id='dscCallInformation', description='DSC Call Information')
    running_bit_offset = 0
    _repeating_field_set_1_offset = 0
    # 1:dsc_format_symbol | Offset: 0, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 0
    dsc_format_symbol_raw = decode_int(_data_raw_, running_bit_offset, 8)
    dsc_format_symbol = master_dict['DSC_FORMAT'].get(dsc_format_symbol_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dscFormatSymbol', 'DSC Format Symbol', None, None, dsc_format_symbol, dsc_format_symbol_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 2:dsc_category_symbol | Offset: 8, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 8
    dsc_category_symbol_raw = decode_int(_data_raw_, running_bit_offset, 8)
    dsc_category_symbol = master_dict['DSC_CATEGORY'].get(dsc_category_symbol_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dscCategorySymbol', 'DSC Category Symbol', None, None, dsc_category_symbol, dsc_category_symbol_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 3:dsc_message_address | Offset: 16, Length: 40, Signed: False Resolution: 1, Field Type: DECIMAL, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 16
    dsc_message_address_raw = decode_int(_data_raw_, running_bit_offset, 40)
    dsc_message_address = decode_decimal(dsc_message_address_raw)
    nmea2000Message.fields.append(NMEA2000Field('dscMessageAddress', 'DSC Message Address', "MMSI, Geographic Area or blank", None, dsc_message_address, dsc_message_address_raw, None, FieldTypes.DECIMAL, False))
    running_bit_offset += 40

    # 4:__1st_telecommand | Offset: 56, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 56
    __1st_telecommand_raw = decode_int(_data_raw_, running_bit_offset, 8)
    __1st_telecommand = master_dict['DSC_FIRST_TELECOMMAND'].get(__1st_telecommand_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('1stTelecommand', '1st Telecommand', None, None, __1st_telecommand, __1st_telecommand_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 5:subsequent_communication_mode_or_2nd_telecommand | Offset: 64, Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 64
    subsequent_communication_mode_or_2nd_telecommand_raw = decode_int(_data_raw_, running_bit_offset, 8)
    subsequent_communication_mode_or_2nd_telecommand = master_dict['DSC_SECOND_TELECOMMAND'].get(subsequent_communication_mode_or_2nd_telecommand_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('subsequentCommunicationModeOr2ndTelecommand', 'Subsequent Communication Mode or 2nd Telecommand', None, None, subsequent_communication_mode_or_2nd_telecommand, subsequent_communication_mode_or_2nd_telecommand_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 6:proposed_rx_frequency_channel | Offset: 72, Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 72
    proposed_rx_frequency_channel, proposed_rx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('proposedRxFrequencyChannel', 'Proposed Rx Frequency/Channel', None, None, proposed_rx_frequency_channel, proposed_rx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 7:proposed_tx_frequency_channel | Offset: 120, Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 120
    proposed_tx_frequency_channel, proposed_tx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('proposedTxFrequencyChannel', 'Proposed Tx Frequency/Channel', None, None, proposed_tx_frequency_channel, proposed_tx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 8:telephone_number | Offset: 168, Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    running_bit_offset = 168
    telephone_number_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    telephone_number = telephone_number_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('telephoneNumber', 'Telephone Number', None, None, telephone_number, telephone_number_raw, None, FieldTypes.STRING_LAU, False))
    

    # 9:latitude_of_vessel_reported | Offset: , Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    latitude_of_vessel_reported = latitude_of_vessel_reported_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -90, 90)
    nmea2000Message.fields.append(NMEA2000Field('latitudeOfVesselReported', 'Latitude of Vessel Reported', None, 'deg', latitude_of_vessel_reported, latitude_of_vessel_reported_raw, PhysicalQuantities.GEOGRAPHICAL_LATITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 10:longitude_of_vessel_reported | Offset: , Length: 32, Signed: True Resolution: 1e-07, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    longitude_of_vessel_reported = longitude_of_vessel_reported_raw = decode_number(_data_raw_, running_bit_offset, 32, True, 1e-07, -180, 180)
    nmea2000Message.fields.append(NMEA2000Field('longitudeOfVesselReported', 'Longitude of Vessel Reported', None, 'deg', longitude_of_vessel_reported, longitude_of_vessel_reported_raw, PhysicalQuantities.GEOGRAPHICAL_LONGITUDE, FieldTypes.NUMBER, False))
    running_bit_offset += 32

    # 11:time_of_position | Offset: , Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    time_of_position_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time_of_position = decode_time(time_of_position_raw)
    nmea2000Message.fields.append(NMEA2000Field('timeOfPosition', 'Time of Position', "Seconds since midnight", 's', time_of_position, time_of_position_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 12:mmsi_of_ship_in_distress | Offset: , Length: 40, Signed: False Resolution: 1, Field Type: DECIMAL, Match: , PartOfPrimaryKey: ,
    mmsi_of_ship_in_distress_raw = decode_int(_data_raw_, running_bit_offset, 40)
    mmsi_of_ship_in_distress = decode_decimal(mmsi_of_ship_in_distress_raw)
    nmea2000Message.fields.append(NMEA2000Field('mmsiOfShipInDistress', 'MMSI of Ship In Distress', None, None, mmsi_of_ship_in_distress, mmsi_of_ship_in_distress_raw, None, FieldTypes.DECIMAL, False))
    running_bit_offset += 40

    # 13:dsc_eos_symbol | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    dsc_eos_symbol = dsc_eos_symbol_raw = decode_number(_data_raw_, running_bit_offset, 8, False, 1, 0, 252)
    nmea2000Message.fields.append(NMEA2000Field('dscEosSymbol', 'DSC EOS Symbol', None, None, dsc_eos_symbol, dsc_eos_symbol_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 8

    # 14:expansion_enabled | Offset: , Length: 2, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    expansion_enabled_raw = decode_int(_data_raw_, running_bit_offset, 2)
    expansion_enabled = master_dict['YES_NO'].get(expansion_enabled_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('expansionEnabled', 'Expansion Enabled', None, None, expansion_enabled, expansion_enabled_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 2

    # 15:reserved_ | Offset: , Length: 6, Signed: False Resolution: 1, Field Type: RESERVED, Match: , PartOfPrimaryKey: ,
    reserved_ = reserved__raw = decode_int(_data_raw_, running_bit_offset, 6)
    nmea2000Message.fields.append(NMEA2000Field('reserved_', 'Reserved', None, None, reserved_, reserved__raw, None, FieldTypes.RESERVED, False))
    running_bit_offset += 6

    # 16:calling_rx_frequency_channel | Offset: , Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    calling_rx_frequency_channel, calling_rx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('callingRxFrequencyChannel', 'Calling Rx Frequency/Channel', None, None, calling_rx_frequency_channel, calling_rx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 17:calling_tx_frequency_channel | Offset: , Length: 48, Signed: False Resolution: , Field Type: STRING_FIX, Match: , PartOfPrimaryKey: ,
    calling_tx_frequency_channel, calling_tx_frequency_channel_raw = decode_string_fix_raw(_data_raw_, running_bit_offset, 48)
    nmea2000Message.fields.append(NMEA2000Field('callingTxFrequencyChannel', 'Calling Tx Frequency/Channel', None, None, calling_tx_frequency_channel, calling_tx_frequency_channel_raw, None, FieldTypes.STRING_FIX, False))
    running_bit_offset += 48

    # 18:time_of_receipt | Offset: , Length: 32, Signed: False Resolution: 0.0001, Field Type: TIME, Match: , PartOfPrimaryKey: ,
    time_of_receipt_raw = decode_number(_data_raw_, running_bit_offset, 32, False, 0.0001, 0, 86401)
    time_of_receipt = decode_time(time_of_receipt_raw)
    nmea2000Message.fields.append(NMEA2000Field('timeOfReceipt', 'Time of Receipt', "Seconds since midnight", 's', time_of_receipt, time_of_receipt_raw, PhysicalQuantities.TIME, FieldTypes.TIME, False))
    running_bit_offset += 32

    # 19:date_of_receipt | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: DATE, Match: , PartOfPrimaryKey: ,
    date_of_receipt_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    date_of_receipt = decode_date(date_of_receipt_raw)
    nmea2000Message.fields.append(NMEA2000Field('dateOfReceipt', 'Date of Receipt', None, 'd', date_of_receipt, date_of_receipt_raw, PhysicalQuantities.DATE, FieldTypes.DATE, False))
    running_bit_offset += 16

    # 20:dsc_equipment_assigned_message_id | Offset: , Length: 16, Signed: False Resolution: 1, Field Type: NUMBER, Match: , PartOfPrimaryKey: ,
    dsc_equipment_assigned_message_id = dsc_equipment_assigned_message_id_raw = decode_number(_data_raw_, running_bit_offset, 16, False, 1, 0, 65532)
    nmea2000Message.fields.append(NMEA2000Field('dscEquipmentAssignedMessageId', 'DSC Equipment Assigned Message ID', None, None, dsc_equipment_assigned_message_id, dsc_equipment_assigned_message_id_raw, None, FieldTypes.NUMBER, False))
    running_bit_offset += 16

    # 21:dsc_expansion_field_symbol | Offset: , Length: 8, Signed: False Resolution: 1, Field Type: LOOKUP, Match: , PartOfPrimaryKey: ,
    _repeating_field_set_1_offset = running_bit_offset
    dsc_expansion_field_symbol_raw = decode_int(_data_raw_, running_bit_offset, 8)
    dsc_expansion_field_symbol = master_dict['DSC_EXPANSION_DATA'].get(dsc_expansion_field_symbol_raw, None)
    nmea2000Message.fields.append(NMEA2000Field('dscExpansionFieldSymbol', 'DSC Expansion Field Symbol', None, None, dsc_expansion_field_symbol, dsc_expansion_field_symbol_raw, None, FieldTypes.LOOKUP, False))
    running_bit_offset += 8

    # 22:dsc_expansion_field_data | Offset: , Length: , Signed: False Resolution: , Field Type: STRING_LAU, Match: , PartOfPrimaryKey: ,
    dsc_expansion_field_data_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
    dsc_expansion_field_data = dsc_expansion_field_data_raw
    running_bit_offset += bits_to_skip
    nmea2000Message.fields.append(NMEA2000Field('dscExpansionFieldData', 'DSC Expansion Field Data', None, None, dsc_expansion_field_data, dsc_expansion_field_data_raw, None, FieldTypes.STRING_LAU, False))
    

    running_bit_offset = _repeating_field_set_1_offset
    repeating_field_set_1_entries = []
    _repeating_field_set_1_count = None
    while (
        (_repeating_field_set_1_count is None and running_bit_offset < _data_length_bits_) or
        (_repeating_field_set_1_count is not None and len(repeating_field_set_1_entries) < _repeating_field_set_1_count)
    ):
        repeating_entry = {}
    
        dsc_expansion_field_symbol_raw = decode_int(_data_raw_, running_bit_offset, 8)
        dsc_expansion_field_symbol = master_dict['DSC_EXPANSION_DATA'].get(dsc_expansion_field_symbol_raw, None)
        running_bit_offset += 8
        repeating_entry["dscExpansionFieldSymbol"] = _repeating_entry_value(dsc_expansion_field_symbol, dsc_expansion_field_symbol_raw)
    
        dsc_expansion_field_data_raw, bits_to_skip = decode_string_lau(_data_raw_, running_bit_offset)
        dsc_expansion_field_data = dsc_expansion_field_data_raw
        running_bit_offset += bits_to_skip
        repeating_entry["dscExpansionFieldData"] = _repeating_entry_value(dsc_expansion_field_data, dsc_expansion_field_data_raw)
        repeating_field_set_1_entries.append(repeating_entry)
    if repeating_field_set_1_entries:
        nmea2000Message.fields = [
            field for field in nmea2000Message.fields
            if field.id not in {
                "dscExpansionFieldSymbol",
                "dscExpansionFieldData",
            }
        ]
        nmea2000Message.fields.append(NMEA2000Field('list', 'List', None, None, repeating_field_set_1_entries, None, None, FieldTypes.VARIABLE, False))
    return nmea2000Message

def encode_pgn_129808_dscCallInformation(nmea2000Message: NMEA2000Message) -> bytes:
    """Encode Nmea2000Message object to binary data for PGN 129808."""
    data_raw = 0
    running_bit_offset = 0
    payload_bit_length = 0
    repeating_field_set_1_entries = _get_repeating_entries(nmea2000Message, "list", (
        "dscExpansionFieldSymbol",
        "dscExpansionFieldData",
    ))
    # dscFormatSymbol | Offset: 0, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 0
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscFormatSymbol")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_FORMAT(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DSC Format Symbol' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Format Symbol' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Format Symbol' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscCategorySymbol | Offset: 8, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 8
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscCategorySymbol")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_CATEGORY(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DSC Category Symbol' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Category Symbol' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Category Symbol' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscMessageAddress | Offset: 16, Length: 40, Resolution: 1, Field Type: DECIMAL
    running_bit_offset = 16
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscMessageAddress")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        field_value = encode_decimal(field.value)
    if field_value is None:
        field_value = 0
    field_bit_length = 40
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'DSC Message Address' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Message Address' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Message Address' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # 1stTelecommand | Offset: 56, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 56
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("1stTelecommand")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_FIRST_TELECOMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, '1st Telecommand' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, '1st Telecommand' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, '1st Telecommand' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # subsequentCommunicationModeOr2ndTelecommand | Offset: 64, Length: 8, Resolution: 1, Field Type: LOOKUP
    running_bit_offset = 64
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("subsequentCommunicationModeOr2ndTelecommand")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_DSC_SECOND_TELECOMMAND(field.value)
    field_bit_length = 8
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Subsequent Communication Mode or 2nd Telecommand' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Subsequent Communication Mode or 2nd Telecommand' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Subsequent Communication Mode or 2nd Telecommand' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # proposedRxFrequencyChannel | Offset: 72, Length: 48, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 72
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("proposedRxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Proposed Rx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Proposed Rx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Proposed Rx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # proposedTxFrequencyChannel | Offset: 120, Length: 48, Resolution: , Field Type: STRING_FIX
    running_bit_offset = 120
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("proposedTxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Proposed Tx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Proposed Tx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Proposed Tx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # telephoneNumber | Offset: 168, Length: , Resolution: , Field Type: STRING_LAU
    running_bit_offset = 168
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("telephoneNumber")

    advance_running_offset = True
    field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
    field_value = encode_little_endian_data(field_bytes)
    field_bit_length = binary_data_bit_length(field_bytes)
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Telephone Number' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Telephone Number' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Telephone Number' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # latitudeOfVesselReported | Offset: , Length: 32, Resolution: 1e-07, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("latitudeOfVesselReported")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Latitude of Vessel Reported' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Latitude of Vessel Reported' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Latitude of Vessel Reported' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # longitudeOfVesselReported | Offset: , Length: 32, Resolution: 1e-07, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("longitudeOfVesselReported")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and raw_number_matches_value(field.raw_value, field.value, 1e-07):
        field_value = encode_number_raw(field.raw_value, 32, True)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, True, 1e-07)
    else:
        assert field.value is None or isinstance(field.value, (int, float))
        field_value = encode_number(field.value, 32, True, 1e-07)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Longitude of Vessel Reported' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Longitude of Vessel Reported' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Longitude of Vessel Reported' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # timeOfPosition | Offset: , Length: 32, Resolution: 0.0001, Field Type: TIME
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeOfPosition")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.0001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time of Position' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time of Position' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time of Position' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # mmsiOfShipInDistress | Offset: , Length: 40, Resolution: 1, Field Type: DECIMAL
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("mmsiOfShipInDistress")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        field_value = encode_decimal(field.value)
    if field_value is None:
        field_value = 0
    field_bit_length = 40
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'MMSI of Ship In Distress' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'MMSI of Ship In Distress' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'MMSI of Ship In Distress' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscEosSymbol | Offset: , Length: 8, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscEosSymbol")

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
        raise ValueError("Cant encode this message, 'DSC EOS Symbol' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC EOS Symbol' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC EOS Symbol' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # expansionEnabled | Offset: , Length: 2, Resolution: 1, Field Type: LOOKUP
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("expansionEnabled")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    elif isinstance(field.value, int):
        field_value = field.value
    else:
        field_value = lookup_encode_YES_NO(field.value)
    field_bit_length = 2
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Expansion Enabled' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Expansion Enabled' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Expansion Enabled' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # reserved_ | Offset: , Length: 6, Resolution: 1, Field Type: RESERVED
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("reserved_")

    advance_running_offset = True
    field_value = field.raw_value if isinstance(field.raw_value, int) else field.value
    if field_value is None:
        field_value = 0
    if not isinstance(field_value, int):
        raise ValueError("Cant encode this message, 'Reserved' must be an integer")
    field_bit_length = 6
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
    # callingRxFrequencyChannel | Offset: , Length: 48, Resolution: , Field Type: STRING_FIX
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("callingRxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Calling Rx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Calling Rx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Calling Rx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # callingTxFrequencyChannel | Offset: , Length: 48, Resolution: , Field Type: STRING_FIX
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("callingTxFrequencyChannel")

    advance_running_offset = True
    field_value = encode_string_fix(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else (field.value if field.value is not None else field.raw_value), 48)
    field_bit_length = 48
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Calling Tx Frequency/Channel' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Calling Tx Frequency/Channel' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Calling Tx Frequency/Channel' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # timeOfReceipt | Offset: , Length: 32, Resolution: 0.0001, Field Type: TIME
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("timeOfReceipt")

    advance_running_offset = True
    if isinstance(field.raw_value, int) and (isinstance(field.value, time)):
        field_value = encode_number_raw(field.raw_value, 32, False)
    elif isinstance(field.raw_value, (int, float)):
        field_value = encode_number(field.raw_value, 32, False, 0.0001)
    elif isinstance(field.value, (int, float)):
        field_value = encode_number(field.value, 32, False, 0.0001)
    else:
        assert field.value is None or isinstance(field.value, time)
        field_value = encode_time(field.value, 32)
    field_bit_length = 32
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Time of Receipt' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Time of Receipt' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Time of Receipt' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dateOfReceipt | Offset: , Length: 16, Resolution: 1, Field Type: DATE
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dateOfReceipt")

    advance_running_offset = True
    if isinstance(field.raw_value, int):
        field_value = field.raw_value
    else:
        assert field.value is None or isinstance(field.value, date)
        field_value = encode_date(field.value, 16)
    field_bit_length = 16
    assert isinstance(field_value, int)
    if field_value < 0:
        raise ValueError("Cant encode this message, 'Date of Receipt' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'Date of Receipt' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'Date of Receipt' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    # dscEquipmentAssignedMessageId | Offset: , Length: 16, Resolution: 1, Field Type: NUMBER
    field_offset = running_bit_offset
    field = nmea2000Message.get_field_by_id("dscEquipmentAssignedMessageId")

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
        raise ValueError("Cant encode this message, 'DSC Equipment Assigned Message ID' cannot be negative")
    if field_bit_length < 0:
        raise ValueError("Cant encode this message, 'DSC Equipment Assigned Message ID' has a negative bit length")
    if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
        raise ValueError("Cant encode this message, 'DSC Equipment Assigned Message ID' exceeds the encoded bit length")
    field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
    data_raw |= (field_value & field_mask) << field_offset
    payload_end_offset = field_offset + field_bit_length
    running_bit_offset = payload_end_offset if advance_running_offset else field_offset
    payload_bit_length = max(payload_bit_length, payload_end_offset)
    for repeating_entry in repeating_field_set_1_entries:
        # dscExpansionFieldSymbol | Offset: , Length: 8, Resolution: 1, Field Type: LOOKUP
        field = repeating_entry.get("dscExpansionFieldSymbol")
        if field is None:
            raise ValueError("Cant encode this message, missing 'DSC Expansion Field Symbol'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        if isinstance(field.raw_value, int):
            field_value = field.raw_value
        elif isinstance(field.value, int):
            field_value = field.value
        else:
            field_value = lookup_encode_DSC_EXPANSION_DATA(field.value)
        field_bit_length = 8
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Symbol' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Symbol' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Symbol' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
        # dscExpansionFieldData | Offset: , Length: , Resolution: , Field Type: STRING_LAU
        field = repeating_entry.get("dscExpansionFieldData")
        if field is None:
            raise ValueError("Cant encode this message, missing 'DSC Expansion Field Data'")
        field_offset = running_bit_offset
    
        advance_running_offset = True
        field_bytes = encode_string_lau(field.raw_value if isinstance(field.raw_value, (bytes, bytearray, memoryview)) else field.value)
        field_value = encode_little_endian_data(field_bytes)
        field_bit_length = binary_data_bit_length(field_bytes)
        assert isinstance(field_value, int)
        if field_value < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Data' cannot be negative")
        if field_bit_length < 0:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Data' has a negative bit length")
        if field_bit_length > 0 and field_value.bit_length() > field_bit_length:
            raise ValueError("Cant encode this message, 'DSC Expansion Field Data' exceeds the encoded bit length")
        field_mask = (1 << field_bit_length) - 1 if field_bit_length > 0 else 0
        data_raw |= (field_value & field_mask) << field_offset
        payload_end_offset = field_offset + field_bit_length
        running_bit_offset = payload_end_offset if advance_running_offset else field_offset
        payload_bit_length = max(payload_bit_length, payload_end_offset)
    
    return data_raw.to_bytes((payload_bit_length + 7) // 8, byteorder="little")
