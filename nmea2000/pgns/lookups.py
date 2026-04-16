# pylint: skip-file
from __future__ import annotations

from ..utils import *
from ..message import NMEA2000Message, NMEA2000Field, LookupFieldTypeEnumeration, int_to_bytes
from ..consts import PhysicalQuantities, FieldTypes, IndirectLookupEncodeMaps

master_dict = {

    'LIGHTING_COMMAND': {
        0: "Idle",
        1: "Detect Devices",
        2: "Reboot",
        3: "Factory Reset",
        4: "Powering Up",
    },
    'INDUSTRY_CODE': {
        0: "Global",
        1: "Highway",
        2: "Agriculture",
        3: "Construction",
        4: "Marine Industry",
        5: "Industrial",
    },
    'MANUFACTURER_CODE': {
        69: "ARKS Enterprises, Inc.",
        78: "FW Murphy/Enovation Controls",
        80: "Twin Disc",
        85: "Kohler Power Systems",
        88: "Hemisphere GPS Inc",
        116: "BEP Marine",
        135: "Airmar",
        137: "Maretron",
        140: "Lowrance",
        144: "Mercury Marine",
        147: "Nautibus Electronic GmbH",
        148: "Blue Water Data",
        154: "Westerbeke",
        157: "ISSPRO Inc",
        161: "Offshore Systems (UK) Ltd.",
        163: "Evinrude/BRP",
        165: "CPAC Systems AB",
        168: "Xantrex Technology Inc.",
        169: "Marlin Technologies, Inc.",
        172: "Yanmar Marine",
        174: "Volvo Penta",
        175: "Honda Marine",
        176: "Carling Technologies Inc. (Moritz Aerospace)",
        185: "Beede Instruments",
        192: "Floscan Instrument Co. Inc.",
        193: "Nobletec",
        198: "Mystic Valley Communications",
        199: "Actia",
        200: "Honda Marine",
        201: "Disenos Y Technologia",
        211: "Digital Switching Systems",
        215: "Xintex/Atena",
        224: "EMMI NETWORK S.L.",
        225: "Honda Marine",
        228: "ZF",
        229: "Garmin",
        233: "Yacht Monitoring Solutions",
        235: "Sailormade Marine Telemetry/Tetra Technology LTD",
        243: "Eride",
        250: "Honda Marine",
        257: "Honda Motor Company LTD",
        272: "Groco",
        273: "Actisense",
        274: "Amphenol LTW Technology",
        275: "Navico",
        283: "Hamilton Jet",
        285: "Sea Recovery",
        286: "Coelmo SRL Italy",
        295: "BEP Marine",
        304: "Empir Bus",
        305: "NovAtel",
        306: "Sleipner Motor AS",
        307: "MBW Technologies",
        311: "Fischer Panda",
        315: "ICOM",
        328: "Qwerty",
        329: "Dief",
        341: "Boening Automationstechnologie GmbH & Co. KG",
        345: "Korean Maritime University",
        351: "Thrane and Thrane",
        355: "Mastervolt",
        356: "Fischer Panda Generators",
        358: "Victron Energy",
        370: "Rolls Royce Marine",
        373: "Electronic Design",
        374: "Northern Lights",
        378: "Glendinning",
        381: "B & G",
        384: "Rose Point Navigation Systems",
        385: "Johnson Outdoors Marine Electronics Inc Geonav",
        394: "Capi 2",
        396: "Beyond Measure",
        400: "Livorsi Marine",
        404: "ComNav",
        409: "Chetco",
        419: "Fusion Electronics",
        421: "Standard Horizon",
        422: "True Heading AB",
        426: "Egersund Marine Electronics AS",
        427: "em-trak Marine Electronics",
        431: "Tohatsu Co, JP",
        437: "Digital Yacht",
        438: "Comar Systems Limited",
        440: "Cummins",
        443: "VDO (aka Continental-Corporation)",
        451: "Parker Hannifin aka Village Marine Tech",
        459: "Alltek Marine Electronics Corp",
        460: "SAN GIORGIO S.E.I.N",
        466: "Veethree Electronics & Marine",
        467: "Humminbird Marine Electronics",
        470: "SI-TEX Marine Electronics",
        471: "Sea Cross Marine AB",
        475: "GME aka Standard Communications Pty LTD",
        476: "Humminbird Marine Electronics",
        478: "Ocean Sat BV",
        481: "Chetco Digitial Instruments",
        493: "Watcheye",
        499: "Lcj Capteurs",
        502: "Attwood Marine",
        503: "Naviop S.R.L.",
        504: "Vesper Marine Ltd",
        510: "Marinesoft Co. LTD",
        513: "Simarine",
        517: "NoLand Engineering",
        518: "Transas USA",
        529: "National Instruments Korea",
        530: "National Marine Electronics Association",
        532: "Onwa Marine",
        540: "Webasto",
        571: "Marinecraft (South Korea)",
        573: "McMurdo Group aka Orolia LTD",
        578: "Advansea",
        579: "KVH",
        580: "San Jose Technology",
        583: "Yacht Control",
        586: "Suzuki Motor Corporation",
        591: "US Coast Guard",
        595: "Ship Module aka Customware",
        600: "Aquatic AV",
        605: "Aventics GmbH",
        606: "Intellian",
        612: "SamwonIT",
        614: "Arlt Tecnologies",
        637: "Bavaria Yacts",
        641: "Diverse Yacht Services",
        644: "Wema U.S.A dba KUS",
        645: "Garmin",
        658: "Shenzhen Jiuzhou Himunication",
        688: "Rockford Corp",
        699: "Harman International",
        704: "JL Audio",
        708: "Lars Thrane",
        715: "Autonnic",
        717: "Yacht Devices",
        734: "REAP Systems",
        735: "Au Electronics Group",
        739: "LxNav",
        741: "Littelfuse, Inc (formerly Carling Technologies)",
        743: "DaeMyung",
        744: "Woosung",
        748: "ISOTTA IFRA srl",
        773: "Clarion US",
        776: "HMI Systems",
        777: "Ocean Signal",
        778: "Seekeeper",
        781: "Poly Planar",
        785: "Fischer Panda DE",
        795: "Broyda Industries",
        796: "Canadian Automotive",
        797: "Tides Marine",
        798: "Lumishore",
        799: "Still Water Designs and Audio",
        802: "BJ Technologies (Beneteau)",
        803: "Gill Sensors",
        811: "Blue Water Desalination",
        815: "FLIR",
        824: "Undheim Systems",
        826: "Lewmar Inc",
        838: "TeamSurv",
        844: "Fell Marine",
        847: "Oceanvolt",
        862: "Prospec",
        868: "Data Panel Corp",
        890: "L3 Technologies",
        894: "Rhodan Marine Systems",
        896: "Nexfour Solutions",
        905: "ASA Electronics",
        909: "Marines Co (South Korea)",
        911: "Nautic-on",
        917: "Sentinel",
        929: "JL Marine ystems",
        930: "Ecotronix",
        944: "Zontisa Marine",
        951: "EXOR International",
        962: "Timbolier Industries",
        963: "TJC Micro",
        968: "Cox Powertrain",
        969: "Blue Seas",
        981: "Kobelt Manufacturing Co. Ltd",
        992: "Blue Ocean IOT",
        997: "Xenta Systems",
        1004: "Ultraflex SpA",
        1008: "Lintest SmartBoat",
        1011: "Soundmax",
        1020: "Team Italia Marine (Onyx Marine Automation s.r.l)",
        1021: "Entratech",
        1022: "ITC Inc.",
        1029: "The Marine Guardian LLC",
        1047: "Sonic Corporation",
        1051: "ProNav",
        1053: "Vetus Maxwell INC.",
        1056: "Lithium Pros",
        1059: "Boatrax",
        1062: "Marol Co ltd",
        1065: "CALYPSO Instruments",
        1066: "Spot Zero Water",
        1069: "Lithionics Battery LLC",
        1070: "Quick-teck Electronics Ltd",
        1075: "Uniden America",
        1083: "Nauticoncept",
        1084: "Shadow-Caster LED lighting LLC",
        1085: "Wet Sounds, LLC",
        1088: "E-T-A Circuit Breakers",
        1092: "Scheiber",
        1100: "Smart Yachts International Limited",
        1109: "Dockmate",
        1114: "Bobs Machine",
        1118: "L3Harris ASV",
        1119: "Balmar LLC",
        1120: "Elettromedia spa",
        1127: "Electromaax",
        1140: "Across Oceans Systems Ltd.",
        1145: "Kiwi Yachting",
        1150: "BSB Artificial Intelligence GmbH",
        1151: "Orca Technologoes AS",
        1154: "TBS Electronics BV",
        1158: "Technoton Electroics",
        1160: "MG Energy Systems B.V.",
        1169: "Sea Macine Robotics Inc.",
        1171: "Vista Manufacturing",
        1183: "Zipwake",
        1186: "Sailmon BV",
        1192: "Airmoniq Pro Kft",
        1194: "Sierra Marine",
        1200: "Xinuo Information Technology (Xiamen)",
        1218: "Septentrio",
        1233: "NKE Marine Elecronics",
        1238: "SuperTrack Aps",
        1239: "Honda Electronics Co., LTD",
        1245: "Raritan Engineering Company, Inc",
        1249: "Integrated Power Solutions AG",
        1260: "Interactive Technologies, Inc.",
        1283: "LTG-Tech",
        1299: "Energy Solutions (UK) LTD.",
        1300: "WATT Fuel Cell Corp",
        1302: "Pro Mainer",
        1305: "Dragonfly Energy",
        1306: "Koden Electronics Co., Ltd",
        1311: "Humphree AB",
        1316: "Hinkley Yachts",
        1317: "Global Marine Management GmbH (GMM)",
        1320: "Triskel Marine Ltd",
        1330: "Warwick Control Technologies",
        1331: "Dolphin Charger",
        1337: "Barnacle Systems Inc",
        1348: "Radian IoT, Inc.",
        1353: "Ocean LED Marine Ltd",
        1359: "BluNav",
        1361: "OVA (Nantong Saiyang Electronics Co., Ltd)",
        1368: "RAD Propulsion",
        1369: "Electric Yacht",
        1372: "Elco Motor Yachts",
        1384: "Tecnoseal Foundry S.r.l",
        1385: "Pro Charging Systems, LLC",
        1389: "EVEX Co., LTD",
        1398: "Gobius Sensor Technology AB",
        1403: "Arco Marine",
        1408: "Lenco Marine Inc.",
        1413: "Naocontrol S.L.",
        1417: "Revatek",
        1438: "Aeolionics",
        1439: "PredictWind Ltd",
        1440: "Egis Mobile Electric",
        1445: "Starboard Yacht Group",
        1446: "Roswell Marine",
        1451: "ePropulsion (Guangdong ePropulsion Technology Ltd.)",
        1452: "Micro-Air LLC",
        1453: "Vital Battery",
        1458: "Ride Controller LLC",
        1460: "Tocaro Blue",
        1461: "Vanquish Yachts",
        1471: "FT Technologies",
        1478: "Alps Alpine Co., Ltd.",
        1481: "E-Force Marine",
        1482: "CMC Marine",
        1483: "Nanjing Sandemarine Information Technology Co., Ltd.",
        1850: "Teleflex Marine (SeaStar Solutions)",
        1851: "Raymarine",
        1852: "Navionics",
        1853: "Japan Radio Co",
        1854: "Northstar Technologies",
        1855: "Furuno",
        1856: "Trimble",
        1857: "Simrad",
        1858: "Litton",
        1859: "Kvasar AB",
        1860: "MMP",
        1861: "Vector Cantech",
        1862: "Yamaha Marine",
        1863: "Faria Instruments",
    },
    'AIS_MESSAGE_ID': {
        1: "Scheduled Class A position report",
        2: "Assigned scheduled Class A position report",
        3: "Interrogated Class A position report",
        4: "Base station report",
        5: "Static and voyage related data",
        6: "Binary addressed message",
        7: "Binary acknowledgement",
        8: "Binary broadcast message",
        9: "Standard SAR aircraft position report",
        10: "UTC/date inquiry",
        11: "UTC/date response",
        12: "Safety related addressed message",
        13: "Safety related acknowledgement",
        14: "Satety related broadcast message",
        15: "Interrogation",
        16: "Assignment mode command",
        17: "DGNSS broadcast binary message",
        18: "Standard Class B position report",
        19: "Extended Class B position report",
        20: "Data link management message",
        21: "ATON report",
        22: "Channel management",
        23: "Group assignment command",
        24: "Static data report",
        25: "Single slot binary message",
        26: "Multiple slot binary message",
        27: "Position report for long range applications",
    },
    'SHIP_TYPE': {
        0: "Unavailable",
        20: "Wing In Ground",
        21: "Wing In Ground (hazard cat X)",
        22: "Wing In Ground (hazard cat Y)",
        23: "Wing In Ground (hazard cat Z)",
        24: "Wing In Ground (hazard cat OS)",
        29: "Wing In Ground (no additional information)",
        30: "Fishing",
        31: "Towing",
        32: "Towing exceeds 200m or wider than 25m",
        33: "Engaged in dredging or underwater operations",
        34: "Engaged in diving operations",
        35: "Engaged in military operations",
        36: "Sailing",
        37: "Pleasure",
        40: "High speed craft",
        41: "High speed craft (hazard cat X)",
        42: "High speed craft (hazard cat Y)",
        43: "High speed craft (hazard cat Z)",
        44: "High speed craft (hazard cat OS)",
        49: "High speed craft (no additional information)",
        50: "Pilot vessel",
        51: "SAR",
        52: "Tug",
        53: "Port tender",
        54: "Anti-pollution",
        55: "Law enforcement",
        56: "Spare",
        57: "Spare #2",
        58: "Medical",
        59: "Ships and aircraft of States not parties to an armed conflict",
        60: "Passenger ship",
        61: "Passenger ship (hazard cat X)",
        62: "Passenger ship (hazard cat Y)",
        63: "Passenger ship (hazard cat Z)",
        64: "Passenger ship (hazard cat OS)",
        69: "Passenger ship (no additional information)",
        70: "Cargo ship",
        71: "Cargo ship (hazard cat X)",
        72: "Cargo ship (hazard cat Y)",
        73: "Cargo ship (hazard cat Z)",
        74: "Cargo ship (hazard cat OS)",
        79: "Cargo ship (no additional information)",
        80: "Tanker",
        81: "Tanker (hazard cat X)",
        82: "Tanker (hazard cat Y)",
        83: "Tanker (hazard cat Z)",
        84: "Tanker (hazard cat OS)",
        89: "Tanker (no additional information)",
        90: "Other",
        91: "Other (hazard cat X)",
        92: "Other (hazard cat Y)",
        93: "Other (hazard cat Z)",
        94: "Other (hazard cat OS)",
        99: "Other (no additional information)",
    },
    'DEVICE_CLASS': {
        0: "Reserved for 2000 Use",
        10: "System tools",
        20: "Safety systems",
        25: "Internetwork device",
        30: "Electrical Distribution",
        35: "Electrical Generation",
        40: "Steering and Control surfaces",
        50: "Propulsion",
        60: "Navigation",
        70: "Communication",
        75: "Sensor Communication Interface",
        80: "Instrumentation/general systems",
        85: "External Environment",
        90: "Internal Environment",
        100: "Deck + cargo + fishing equipment systems",
        110: "Human Interface",
        120: "Display",
        125: "Entertainment",
    },
    'REPEAT_INDICATOR': {
        0: "Initial",
        1: "First retransmission",
        2: "Second retransmission",
        3: "Final retransmission",
    },
    'TX_RX_MODE': {
        0: "Tx A/Tx B, Rx A/Rx B",
        1: "Tx A, Rx A/Rx B",
        2: "Tx B, Rx A/Rx B",
    },
    'STATION_TYPE': {
        0: "All types of mobile station",
        2: "All types of Class B mobile station",
        3: "SAR airborne mobile station",
        4: "AtoN station",
        5: "Class B CS shipborne mobile station",
        6: "Inland waterways",
        7: "Regional use 7",
        8: "Regional use 8",
        9: "Regional use 9",
    },
    'REPORTING_INTERVAL': {
        0: "As given by the autonomous mode",
        1: "10 min",
        2: "6 min",
        3: "3 min",
        4: "1 min",
        5: "30 sec",
        6: "15 sec",
        7: "10 sec",
        8: "5 sec",
        9: "2 sec (not applicable to Class B CS)",
        10: "Next shorter reporting interval",
        11: "Next longer reporting interval",
    },
    'AIS_TRANSCEIVER': {
        0: "Channel A VDL reception",
        1: "Channel B VDL reception",
        2: "Channel A VDL transmission",
        3: "Channel B VDL transmission",
        4: "Own information not broadcast",
        5: "Reserved",
    },
    'AIS_ASSIGNED_MODE': {
        0: "Autonomous and continuous",
        1: "Assigned mode",
    },
    'ATON_TYPE': {
        0: "Default: Type of AtoN not specified",
        1: "Reference point",
        2: "RACON",
        3: "Fixed structure off-shore",
        4: "Reserved for future use",
        5: "Fixed light: without sectors",
        6: "Fixed light: with sectors",
        7: "Fixed leading light front",
        8: "Fixed leading light rear",
        9: "Fixed beacon: cardinal N",
        10: "Fixed beacon: cardinal E",
        11: "Fixed beacon: cardinal S",
        12: "Fixed beacon: cardinal W",
        13: "Fixed beacon: port hand",
        14: "Fixed beacon: starboard hand",
        15: "Fixed beacon: preferred channel port hand",
        16: "Fixed beacon: preferred channel starboard hand",
        17: "Fixed beacon: isolated danger",
        18: "Fixed beacon: safe water",
        19: "Fixed beacon: special mark",
        20: "Floating AtoN: cardinal N",
        21: "Floating AtoN: cardinal E",
        22: "Floating AtoN: cardinal S",
        23: "Floating AtoN: cardinal W",
        24: "Floating AtoN: port hand mark",
        25: "Floating AtoN: starboard hand mark",
        26: "Floating AtoN: preferred channel port hand",
        27: "Floating AtoN: preferred channel starboard hand",
        28: "Floating AtoN: isolated danger",
        29: "Floating AtoN: safe water",
        30: "Floating AtoN: special mark",
        31: "Floating AtoN: light vessel/LANBY/rigs",
    },
    'AIS_SPECIAL_MANEUVER': {
        0: "Not available",
        1: "Not engaged in special maneuver",
        2: "Engaged in special maneuver",
        3: "Reserved",
    },
    'POSITION_FIX_DEVICE': {
        0: "Default: undefined",
        1: "GPS",
        2: "GLONASS",
        3: "Combined GPS/GLONASS",
        4: "Loran-C",
        5: "Chayka",
        6: "Integrated navigation system",
        7: "Surveyed",
        8: "Galileo",
    },
    'GNS': {
        0: "GPS",
        1: "GLONASS",
        2: "GPS+GLONASS",
        3: "GPS+SBAS/WAAS",
        4: "GPS+SBAS/WAAS+GLONASS",
        5: "Chayka",
        6: "integrated",
        7: "surveyed",
        8: "Galileo",
    },
    'ENGINE_INSTANCE': {
        0: "Single Engine or Dual Engine Port",
        1: "Dual Engine Starboard",
    },
    'GEAR_STATUS': {
        0: "Forward",
        1: "Neutral",
        2: "Reverse",
    },
    'DIRECTION': {
        0: "Forward",
        1: "Reverse",
    },
    'POSITION_ACCURACY': {
        0: "Low",
        1: "High",
    },
    'RAIM_FLAG': {
        0: "not in use",
        1: "in use",
    },
    'TIME_STAMP': {
        60: "Not available",
        61: "Manual input mode",
        62: "Dead reckoning mode",
        63: "Positioning system is inoperative",
    },
    'GNS_METHOD': {
        0: "no GNSS",
        1: "GNSS fix",
        2: "DGNSS fix",
        3: "Precise GNSS",
        4: "RTK Fixed Integer",
        5: "RTK float",
        6: "Estimated (DR) mode",
        7: "Manual Input",
        8: "Simulate mode",
    },
    'GNS_INTEGRITY': {
        0: "No integrity checking",
        1: "Safe",
        2: "Caution",
        3: "Unsafe",
    },
    'SYSTEM_TIME': {
        0: "GPS",
        1: "GLONASS",
        2: "Radio Station",
        3: "Local Cesium clock",
        4: "Local Rubidium clock",
        5: "Local Crystal clock",
    },
    'MAGNETIC_VARIATION': {
        0: "Manual",
        1: "Automatic Chart",
        2: "Automatic Table",
        3: "Automatic Calculation",
        4: "WMM 2000",
        5: "WMM 2005",
        6: "WMM 2010",
        7: "WMM 2015",
        8: "WMM 2020",
        9: "WMM 2025",
    },
    'RESIDUAL_MODE': {
        0: "Autonomous",
        1: "Differential enhanced",
        2: "Estimated",
        3: "Simulator",
        4: "Manual",
    },
    'WIND_REFERENCE': {
        0: "True (ground referenced to North)",
        1: "Magnetic (ground referenced to Magnetic North)",
        2: "Apparent",
        3: "True (boat referenced)",
        4: "True (water referenced)",
    },
    'WATER_REFERENCE': {
        0: "Paddle wheel",
        1: "Pitot tube",
        2: "Doppler",
        3: "Correlation (ultra sound)",
        4: "Electro Magnetic",
    },
    'YES_NO': {
        0: "No",
        1: "Yes",
    },
    'OK_WARNING': {
        0: "OK",
        1: "Warning",
    },
    'OFF_ON': {
        0: "Off",
        1: "On",
    },
    'OFF_ON_CONTROL': {
        0: "Off",
        1: "On",
        2: "Reserved",
        3: "Take no action (no change)",
    },
    'DIRECTION_REFERENCE': {
        0: "True",
        1: "Magnetic",
        2: "Error",
    },
    'DIRECTION_RUDDER': {
        0: "No Order",
        1: "Move to starboard",
        2: "Move to port",
    },
    'NAV_STATUS': {
        0: "Under way using engine",
        1: "At anchor",
        2: "Not under command",
        3: "Restricted maneuverability",
        4: "Constrained by her draught",
        5: "Moored",
        6: "Aground",
        7: "Engaged in Fishing",
        8: "Under way sailing",
        9: "Hazardous material - High Speed",
        10: "Hazardous material - Wing in Ground",
        11: "Power-driven vessel towing astern",
        12: "Power-driven vessel pushing ahead or towing alongside",
        14: "AIS-SART",
    },
    'POWER_FACTOR': {
        0: "Leading",
        1: "Lagging",
        2: "Error",
    },
    'TEMPERATURE_SOURCE': {
        0: "Sea Temperature",
        1: "Outside Temperature",
        2: "Inside Temperature",
        3: "Engine Room Temperature",
        4: "Main Cabin Temperature",
        5: "Live Well Temperature",
        6: "Bait Well Temperature",
        7: "Refrigeration Temperature",
        8: "Heating System Temperature",
        9: "Dew Point Temperature",
        10: "Apparent Wind Chill Temperature",
        11: "Theoretical Wind Chill Temperature",
        12: "Heat Index Temperature",
        13: "Freezer Temperature",
        14: "Exhaust Gas Temperature",
        15: "Shaft Seal Temperature",
    },
    'HUMIDITY_SOURCE': {
        0: "Inside",
        1: "Outside",
    },
    'PRESSURE_SOURCE': {
        0: "Atmospheric",
        1: "Water",
        2: "Steam",
        3: "Compressed Air",
        4: "Hydraulic",
        5: "Filter",
        6: "AltimeterSetting",
        7: "Oil",
        8: "Fuel",
    },
    'DSC_FORMAT': {
        102: "Geographical area",
        112: "Distress",
        114: "Common interest",
        116: "All ships",
        120: "Individual stations",
        121: "Non-calling purpose",
        123: "Individual station automatic",
    },
    'DSC_CATEGORY': {
        100: "Routine",
        108: "Safety",
        110: "Urgency",
        112: "Distress",
    },
    'DSC_NATURE': {
        100: "Fire",
        101: "Flooding",
        102: "Collision",
        103: "Grounding",
        104: "Listing",
        105: "Sinking",
        106: "Disabled and adrift",
        107: "Undesignated",
        108: "Abandoning ship",
        109: "Piracy",
        110: "Man overboard",
        112: "EPIRB emission",
    },
    'DSC_FIRST_TELECOMMAND': {
        100: "F3E/G3E All modes TP",
        101: "F3E/G3E duplex TP",
        103: "Polling",
        104: "Unable to comply",
        105: "End of call",
        106: "Data",
        109: "J3E TP",
        110: "Distress acknowledgement",
        112: "Distress relay",
        113: "F1B/J2B TTY-FEC",
        115: "F1B/J2B TTY-ARQ",
        118: "Test",
        121: "Ship position or location registration updating",
        126: "No information",
    },
    'DSC_SECOND_TELECOMMAND': {
        100: "No reason given",
        101: "Congestion at MSC",
        102: "Busy",
        103: "Queue indication",
        104: "Station barred",
        105: "No operator available",
        106: "Operator temporarily unavailable",
        107: "Equipment disabled",
        108: "Unable to use proposed channel",
        109: "Unable to use proposed mode",
        110: "Ships and aircraft of States not parties to an armed conflict",
        111: "Medical transports",
        112: "Pay phone/public call office",
        113: "Fax/data",
        126: "No information",
    },
    'DSC_EXPANSION_DATA': {
        100: "Enhanced position",
        101: "Source and datum of position",
        102: "SOG",
        103: "COG",
        104: "Additional station identification",
        105: "Enhanced geographic area",
        106: "Number of persons on board",
    },
    'SEATALK_MESSAGE_ID': {
        240: "Seatalk 1 Encoded",
        140: "Display",
        108: "Pilot Configuration",
    },
    'SEATALK_COMMAND': {
        129: "Seatalk1",
        22: "Hull Type",
        38: "Auto Turn",
        12: "Settings",
    },
    'SEATALK1_COMMAND': {
        0: "Depth Below Transducer",
        1: "Equipment ID",
        5: "Engine RPM and PITCH",
        16: "Apparent Wind Angle",
        17: "Apparent Wind Speed",
        32: "Speed through water",
        33: "Trip Mileage",
        34: "Total Mileage",
        35: "Water temperature (ST50)",
        36: "Display units for Mileage & Speed",
        37: "Total & Trip Log",
        38: "Speed through water (with average)",
        39: "Water temperature",
        48: "Set lamp Intensity",
        54: "Cancel MOB (Man Over Board) condition",
        56: "Codelock data",
        80: "LAT position",
        81: "LON position",
        82: "Speed over Ground",
        83: "Course over Ground (COG)",
        84: "GMT-time",
        85: "TRACK keystroke on GPS unit",
        86: "Date",
        87: "Sat Info",
        88: "LAT/LON (raw unfiltered)",
        89: "Set Count Down Timer",
        97: "Issued by E-80 multifunction display at initialization",
        101: "Select Fathom display units for depth display",
        102: "Wind alarm",
        104: "Alarm acknowledgment keystroke",
        108: "Second equipment-ID datagram",
        110: "MOB (Man Over Board)",
        112: "Keystroke on Raymarine A25006 ST60 Maxiview Remote Control",
        128: "Set Lamp Intensity",
        129: "Sent by course computer during setup",
        130: "Target waypoint name",
        131: "Sent by course computer",
        132: "Compass heading Autopilot course and Rudder position",
        133: "Navigation to waypoint information",
        134: "Keystroke",
        135: "Set Response level",
        136: "Autopilot Parameter",
        137: "Compass heading sent by ST40 compass instrument",
        144: "Device Indentification",
        145: "Set Rudder gain",
        146: "Set Autopilot Parameter",
        147: "Enter AP-Setup",
        149: "Replaces command 84 while autopilot is in value setting mode",
        153: "Compass variation",
        154: "Version String",
        156: "Compass heading and Rudder position",
        158: "Waypoint definition",
        161: "Destination Waypoint Info",
        162: "Arrival Info",
        164: "Broadcast query/response to identify devices",
        165: "GPS and DGPS Info",
        167: "Unknown meaning",
        168: "Alarm ON/OFF for Guard",
        171: "Alarm ON/OFF for Guard",
    },
    'SEATALK1_ATT': {
        0: "Depth Below Transducer",
    },
    'SEATALK_ALARM_STATUS': {
        0: "Alarm condition not met",
        1: "Alarm condition met and not silenced",
        2: "Alarm condition met and silenced",
    },
    'SEATALK_ALARM_ID': {
        0: "No Alarm",
        1: "Shallow Depth",
        2: "Deep Depth",
        3: "Shallow Anchor",
        4: "Deep Anchor",
        5: "Off Course",
        6: "AWA High",
        7: "AWA Low",
        8: "AWS High",
        9: "AWS Low",
        10: "TWA High",
        11: "TWA Low",
        12: "TWS High",
        13: "TWS Low",
        14: "WP Arrival",
        15: "Boat Speed High",
        16: "Boat Speed Low",
        17: "Sea Temperature High",
        18: "Sea Temperature Low",
        19: "Pilot Watch",
        20: "Pilot Off Course",
        21: "Pilot Wind Shift",
        22: "Pilot Low Battery",
        23: "Pilot Last Minute Of Watch",
        24: "Pilot No NMEA Data",
        25: "Pilot Large XTE",
        26: "Pilot NMEA DataError",
        27: "Pilot CU Disconnected",
        28: "Pilot Auto Release",
        29: "Pilot Way Point Advance",
        30: "Pilot Drive Stopped",
        31: "Pilot Type Unspecified",
        32: "Pilot Calibration Required",
        33: "Pilot Last Heading",
        34: "Pilot No Pilot",
        35: "Pilot Route Complete",
        36: "Pilot Variable Text",
        37: "GPS Failure",
        38: "MOB",
        39: "Seatalk1 Anchor",
        40: "Pilot Swapped Motor Power",
        41: "Pilot Standby Too Fast To Fish",
        42: "Pilot No GPS Fix",
        43: "Pilot No GPS COG",
        44: "Pilot Start Up",
        45: "Pilot Too Slow",
        46: "Pilot No Compass",
        47: "Pilot Rate Gyro Fault",
        48: "Pilot Current Limit",
        49: "Pilot Way Point Advance Port",
        50: "Pilot Way Point Advance Stbd",
        51: "Pilot No Wind Data",
        52: "Pilot No Speed Data",
        53: "Pilot Seatalk Fail1",
        54: "Pilot Seatalk Fail2",
        55: "Pilot Warning Too Fast To Fish",
        56: "Pilot Auto Dockside Fail",
        57: "Pilot Turn Too Fast",
        58: "Pilot No Nav Data",
        59: "Pilot Lost Waypoint Data",
        60: "Pilot EEPROM Corrupt",
        61: "Pilot Rudder Feedback Fail",
        62: "Pilot Autolearn Fail1",
        63: "Pilot Autolearn Fail2",
        64: "Pilot Autolearn Fail3",
        65: "Pilot Autolearn Fail4",
        66: "Pilot Autolearn Fail5",
        67: "Pilot Autolearn Fail6",
        68: "Pilot Warning Cal Required",
        69: "Pilot Warning OffCourse",
        70: "Pilot Warning XTE",
        71: "Pilot Warning Wind Shift",
        72: "Pilot Warning Drive Short",
        73: "Pilot Warning Clutch Short",
        74: "Pilot Warning Solenoid Short",
        75: "Pilot Joystick Fault",
        76: "Pilot No Joystick Data",
        80: "Pilot Invalid Command",
        81: "AIS TX Malfunction",
        82: "AIS Antenna VSWR fault",
        83: "AIS Rx channel 1 malfunction",
        84: "AIS Rx channel 2 malfunction",
        85: "AIS No sensor position in use",
        86: "AIS No valid SOG information",
        87: "AIS No valid COG information",
        88: "AIS 12V alarm",
        89: "AIS 6V alarm",
        90: "AIS Noise threshold exceeded channel A",
        91: "AIS Noise threshold exceeded channel B",
        92: "AIS Transmitter PA fault",
        93: "AIS 3V3 alarm",
        94: "AIS Rx channel 70 malfunction",
        95: "AIS Heading lost/invalid",
        96: "AIS internal GPS lost",
        97: "AIS No sensor position",
        98: "AIS Lock failure",
        99: "AIS Internal GGA timeout",
        100: "AIS Protocol stack restart",
        101: "Pilot No IPS communications",
        102: "Pilot Power-On or Sleep-Switch Reset While Engaged",
        103: "Pilot Unexpected Reset While Engaged",
        104: "AIS Dangerous Target",
        105: "AIS Lost Target",
        106: "AIS Safety Related Message (used to silence)",
        107: "AIS Connection Lost",
        108: "No Fix",
    },
    'SEATALK_ALARM_GROUP': {
        0: "Instrument",
        1: "Autopilot",
        2: "Radar",
        3: "Chart Plotter",
        4: "AIS",
    },
    'SEATALK_PILOT_MODE': {
        64: "Standby",
        66: "Auto",
        70: "Wind",
        74: "Track",
    },
    'SEATALK_PILOT_HULL_TYPE': {
        0: "Sail",
        1: "Sail (slow turn)",
        2: "Sail Catamaran",
        3: "Power (slow turn)",
        4: "Power (fast turn)",
        8: "Power",
    },
    'SEATALK_SHARED': {
        1: "Shared",
        2: "Not Shared",
    },
    'ENTERTAINMENT_ZONE': {
        0: "All zones",
        1: "Zone 1",
        2: "Zone 2",
        3: "Zone 3",
        4: "Zone 4",
    },
    'ENTERTAINMENT_SOURCE': {
        0: "Vessel alarm",
        1: "AM",
        2: "FM",
        3: "Weather",
        4: "DAB",
        5: "Aux",
        6: "USB",
        7: "CD",
        8: "MP3",
        9: "Apple iOS",
        10: "Android",
        11: "Bluetooth",
        12: "Sirius XM",
        13: "Pandora",
        14: "Spotify",
        15: "Slacker",
        16: "Songza",
        17: "Apple Radio",
        18: "Last FM",
        19: "Ethernet",
        20: "Video MP4",
        21: "Video DVD",
        22: "Video BluRay",
        23: "HDMI",
        24: "Video",
    },
    'ENTERTAINMENT_PLAY_STATUS': {
        0: "Play",
        1: "Pause",
        2: "Stop",
        3: "FF 1x",
        4: "FF 2x",
        5: "FF 3x",
        6: "FF 4x",
        7: "RW 1x",
        8: "RW 2x",
        9: "RW 3x",
        10: "RW 4x",
        11: "Skip ahead",
        12: "Skip back",
        13: "Jog ahead",
        14: "Jog back",
        15: "Seek up",
        16: "Seek down",
        17: "Scan up",
        18: "Scan down",
        19: "Tune up",
        20: "Tune down",
        21: "Slow motion .75x",
        22: "Slow motion .5x",
        23: "Slow motion .25x",
        24: "Slow motion .125x",
    },
    'ENTERTAINMENT_REPEAT_STATUS': {
        0: "Off",
        1: "One",
        2: "All",
    },
    'ENTERTAINMENT_SHUFFLE_STATUS': {
        0: "Off",
        1: "Play queue",
        2: "All",
    },
    'ENTERTAINMENT_LIKE_STATUS': {
        0: "None",
        1: "Thumbs up",
        2: "Thumbs down",
    },
    'ENTERTAINMENT_TYPE': {
        0: "File",
        1: "Playlist Name",
        2: "Genre Name",
        3: "Album Name",
        4: "Artist Name",
        5: "Track Name",
        6: "Station Name",
        7: "Station Number",
        8: "Favourite Number",
        9: "Play Queue",
        10: "Content Info",
    },
    'ENTERTAINMENT_GROUP': {
        0: "File",
        1: "Playlist Name",
        2: "Genre Name",
        3: "Album Name",
        4: "Artist Name",
        5: "Track Name",
        6: "Station Name",
        7: "Station Number",
        8: "Favourite Number",
        9: "Play Queue",
        10: "Content Info",
    },
    'ENTERTAINMENT_CHANNEL': {
        0: "All channels",
        1: "Stereo full range",
        2: "Stereo front",
        3: "Stereo back",
        4: "Stereo surround",
        5: "Center",
        6: "Subwoofer",
        7: "Front left",
        8: "Front right",
        9: "Back left",
        10: "Back right",
        11: "Surround left",
        12: "Surround right",
    },
    'ENTERTAINMENT_EQ': {
        0: "Flat",
        1: "Rock",
        2: "Hall",
        3: "Jazz",
        4: "Pop",
        5: "Live",
        6: "Classic",
        7: "Vocal",
        8: "Arena",
        9: "Cinema",
        10: "Custom",
    },
    'ENTERTAINMENT_FILTER': {
        0: "Full range",
        1: "High pass",
        2: "Low pass",
        3: "Band pass",
        4: "Notch filter",
    },
    'ALERT_TYPE': {
        1: "Emergency Alarm",
        2: "Alarm",
        5: "Warning",
        8: "Caution",
    },
    'ALERT_CATEGORY': {
        0: "Navigational",
        1: "Technical",
    },
    'ALERT_TRIGGER_CONDITION': {
        0: "Manual",
        1: "Auto",
        2: "Test",
        3: "Disabled",
    },
    'ALERT_THRESHOLD_STATUS': {
        0: "Normal",
        1: "Threshold Exceeded",
        2: "Extreme Threshold Exceeded",
        3: "Low Threshold Exceeded",
        4: "Acknowledged",
        5: "Awaiting Acknowledge",
    },
    'ALERT_STATE': {
        0: "Disabled",
        1: "Normal",
        2: "Active",
        3: "Silenced",
        4: "Acknowledged",
        5: "Awaiting Acknowledge",
    },
    'ALERT_LANGUAGE_ID': {
        0: "English (US)",
        1: "English (UK)",
        2: "Arabic",
        3: "Chinese (simplified)",
        4: "Croatian",
        5: "Danish",
        6: "Dutch",
        7: "Finnish",
        8: "French",
        9: "German",
        10: "Greek",
        11: "Italian",
        12: "Japanese",
        13: "Korean",
        14: "Norwegian",
        15: "Polish",
        16: "Portuguese",
        17: "Russian",
        18: "Spanish",
        19: "Swedish",
    },
    'ALERT_RESPONSE_COMMAND': {
        0: "Acknowledge",
        1: "Temporary Silence",
        2: "Test Command off",
        3: "Test Command on",
    },
    'CONVERTER_STATE': {
        0: "Off",
        1: "Low Power Mode",
        2: "Fault",
        3: "Bulk",
        4: "Absorption",
        5: "Float",
        6: "Storage",
        7: "Equalize",
        8: "Pass thru",
        9: "Inverting",
        10: "Assisting",
    },
    'THRUSTER_DIRECTION_CONTROL': {
        0: "Off",
        1: "Ready",
        2: "To Port",
        3: "To Starboard",
    },
    'THRUSTER_RETRACT_CONTROL': {
        0: "Off",
        1: "Extend",
        2: "Retract",
    },
    'THRUSTER_MOTOR_TYPE': {
        0: "12VDC",
        1: "24VDC",
        2: "48VDC",
        3: "24VAC",
        4: "Hydraulic",
    },
    'BOOT_STATE': {
        0: "in Startup Monitor",
        1: "running Bootloader",
        2: "running Application",
    },
    'ACCESS_LEVEL': {
        0: "Locked",
        1: "unlocked level 1",
        2: "unlocked level 2",
    },
    'TRANSMISSION_INTERVAL': {
        0: "Acknowledge",
        1: "Transmit Interval/Priority not supported",
        2: "Transmit Interval too low",
        3: "Access denied",
        4: "Not supported",
    },
    'PARAMETER_FIELD': {
        0: "Acknowledge",
        1: "Invalid parameter field",
        2: "Temporary error",
        3: "Parameter out of range",
        4: "Access denied",
        5: "Not supported",
        6: "Read or Write not supported",
    },
    'PGN_LIST_FUNCTION': {
        0: "Transmit PGN list",
        1: "Receive PGN list",
    },
    'FUSION_COMMAND': {
        1: "Play",
        2: "Pause",
        4: "Next",
        6: "Prev",
    },
    'FUSION_SIRIUS_COMMAND': {
        1: "Next",
        2: "Prev",
    },
    'FUSION_MUTE_COMMAND': {
        1: "Mute On",
        2: "Mute Off",
    },
    'SEATALK_KEYSTROKE': {
        1: "Auto",
        2: "Standby",
        3: "Wind",
        5: "-1",
        6: "-10",
        7: "+1",
        8: "+10",
        33: "-1 and -10",
        34: "+1 and +10",
        35: "Track",
    },
    'SEATALK_DEVICE_ID': {
        3: "S100",
        5: "Course Computer",
    },
    'SEATALK_NETWORK_GROUP': {
        0: "None",
        1: "Helm 1",
        2: "Helm 2",
        3: "Cockpit",
        4: "Flybridge",
        5: "Mast",
        6: "Group 1",
        7: "Group 2",
        8: "Group 3",
        9: "Group 4",
        10: "Group 5",
    },
    'SEATALK_DISPLAY_COLOR': {
        0: "Day 1",
        2: "Day 2",
        3: "Red/Black",
        4: "Inverse",
    },
    'AIRMAR_CALIBRATE_FUNCTION': {
        0: "Normal/cancel calibration",
        1: "Enter calibration mode",
        2: "Reset calibration to 0",
        3: "Verify",
        4: "Reset compass to defaults",
        5: "Reset damping to defaults",
    },
    'AIRMAR_CALIBRATE_STATUS': {
        0: "Queried",
        1: "Passed",
        2: "Failed - timeout",
        3: "Failed - tilt error",
        4: "Failed - other",
        5: "In progress",
    },
    'AIRMAR_TEMPERATURE_INSTANCE': {
        0: "Device Sensor",
        1: "Onboard Water Sensor",
        2: "Optional Water Sensor",
    },
    'AIRMAR_FILTER': {
        0: "No filter",
        1: "Basic IIR filter",
    },
    'CONTROLLER_STATE': {
        0: "Error Active",
        1: "Error Passive",
        2: "Bus Off",
    },
    'EQUIPMENT_STATUS': {
        0: "Operational",
        1: "Fault",
    },
    'MOB_STATUS': {
        0: "MOB Emitter Activated",
        1: "Manual on-board MOB Button Activation",
        2: "Test mode",
    },
    'LOW_BATTERY': {
        0: "Good",
        1: "Low",
    },
    'TURN_MODE': {
        0: "Rudder limit controlled",
        1: "Turn rate controlled",
        2: "Radius controlled",
    },
    'ACCEPTABILITY': {
        0: "Bad level",
        1: "Bad frequency",
        2: "Being qualified",
        3: "Good",
    },
    'LINE': {
        0: "Line 1",
        1: "Line 2",
        2: "Line 3",
    },
    'WAVEFORM': {
        0: "Sine wave",
        1: "Modified sine wave",
    },
    'TANK_TYPE': {
        0: "Fuel",
        1: "Water",
        2: "Gray water",
        3: "Live well",
        4: "Oil",
        5: "Black water",
    },
    'DC_SOURCE': {
        0: "Battery",
        1: "Alternator",
        2: "Convertor",
        3: "Solar cell",
        4: "Wind generator",
    },
    'CHARGER_STATE': {
        0: "Not charging",
        1: "Bulk",
        2: "Absorption",
        3: "Overcharge",
        4: "Equalise",
        5: "Float",
        6: "No float",
        7: "Constant VI",
        8: "Disabled",
        9: "Fault",
    },
    'CHARGING_ALGORITHM': {
        0: "Trickle",
        1: "Constant voltage / Constant current",
        2: "2 stage (no float)",
        3: "3 stage",
    },
    'CHARGER_MODE': {
        0: "Standalone",
        1: "Primary",
        2: "Secondary",
        3: "Echo",
    },
    'INVERTER_STATE': {
        0: "Invert",
        1: "AC passthru",
        2: "Load sense",
        3: "Fault",
        4: "Disabled",
    },
    'BATTERY_TYPE': {
        0: "Flooded",
        1: "Gel",
        2: "AGM",
    },
    'BATTERY_VOLTAGE': {
        0: "6V",
        1: "12V",
        2: "24V",
        3: "32V",
        4: "36V",
        5: "42V",
        6: "48V",
    },
    'BATTERY_CHEMISTRY': {
        0: "Pb (Lead)",
        1: "Li",
        2: "NiCd",
        3: "ZnO",
        4: "NiMH",
    },
    'GOOD_WARNING_ERROR': {
        0: "Good",
        1: "Warning",
        2: "Error",
    },
    'TRACKING': {
        0: "Cancelled",
        1: "Acquiring",
        2: "Tracking",
        3: "Lost",
    },
    'TARGET_ACQUISITION': {
        0: "Manual",
        1: "Automatic",
    },
    'WINDLASS_DIRECTION': {
        0: "Off",
        1: "Down",
        2: "Up",
    },
    'SPEED_TYPE': {
        0: "Single speed",
        1: "Dual speed",
        2: "Proportional speed",
    },
    'WINDLASS_MOTION': {
        0: "Windlass stopped",
        1: "Deployment occurring",
        2: "Retrieval occurring",
    },
    'RODE_TYPE': {
        0: "Chain presently detected",
        1: "Rope presently detected",
    },
    'DOCKING_STATUS': {
        0: "Not docked",
        1: "Fully docked",
    },
    'AIS_TYPE': {
        0: "SOTDMA",
        1: "CS",
    },
    'AIS_BAND': {
        0: "Top 525 kHz of marine band",
        1: "Entire marine band",
    },
    'AIS_MODE': {
        0: "Autonomous",
        1: "Assigned",
    },
    'AIS_COMMUNICATION_STATE': {
        0: "SOTDMA",
        1: "ITDMA",
    },
    'AVAILABLE': {
        0: "Available",
        1: "Not available",
    },
    'BEARING_MODE': {
        0: "Great Circle",
        1: "Rhumbline",
    },
    'MARK_TYPE': {
        0: "Collision",
        1: "Turning point",
        2: "Reference",
        3: "Wheelover",
        4: "Waypoint",
    },
    'GNSS_MODE': {
        0: "1D",
        1: "2D",
        2: "3D",
        3: "Auto",
    },
    'RANGE_RESIDUAL_MODE': {
        0: "Range residuals were used to calculate data",
        1: "Range residuals were calculated after the position",
    },
    'DGNSS_MODE': {
        0: "None",
        1: "SBAS if available",
        3: "SBAS",
    },
    'SATELLITE_STATUS': {
        0: "Not tracked",
        1: "Tracked",
        2: "Used",
        3: "Not tracked+Diff",
        4: "Tracked+Diff",
        5: "Used+Diff",
    },
    'AIS_VERSION': {
        0: "ITU-R M.1371-1",
        1: "ITU-R M.1371-3",
        2: "ITU-R M.1371-5",
        3: "ITU-R M.1371 future edition",
    },
    'TIDE': {
        0: "Falling",
        1: "Rising",
    },
    'WATERMAKER_STATE': {
        0: "Stopped",
        1: "Starting",
        2: "Running",
        3: "Stopping",
        4: "Flushing",
        5: "Rinsing",
        6: "Initiating",
        7: "Manual",
    },
    'ENTERTAINMENT_ID_TYPE': {
        0: "Group",
        1: "File",
        2: "Encrypted group",
        3: "Encrypted file",
    },
    'ENTERTAINMENT_DEFAULT_SETTINGS': {
        0: "Save current settings as user default",
        1: "Load user default",
        2: "Load manufacturer default",
    },
    'ENTERTAINMENT_REGIONS': {
        0: "USA",
        1: "Europe",
        2: "Asia",
        3: "Middle East",
        4: "Latin America",
        5: "Australia",
        6: "Russia",
        7: "Japan",
    },
    'VIDEO_PROTOCOLS': {
        0: "PAL",
        1: "NTSC",
    },
    'ENTERTAINMENT_VOLUME_CONTROL': {
        0: "Up",
        1: "Down",
    },
    'BLUETOOTH_STATUS': {
        0: "Connected",
        1: "Not connected",
        2: "Not paired",
    },
    'BLUETOOTH_SOURCE_STATUS': {
        0: "Reserved",
        1: "Connected",
        2: "Connecting",
        3: "Not connected",
    },
    'SONICHUB_COMMAND': {
        1: "Init #2",
        4: "AM Radio",
        5: "Zone Info",
        6: "Source",
        8: "Source List",
        9: "Control",
        12: "FM Radio",
        13: "Playlist",
        14: "Track",
        15: "Artist",
        16: "Album",
        19: "Menu Item",
        20: "Zones",
        23: "Max Volume",
        24: "Volume",
        25: "Init #1",
        48: "Position",
        50: "Init #3",
    },
    'SIMNET_AP_MODE': {
        2: "Heading",
        3: "Wind",
        10: "Nav",
        11: "No Drift",
    },
    'SIMNET_DEVICE_MODEL': {
        0: "AC",
        1: "Other device",
        100: "NAC",
    },
    'SIMNET_DEVICE_REPORT': {
        2: "Status",
        3: "Send Status",
        10: "Mode",
        11: "Send Mode",
        23: "Sailing Processor Status",
    },
    'SIMNET_AP_STATUS': {
        2: "Manual",
        16: "Automatic",
    },
    'SIMNET_COMMAND': {
        50: "Text",
    },
    'SIMNET_EVENT_COMMAND': {
        1: "Alarm",
        2: "AP command",
        255: "Autopilot",
    },
    'SIMNET_NIGHT_MODE': {
        2: "Day",
        4: "Night",
    },
    'SIMNET_NIGHT_MODE_COLOR': {
        0: "Red",
        1: "Green",
        2: "Blue",
        3: "White",
        4: "Magenta",
    },
    'SIMNET_DISPLAY_GROUP': {
        1: "Default",
        2: "Group 1",
        3: "Group 2",
        4: "Group 3",
        5: "Group 4",
        6: "Group 5",
        7: "Group 6",
    },
    'SIMNET_HOUR_DISPLAY': {
        0: "24 hour",
        1: "12 hour",
    },
    'SIMNET_TIME_FORMAT': {
        1: "MM/dd/yyyy",
        2: "dd/MM/yyyy",
    },
    'SIMNET_BACKLIGHT_LEVEL': {
        0: "10% (Min)",
        1: "Day mode",
        4: "Night mode",
        11: "20%",
        22: "30%",
        33: "40%",
        44: "50%",
        55: "60%",
        66: "70%",
        77: "80%",
        88: "90%",
        99: "100% (Max)",
    },
    'SIMNET_AP_EVENTS': {
        2: "Follow/Non Follow",
        6: "Standby",
        9: "Heading mode",
        10: "Nav mode",
        12: "No Drift mode",
        13: "Non Follow Up mode",
        14: "Follow Up mode",
        15: "Wind mode",
        17: "Tack",
        18: "Square (Turn)",
        19: "C-Turn",
        20: "U-Turn",
        21: "Spiral (Turn)",
        22: "Zig Zag (Turn)",
        23: "Lazy-S (Turn)",
        24: "Depth (Turn)",
        26: "Change course",
        61: "Timer sync",
        112: "Ping port end",
        113: "Ping starboard end",
    },
    'SIMNET_DIRECTION': {
        2: "Port",
        3: "Starboard",
        4: "Left rudder (port)",
        5: "Right rudder (starboard)",
    },
    'SIMNET_AP_COMMAND_TYPE': {
        2: "Follow Up",
        10: "AP Command",
    },
    'SIMNET_ALARM': {
        57: "Low boat speed",
        56: "No autopilot computer",
        58: "Nav data missing",
    },
    'FUSION_MESSAGE_ID': {
        1: "Request Status",
        2: "Set Source",
        3: "Media Command",
        5: "Tuner Command",
        6: "Marine Tuner Command",
        7: "Set Marine Tuner Squelch",
        8: "Set Marine Tuner Scan Mode",
        9: "Menu Action",
        10: "Request Menu Count",
        11: "Request Menu Item",
        12: "Request Menu Lock ID",
        13: "Set Aux Gain",
        15: "Set Settings",
        16: "DAB Updtate Command",
        17: "Set Mute",
        18: "Set Balance",
        19: "Set Low Pass Filer",
        20: "Set Sublevel",
        22: "Set Equalizer",
        23: "Set Volume Limit",
        24: "Set Zone Volume",
        25: "Set All Volumes",
        27: "Set Line Level Control",
        28: "Power",
        29: "Set Device Name",
        30: "Send Sirius Command",
        31: "Set Sirius Parental",
        33: "Send Factory Reset Command",
        34: "Set Zone Name",
        35: "Send Dvd Command",
        36: "Dvd Press Ir Key",
        39: "Send Select Sirius Team",
        40: "Send Select Sirius Artist",
        41: "Send Sirius Sport Alert User Action",
        45: "Send Sirius Artist Song User Action",
        50: "Send Multiroom Command",
        51: "Get Multiroom Device Record",
        52: "Scan Multirooom Devices",
        53: "Send File Transfer",
        54: "Set Loud",
        56: "Fapi Set Source Multiroom Enabled",
        57: "Request Head Unit Dsp Settings",
        64: "Send Transfer Status",
        65: "Fapi Get Server Info",
        69: "Fapi Set Source Enabled",
        70: "Fapi Set Source Name",
        73: "Send External Amp Gain",
        74: "Send Internal Amp Gain",
        75: "Send Mono",
    },
    'FUSION_PLAY_STATUS': {
        0: "Invalid",
        1: "Playing",
        2: "Paused",
        3: "Stopped",
        4: "Skip Forward",
        5: "Skip Rewind",
    },
    'FUSION_SOURCE_TYPE': {
        0: "AM",
        1: "FM",
        2: "Aux",
        3: "Sirius",
        4: "Ipod",
        5: "USB",
        6: "DVD",
        7: "VHF",
        8: "Invalid",
        9: "MTP",
        10: "Bluetooth",
        11: "ARC",
        12: "Android",
        13: "Pandora",
        14: "DAB",
        15: "AirPlay",
        16: "UPNP",
        17: "Unknown",
    },
    'FUSION_SIRIUS_COM_STATE': {
        255: "Unknown",
        1: "Off",
        2: "Initialising",
        3: "On",
    },
    'FUSION_SIRIUS_ALERT': {
        255: "Unknown",
        1: "None",
        2: "Antenna",
        3: "NoSignal",
        4: "Subscription Update",
    },
    'FUSION_SIRIUS_TUNING_MODE': {
        1: "Normal",
        2: "Category",
        3: "Preset",
    },
    'FUSION_STATUS_MESSAGE_ID': {
        0: "Unknown",
        32769: "API Version",
        32770: "Source",
        32771: "Source Count",
        32772: "Track Info",
        32773: "Track Title",
        32774: "Track Artist",
        32775: "Track Album",
        32776: "Cover Art",
        32777: "Track Progress",
        32778: "Tuner Align",
        32779: "Tuner",
        32780: "Marine Tuner",
        32781: "Marine Squelch",
        32782: "Marine Scan Mode",
        32783: "Menu Action",
        32784: "Menu Count",
        32785: "Menu Item",
        32786: "Menu Lock ID",
        32787: "Aux Gain",
        32788: "Setting",
        32789: "Settings",
        32790: "Update Firmware Result",
        32791: "Mute",
        32792: "Balance",
        32793: "Low Pass Filter",
        32794: "Sublevels",
        32795: "Tone",
        32796: "Volume Limits",
        32797: "Volume",
        32798: "Capabilities",
        32799: "Line Level Control",
        32800: "Power",
        32801: "Unit Name",
        32802: "Sirius",
        32803: "SiriusXM Preset Event",
        32804: "SiriusXM Channel",
        32805: "SiriusXM Title",
        32806: "SiriusXM Artist",
        32807: "SiriusXM Genre",
        32808: "SiriusXM Category",
        32809: "SiriusXm Signal",
        32810: "SiriusXM Parental Request",
        32811: "SiriusXM Diagnostics",
        32812: "SiriusXM Presets",
        32813: "Zone Name",
    },
    'SONICHUB_CONTROL': {
        0: "Set",
        128: "Ack",
    },
    'SONICHUB_SOURCE': {
        0: "AM",
        1: "FM",
        2: "iPod",
        3: "USB",
        4: "AUX",
        5: "AUX 2",
        6: "Mic",
    },
    'ISO_CONTROL': {
        0: "ACK",
        1: "NAK",
        2: "Access Denied",
        3: "Address Busy",
    },
    'ISO_COMMAND': {
        0: "ACK",
        16: "RTS",
        17: "CTS",
        19: "EOM",
        32: "BAM",
        255: "Abort",
    },
    'GROUP_FUNCTION': {
        0: "Request",
        1: "Command",
        2: "Acknowledge",
        3: "Read Fields",
        4: "Read Fields Reply",
        5: "Write Fields",
        6: "Write Fields Reply",
    },
    'AIRMAR_COMMAND': {
        32: "Attitude Offsets",
        33: "Calibrate Compass",
        34: "True Wind Options",
        35: "Simulate Mode",
        40: "Calibrate Depth",
        41: "Calibrate Speed",
        42: "Calibrate Temperature",
        43: "Speed Filter",
        44: "Temperature Filter",
        46: "NMEA 2000 options",
    },
    'AIRMAR_DEPTH_QUALITY_FACTOR': {
        0: "Depth unlocked",
        1: "Quality 10%",
        2: "Quality 20%",
        3: "Quality 30%",
        4: "Quality 40%",
        5: "Quality 50%",
        6: "Quality 60%",
        7: "Quality 70%",
        8: "Quality 80%",
        9: "Quality 90%",
        10: "Quality 100%",
    },
    'PGN_ERROR_CODE': {
        0: "Acknowledge",
        1: "PGN not supported",
        2: "PGN not available",
        3: "Access denied",
        4: "Not supported",
        5: "Tag not supported",
        6: "Read or Write not supported",
    },
    'AIRMAR_TRANSMISSION_INTERVAL': {
        0: "Measure interval",
        1: "Requested by user",
    },
    'MOB_POSITION_SOURCE': {
        0: "Position estimated by the vessel",
        1: "Position reported by MOB emitter",
    },
    'STEERING_MODE': {
        0: "Main Steering",
        1: "Non-Follow-Up Device",
        2: "Follow-Up Device",
        3: "Heading Control Standalone",
        4: "Heading Control",
        5: "Track Control",
    },
    'FUSION_RADIO_SOURCE': {
        0: "AM",
        1: "FM",
    },
    'FUSION_SETTING': {
        0: "Alpha Search Threshold",
        1: "iPod Subtitles",
        2: "Zone 2 Linked",
        3: "Zone 2 Enabled",
        4: "Zone 3 Enabled",
        5: "Zone 4 Enabled",
        6: "Telemute",
        7: "Tuner Region",
        8: "Marine Zone",
        9: "USB repeat",
        10: "USB shuffle",
        11: "iPod Album Artwork",
        12: "iPod repeat",
        13: "iPod shuffle",
        14: "AM Preset 0",
        15: "AM Preset 1",
        16: "AM Preset 2",
        17: "AM Preset 3",
        18: "AM Preset 4",
        19: "AM Preset 5",
        20: "AM Preset 6",
        21: "AM Preset 7",
        22: "AM Preset 8",
        23: "AM Preset 9",
        24: "AM Preset 10",
        25: "AM Preset 11",
        26: "AM Preset 12",
        27: "AM Preset 13",
        28: "AM Preset 14",
        29: "FM Preset 0",
        30: "FM Preset 1",
        31: "FM Preset 2",
        32: "FM Preset 3",
        33: "FM Preset 4",
        34: "FM Preset 5",
        35: "FM Preset 6",
        36: "FM Preset 7",
        37: "FM Preset 8",
        38: "FM Preset 9",
        39: "FM Preset 10",
        40: "FM Preset 11",
        41: "FM Preset 12",
        42: "FM Preset 13",
        43: "FM Preset 14",
        44: "VHF Preset 0",
        45: "VHF Preset 1",
        46: "VHF Preset 2",
        47: "VHF Preset 3",
        48: "VHF Preset 4",
        49: "VHF Preset 5",
        50: "VHF Preset 6",
        51: "VHF Preset 7",
        52: "VHF Preset 8",
        53: "VHF Preset 9",
        54: "VHF Preset 10",
        55: "VHF Preset 11",
        56: "VHF Preset 12",
        57: "VHF Preset 13",
        58: "VHF Preset 14",
        59: "Clock Time",
        60: "Clock Alarm",
        61: "iPod Video Signal",
        62: "iPod Monitor Aspect",
        63: "Aux Name Index",
        64: "AM Enabled",
        65: "VHF Enabled",
        66: "Language",
        67: "Internal Amps On",
        68: "MTP Repeat",
        69: "MTP Shuffle",
        70: "Id Accessory Source",
        71: "NMEA Power",
        72: "Low Power Mode",
        73: "DVD region",
        74: "Volume Zone Sync",
        75: "Max Volume Start",
        76: "BT Auto Connect",
        77: "Null Setting",
    },
    'FUSION_REPEAT_STATUS': {
        0: "Off",
        1: "One/track",
        2: "All/album",
    },
    'AIRMAR_POST_CONTROL': {
        0: "Report previous values",
        1: "Generate new values",
    },
    'AIRMAR_POST_ID': {
        1: "Format Code",
        2: "Factory EEPROM",
        3: "User EEPROM",
        4: "Water Temperature Sensor",
        5: "Sonar Transceiver",
        6: "Speed sensor",
        7: "Internal temperature sensor",
        8: "Battery voltage sensor",
    },
    'SONICHUB_TUNING': {
        1: "Seeking up",
        2: "Tuned",
        3: "Seeking down",
    },
    'SONICHUB_PLAYLIST': {
        1: "Report",
        4: "Next song",
        6: "Previous song",
    },
    'FUSION_POWER_STATE': {
        1: "On",
        2: "Off",
    },
    'PRIORITY': {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "Leave unchanged",
        9: "Reset to default",
    },
    'DEVICE_TEMP_STATE': {
        0: "Cold",
        1: "Warm",
        2: "Hot",
    },
    'BANDG_DECIMALS': {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        254: "Auto",
    },
    'GARMIN_COLOR_MODE': {
        0: "Day",
        1: "Night",
        13: "Color",
    },
    'GARMIN_COLOR': {
        0: "Day full color",
        1: "Day high contrast",
        2: "Night full color",
        3: "Night red/black",
        4: "Night green/black",
    },
    'GARMIN_BACKLIGHT_LEVEL': {
        0: "0%",
        1: "5%",
        2: "10%",
        3: "15%",
        4: "20%",
        5: "25%",
        6: "30%",
        7: "35%",
        8: "40%",
        9: "45%",
        10: "50%",
        11: "55%",
        12: "60%",
        13: "65%",
        14: "70%",
        15: "75%",
        16: "80%",
        17: "85%",
        18: "90%",
        19: "95%",
        20: "100%",
    },
    'SEATALK_PILOT_MODE_16': {
        0: "Standby",
        64: "Auto, compass commanded",
        256: "Vane, Wind Mode",
        384: "Track Mode",
        385: "No Drift, COG referenced (In track, course changes)",
    },
    'STATION_HEALTH': {
        0: "Not Working",
        1: "Unmonitored",
        2: "Healthy Operational",
        3: "Healthy Test Mode",
        4: "Test Mode",
    },
    'SERIAL_BIT_RATE': {
        0: "25",
        1: "50",
        2: "100",
        3: "200",
        4: "300",
        5: "600",
        6: "1200",
        7: "2400",
        8: "4800",
        9: "9600",
        10: "19200",
        11: "38400",
        12: "57600",
    },
    'SERIAL_DETECTION_MODE': {
        0: "Auto bit rate",
        1: "Manual bit rate",
    },
    'DIFFERENTIAL_SOURCE': {
        0: "Auto",
        1: "Loran",
        2: "MSK Beacon",
        3: "FM Subcarrier",
        4: "AIS",
        5: "Ground based radio",
        6: "SBAS",
        7: "Satellite",
    },
    'DIFFERENTIAL_MODE': {
        0: "Manual",
        1: "Auto Power",
        2: "Auto Range",
    },
    'WP_POSITION_RESOLUTION': {
        0: "more than 0.1 min",
        1: "<0.01 .. 0.1] min",
        2: "<0.001 .. 0.01] min",
        3: "<0.0001 .. 0.001] min",
        4: "<0 .. 0.0001] min",
    },
    'WP_IDENTIFICATION_METHOD': {
        0: "Waypoints in WP list",
        1: "Waypoints embedded in route",
    },
    'WP_ROUTE_STATUS': {
        0: "Active",
        1: "Inactive",
        2: "Deleted",
    },
    'WP_NAVIGATION_METHOD': {
        0: "Great Circle",
        1: "Rhumb Line",
    },
    'INVERTER_MODE': {
        0: "Standalone",
        1: "Series Master",
        2: "Series Slave",
        3: "Parallel Master",
        4: "Parallel Slave",
    },
    'CERTIFICATION_LEVEL': {
        0: "Level A",
        1: "Level B",
    },
    'AGS_MODE': {
        0: "Off",
        1: "On",
        2: "Automatic",
    },
    'AGS_OPERATING_STATE': {
        0: "Quiet time",
        1: "Auto on",
        2: "Auto off",
        3: "Manual On",
        4: "Manual Off",
        5: "Generator shutdown",
        6: "External shutdown",
        7: "Fault",
        8: "Suspend",
        9: "Not operating",
    },
    'AGS_GENERATING_STATE': {
        0: "Preheating",
        1: "Start delay",
        2: "Cranking",
        3: "Starter cooling",
        4: "Warming up",
        5: "Cooling down",
        6: "Spinning up",
        7: "Shutdown bypass",
        8: "Stopping",
        9: "Running",
        10: "Stopped",
        11: "Crank delaty",
    },
    'AGS_ON_REASON': {
        0: "Not on",
        1: "DC voltage low",
        2: "Battery state of charge low",
        3: "AC current high",
        4: "Contact closed",
        5: "Manual on",
        6: "Exercise",
        7: "Non Quiet time",
        8: "External on via AGS",
        9: "External on via generator",
        10: "Unable to stop",
    },
    'AGS_OFF_REASON': {
        0: "Not off",
        1: "DC voltage high",
        2: "Battery state of charge high",
        3: "AC current low",
        4: "Contact opened",
        5: "Reached absorption",
        6: "Reached float",
        7: "Manual off",
        8: "Max run time",
        9: "Max auto cycle",
        10: "Exercise done",
        11: "Quiet time",
        12: "External off via AGS",
        13: "Safe mode",
        14: "External off via generator",
        15: "External shutdown",
        16: "Auto off",
        17: "Fault",
        18: "Unable to start",
    },
    'TELEPHONE_MODE': {
        0: "F3E/G3E simplex, telephone",
        1: "F3E/G3E duplex, telephone",
        2: "J3E, telephone",
        3: "H3E, telephone",
        4: "F1B/J2B FEC NBDP, telex/teleprinter",
        5: "F1B/J2B ARQ NBDP, telex/teleprinter",
        6: "F1B/J2B receive only, teleprinter/DSC",
        7: "F1B/J2B, teleprinter/DSC",
        8: "A1A Morse, tape recorder",
        9: "A1A Morse, Morse key/head set",
        10: "F1C/F2C/F3C, FAX machine",
    },
    'POWER_MODE': {
        0: "High",
        1: "Low",
    },
    'BROADCAST_INDICATOR': {
        0: "Broadcast geo area message",
        1: "Addressed message",
    },
    'BANDWIDTH': {
        0: "Default",
        1: "12.5 kHz",
    },
    'FLOOD_STATE': {
        0: "Flood",
        1: "Slack",
        2: "Ebb",
    },
    'AC_LINE': {
        0: "Line 1",
        1: "Line 2",
        2: "Line 3",
    },
    'ZONE_SIZE': {
        0: "1 nm",
        1: "2 nm",
        2: "3 nm",
        3: "4 nm",
        4: "5 nm",
        5: "6 nm",
    },
    'MARETRON_PRODUCT_CODE': {
        434: "SSC200",
        2686: "SSC300",
        2781: "TLA100",
        3979: "NBE100",
        4078: "RIM100",
        8165: "ALM100",
        20067: "TMP100",
        22585: "DCR100",
        23603: "SIM100",
        26493: "ACM100",
    },
    'MARETRON_OPCODE': {
        0: "Read All",
        1: "Write Register",
        2: "Read Config",
        3: "Write Config",
        4: "Calibrate",
        5: "Clear Calibration",
        6: "Status",
        7: "Clear Status",
        8: "Reset Factory Default",
        9: "Debug",
        16: "Write Instance",
        17: "Read Instance",
        32: "Write Label",
        33: "Read Label",
        48: "Write Switch Config",
        49: "Read Switch Config",
        64: "Write Alert Config",
        65: "Read Alert Config",
        80: "Write Channel Config",
        81: "Read Channel Config",
        86: "Read Channel Config Extended",
        87: "Write Channel Config Extended",
    },
    'MARETRON_SOFTWARE_CODE': {
        1: "Version 1",
    },
    'MARETRON_COMMAND': {
        80: "Deviation calibration",
    },
    'MARETRON_STATUS_DEVIATION': {
        1: "Started",
        2: "Completed successfully",
        3: "Failed to complete",
        4: "Turning too fast",
        5: "Turning too slow",
        6: "Invalid movement",
    },
    'AUTOMATIC_MANUAL': {
        0: "Automatic",
        1: "Manual",
    },
    'SBAS_SV': {
        0: "120",
        1: "121",
        2: "122",
        3: "123",
        4: "124",
        5: "125",
        6: "126",
        7: "127",
        8: "128",
        9: "129",
        10: "130",
        11: "131",
        12: "132",
        13: "133",
        14: "134",
        15: "135",
        16: "136",
        17: "137",
        18: "138",
    },
}

master_flags_dict = {

    'STATION_STATUS': {
        0: "Station in use",
        1: "Low SNR",
        2: "Cycle Error",
        3: "Blink",
    },
    'ENGINE_STATUS_1': {
        0: "Check Engine",
        1: "Over Temperature",
        2: "Low Oil Pressure",
        3: "Low Oil Level",
        4: "Low Fuel Pressure",
        5: "Low System Voltage",
        6: "Low Coolant Level",
        7: "Water Flow",
        8: "Water In Fuel",
        9: "Charge Indicator",
        10: "Preheat Indicator",
        11: "High Boost Pressure",
        12: "Rev Limit Exceeded",
        13: "EGR System",
        14: "Throttle Position Sensor",
        15: "Emergency Stop",
    },
    'ENGINE_STATUS_2': {
        0: "Warning Level 1",
        1: "Warning Level 2",
        2: "Power Reduction",
        3: "Maintenance Needed",
        4: "Engine Comm Error",
        5: "Sub or Secondary Throttle",
        6: "Neutral Start Protect",
        7: "Engine Shutting Down",
    },
    'ENTERTAINMENT_PLAY_STATUS_BITFIELD': {
        0: "Play",
        1: "Pause",
        2: "Stop",
        3: "FF 1x",
        4: "FF 2x",
        5: "FF 3x",
        6: "FF 4x",
        7: "RW 1x",
        8: "RW 2x",
        9: "RW 3x",
        10: "RW 4x",
        11: "Skip ahead",
        12: "Skip back",
        13: "Jog ahead",
        14: "Jog back",
        15: "Seek up",
        16: "Seek down",
        17: "Scan up",
        18: "Scan down",
        19: "Tune up",
        20: "Tune down",
        21: "Slow motion .75x",
        22: "Slow motion .5x",
        23: "Slow motion .25x",
        24: "Slow motion .125x",
        25: "Source renaming",
    },
    'ENTERTAINMENT_GROUP_BITFIELD': {
        0: "File",
        1: "Playlist Name",
        2: "Genre Name",
        3: "Album Name",
        4: "Artist Name",
        5: "Track Name",
        6: "Station Name",
        7: "Station Number",
        8: "Favourite Number",
        9: "Play Queue",
        10: "Content Info",
    },
    'THRUSTER_CONTROL_EVENTS': {
        0: "Another device controlling thruster",
        1: "Boat speed too fast to safely use thruster",
    },
    'THRUSTER_MOTOR_EVENTS': {
        0: "Motor over temperature cutout",
        1: "Motor over current cutout",
        2: "Low oil level warning",
        3: "Oil over temperature warning",
        4: "Controller under voltage cutout",
        5: "Manufacturer defined",
    },
    'WINDLASS_CONTROL': {
        0: "Another device controlling windlass",
    },
    'WINDLASS_OPERATION': {
        0: "System error",
        1: "Sensor error",
        2: "No windlass motion detected",
        3: "Retrieval docking distance reached",
        4: "End of rode reached",
    },
    'WINDLASS_MONITORING': {
        0: "Controller under voltage cut-out",
        1: "Controller over current cut-out",
        2: "Controller over temperature cut-out",
        3: "Manufacturer defined",
    },
    'SIMNET_AP_MODE_BITFIELD': {
        3: "Standby",
        4: "Heading",
        6: "Nav",
        8: "No Drift",
        10: "Wind",
    },
    'SIMNET_ALERT_BITFIELD': {
        0: "No GPS fix",
        2: "No active autopilot control unit",
        4: "No autopilot computer",
        6: "AP clutch overload",
        8: "AP clutch disengaged",
        10: "Rudder controller fault",
        12: "No rudder response",
        14: "Rudder drive overload",
        16: "High drive supply",
        18: "Low drive supply",
        20: "Memory fail",
        22: "AP position data missing",
        24: "AP speed data missing",
        26: "AP depth data missing",
        28: "AP heading data missing",
        30: "AP nav data missing",
        32: "AP rudder data missing",
        34: "AP wind data missing",
        36: "AP off course",
        38: "High drive temperature",
        40: "Drive inhibit",
        42: "Rudder limit",
        44: "Drive computer missing",
        46: "Drive ready missing",
        48: "EVC com error",
        50: "EVC override",
        52: "Low CAN bus voltage",
        54: "CAN bus supply overload",
        56: "Wind sensor battery low",
    },
    'ENTERTAINMENT_REPEAT_BITFIELD': {
        0: "Song",
        1: "Play queue",
    },
    'ENTERTAINMENT_SHUFFLE_BITFIELD': {
        0: "Play queue",
        1: "All",
    },
    'WP_CHANGE': {
        0: "Change in main data (Position, Name)",
        1: "Change in supplementary parameters (or new added)",
        2: "Changed number of WPs in Route/WP-List, and/or name changed/added",
        3: "Route: Change supplementary parameters (or new added)",
        6: "Other not specified changed",
    },
    'WP_CRITICAL_PARAMETERS': {
        0: "Navigation Method",
        1: "XTE Limit",
    },
    'DISABLED_SATELLITES': {
        0: "Disable SV #1",
        1: "Disable SV #2",
        2: "Disable SV #3",
        3: "Disable SV #4",
        4: "Disable SV #5",
        5: "Disable SV #6",
        6: "Disable SV #7",
        7: "Disable SV #8",
        8: "Disable SV #9",
        9: "Disable SV #10",
        10: "Disable SV #11",
        11: "Disable SV #12",
        12: "Disable SV #13",
        13: "Disable SV #14",
        14: "Disable SV #15",
        15: "Disable SV #16",
        16: "Disable SV #17",
        17: "Disable SV #18",
        18: "Disable SV #19",
        19: "Disable SV #20",
        20: "Disable SV #21",
        21: "Disable SV #22",
        22: "Disable SV #23",
        23: "Disable SV #24",
        24: "Disable SV #25",
        25: "Disable SV #26",
        26: "Disable SV #27",
        27: "Disable SV #28",
        28: "Disable SV #29",
        29: "Disable SV #30",
        30: "Disable SV #31",
        31: "Disable SV #32",
        32: "Disable SV #33",
        33: "Disable SV #34",
        34: "Disable SV #35",
        35: "Disable SV #36",
        36: "Disable SV #37",
        37: "Disable SV #38",
        38: "Disable SV #39",
        39: "Disable SV #40",
    },
}

master_indirect_lookup_dict = {

    'DEVICE_FUNCTION': {
        "10_130": "Diagnostic",
        "10_140": "Bus Traffic Logger",
        "20_110": "Alarm Enunciator",
        "20_130": "Emergency Position Indicating Radio Beacon (EPIRB)",
        "20_135": "Man Overboard",
        "20_140": "Voyage Data Recorder",
        "20_150": "Camera",
        "25_130": "PC Gateway",
        "25_131": "NMEA 2000 to Analog Gateway",
        "25_132": "Analog to NMEA 2000 Gateway",
        "25_133": "NMEA 2000 to Serial Gateway",
        "25_135": "NMEA 0183 Gateway",
        "25_136": "NMEA Network Gateway",
        "25_137": "NMEA 2000 Wireless Gateway",
        "25_140": "Router",
        "25_150": "Bridge",
        "25_160": "Repeater",
        "30_130": "Binary Event Monitor",
        "30_140": "Load Controller",
        "30_141": "AC/DC Input",
        "30_150": "Function Controller",
        "35_140": "Engine",
        "35_141": "DC Generator/Alternator",
        "35_142": "Solar Panel (Solar Array)",
        "35_143": "Wind Generator (DC)",
        "35_144": "Fuel Cell",
        "35_145": "Network Power Supply",
        "35_151": "AC Generator",
        "35_152": "AC Bus",
        "35_153": "AC Mains (Utility/Shore)",
        "35_154": "AC Output",
        "35_160": "Power Converter - Battery Charger",
        "35_161": "Power Converter - Battery Charger+Inverter",
        "35_162": "Power Converter - Inverter",
        "35_163": "Power Converter - DC",
        "35_170": "Battery",
        "35_180": "Engine Gateway",
        "40_130": "Follow-up Controller",
        "40_140": "Mode Controller",
        "40_150": "Autopilot",
        "40_155": "Rudder",
        "40_160": "Heading Sensors",
        "40_170": "Trim (Tabs)/Interceptors",
        "40_180": "Attitude (Pitch, Roll, Yaw) Control",
        "50_130": "Engineroom Monitoring",
        "50_140": "Engine",
        "50_141": "DC Generator/Alternator",
        "50_150": "Engine Controller",
        "50_151": "AC Generator",
        "50_155": "Motor",
        "50_160": "Engine Gateway",
        "50_165": "Transmission",
        "50_170": "Throttle/Shift Control",
        "50_180": "Actuator",
        "50_190": "Gauge Interface",
        "50_200": "Gauge Large",
        "50_210": "Gauge Small",
        "60_130": "Bottom Depth",
        "60_135": "Bottom Depth/Speed",
        "60_136": "Bottom Depth/Speed/Temperature",
        "60_140": "Ownship Attitude",
        "60_145": "Ownship Position (GNSS)",
        "60_150": "Ownship Position (Loran C)",
        "60_155": "Speed",
        "60_160": "Turn Rate Indicator",
        "60_170": "Integrated Navigation",
        "60_175": "Integrated Navigation System",
        "60_190": "Navigation Management",
        "60_195": "Automatic Identification System (AIS)",
        "60_200": "Radar",
        "60_201": "Infrared Imaging",
        "60_205": "ECDIS",
        "60_210": "ECS",
        "60_220": "Direction Finder",
        "60_230": "Voyage Status",
        "70_130": "EPIRB",
        "70_140": "AIS",
        "70_150": "DSC",
        "70_160": "Data Receiver/Transceiver",
        "70_170": "Satellite",
        "70_180": "Radio-telephone (MF/HF)",
        "70_190": "Radiotelephone",
        "75_130": "Temperature",
        "75_140": "Pressure",
        "75_150": "Fluid Level",
        "75_160": "Flow",
        "75_170": "Humidity",
        "80_130": "Time/Date Systems",
        "80_140": "VDR",
        "80_150": "Integrated Instrumentation",
        "80_160": "General Purpose Displays",
        "80_170": "General Sensor Box",
        "80_180": "Weather Instruments",
        "80_190": "Transducer/General",
        "80_200": "NMEA 0183 Converter",
        "85_130": "Atmospheric",
        "85_160": "Aquatic",
        "90_130": "HVAC",
        "100_130": "Scale (Catch)",
        "110_130": "Button Interface",
        "110_135": "Switch Interface",
        "110_140": "Analog Interface",
        "120_130": "Display",
        "120_140": "Alarm Enunciator",
        "125_130": "Multimedia Player",
        "125_140": "Multimedia Controller",
    },
}


def _optional_message_field(nmea2000Message: NMEA2000Message, field_id: str) -> NMEA2000Field | None:
    return next((field for field in nmea2000Message.fields if field.id == field_id), None)


def _coerce_repeating_field(field_id: str, value) -> NMEA2000Field:
    if isinstance(value, NMEA2000Field):
        return value
    if isinstance(value, dict):
        return NMEA2000Field(field_id, value=value.get('value'), raw_value=value.get('raw_value'))
    return NMEA2000Field(field_id, value=value, raw_value=None)


def _get_repeating_entries(
    nmea2000Message: NMEA2000Message,
    list_field_id: str,
    field_ids: tuple[str, ...]
) -> list[dict[str, NMEA2000Field]]:
    list_field = _optional_message_field(nmea2000Message, list_field_id)
    if list_field is not None:
        if list_field.value is None:
            return []
        if not isinstance(list_field.value, list):
            raise ValueError(f"{list_field_id} must be a list of dict items")

        entries: list[dict[str, NMEA2000Field]] = []
        for item in list_field.value:
            if not isinstance(item, dict):
                raise ValueError(f"{list_field_id} must contain dict items")
            entry: dict[str, NMEA2000Field] = {}
            for field_id in field_ids:
                if field_id in item:
                    entry[field_id] = _coerce_repeating_field(field_id, item[field_id])
            entries.append(entry)
        return entries

    entry: dict[str, NMEA2000Field] = {}
    for field_id in field_ids:
        field = _optional_message_field(nmea2000Message, field_id)
        if field is not None:
            entry[field_id] = field
    return [entry] if entry else []


def _repeating_entry_value(value, raw_value):
    if value is None and raw_value is not None:
        return {"value": value, "raw_value": raw_value}
    return value


def _lookup_field_type_encode(value, lookup_dict: dict[int, LookupFieldTypeEnumeration], lookup_name: str) -> int:
    for raw_value, metadata in lookup_dict.items():
        if metadata.name == value:
            return raw_value
    raise ValueError(f"Cant encode this message, {value} is missing from {lookup_name}")



lookup_dict_encode_LIGHTING_COMMAND = {
    "Idle" : 0,
    "Detect Devices" : 1,
    "Reboot" : 2,
    "Factory Reset" : 3,
    "Powering Up" : 4,
}
def lookup_encode_LIGHTING_COMMAND(value):
    result = lookup_dict_encode_LIGHTING_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from LIGHTING_COMMAND")
    return result

lookup_dict_encode_INDUSTRY_CODE = {
    "Global" : 0,
    "Highway" : 1,
    "Agriculture" : 2,
    "Construction" : 3,
    "Marine Industry" : 4,
    "Industrial" : 5,
}
def lookup_encode_INDUSTRY_CODE(value):
    result = lookup_dict_encode_INDUSTRY_CODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from INDUSTRY_CODE")
    return result

lookup_dict_encode_MANUFACTURER_CODE = {
    "ARKS Enterprises, Inc." : 69,
    "FW Murphy/Enovation Controls" : 78,
    "Twin Disc" : 80,
    "Kohler Power Systems" : 85,
    "Hemisphere GPS Inc" : 88,
    "BEP Marine" : 116,
    "Airmar" : 135,
    "Maretron" : 137,
    "Lowrance" : 140,
    "Mercury Marine" : 144,
    "Nautibus Electronic GmbH" : 147,
    "Blue Water Data" : 148,
    "Westerbeke" : 154,
    "ISSPRO Inc" : 157,
    "Offshore Systems (UK) Ltd." : 161,
    "Evinrude/BRP" : 163,
    "CPAC Systems AB" : 165,
    "Xantrex Technology Inc." : 168,
    "Marlin Technologies, Inc." : 169,
    "Yanmar Marine" : 172,
    "Volvo Penta" : 174,
    "Honda Marine" : 175,
    "Carling Technologies Inc. (Moritz Aerospace)" : 176,
    "Beede Instruments" : 185,
    "Floscan Instrument Co. Inc." : 192,
    "Nobletec" : 193,
    "Mystic Valley Communications" : 198,
    "Actia" : 199,
    "Honda Marine" : 200,
    "Disenos Y Technologia" : 201,
    "Digital Switching Systems" : 211,
    "Xintex/Atena" : 215,
    "EMMI NETWORK S.L." : 224,
    "Honda Marine" : 225,
    "ZF" : 228,
    "Garmin" : 229,
    "Yacht Monitoring Solutions" : 233,
    "Sailormade Marine Telemetry/Tetra Technology LTD" : 235,
    "Eride" : 243,
    "Honda Marine" : 250,
    "Honda Motor Company LTD" : 257,
    "Groco" : 272,
    "Actisense" : 273,
    "Amphenol LTW Technology" : 274,
    "Navico" : 275,
    "Hamilton Jet" : 283,
    "Sea Recovery" : 285,
    "Coelmo SRL Italy" : 286,
    "BEP Marine" : 295,
    "Empir Bus" : 304,
    "NovAtel" : 305,
    "Sleipner Motor AS" : 306,
    "MBW Technologies" : 307,
    "Fischer Panda" : 311,
    "ICOM" : 315,
    "Qwerty" : 328,
    "Dief" : 329,
    "Boening Automationstechnologie GmbH & Co. KG" : 341,
    "Korean Maritime University" : 345,
    "Thrane and Thrane" : 351,
    "Mastervolt" : 355,
    "Fischer Panda Generators" : 356,
    "Victron Energy" : 358,
    "Rolls Royce Marine" : 370,
    "Electronic Design" : 373,
    "Northern Lights" : 374,
    "Glendinning" : 378,
    "B & G" : 381,
    "Rose Point Navigation Systems" : 384,
    "Johnson Outdoors Marine Electronics Inc Geonav" : 385,
    "Capi 2" : 394,
    "Beyond Measure" : 396,
    "Livorsi Marine" : 400,
    "ComNav" : 404,
    "Chetco" : 409,
    "Fusion Electronics" : 419,
    "Standard Horizon" : 421,
    "True Heading AB" : 422,
    "Egersund Marine Electronics AS" : 426,
    "em-trak Marine Electronics" : 427,
    "Tohatsu Co, JP" : 431,
    "Digital Yacht" : 437,
    "Comar Systems Limited" : 438,
    "Cummins" : 440,
    "VDO (aka Continental-Corporation)" : 443,
    "Parker Hannifin aka Village Marine Tech" : 451,
    "Alltek Marine Electronics Corp" : 459,
    "SAN GIORGIO S.E.I.N" : 460,
    "Veethree Electronics & Marine" : 466,
    "Humminbird Marine Electronics" : 467,
    "SI-TEX Marine Electronics" : 470,
    "Sea Cross Marine AB" : 471,
    "GME aka Standard Communications Pty LTD" : 475,
    "Humminbird Marine Electronics" : 476,
    "Ocean Sat BV" : 478,
    "Chetco Digitial Instruments" : 481,
    "Watcheye" : 493,
    "Lcj Capteurs" : 499,
    "Attwood Marine" : 502,
    "Naviop S.R.L." : 503,
    "Vesper Marine Ltd" : 504,
    "Marinesoft Co. LTD" : 510,
    "Simarine" : 513,
    "NoLand Engineering" : 517,
    "Transas USA" : 518,
    "National Instruments Korea" : 529,
    "National Marine Electronics Association" : 530,
    "Onwa Marine" : 532,
    "Webasto" : 540,
    "Marinecraft (South Korea)" : 571,
    "McMurdo Group aka Orolia LTD" : 573,
    "Advansea" : 578,
    "KVH" : 579,
    "San Jose Technology" : 580,
    "Yacht Control" : 583,
    "Suzuki Motor Corporation" : 586,
    "US Coast Guard" : 591,
    "Ship Module aka Customware" : 595,
    "Aquatic AV" : 600,
    "Aventics GmbH" : 605,
    "Intellian" : 606,
    "SamwonIT" : 612,
    "Arlt Tecnologies" : 614,
    "Bavaria Yacts" : 637,
    "Diverse Yacht Services" : 641,
    "Wema U.S.A dba KUS" : 644,
    "Garmin" : 645,
    "Shenzhen Jiuzhou Himunication" : 658,
    "Rockford Corp" : 688,
    "Harman International" : 699,
    "JL Audio" : 704,
    "Lars Thrane" : 708,
    "Autonnic" : 715,
    "Yacht Devices" : 717,
    "REAP Systems" : 734,
    "Au Electronics Group" : 735,
    "LxNav" : 739,
    "Littelfuse, Inc (formerly Carling Technologies)" : 741,
    "DaeMyung" : 743,
    "Woosung" : 744,
    "ISOTTA IFRA srl" : 748,
    "Clarion US" : 773,
    "HMI Systems" : 776,
    "Ocean Signal" : 777,
    "Seekeeper" : 778,
    "Poly Planar" : 781,
    "Fischer Panda DE" : 785,
    "Broyda Industries" : 795,
    "Canadian Automotive" : 796,
    "Tides Marine" : 797,
    "Lumishore" : 798,
    "Still Water Designs and Audio" : 799,
    "BJ Technologies (Beneteau)" : 802,
    "Gill Sensors" : 803,
    "Blue Water Desalination" : 811,
    "FLIR" : 815,
    "Undheim Systems" : 824,
    "Lewmar Inc" : 826,
    "TeamSurv" : 838,
    "Fell Marine" : 844,
    "Oceanvolt" : 847,
    "Prospec" : 862,
    "Data Panel Corp" : 868,
    "L3 Technologies" : 890,
    "Rhodan Marine Systems" : 894,
    "Nexfour Solutions" : 896,
    "ASA Electronics" : 905,
    "Marines Co (South Korea)" : 909,
    "Nautic-on" : 911,
    "Sentinel" : 917,
    "JL Marine ystems" : 929,
    "Ecotronix" : 930,
    "Zontisa Marine" : 944,
    "EXOR International" : 951,
    "Timbolier Industries" : 962,
    "TJC Micro" : 963,
    "Cox Powertrain" : 968,
    "Blue Seas" : 969,
    "Kobelt Manufacturing Co. Ltd" : 981,
    "Blue Ocean IOT" : 992,
    "Xenta Systems" : 997,
    "Ultraflex SpA" : 1004,
    "Lintest SmartBoat" : 1008,
    "Soundmax" : 1011,
    "Team Italia Marine (Onyx Marine Automation s.r.l)" : 1020,
    "Entratech" : 1021,
    "ITC Inc." : 1022,
    "The Marine Guardian LLC" : 1029,
    "Sonic Corporation" : 1047,
    "ProNav" : 1051,
    "Vetus Maxwell INC." : 1053,
    "Lithium Pros" : 1056,
    "Boatrax" : 1059,
    "Marol Co ltd" : 1062,
    "CALYPSO Instruments" : 1065,
    "Spot Zero Water" : 1066,
    "Lithionics Battery LLC" : 1069,
    "Quick-teck Electronics Ltd" : 1070,
    "Uniden America" : 1075,
    "Nauticoncept" : 1083,
    "Shadow-Caster LED lighting LLC" : 1084,
    "Wet Sounds, LLC" : 1085,
    "E-T-A Circuit Breakers" : 1088,
    "Scheiber" : 1092,
    "Smart Yachts International Limited" : 1100,
    "Dockmate" : 1109,
    "Bobs Machine" : 1114,
    "L3Harris ASV" : 1118,
    "Balmar LLC" : 1119,
    "Elettromedia spa" : 1120,
    "Electromaax" : 1127,
    "Across Oceans Systems Ltd." : 1140,
    "Kiwi Yachting" : 1145,
    "BSB Artificial Intelligence GmbH" : 1150,
    "Orca Technologoes AS" : 1151,
    "TBS Electronics BV" : 1154,
    "Technoton Electroics" : 1158,
    "MG Energy Systems B.V." : 1160,
    "Sea Macine Robotics Inc." : 1169,
    "Vista Manufacturing" : 1171,
    "Zipwake" : 1183,
    "Sailmon BV" : 1186,
    "Airmoniq Pro Kft" : 1192,
    "Sierra Marine" : 1194,
    "Xinuo Information Technology (Xiamen)" : 1200,
    "Septentrio" : 1218,
    "NKE Marine Elecronics" : 1233,
    "SuperTrack Aps" : 1238,
    "Honda Electronics Co., LTD" : 1239,
    "Raritan Engineering Company, Inc" : 1245,
    "Integrated Power Solutions AG" : 1249,
    "Interactive Technologies, Inc." : 1260,
    "LTG-Tech" : 1283,
    "Energy Solutions (UK) LTD." : 1299,
    "WATT Fuel Cell Corp" : 1300,
    "Pro Mainer" : 1302,
    "Dragonfly Energy" : 1305,
    "Koden Electronics Co., Ltd" : 1306,
    "Humphree AB" : 1311,
    "Hinkley Yachts" : 1316,
    "Global Marine Management GmbH (GMM)" : 1317,
    "Triskel Marine Ltd" : 1320,
    "Warwick Control Technologies" : 1330,
    "Dolphin Charger" : 1331,
    "Barnacle Systems Inc" : 1337,
    "Radian IoT, Inc." : 1348,
    "Ocean LED Marine Ltd" : 1353,
    "BluNav" : 1359,
    "OVA (Nantong Saiyang Electronics Co., Ltd)" : 1361,
    "RAD Propulsion" : 1368,
    "Electric Yacht" : 1369,
    "Elco Motor Yachts" : 1372,
    "Tecnoseal Foundry S.r.l" : 1384,
    "Pro Charging Systems, LLC" : 1385,
    "EVEX Co., LTD" : 1389,
    "Gobius Sensor Technology AB" : 1398,
    "Arco Marine" : 1403,
    "Lenco Marine Inc." : 1408,
    "Naocontrol S.L." : 1413,
    "Revatek" : 1417,
    "Aeolionics" : 1438,
    "PredictWind Ltd" : 1439,
    "Egis Mobile Electric" : 1440,
    "Starboard Yacht Group" : 1445,
    "Roswell Marine" : 1446,
    "ePropulsion (Guangdong ePropulsion Technology Ltd.)" : 1451,
    "Micro-Air LLC" : 1452,
    "Vital Battery" : 1453,
    "Ride Controller LLC" : 1458,
    "Tocaro Blue" : 1460,
    "Vanquish Yachts" : 1461,
    "FT Technologies" : 1471,
    "Alps Alpine Co., Ltd." : 1478,
    "E-Force Marine" : 1481,
    "CMC Marine" : 1482,
    "Nanjing Sandemarine Information Technology Co., Ltd." : 1483,
    "Teleflex Marine (SeaStar Solutions)" : 1850,
    "Raymarine" : 1851,
    "Navionics" : 1852,
    "Japan Radio Co" : 1853,
    "Northstar Technologies" : 1854,
    "Furuno" : 1855,
    "Trimble" : 1856,
    "Simrad" : 1857,
    "Litton" : 1858,
    "Kvasar AB" : 1859,
    "MMP" : 1860,
    "Vector Cantech" : 1861,
    "Yamaha Marine" : 1862,
    "Faria Instruments" : 1863,
}
def lookup_encode_MANUFACTURER_CODE(value):
    result = lookup_dict_encode_MANUFACTURER_CODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MANUFACTURER_CODE")
    return result

lookup_dict_encode_AIS_MESSAGE_ID = {
    "Scheduled Class A position report" : 1,
    "Assigned scheduled Class A position report" : 2,
    "Interrogated Class A position report" : 3,
    "Base station report" : 4,
    "Static and voyage related data" : 5,
    "Binary addressed message" : 6,
    "Binary acknowledgement" : 7,
    "Binary broadcast message" : 8,
    "Standard SAR aircraft position report" : 9,
    "UTC/date inquiry" : 10,
    "UTC/date response" : 11,
    "Safety related addressed message" : 12,
    "Safety related acknowledgement" : 13,
    "Satety related broadcast message" : 14,
    "Interrogation" : 15,
    "Assignment mode command" : 16,
    "DGNSS broadcast binary message" : 17,
    "Standard Class B position report" : 18,
    "Extended Class B position report" : 19,
    "Data link management message" : 20,
    "ATON report" : 21,
    "Channel management" : 22,
    "Group assignment command" : 23,
    "Static data report" : 24,
    "Single slot binary message" : 25,
    "Multiple slot binary message" : 26,
    "Position report for long range applications" : 27,
}
def lookup_encode_AIS_MESSAGE_ID(value):
    result = lookup_dict_encode_AIS_MESSAGE_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_MESSAGE_ID")
    return result

lookup_dict_encode_SHIP_TYPE = {
    "Unavailable" : 0,
    "Wing In Ground" : 20,
    "Wing In Ground (hazard cat X)" : 21,
    "Wing In Ground (hazard cat Y)" : 22,
    "Wing In Ground (hazard cat Z)" : 23,
    "Wing In Ground (hazard cat OS)" : 24,
    "Wing In Ground (no additional information)" : 29,
    "Fishing" : 30,
    "Towing" : 31,
    "Towing exceeds 200m or wider than 25m" : 32,
    "Engaged in dredging or underwater operations" : 33,
    "Engaged in diving operations" : 34,
    "Engaged in military operations" : 35,
    "Sailing" : 36,
    "Pleasure" : 37,
    "High speed craft" : 40,
    "High speed craft (hazard cat X)" : 41,
    "High speed craft (hazard cat Y)" : 42,
    "High speed craft (hazard cat Z)" : 43,
    "High speed craft (hazard cat OS)" : 44,
    "High speed craft (no additional information)" : 49,
    "Pilot vessel" : 50,
    "SAR" : 51,
    "Tug" : 52,
    "Port tender" : 53,
    "Anti-pollution" : 54,
    "Law enforcement" : 55,
    "Spare" : 56,
    "Spare #2" : 57,
    "Medical" : 58,
    "Ships and aircraft of States not parties to an armed conflict" : 59,
    "Passenger ship" : 60,
    "Passenger ship (hazard cat X)" : 61,
    "Passenger ship (hazard cat Y)" : 62,
    "Passenger ship (hazard cat Z)" : 63,
    "Passenger ship (hazard cat OS)" : 64,
    "Passenger ship (no additional information)" : 69,
    "Cargo ship" : 70,
    "Cargo ship (hazard cat X)" : 71,
    "Cargo ship (hazard cat Y)" : 72,
    "Cargo ship (hazard cat Z)" : 73,
    "Cargo ship (hazard cat OS)" : 74,
    "Cargo ship (no additional information)" : 79,
    "Tanker" : 80,
    "Tanker (hazard cat X)" : 81,
    "Tanker (hazard cat Y)" : 82,
    "Tanker (hazard cat Z)" : 83,
    "Tanker (hazard cat OS)" : 84,
    "Tanker (no additional information)" : 89,
    "Other" : 90,
    "Other (hazard cat X)" : 91,
    "Other (hazard cat Y)" : 92,
    "Other (hazard cat Z)" : 93,
    "Other (hazard cat OS)" : 94,
    "Other (no additional information)" : 99,
}
def lookup_encode_SHIP_TYPE(value):
    result = lookup_dict_encode_SHIP_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SHIP_TYPE")
    return result

lookup_dict_encode_DEVICE_CLASS = {
    "Reserved for 2000 Use" : 0,
    "System tools" : 10,
    "Safety systems" : 20,
    "Internetwork device" : 25,
    "Electrical Distribution" : 30,
    "Electrical Generation" : 35,
    "Steering and Control surfaces" : 40,
    "Propulsion" : 50,
    "Navigation" : 60,
    "Communication" : 70,
    "Sensor Communication Interface" : 75,
    "Instrumentation/general systems" : 80,
    "External Environment" : 85,
    "Internal Environment" : 90,
    "Deck + cargo + fishing equipment systems" : 100,
    "Human Interface" : 110,
    "Display" : 120,
    "Entertainment" : 125,
}
def lookup_encode_DEVICE_CLASS(value):
    result = lookup_dict_encode_DEVICE_CLASS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DEVICE_CLASS")
    return result

lookup_dict_encode_REPEAT_INDICATOR = {
    "Initial" : 0,
    "First retransmission" : 1,
    "Second retransmission" : 2,
    "Final retransmission" : 3,
}
def lookup_encode_REPEAT_INDICATOR(value):
    result = lookup_dict_encode_REPEAT_INDICATOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from REPEAT_INDICATOR")
    return result

lookup_dict_encode_TX_RX_MODE = {
    "Tx A/Tx B, Rx A/Rx B" : 0,
    "Tx A, Rx A/Rx B" : 1,
    "Tx B, Rx A/Rx B" : 2,
}
def lookup_encode_TX_RX_MODE(value):
    result = lookup_dict_encode_TX_RX_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TX_RX_MODE")
    return result

lookup_dict_encode_STATION_TYPE = {
    "All types of mobile station" : 0,
    "All types of Class B mobile station" : 2,
    "SAR airborne mobile station" : 3,
    "AtoN station" : 4,
    "Class B CS shipborne mobile station" : 5,
    "Inland waterways" : 6,
    "Regional use 7" : 7,
    "Regional use 8" : 8,
    "Regional use 9" : 9,
}
def lookup_encode_STATION_TYPE(value):
    result = lookup_dict_encode_STATION_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from STATION_TYPE")
    return result

lookup_dict_encode_REPORTING_INTERVAL = {
    "As given by the autonomous mode" : 0,
    "10 min" : 1,
    "6 min" : 2,
    "3 min" : 3,
    "1 min" : 4,
    "30 sec" : 5,
    "15 sec" : 6,
    "10 sec" : 7,
    "5 sec" : 8,
    "2 sec (not applicable to Class B CS)" : 9,
    "Next shorter reporting interval" : 10,
    "Next longer reporting interval" : 11,
}
def lookup_encode_REPORTING_INTERVAL(value):
    result = lookup_dict_encode_REPORTING_INTERVAL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from REPORTING_INTERVAL")
    return result

lookup_dict_encode_AIS_TRANSCEIVER = {
    "Channel A VDL reception" : 0,
    "Channel B VDL reception" : 1,
    "Channel A VDL transmission" : 2,
    "Channel B VDL transmission" : 3,
    "Own information not broadcast" : 4,
    "Reserved" : 5,
}
def lookup_encode_AIS_TRANSCEIVER(value):
    result = lookup_dict_encode_AIS_TRANSCEIVER.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_TRANSCEIVER")
    return result

lookup_dict_encode_AIS_ASSIGNED_MODE = {
    "Autonomous and continuous" : 0,
    "Assigned mode" : 1,
}
def lookup_encode_AIS_ASSIGNED_MODE(value):
    result = lookup_dict_encode_AIS_ASSIGNED_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_ASSIGNED_MODE")
    return result

lookup_dict_encode_ATON_TYPE = {
    "Default: Type of AtoN not specified" : 0,
    "Reference point" : 1,
    "RACON" : 2,
    "Fixed structure off-shore" : 3,
    "Reserved for future use" : 4,
    "Fixed light: without sectors" : 5,
    "Fixed light: with sectors" : 6,
    "Fixed leading light front" : 7,
    "Fixed leading light rear" : 8,
    "Fixed beacon: cardinal N" : 9,
    "Fixed beacon: cardinal E" : 10,
    "Fixed beacon: cardinal S" : 11,
    "Fixed beacon: cardinal W" : 12,
    "Fixed beacon: port hand" : 13,
    "Fixed beacon: starboard hand" : 14,
    "Fixed beacon: preferred channel port hand" : 15,
    "Fixed beacon: preferred channel starboard hand" : 16,
    "Fixed beacon: isolated danger" : 17,
    "Fixed beacon: safe water" : 18,
    "Fixed beacon: special mark" : 19,
    "Floating AtoN: cardinal N" : 20,
    "Floating AtoN: cardinal E" : 21,
    "Floating AtoN: cardinal S" : 22,
    "Floating AtoN: cardinal W" : 23,
    "Floating AtoN: port hand mark" : 24,
    "Floating AtoN: starboard hand mark" : 25,
    "Floating AtoN: preferred channel port hand" : 26,
    "Floating AtoN: preferred channel starboard hand" : 27,
    "Floating AtoN: isolated danger" : 28,
    "Floating AtoN: safe water" : 29,
    "Floating AtoN: special mark" : 30,
    "Floating AtoN: light vessel/LANBY/rigs" : 31,
}
def lookup_encode_ATON_TYPE(value):
    result = lookup_dict_encode_ATON_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ATON_TYPE")
    return result

lookup_dict_encode_AIS_SPECIAL_MANEUVER = {
    "Not available" : 0,
    "Not engaged in special maneuver" : 1,
    "Engaged in special maneuver" : 2,
    "Reserved" : 3,
}
def lookup_encode_AIS_SPECIAL_MANEUVER(value):
    result = lookup_dict_encode_AIS_SPECIAL_MANEUVER.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_SPECIAL_MANEUVER")
    return result

lookup_dict_encode_POSITION_FIX_DEVICE = {
    "Default: undefined" : 0,
    "GPS" : 1,
    "GLONASS" : 2,
    "Combined GPS/GLONASS" : 3,
    "Loran-C" : 4,
    "Chayka" : 5,
    "Integrated navigation system" : 6,
    "Surveyed" : 7,
    "Galileo" : 8,
}
def lookup_encode_POSITION_FIX_DEVICE(value):
    result = lookup_dict_encode_POSITION_FIX_DEVICE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from POSITION_FIX_DEVICE")
    return result

lookup_dict_encode_GNS = {
    "GPS" : 0,
    "GLONASS" : 1,
    "GPS+GLONASS" : 2,
    "GPS+SBAS/WAAS" : 3,
    "GPS+SBAS/WAAS+GLONASS" : 4,
    "Chayka" : 5,
    "integrated" : 6,
    "surveyed" : 7,
    "Galileo" : 8,
}
def lookup_encode_GNS(value):
    result = lookup_dict_encode_GNS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GNS")
    return result

lookup_dict_encode_ENGINE_INSTANCE = {
    "Single Engine or Dual Engine Port" : 0,
    "Dual Engine Starboard" : 1,
}
def lookup_encode_ENGINE_INSTANCE(value):
    result = lookup_dict_encode_ENGINE_INSTANCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENGINE_INSTANCE")
    return result

lookup_dict_encode_GEAR_STATUS = {
    "Forward" : 0,
    "Neutral" : 1,
    "Reverse" : 2,
}
def lookup_encode_GEAR_STATUS(value):
    result = lookup_dict_encode_GEAR_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GEAR_STATUS")
    return result

lookup_dict_encode_DIRECTION = {
    "Forward" : 0,
    "Reverse" : 1,
}
def lookup_encode_DIRECTION(value):
    result = lookup_dict_encode_DIRECTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DIRECTION")
    return result

lookup_dict_encode_POSITION_ACCURACY = {
    "Low" : 0,
    "High" : 1,
}
def lookup_encode_POSITION_ACCURACY(value):
    result = lookup_dict_encode_POSITION_ACCURACY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from POSITION_ACCURACY")
    return result

lookup_dict_encode_RAIM_FLAG = {
    "not in use" : 0,
    "in use" : 1,
}
def lookup_encode_RAIM_FLAG(value):
    result = lookup_dict_encode_RAIM_FLAG.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from RAIM_FLAG")
    return result

lookup_dict_encode_TIME_STAMP = {
    "Not available" : 60,
    "Manual input mode" : 61,
    "Dead reckoning mode" : 62,
    "Positioning system is inoperative" : 63,
}
def lookup_encode_TIME_STAMP(value):
    result = lookup_dict_encode_TIME_STAMP.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TIME_STAMP")
    return result

lookup_dict_encode_GNS_METHOD = {
    "no GNSS" : 0,
    "GNSS fix" : 1,
    "DGNSS fix" : 2,
    "Precise GNSS" : 3,
    "RTK Fixed Integer" : 4,
    "RTK float" : 5,
    "Estimated (DR) mode" : 6,
    "Manual Input" : 7,
    "Simulate mode" : 8,
}
def lookup_encode_GNS_METHOD(value):
    result = lookup_dict_encode_GNS_METHOD.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GNS_METHOD")
    return result

lookup_dict_encode_GNS_INTEGRITY = {
    "No integrity checking" : 0,
    "Safe" : 1,
    "Caution" : 2,
    "Unsafe" : 3,
}
def lookup_encode_GNS_INTEGRITY(value):
    result = lookup_dict_encode_GNS_INTEGRITY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GNS_INTEGRITY")
    return result

lookup_dict_encode_SYSTEM_TIME = {
    "GPS" : 0,
    "GLONASS" : 1,
    "Radio Station" : 2,
    "Local Cesium clock" : 3,
    "Local Rubidium clock" : 4,
    "Local Crystal clock" : 5,
}
def lookup_encode_SYSTEM_TIME(value):
    result = lookup_dict_encode_SYSTEM_TIME.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SYSTEM_TIME")
    return result

lookup_dict_encode_MAGNETIC_VARIATION = {
    "Manual" : 0,
    "Automatic Chart" : 1,
    "Automatic Table" : 2,
    "Automatic Calculation" : 3,
    "WMM 2000" : 4,
    "WMM 2005" : 5,
    "WMM 2010" : 6,
    "WMM 2015" : 7,
    "WMM 2020" : 8,
    "WMM 2025" : 9,
}
def lookup_encode_MAGNETIC_VARIATION(value):
    result = lookup_dict_encode_MAGNETIC_VARIATION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MAGNETIC_VARIATION")
    return result

lookup_dict_encode_RESIDUAL_MODE = {
    "Autonomous" : 0,
    "Differential enhanced" : 1,
    "Estimated" : 2,
    "Simulator" : 3,
    "Manual" : 4,
}
def lookup_encode_RESIDUAL_MODE(value):
    result = lookup_dict_encode_RESIDUAL_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from RESIDUAL_MODE")
    return result

lookup_dict_encode_WIND_REFERENCE = {
    "True (ground referenced to North)" : 0,
    "Magnetic (ground referenced to Magnetic North)" : 1,
    "Apparent" : 2,
    "True (boat referenced)" : 3,
    "True (water referenced)" : 4,
}
def lookup_encode_WIND_REFERENCE(value):
    result = lookup_dict_encode_WIND_REFERENCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WIND_REFERENCE")
    return result

lookup_dict_encode_WATER_REFERENCE = {
    "Paddle wheel" : 0,
    "Pitot tube" : 1,
    "Doppler" : 2,
    "Correlation (ultra sound)" : 3,
    "Electro Magnetic" : 4,
}
def lookup_encode_WATER_REFERENCE(value):
    result = lookup_dict_encode_WATER_REFERENCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WATER_REFERENCE")
    return result

lookup_dict_encode_YES_NO = {
    "No" : 0,
    "Yes" : 1,
}
def lookup_encode_YES_NO(value):
    result = lookup_dict_encode_YES_NO.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from YES_NO")
    return result

lookup_dict_encode_OK_WARNING = {
    "OK" : 0,
    "Warning" : 1,
}
def lookup_encode_OK_WARNING(value):
    result = lookup_dict_encode_OK_WARNING.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from OK_WARNING")
    return result

lookup_dict_encode_OFF_ON = {
    "Off" : 0,
    "On" : 1,
}
def lookup_encode_OFF_ON(value):
    result = lookup_dict_encode_OFF_ON.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from OFF_ON")
    return result

lookup_dict_encode_OFF_ON_CONTROL = {
    "Off" : 0,
    "On" : 1,
    "Reserved" : 2,
    "Take no action (no change)" : 3,
}
def lookup_encode_OFF_ON_CONTROL(value):
    result = lookup_dict_encode_OFF_ON_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from OFF_ON_CONTROL")
    return result

lookup_dict_encode_DIRECTION_REFERENCE = {
    "True" : 0,
    "Magnetic" : 1,
    "Error" : 2,
}
def lookup_encode_DIRECTION_REFERENCE(value):
    result = lookup_dict_encode_DIRECTION_REFERENCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DIRECTION_REFERENCE")
    return result

lookup_dict_encode_DIRECTION_RUDDER = {
    "No Order" : 0,
    "Move to starboard" : 1,
    "Move to port" : 2,
}
def lookup_encode_DIRECTION_RUDDER(value):
    result = lookup_dict_encode_DIRECTION_RUDDER.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DIRECTION_RUDDER")
    return result

lookup_dict_encode_NAV_STATUS = {
    "Under way using engine" : 0,
    "At anchor" : 1,
    "Not under command" : 2,
    "Restricted maneuverability" : 3,
    "Constrained by her draught" : 4,
    "Moored" : 5,
    "Aground" : 6,
    "Engaged in Fishing" : 7,
    "Under way sailing" : 8,
    "Hazardous material - High Speed" : 9,
    "Hazardous material - Wing in Ground" : 10,
    "Power-driven vessel towing astern" : 11,
    "Power-driven vessel pushing ahead or towing alongside" : 12,
    "AIS-SART" : 14,
}
def lookup_encode_NAV_STATUS(value):
    result = lookup_dict_encode_NAV_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from NAV_STATUS")
    return result

lookup_dict_encode_POWER_FACTOR = {
    "Leading" : 0,
    "Lagging" : 1,
    "Error" : 2,
}
def lookup_encode_POWER_FACTOR(value):
    result = lookup_dict_encode_POWER_FACTOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from POWER_FACTOR")
    return result

lookup_dict_encode_TEMPERATURE_SOURCE = {
    "Sea Temperature" : 0,
    "Outside Temperature" : 1,
    "Inside Temperature" : 2,
    "Engine Room Temperature" : 3,
    "Main Cabin Temperature" : 4,
    "Live Well Temperature" : 5,
    "Bait Well Temperature" : 6,
    "Refrigeration Temperature" : 7,
    "Heating System Temperature" : 8,
    "Dew Point Temperature" : 9,
    "Apparent Wind Chill Temperature" : 10,
    "Theoretical Wind Chill Temperature" : 11,
    "Heat Index Temperature" : 12,
    "Freezer Temperature" : 13,
    "Exhaust Gas Temperature" : 14,
    "Shaft Seal Temperature" : 15,
}
def lookup_encode_TEMPERATURE_SOURCE(value):
    result = lookup_dict_encode_TEMPERATURE_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TEMPERATURE_SOURCE")
    return result

lookup_dict_encode_HUMIDITY_SOURCE = {
    "Inside" : 0,
    "Outside" : 1,
}
def lookup_encode_HUMIDITY_SOURCE(value):
    result = lookup_dict_encode_HUMIDITY_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from HUMIDITY_SOURCE")
    return result

lookup_dict_encode_PRESSURE_SOURCE = {
    "Atmospheric" : 0,
    "Water" : 1,
    "Steam" : 2,
    "Compressed Air" : 3,
    "Hydraulic" : 4,
    "Filter" : 5,
    "AltimeterSetting" : 6,
    "Oil" : 7,
    "Fuel" : 8,
}
def lookup_encode_PRESSURE_SOURCE(value):
    result = lookup_dict_encode_PRESSURE_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from PRESSURE_SOURCE")
    return result

lookup_dict_encode_DSC_FORMAT = {
    "Geographical area" : 102,
    "Distress" : 112,
    "Common interest" : 114,
    "All ships" : 116,
    "Individual stations" : 120,
    "Non-calling purpose" : 121,
    "Individual station automatic" : 123,
}
def lookup_encode_DSC_FORMAT(value):
    result = lookup_dict_encode_DSC_FORMAT.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DSC_FORMAT")
    return result

lookup_dict_encode_DSC_CATEGORY = {
    "Routine" : 100,
    "Safety" : 108,
    "Urgency" : 110,
    "Distress" : 112,
}
def lookup_encode_DSC_CATEGORY(value):
    result = lookup_dict_encode_DSC_CATEGORY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DSC_CATEGORY")
    return result

lookup_dict_encode_DSC_NATURE = {
    "Fire" : 100,
    "Flooding" : 101,
    "Collision" : 102,
    "Grounding" : 103,
    "Listing" : 104,
    "Sinking" : 105,
    "Disabled and adrift" : 106,
    "Undesignated" : 107,
    "Abandoning ship" : 108,
    "Piracy" : 109,
    "Man overboard" : 110,
    "EPIRB emission" : 112,
}
def lookup_encode_DSC_NATURE(value):
    result = lookup_dict_encode_DSC_NATURE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DSC_NATURE")
    return result

lookup_dict_encode_DSC_FIRST_TELECOMMAND = {
    "F3E/G3E All modes TP" : 100,
    "F3E/G3E duplex TP" : 101,
    "Polling" : 103,
    "Unable to comply" : 104,
    "End of call" : 105,
    "Data" : 106,
    "J3E TP" : 109,
    "Distress acknowledgement" : 110,
    "Distress relay" : 112,
    "F1B/J2B TTY-FEC" : 113,
    "F1B/J2B TTY-ARQ" : 115,
    "Test" : 118,
    "Ship position or location registration updating" : 121,
    "No information" : 126,
}
def lookup_encode_DSC_FIRST_TELECOMMAND(value):
    result = lookup_dict_encode_DSC_FIRST_TELECOMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DSC_FIRST_TELECOMMAND")
    return result

lookup_dict_encode_DSC_SECOND_TELECOMMAND = {
    "No reason given" : 100,
    "Congestion at MSC" : 101,
    "Busy" : 102,
    "Queue indication" : 103,
    "Station barred" : 104,
    "No operator available" : 105,
    "Operator temporarily unavailable" : 106,
    "Equipment disabled" : 107,
    "Unable to use proposed channel" : 108,
    "Unable to use proposed mode" : 109,
    "Ships and aircraft of States not parties to an armed conflict" : 110,
    "Medical transports" : 111,
    "Pay phone/public call office" : 112,
    "Fax/data" : 113,
    "No information" : 126,
}
def lookup_encode_DSC_SECOND_TELECOMMAND(value):
    result = lookup_dict_encode_DSC_SECOND_TELECOMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DSC_SECOND_TELECOMMAND")
    return result

lookup_dict_encode_DSC_EXPANSION_DATA = {
    "Enhanced position" : 100,
    "Source and datum of position" : 101,
    "SOG" : 102,
    "COG" : 103,
    "Additional station identification" : 104,
    "Enhanced geographic area" : 105,
    "Number of persons on board" : 106,
}
def lookup_encode_DSC_EXPANSION_DATA(value):
    result = lookup_dict_encode_DSC_EXPANSION_DATA.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DSC_EXPANSION_DATA")
    return result

lookup_dict_encode_SEATALK_MESSAGE_ID = {
    "Seatalk 1 Encoded" : 240,
    "Display" : 140,
    "Pilot Configuration" : 108,
}
def lookup_encode_SEATALK_MESSAGE_ID(value):
    result = lookup_dict_encode_SEATALK_MESSAGE_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_MESSAGE_ID")
    return result

lookup_dict_encode_SEATALK_COMMAND = {
    "Seatalk1" : 129,
    "Hull Type" : 22,
    "Auto Turn" : 38,
    "Settings" : 12,
}
def lookup_encode_SEATALK_COMMAND(value):
    result = lookup_dict_encode_SEATALK_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_COMMAND")
    return result

lookup_dict_encode_SEATALK1_COMMAND = {
    "Depth Below Transducer" : 0,
    "Equipment ID" : 1,
    "Engine RPM and PITCH" : 5,
    "Apparent Wind Angle" : 16,
    "Apparent Wind Speed" : 17,
    "Speed through water" : 32,
    "Trip Mileage" : 33,
    "Total Mileage" : 34,
    "Water temperature (ST50)" : 35,
    "Display units for Mileage & Speed" : 36,
    "Total & Trip Log" : 37,
    "Speed through water (with average)" : 38,
    "Water temperature" : 39,
    "Set lamp Intensity" : 48,
    "Cancel MOB (Man Over Board) condition" : 54,
    "Codelock data" : 56,
    "LAT position" : 80,
    "LON position" : 81,
    "Speed over Ground" : 82,
    "Course over Ground (COG)" : 83,
    "GMT-time" : 84,
    "TRACK keystroke on GPS unit" : 85,
    "Date" : 86,
    "Sat Info" : 87,
    "LAT/LON (raw unfiltered)" : 88,
    "Set Count Down Timer" : 89,
    "Issued by E-80 multifunction display at initialization" : 97,
    "Select Fathom display units for depth display" : 101,
    "Wind alarm" : 102,
    "Alarm acknowledgment keystroke" : 104,
    "Second equipment-ID datagram" : 108,
    "MOB (Man Over Board)" : 110,
    "Keystroke on Raymarine A25006 ST60 Maxiview Remote Control" : 112,
    "Set Lamp Intensity" : 128,
    "Sent by course computer during setup" : 129,
    "Target waypoint name" : 130,
    "Sent by course computer" : 131,
    "Compass heading Autopilot course and Rudder position" : 132,
    "Navigation to waypoint information" : 133,
    "Keystroke" : 134,
    "Set Response level" : 135,
    "Autopilot Parameter" : 136,
    "Compass heading sent by ST40 compass instrument" : 137,
    "Device Indentification" : 144,
    "Set Rudder gain" : 145,
    "Set Autopilot Parameter" : 146,
    "Enter AP-Setup" : 147,
    "Replaces command 84 while autopilot is in value setting mode" : 149,
    "Compass variation" : 153,
    "Version String" : 154,
    "Compass heading and Rudder position" : 156,
    "Waypoint definition" : 158,
    "Destination Waypoint Info" : 161,
    "Arrival Info" : 162,
    "Broadcast query/response to identify devices" : 164,
    "GPS and DGPS Info" : 165,
    "Unknown meaning" : 167,
    "Alarm ON/OFF for Guard" : 168,
    "Alarm ON/OFF for Guard" : 171,
}
def lookup_encode_SEATALK1_COMMAND(value):
    result = lookup_dict_encode_SEATALK1_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK1_COMMAND")
    return result

lookup_dict_encode_SEATALK1_ATT = {
    "Depth Below Transducer" : 0,
}
def lookup_encode_SEATALK1_ATT(value):
    result = lookup_dict_encode_SEATALK1_ATT.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK1_ATT")
    return result

lookup_dict_encode_SEATALK_ALARM_STATUS = {
    "Alarm condition not met" : 0,
    "Alarm condition met and not silenced" : 1,
    "Alarm condition met and silenced" : 2,
}
def lookup_encode_SEATALK_ALARM_STATUS(value):
    result = lookup_dict_encode_SEATALK_ALARM_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_ALARM_STATUS")
    return result

lookup_dict_encode_SEATALK_ALARM_ID = {
    "No Alarm" : 0,
    "Shallow Depth" : 1,
    "Deep Depth" : 2,
    "Shallow Anchor" : 3,
    "Deep Anchor" : 4,
    "Off Course" : 5,
    "AWA High" : 6,
    "AWA Low" : 7,
    "AWS High" : 8,
    "AWS Low" : 9,
    "TWA High" : 10,
    "TWA Low" : 11,
    "TWS High" : 12,
    "TWS Low" : 13,
    "WP Arrival" : 14,
    "Boat Speed High" : 15,
    "Boat Speed Low" : 16,
    "Sea Temperature High" : 17,
    "Sea Temperature Low" : 18,
    "Pilot Watch" : 19,
    "Pilot Off Course" : 20,
    "Pilot Wind Shift" : 21,
    "Pilot Low Battery" : 22,
    "Pilot Last Minute Of Watch" : 23,
    "Pilot No NMEA Data" : 24,
    "Pilot Large XTE" : 25,
    "Pilot NMEA DataError" : 26,
    "Pilot CU Disconnected" : 27,
    "Pilot Auto Release" : 28,
    "Pilot Way Point Advance" : 29,
    "Pilot Drive Stopped" : 30,
    "Pilot Type Unspecified" : 31,
    "Pilot Calibration Required" : 32,
    "Pilot Last Heading" : 33,
    "Pilot No Pilot" : 34,
    "Pilot Route Complete" : 35,
    "Pilot Variable Text" : 36,
    "GPS Failure" : 37,
    "MOB" : 38,
    "Seatalk1 Anchor" : 39,
    "Pilot Swapped Motor Power" : 40,
    "Pilot Standby Too Fast To Fish" : 41,
    "Pilot No GPS Fix" : 42,
    "Pilot No GPS COG" : 43,
    "Pilot Start Up" : 44,
    "Pilot Too Slow" : 45,
    "Pilot No Compass" : 46,
    "Pilot Rate Gyro Fault" : 47,
    "Pilot Current Limit" : 48,
    "Pilot Way Point Advance Port" : 49,
    "Pilot Way Point Advance Stbd" : 50,
    "Pilot No Wind Data" : 51,
    "Pilot No Speed Data" : 52,
    "Pilot Seatalk Fail1" : 53,
    "Pilot Seatalk Fail2" : 54,
    "Pilot Warning Too Fast To Fish" : 55,
    "Pilot Auto Dockside Fail" : 56,
    "Pilot Turn Too Fast" : 57,
    "Pilot No Nav Data" : 58,
    "Pilot Lost Waypoint Data" : 59,
    "Pilot EEPROM Corrupt" : 60,
    "Pilot Rudder Feedback Fail" : 61,
    "Pilot Autolearn Fail1" : 62,
    "Pilot Autolearn Fail2" : 63,
    "Pilot Autolearn Fail3" : 64,
    "Pilot Autolearn Fail4" : 65,
    "Pilot Autolearn Fail5" : 66,
    "Pilot Autolearn Fail6" : 67,
    "Pilot Warning Cal Required" : 68,
    "Pilot Warning OffCourse" : 69,
    "Pilot Warning XTE" : 70,
    "Pilot Warning Wind Shift" : 71,
    "Pilot Warning Drive Short" : 72,
    "Pilot Warning Clutch Short" : 73,
    "Pilot Warning Solenoid Short" : 74,
    "Pilot Joystick Fault" : 75,
    "Pilot No Joystick Data" : 76,
    "Pilot Invalid Command" : 80,
    "AIS TX Malfunction" : 81,
    "AIS Antenna VSWR fault" : 82,
    "AIS Rx channel 1 malfunction" : 83,
    "AIS Rx channel 2 malfunction" : 84,
    "AIS No sensor position in use" : 85,
    "AIS No valid SOG information" : 86,
    "AIS No valid COG information" : 87,
    "AIS 12V alarm" : 88,
    "AIS 6V alarm" : 89,
    "AIS Noise threshold exceeded channel A" : 90,
    "AIS Noise threshold exceeded channel B" : 91,
    "AIS Transmitter PA fault" : 92,
    "AIS 3V3 alarm" : 93,
    "AIS Rx channel 70 malfunction" : 94,
    "AIS Heading lost/invalid" : 95,
    "AIS internal GPS lost" : 96,
    "AIS No sensor position" : 97,
    "AIS Lock failure" : 98,
    "AIS Internal GGA timeout" : 99,
    "AIS Protocol stack restart" : 100,
    "Pilot No IPS communications" : 101,
    "Pilot Power-On or Sleep-Switch Reset While Engaged" : 102,
    "Pilot Unexpected Reset While Engaged" : 103,
    "AIS Dangerous Target" : 104,
    "AIS Lost Target" : 105,
    "AIS Safety Related Message (used to silence)" : 106,
    "AIS Connection Lost" : 107,
    "No Fix" : 108,
}
def lookup_encode_SEATALK_ALARM_ID(value):
    result = lookup_dict_encode_SEATALK_ALARM_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_ALARM_ID")
    return result

lookup_dict_encode_SEATALK_ALARM_GROUP = {
    "Instrument" : 0,
    "Autopilot" : 1,
    "Radar" : 2,
    "Chart Plotter" : 3,
    "AIS" : 4,
}
def lookup_encode_SEATALK_ALARM_GROUP(value):
    result = lookup_dict_encode_SEATALK_ALARM_GROUP.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_ALARM_GROUP")
    return result

lookup_dict_encode_SEATALK_PILOT_MODE = {
    "Standby" : 64,
    "Auto" : 66,
    "Wind" : 70,
    "Track" : 74,
}
def lookup_encode_SEATALK_PILOT_MODE(value):
    result = lookup_dict_encode_SEATALK_PILOT_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_PILOT_MODE")
    return result

lookup_dict_encode_SEATALK_PILOT_HULL_TYPE = {
    "Sail" : 0,
    "Sail (slow turn)" : 1,
    "Sail Catamaran" : 2,
    "Power (slow turn)" : 3,
    "Power (fast turn)" : 4,
    "Power" : 8,
}
def lookup_encode_SEATALK_PILOT_HULL_TYPE(value):
    result = lookup_dict_encode_SEATALK_PILOT_HULL_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_PILOT_HULL_TYPE")
    return result

lookup_dict_encode_SEATALK_SHARED = {
    "Shared" : 1,
    "Not Shared" : 2,
}
def lookup_encode_SEATALK_SHARED(value):
    result = lookup_dict_encode_SEATALK_SHARED.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_SHARED")
    return result

lookup_dict_encode_ENTERTAINMENT_ZONE = {
    "All zones" : 0,
    "Zone 1" : 1,
    "Zone 2" : 2,
    "Zone 3" : 3,
    "Zone 4" : 4,
}
def lookup_encode_ENTERTAINMENT_ZONE(value):
    result = lookup_dict_encode_ENTERTAINMENT_ZONE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_ZONE")
    return result

lookup_dict_encode_ENTERTAINMENT_SOURCE = {
    "Vessel alarm" : 0,
    "AM" : 1,
    "FM" : 2,
    "Weather" : 3,
    "DAB" : 4,
    "Aux" : 5,
    "USB" : 6,
    "CD" : 7,
    "MP3" : 8,
    "Apple iOS" : 9,
    "Android" : 10,
    "Bluetooth" : 11,
    "Sirius XM" : 12,
    "Pandora" : 13,
    "Spotify" : 14,
    "Slacker" : 15,
    "Songza" : 16,
    "Apple Radio" : 17,
    "Last FM" : 18,
    "Ethernet" : 19,
    "Video MP4" : 20,
    "Video DVD" : 21,
    "Video BluRay" : 22,
    "HDMI" : 23,
    "Video" : 24,
}
def lookup_encode_ENTERTAINMENT_SOURCE(value):
    result = lookup_dict_encode_ENTERTAINMENT_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_SOURCE")
    return result

lookup_dict_encode_ENTERTAINMENT_PLAY_STATUS = {
    "Play" : 0,
    "Pause" : 1,
    "Stop" : 2,
    "FF 1x" : 3,
    "FF 2x" : 4,
    "FF 3x" : 5,
    "FF 4x" : 6,
    "RW 1x" : 7,
    "RW 2x" : 8,
    "RW 3x" : 9,
    "RW 4x" : 10,
    "Skip ahead" : 11,
    "Skip back" : 12,
    "Jog ahead" : 13,
    "Jog back" : 14,
    "Seek up" : 15,
    "Seek down" : 16,
    "Scan up" : 17,
    "Scan down" : 18,
    "Tune up" : 19,
    "Tune down" : 20,
    "Slow motion .75x" : 21,
    "Slow motion .5x" : 22,
    "Slow motion .25x" : 23,
    "Slow motion .125x" : 24,
}
def lookup_encode_ENTERTAINMENT_PLAY_STATUS(value):
    result = lookup_dict_encode_ENTERTAINMENT_PLAY_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_PLAY_STATUS")
    return result

lookup_dict_encode_ENTERTAINMENT_REPEAT_STATUS = {
    "Off" : 0,
    "One" : 1,
    "All" : 2,
}
def lookup_encode_ENTERTAINMENT_REPEAT_STATUS(value):
    result = lookup_dict_encode_ENTERTAINMENT_REPEAT_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_REPEAT_STATUS")
    return result

lookup_dict_encode_ENTERTAINMENT_SHUFFLE_STATUS = {
    "Off" : 0,
    "Play queue" : 1,
    "All" : 2,
}
def lookup_encode_ENTERTAINMENT_SHUFFLE_STATUS(value):
    result = lookup_dict_encode_ENTERTAINMENT_SHUFFLE_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_SHUFFLE_STATUS")
    return result

lookup_dict_encode_ENTERTAINMENT_LIKE_STATUS = {
    "None" : 0,
    "Thumbs up" : 1,
    "Thumbs down" : 2,
}
def lookup_encode_ENTERTAINMENT_LIKE_STATUS(value):
    result = lookup_dict_encode_ENTERTAINMENT_LIKE_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_LIKE_STATUS")
    return result

lookup_dict_encode_ENTERTAINMENT_TYPE = {
    "File" : 0,
    "Playlist Name" : 1,
    "Genre Name" : 2,
    "Album Name" : 3,
    "Artist Name" : 4,
    "Track Name" : 5,
    "Station Name" : 6,
    "Station Number" : 7,
    "Favourite Number" : 8,
    "Play Queue" : 9,
    "Content Info" : 10,
}
def lookup_encode_ENTERTAINMENT_TYPE(value):
    result = lookup_dict_encode_ENTERTAINMENT_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_TYPE")
    return result

lookup_dict_encode_ENTERTAINMENT_GROUP = {
    "File" : 0,
    "Playlist Name" : 1,
    "Genre Name" : 2,
    "Album Name" : 3,
    "Artist Name" : 4,
    "Track Name" : 5,
    "Station Name" : 6,
    "Station Number" : 7,
    "Favourite Number" : 8,
    "Play Queue" : 9,
    "Content Info" : 10,
}
def lookup_encode_ENTERTAINMENT_GROUP(value):
    result = lookup_dict_encode_ENTERTAINMENT_GROUP.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_GROUP")
    return result

lookup_dict_encode_ENTERTAINMENT_CHANNEL = {
    "All channels" : 0,
    "Stereo full range" : 1,
    "Stereo front" : 2,
    "Stereo back" : 3,
    "Stereo surround" : 4,
    "Center" : 5,
    "Subwoofer" : 6,
    "Front left" : 7,
    "Front right" : 8,
    "Back left" : 9,
    "Back right" : 10,
    "Surround left" : 11,
    "Surround right" : 12,
}
def lookup_encode_ENTERTAINMENT_CHANNEL(value):
    result = lookup_dict_encode_ENTERTAINMENT_CHANNEL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_CHANNEL")
    return result

lookup_dict_encode_ENTERTAINMENT_EQ = {
    "Flat" : 0,
    "Rock" : 1,
    "Hall" : 2,
    "Jazz" : 3,
    "Pop" : 4,
    "Live" : 5,
    "Classic" : 6,
    "Vocal" : 7,
    "Arena" : 8,
    "Cinema" : 9,
    "Custom" : 10,
}
def lookup_encode_ENTERTAINMENT_EQ(value):
    result = lookup_dict_encode_ENTERTAINMENT_EQ.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_EQ")
    return result

lookup_dict_encode_ENTERTAINMENT_FILTER = {
    "Full range" : 0,
    "High pass" : 1,
    "Low pass" : 2,
    "Band pass" : 3,
    "Notch filter" : 4,
}
def lookup_encode_ENTERTAINMENT_FILTER(value):
    result = lookup_dict_encode_ENTERTAINMENT_FILTER.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_FILTER")
    return result

lookup_dict_encode_ALERT_TYPE = {
    "Emergency Alarm" : 1,
    "Alarm" : 2,
    "Warning" : 5,
    "Caution" : 8,
}
def lookup_encode_ALERT_TYPE(value):
    result = lookup_dict_encode_ALERT_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_TYPE")
    return result

lookup_dict_encode_ALERT_CATEGORY = {
    "Navigational" : 0,
    "Technical" : 1,
}
def lookup_encode_ALERT_CATEGORY(value):
    result = lookup_dict_encode_ALERT_CATEGORY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_CATEGORY")
    return result

lookup_dict_encode_ALERT_TRIGGER_CONDITION = {
    "Manual" : 0,
    "Auto" : 1,
    "Test" : 2,
    "Disabled" : 3,
}
def lookup_encode_ALERT_TRIGGER_CONDITION(value):
    result = lookup_dict_encode_ALERT_TRIGGER_CONDITION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_TRIGGER_CONDITION")
    return result

lookup_dict_encode_ALERT_THRESHOLD_STATUS = {
    "Normal" : 0,
    "Threshold Exceeded" : 1,
    "Extreme Threshold Exceeded" : 2,
    "Low Threshold Exceeded" : 3,
    "Acknowledged" : 4,
    "Awaiting Acknowledge" : 5,
}
def lookup_encode_ALERT_THRESHOLD_STATUS(value):
    result = lookup_dict_encode_ALERT_THRESHOLD_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_THRESHOLD_STATUS")
    return result

lookup_dict_encode_ALERT_STATE = {
    "Disabled" : 0,
    "Normal" : 1,
    "Active" : 2,
    "Silenced" : 3,
    "Acknowledged" : 4,
    "Awaiting Acknowledge" : 5,
}
def lookup_encode_ALERT_STATE(value):
    result = lookup_dict_encode_ALERT_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_STATE")
    return result

lookup_dict_encode_ALERT_LANGUAGE_ID = {
    "English (US)" : 0,
    "English (UK)" : 1,
    "Arabic" : 2,
    "Chinese (simplified)" : 3,
    "Croatian" : 4,
    "Danish" : 5,
    "Dutch" : 6,
    "Finnish" : 7,
    "French" : 8,
    "German" : 9,
    "Greek" : 10,
    "Italian" : 11,
    "Japanese" : 12,
    "Korean" : 13,
    "Norwegian" : 14,
    "Polish" : 15,
    "Portuguese" : 16,
    "Russian" : 17,
    "Spanish" : 18,
    "Swedish" : 19,
}
def lookup_encode_ALERT_LANGUAGE_ID(value):
    result = lookup_dict_encode_ALERT_LANGUAGE_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_LANGUAGE_ID")
    return result

lookup_dict_encode_ALERT_RESPONSE_COMMAND = {
    "Acknowledge" : 0,
    "Temporary Silence" : 1,
    "Test Command off" : 2,
    "Test Command on" : 3,
}
def lookup_encode_ALERT_RESPONSE_COMMAND(value):
    result = lookup_dict_encode_ALERT_RESPONSE_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ALERT_RESPONSE_COMMAND")
    return result

lookup_dict_encode_CONVERTER_STATE = {
    "Off" : 0,
    "Low Power Mode" : 1,
    "Fault" : 2,
    "Bulk" : 3,
    "Absorption" : 4,
    "Float" : 5,
    "Storage" : 6,
    "Equalize" : 7,
    "Pass thru" : 8,
    "Inverting" : 9,
    "Assisting" : 10,
}
def lookup_encode_CONVERTER_STATE(value):
    result = lookup_dict_encode_CONVERTER_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from CONVERTER_STATE")
    return result

lookup_dict_encode_THRUSTER_DIRECTION_CONTROL = {
    "Off" : 0,
    "Ready" : 1,
    "To Port" : 2,
    "To Starboard" : 3,
}
def lookup_encode_THRUSTER_DIRECTION_CONTROL(value):
    result = lookup_dict_encode_THRUSTER_DIRECTION_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from THRUSTER_DIRECTION_CONTROL")
    return result

lookup_dict_encode_THRUSTER_RETRACT_CONTROL = {
    "Off" : 0,
    "Extend" : 1,
    "Retract" : 2,
}
def lookup_encode_THRUSTER_RETRACT_CONTROL(value):
    result = lookup_dict_encode_THRUSTER_RETRACT_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from THRUSTER_RETRACT_CONTROL")
    return result

lookup_dict_encode_THRUSTER_MOTOR_TYPE = {
    "12VDC" : 0,
    "24VDC" : 1,
    "48VDC" : 2,
    "24VAC" : 3,
    "Hydraulic" : 4,
}
def lookup_encode_THRUSTER_MOTOR_TYPE(value):
    result = lookup_dict_encode_THRUSTER_MOTOR_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from THRUSTER_MOTOR_TYPE")
    return result

lookup_dict_encode_BOOT_STATE = {
    "in Startup Monitor" : 0,
    "running Bootloader" : 1,
    "running Application" : 2,
}
def lookup_encode_BOOT_STATE(value):
    result = lookup_dict_encode_BOOT_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BOOT_STATE")
    return result

lookup_dict_encode_ACCESS_LEVEL = {
    "Locked" : 0,
    "unlocked level 1" : 1,
    "unlocked level 2" : 2,
}
def lookup_encode_ACCESS_LEVEL(value):
    result = lookup_dict_encode_ACCESS_LEVEL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ACCESS_LEVEL")
    return result

lookup_dict_encode_TRANSMISSION_INTERVAL = {
    "Acknowledge" : 0,
    "Transmit Interval/Priority not supported" : 1,
    "Transmit Interval too low" : 2,
    "Access denied" : 3,
    "Not supported" : 4,
}
def lookup_encode_TRANSMISSION_INTERVAL(value):
    result = lookup_dict_encode_TRANSMISSION_INTERVAL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TRANSMISSION_INTERVAL")
    return result

lookup_dict_encode_PARAMETER_FIELD = {
    "Acknowledge" : 0,
    "Invalid parameter field" : 1,
    "Temporary error" : 2,
    "Parameter out of range" : 3,
    "Access denied" : 4,
    "Not supported" : 5,
    "Read or Write not supported" : 6,
}
def lookup_encode_PARAMETER_FIELD(value):
    result = lookup_dict_encode_PARAMETER_FIELD.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from PARAMETER_FIELD")
    return result

lookup_dict_encode_PGN_LIST_FUNCTION = {
    "Transmit PGN list" : 0,
    "Receive PGN list" : 1,
}
def lookup_encode_PGN_LIST_FUNCTION(value):
    result = lookup_dict_encode_PGN_LIST_FUNCTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from PGN_LIST_FUNCTION")
    return result

lookup_dict_encode_FUSION_COMMAND = {
    "Play" : 1,
    "Pause" : 2,
    "Next" : 4,
    "Prev" : 6,
}
def lookup_encode_FUSION_COMMAND(value):
    result = lookup_dict_encode_FUSION_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_COMMAND")
    return result

lookup_dict_encode_FUSION_SIRIUS_COMMAND = {
    "Next" : 1,
    "Prev" : 2,
}
def lookup_encode_FUSION_SIRIUS_COMMAND(value):
    result = lookup_dict_encode_FUSION_SIRIUS_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_SIRIUS_COMMAND")
    return result

lookup_dict_encode_FUSION_MUTE_COMMAND = {
    "Mute On" : 1,
    "Mute Off" : 2,
}
def lookup_encode_FUSION_MUTE_COMMAND(value):
    result = lookup_dict_encode_FUSION_MUTE_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_MUTE_COMMAND")
    return result

lookup_dict_encode_SEATALK_KEYSTROKE = {
    "Auto" : 1,
    "Standby" : 2,
    "Wind" : 3,
    "-1" : 5,
    "-10" : 6,
    "+1" : 7,
    "+10" : 8,
    "-1 and -10" : 33,
    "+1 and +10" : 34,
    "Track" : 35,
}
def lookup_encode_SEATALK_KEYSTROKE(value):
    result = lookup_dict_encode_SEATALK_KEYSTROKE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_KEYSTROKE")
    return result

lookup_dict_encode_SEATALK_DEVICE_ID = {
    "S100" : 3,
    "Course Computer" : 5,
}
def lookup_encode_SEATALK_DEVICE_ID(value):
    result = lookup_dict_encode_SEATALK_DEVICE_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_DEVICE_ID")
    return result

lookup_dict_encode_SEATALK_NETWORK_GROUP = {
    "None" : 0,
    "Helm 1" : 1,
    "Helm 2" : 2,
    "Cockpit" : 3,
    "Flybridge" : 4,
    "Mast" : 5,
    "Group 1" : 6,
    "Group 2" : 7,
    "Group 3" : 8,
    "Group 4" : 9,
    "Group 5" : 10,
}
def lookup_encode_SEATALK_NETWORK_GROUP(value):
    result = lookup_dict_encode_SEATALK_NETWORK_GROUP.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_NETWORK_GROUP")
    return result

lookup_dict_encode_SEATALK_DISPLAY_COLOR = {
    "Day 1" : 0,
    "Day 2" : 2,
    "Red/Black" : 3,
    "Inverse" : 4,
}
def lookup_encode_SEATALK_DISPLAY_COLOR(value):
    result = lookup_dict_encode_SEATALK_DISPLAY_COLOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_DISPLAY_COLOR")
    return result

lookup_dict_encode_AIRMAR_CALIBRATE_FUNCTION = {
    "Normal/cancel calibration" : 0,
    "Enter calibration mode" : 1,
    "Reset calibration to 0" : 2,
    "Verify" : 3,
    "Reset compass to defaults" : 4,
    "Reset damping to defaults" : 5,
}
def lookup_encode_AIRMAR_CALIBRATE_FUNCTION(value):
    result = lookup_dict_encode_AIRMAR_CALIBRATE_FUNCTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_CALIBRATE_FUNCTION")
    return result

lookup_dict_encode_AIRMAR_CALIBRATE_STATUS = {
    "Queried" : 0,
    "Passed" : 1,
    "Failed - timeout" : 2,
    "Failed - tilt error" : 3,
    "Failed - other" : 4,
    "In progress" : 5,
}
def lookup_encode_AIRMAR_CALIBRATE_STATUS(value):
    result = lookup_dict_encode_AIRMAR_CALIBRATE_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_CALIBRATE_STATUS")
    return result

lookup_dict_encode_AIRMAR_TEMPERATURE_INSTANCE = {
    "Device Sensor" : 0,
    "Onboard Water Sensor" : 1,
    "Optional Water Sensor" : 2,
}
def lookup_encode_AIRMAR_TEMPERATURE_INSTANCE(value):
    result = lookup_dict_encode_AIRMAR_TEMPERATURE_INSTANCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_TEMPERATURE_INSTANCE")
    return result

lookup_dict_encode_AIRMAR_FILTER = {
    "No filter" : 0,
    "Basic IIR filter" : 1,
}
def lookup_encode_AIRMAR_FILTER(value):
    result = lookup_dict_encode_AIRMAR_FILTER.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_FILTER")
    return result

lookup_dict_encode_CONTROLLER_STATE = {
    "Error Active" : 0,
    "Error Passive" : 1,
    "Bus Off" : 2,
}
def lookup_encode_CONTROLLER_STATE(value):
    result = lookup_dict_encode_CONTROLLER_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from CONTROLLER_STATE")
    return result

lookup_dict_encode_EQUIPMENT_STATUS = {
    "Operational" : 0,
    "Fault" : 1,
}
def lookup_encode_EQUIPMENT_STATUS(value):
    result = lookup_dict_encode_EQUIPMENT_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from EQUIPMENT_STATUS")
    return result

lookup_dict_encode_MOB_STATUS = {
    "MOB Emitter Activated" : 0,
    "Manual on-board MOB Button Activation" : 1,
    "Test mode" : 2,
}
def lookup_encode_MOB_STATUS(value):
    result = lookup_dict_encode_MOB_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MOB_STATUS")
    return result

lookup_dict_encode_LOW_BATTERY = {
    "Good" : 0,
    "Low" : 1,
}
def lookup_encode_LOW_BATTERY(value):
    result = lookup_dict_encode_LOW_BATTERY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from LOW_BATTERY")
    return result

lookup_dict_encode_TURN_MODE = {
    "Rudder limit controlled" : 0,
    "Turn rate controlled" : 1,
    "Radius controlled" : 2,
}
def lookup_encode_TURN_MODE(value):
    result = lookup_dict_encode_TURN_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TURN_MODE")
    return result

lookup_dict_encode_ACCEPTABILITY = {
    "Bad level" : 0,
    "Bad frequency" : 1,
    "Being qualified" : 2,
    "Good" : 3,
}
def lookup_encode_ACCEPTABILITY(value):
    result = lookup_dict_encode_ACCEPTABILITY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ACCEPTABILITY")
    return result

lookup_dict_encode_LINE = {
    "Line 1" : 0,
    "Line 2" : 1,
    "Line 3" : 2,
}
def lookup_encode_LINE(value):
    result = lookup_dict_encode_LINE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from LINE")
    return result

lookup_dict_encode_WAVEFORM = {
    "Sine wave" : 0,
    "Modified sine wave" : 1,
}
def lookup_encode_WAVEFORM(value):
    result = lookup_dict_encode_WAVEFORM.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WAVEFORM")
    return result

lookup_dict_encode_TANK_TYPE = {
    "Fuel" : 0,
    "Water" : 1,
    "Gray water" : 2,
    "Live well" : 3,
    "Oil" : 4,
    "Black water" : 5,
}
def lookup_encode_TANK_TYPE(value):
    result = lookup_dict_encode_TANK_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TANK_TYPE")
    return result

lookup_dict_encode_DC_SOURCE = {
    "Battery" : 0,
    "Alternator" : 1,
    "Convertor" : 2,
    "Solar cell" : 3,
    "Wind generator" : 4,
}
def lookup_encode_DC_SOURCE(value):
    result = lookup_dict_encode_DC_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DC_SOURCE")
    return result

lookup_dict_encode_CHARGER_STATE = {
    "Not charging" : 0,
    "Bulk" : 1,
    "Absorption" : 2,
    "Overcharge" : 3,
    "Equalise" : 4,
    "Float" : 5,
    "No float" : 6,
    "Constant VI" : 7,
    "Disabled" : 8,
    "Fault" : 9,
}
def lookup_encode_CHARGER_STATE(value):
    result = lookup_dict_encode_CHARGER_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from CHARGER_STATE")
    return result

lookup_dict_encode_CHARGING_ALGORITHM = {
    "Trickle" : 0,
    "Constant voltage / Constant current" : 1,
    "2 stage (no float)" : 2,
    "3 stage" : 3,
}
def lookup_encode_CHARGING_ALGORITHM(value):
    result = lookup_dict_encode_CHARGING_ALGORITHM.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from CHARGING_ALGORITHM")
    return result

lookup_dict_encode_CHARGER_MODE = {
    "Standalone" : 0,
    "Primary" : 1,
    "Secondary" : 2,
    "Echo" : 3,
}
def lookup_encode_CHARGER_MODE(value):
    result = lookup_dict_encode_CHARGER_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from CHARGER_MODE")
    return result

lookup_dict_encode_INVERTER_STATE = {
    "Invert" : 0,
    "AC passthru" : 1,
    "Load sense" : 2,
    "Fault" : 3,
    "Disabled" : 4,
}
def lookup_encode_INVERTER_STATE(value):
    result = lookup_dict_encode_INVERTER_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from INVERTER_STATE")
    return result

lookup_dict_encode_BATTERY_TYPE = {
    "Flooded" : 0,
    "Gel" : 1,
    "AGM" : 2,
}
def lookup_encode_BATTERY_TYPE(value):
    result = lookup_dict_encode_BATTERY_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BATTERY_TYPE")
    return result

lookup_dict_encode_BATTERY_VOLTAGE = {
    "6V" : 0,
    "12V" : 1,
    "24V" : 2,
    "32V" : 3,
    "36V" : 4,
    "42V" : 5,
    "48V" : 6,
}
def lookup_encode_BATTERY_VOLTAGE(value):
    result = lookup_dict_encode_BATTERY_VOLTAGE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BATTERY_VOLTAGE")
    return result

lookup_dict_encode_BATTERY_CHEMISTRY = {
    "Pb (Lead)" : 0,
    "Li" : 1,
    "NiCd" : 2,
    "ZnO" : 3,
    "NiMH" : 4,
}
def lookup_encode_BATTERY_CHEMISTRY(value):
    result = lookup_dict_encode_BATTERY_CHEMISTRY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BATTERY_CHEMISTRY")
    return result

lookup_dict_encode_GOOD_WARNING_ERROR = {
    "Good" : 0,
    "Warning" : 1,
    "Error" : 2,
}
def lookup_encode_GOOD_WARNING_ERROR(value):
    result = lookup_dict_encode_GOOD_WARNING_ERROR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GOOD_WARNING_ERROR")
    return result

lookup_dict_encode_TRACKING = {
    "Cancelled" : 0,
    "Acquiring" : 1,
    "Tracking" : 2,
    "Lost" : 3,
}
def lookup_encode_TRACKING(value):
    result = lookup_dict_encode_TRACKING.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TRACKING")
    return result

lookup_dict_encode_TARGET_ACQUISITION = {
    "Manual" : 0,
    "Automatic" : 1,
}
def lookup_encode_TARGET_ACQUISITION(value):
    result = lookup_dict_encode_TARGET_ACQUISITION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TARGET_ACQUISITION")
    return result

lookup_dict_encode_WINDLASS_DIRECTION = {
    "Off" : 0,
    "Down" : 1,
    "Up" : 2,
}
def lookup_encode_WINDLASS_DIRECTION(value):
    result = lookup_dict_encode_WINDLASS_DIRECTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WINDLASS_DIRECTION")
    return result

lookup_dict_encode_SPEED_TYPE = {
    "Single speed" : 0,
    "Dual speed" : 1,
    "Proportional speed" : 2,
}
def lookup_encode_SPEED_TYPE(value):
    result = lookup_dict_encode_SPEED_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SPEED_TYPE")
    return result

lookup_dict_encode_WINDLASS_MOTION = {
    "Windlass stopped" : 0,
    "Deployment occurring" : 1,
    "Retrieval occurring" : 2,
}
def lookup_encode_WINDLASS_MOTION(value):
    result = lookup_dict_encode_WINDLASS_MOTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WINDLASS_MOTION")
    return result

lookup_dict_encode_RODE_TYPE = {
    "Chain presently detected" : 0,
    "Rope presently detected" : 1,
}
def lookup_encode_RODE_TYPE(value):
    result = lookup_dict_encode_RODE_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from RODE_TYPE")
    return result

lookup_dict_encode_DOCKING_STATUS = {
    "Not docked" : 0,
    "Fully docked" : 1,
}
def lookup_encode_DOCKING_STATUS(value):
    result = lookup_dict_encode_DOCKING_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DOCKING_STATUS")
    return result

lookup_dict_encode_AIS_TYPE = {
    "SOTDMA" : 0,
    "CS" : 1,
}
def lookup_encode_AIS_TYPE(value):
    result = lookup_dict_encode_AIS_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_TYPE")
    return result

lookup_dict_encode_AIS_BAND = {
    "Top 525 kHz of marine band" : 0,
    "Entire marine band" : 1,
}
def lookup_encode_AIS_BAND(value):
    result = lookup_dict_encode_AIS_BAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_BAND")
    return result

lookup_dict_encode_AIS_MODE = {
    "Autonomous" : 0,
    "Assigned" : 1,
}
def lookup_encode_AIS_MODE(value):
    result = lookup_dict_encode_AIS_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_MODE")
    return result

lookup_dict_encode_AIS_COMMUNICATION_STATE = {
    "SOTDMA" : 0,
    "ITDMA" : 1,
}
def lookup_encode_AIS_COMMUNICATION_STATE(value):
    result = lookup_dict_encode_AIS_COMMUNICATION_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_COMMUNICATION_STATE")
    return result

lookup_dict_encode_AVAILABLE = {
    "Available" : 0,
    "Not available" : 1,
}
def lookup_encode_AVAILABLE(value):
    result = lookup_dict_encode_AVAILABLE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AVAILABLE")
    return result

lookup_dict_encode_BEARING_MODE = {
    "Great Circle" : 0,
    "Rhumbline" : 1,
}
def lookup_encode_BEARING_MODE(value):
    result = lookup_dict_encode_BEARING_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BEARING_MODE")
    return result

lookup_dict_encode_MARK_TYPE = {
    "Collision" : 0,
    "Turning point" : 1,
    "Reference" : 2,
    "Wheelover" : 3,
    "Waypoint" : 4,
}
def lookup_encode_MARK_TYPE(value):
    result = lookup_dict_encode_MARK_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MARK_TYPE")
    return result

lookup_dict_encode_GNSS_MODE = {
    "1D" : 0,
    "2D" : 1,
    "3D" : 2,
    "Auto" : 3,
}
def lookup_encode_GNSS_MODE(value):
    result = lookup_dict_encode_GNSS_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GNSS_MODE")
    return result

lookup_dict_encode_RANGE_RESIDUAL_MODE = {
    "Range residuals were used to calculate data" : 0,
    "Range residuals were calculated after the position" : 1,
}
def lookup_encode_RANGE_RESIDUAL_MODE(value):
    result = lookup_dict_encode_RANGE_RESIDUAL_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from RANGE_RESIDUAL_MODE")
    return result

lookup_dict_encode_DGNSS_MODE = {
    "None" : 0,
    "SBAS if available" : 1,
    "SBAS" : 3,
}
def lookup_encode_DGNSS_MODE(value):
    result = lookup_dict_encode_DGNSS_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DGNSS_MODE")
    return result

lookup_dict_encode_SATELLITE_STATUS = {
    "Not tracked" : 0,
    "Tracked" : 1,
    "Used" : 2,
    "Not tracked+Diff" : 3,
    "Tracked+Diff" : 4,
    "Used+Diff" : 5,
}
def lookup_encode_SATELLITE_STATUS(value):
    result = lookup_dict_encode_SATELLITE_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SATELLITE_STATUS")
    return result

lookup_dict_encode_AIS_VERSION = {
    "ITU-R M.1371-1" : 0,
    "ITU-R M.1371-3" : 1,
    "ITU-R M.1371-5" : 2,
    "ITU-R M.1371 future edition" : 3,
}
def lookup_encode_AIS_VERSION(value):
    result = lookup_dict_encode_AIS_VERSION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIS_VERSION")
    return result

lookup_dict_encode_TIDE = {
    "Falling" : 0,
    "Rising" : 1,
}
def lookup_encode_TIDE(value):
    result = lookup_dict_encode_TIDE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TIDE")
    return result

lookup_dict_encode_WATERMAKER_STATE = {
    "Stopped" : 0,
    "Starting" : 1,
    "Running" : 2,
    "Stopping" : 3,
    "Flushing" : 4,
    "Rinsing" : 5,
    "Initiating" : 6,
    "Manual" : 7,
}
def lookup_encode_WATERMAKER_STATE(value):
    result = lookup_dict_encode_WATERMAKER_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WATERMAKER_STATE")
    return result

lookup_dict_encode_ENTERTAINMENT_ID_TYPE = {
    "Group" : 0,
    "File" : 1,
    "Encrypted group" : 2,
    "Encrypted file" : 3,
}
def lookup_encode_ENTERTAINMENT_ID_TYPE(value):
    result = lookup_dict_encode_ENTERTAINMENT_ID_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_ID_TYPE")
    return result

lookup_dict_encode_ENTERTAINMENT_DEFAULT_SETTINGS = {
    "Save current settings as user default" : 0,
    "Load user default" : 1,
    "Load manufacturer default" : 2,
}
def lookup_encode_ENTERTAINMENT_DEFAULT_SETTINGS(value):
    result = lookup_dict_encode_ENTERTAINMENT_DEFAULT_SETTINGS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_DEFAULT_SETTINGS")
    return result

lookup_dict_encode_ENTERTAINMENT_REGIONS = {
    "USA" : 0,
    "Europe" : 1,
    "Asia" : 2,
    "Middle East" : 3,
    "Latin America" : 4,
    "Australia" : 5,
    "Russia" : 6,
    "Japan" : 7,
}
def lookup_encode_ENTERTAINMENT_REGIONS(value):
    result = lookup_dict_encode_ENTERTAINMENT_REGIONS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_REGIONS")
    return result

lookup_dict_encode_VIDEO_PROTOCOLS = {
    "PAL" : 0,
    "NTSC" : 1,
}
def lookup_encode_VIDEO_PROTOCOLS(value):
    result = lookup_dict_encode_VIDEO_PROTOCOLS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from VIDEO_PROTOCOLS")
    return result

lookup_dict_encode_ENTERTAINMENT_VOLUME_CONTROL = {
    "Up" : 0,
    "Down" : 1,
}
def lookup_encode_ENTERTAINMENT_VOLUME_CONTROL(value):
    result = lookup_dict_encode_ENTERTAINMENT_VOLUME_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ENTERTAINMENT_VOLUME_CONTROL")
    return result

lookup_dict_encode_BLUETOOTH_STATUS = {
    "Connected" : 0,
    "Not connected" : 1,
    "Not paired" : 2,
}
def lookup_encode_BLUETOOTH_STATUS(value):
    result = lookup_dict_encode_BLUETOOTH_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BLUETOOTH_STATUS")
    return result

lookup_dict_encode_BLUETOOTH_SOURCE_STATUS = {
    "Reserved" : 0,
    "Connected" : 1,
    "Connecting" : 2,
    "Not connected" : 3,
}
def lookup_encode_BLUETOOTH_SOURCE_STATUS(value):
    result = lookup_dict_encode_BLUETOOTH_SOURCE_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BLUETOOTH_SOURCE_STATUS")
    return result

lookup_dict_encode_SONICHUB_COMMAND = {
    "Init #2" : 1,
    "AM Radio" : 4,
    "Zone Info" : 5,
    "Source" : 6,
    "Source List" : 8,
    "Control" : 9,
    "FM Radio" : 12,
    "Playlist" : 13,
    "Track" : 14,
    "Artist" : 15,
    "Album" : 16,
    "Menu Item" : 19,
    "Zones" : 20,
    "Max Volume" : 23,
    "Volume" : 24,
    "Init #1" : 25,
    "Position" : 48,
    "Init #3" : 50,
}
def lookup_encode_SONICHUB_COMMAND(value):
    result = lookup_dict_encode_SONICHUB_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SONICHUB_COMMAND")
    return result

lookup_dict_encode_SIMNET_AP_MODE = {
    "Heading" : 2,
    "Wind" : 3,
    "Nav" : 10,
    "No Drift" : 11,
}
def lookup_encode_SIMNET_AP_MODE(value):
    result = lookup_dict_encode_SIMNET_AP_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_AP_MODE")
    return result

lookup_dict_encode_SIMNET_DEVICE_MODEL = {
    "AC" : 0,
    "Other device" : 1,
    "NAC" : 100,
}
def lookup_encode_SIMNET_DEVICE_MODEL(value):
    result = lookup_dict_encode_SIMNET_DEVICE_MODEL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_DEVICE_MODEL")
    return result

lookup_dict_encode_SIMNET_DEVICE_REPORT = {
    "Status" : 2,
    "Send Status" : 3,
    "Mode" : 10,
    "Send Mode" : 11,
    "Sailing Processor Status" : 23,
}
def lookup_encode_SIMNET_DEVICE_REPORT(value):
    result = lookup_dict_encode_SIMNET_DEVICE_REPORT.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_DEVICE_REPORT")
    return result

lookup_dict_encode_SIMNET_AP_STATUS = {
    "Manual" : 2,
    "Automatic" : 16,
}
def lookup_encode_SIMNET_AP_STATUS(value):
    result = lookup_dict_encode_SIMNET_AP_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_AP_STATUS")
    return result

lookup_dict_encode_SIMNET_COMMAND = {
    "Text" : 50,
}
def lookup_encode_SIMNET_COMMAND(value):
    result = lookup_dict_encode_SIMNET_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_COMMAND")
    return result

lookup_dict_encode_SIMNET_EVENT_COMMAND = {
    "Alarm" : 1,
    "AP command" : 2,
    "Autopilot" : 255,
}
def lookup_encode_SIMNET_EVENT_COMMAND(value):
    result = lookup_dict_encode_SIMNET_EVENT_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_EVENT_COMMAND")
    return result

lookup_dict_encode_SIMNET_NIGHT_MODE = {
    "Day" : 2,
    "Night" : 4,
}
def lookup_encode_SIMNET_NIGHT_MODE(value):
    result = lookup_dict_encode_SIMNET_NIGHT_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_NIGHT_MODE")
    return result

lookup_dict_encode_SIMNET_NIGHT_MODE_COLOR = {
    "Red" : 0,
    "Green" : 1,
    "Blue" : 2,
    "White" : 3,
    "Magenta" : 4,
}
def lookup_encode_SIMNET_NIGHT_MODE_COLOR(value):
    result = lookup_dict_encode_SIMNET_NIGHT_MODE_COLOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_NIGHT_MODE_COLOR")
    return result

lookup_dict_encode_SIMNET_DISPLAY_GROUP = {
    "Default" : 1,
    "Group 1" : 2,
    "Group 2" : 3,
    "Group 3" : 4,
    "Group 4" : 5,
    "Group 5" : 6,
    "Group 6" : 7,
}
def lookup_encode_SIMNET_DISPLAY_GROUP(value):
    result = lookup_dict_encode_SIMNET_DISPLAY_GROUP.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_DISPLAY_GROUP")
    return result

lookup_dict_encode_SIMNET_HOUR_DISPLAY = {
    "24 hour" : 0,
    "12 hour" : 1,
}
def lookup_encode_SIMNET_HOUR_DISPLAY(value):
    result = lookup_dict_encode_SIMNET_HOUR_DISPLAY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_HOUR_DISPLAY")
    return result

lookup_dict_encode_SIMNET_TIME_FORMAT = {
    "MM/dd/yyyy" : 1,
    "dd/MM/yyyy" : 2,
}
def lookup_encode_SIMNET_TIME_FORMAT(value):
    result = lookup_dict_encode_SIMNET_TIME_FORMAT.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_TIME_FORMAT")
    return result

lookup_dict_encode_SIMNET_BACKLIGHT_LEVEL = {
    "10% (Min)" : 0,
    "Day mode" : 1,
    "Night mode" : 4,
    "20%" : 11,
    "30%" : 22,
    "40%" : 33,
    "50%" : 44,
    "60%" : 55,
    "70%" : 66,
    "80%" : 77,
    "90%" : 88,
    "100% (Max)" : 99,
}
def lookup_encode_SIMNET_BACKLIGHT_LEVEL(value):
    result = lookup_dict_encode_SIMNET_BACKLIGHT_LEVEL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_BACKLIGHT_LEVEL")
    return result

lookup_dict_encode_SIMNET_AP_EVENTS = {
    "Follow/Non Follow" : 2,
    "Standby" : 6,
    "Heading mode" : 9,
    "Nav mode" : 10,
    "No Drift mode" : 12,
    "Non Follow Up mode" : 13,
    "Follow Up mode" : 14,
    "Wind mode" : 15,
    "Tack" : 17,
    "Square (Turn)" : 18,
    "C-Turn" : 19,
    "U-Turn" : 20,
    "Spiral (Turn)" : 21,
    "Zig Zag (Turn)" : 22,
    "Lazy-S (Turn)" : 23,
    "Depth (Turn)" : 24,
    "Change course" : 26,
    "Timer sync" : 61,
    "Ping port end" : 112,
    "Ping starboard end" : 113,
}
def lookup_encode_SIMNET_AP_EVENTS(value):
    result = lookup_dict_encode_SIMNET_AP_EVENTS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_AP_EVENTS")
    return result

lookup_dict_encode_SIMNET_DIRECTION = {
    "Port" : 2,
    "Starboard" : 3,
    "Left rudder (port)" : 4,
    "Right rudder (starboard)" : 5,
}
def lookup_encode_SIMNET_DIRECTION(value):
    result = lookup_dict_encode_SIMNET_DIRECTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_DIRECTION")
    return result

lookup_dict_encode_SIMNET_AP_COMMAND_TYPE = {
    "Follow Up" : 2,
    "AP Command" : 10,
}
def lookup_encode_SIMNET_AP_COMMAND_TYPE(value):
    result = lookup_dict_encode_SIMNET_AP_COMMAND_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_AP_COMMAND_TYPE")
    return result

lookup_dict_encode_SIMNET_ALARM = {
    "Low boat speed" : 57,
    "No autopilot computer" : 56,
    "Nav data missing" : 58,
}
def lookup_encode_SIMNET_ALARM(value):
    result = lookup_dict_encode_SIMNET_ALARM.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SIMNET_ALARM")
    return result

lookup_dict_encode_FUSION_MESSAGE_ID = {
    "Request Status" : 1,
    "Set Source" : 2,
    "Media Command" : 3,
    "Tuner Command" : 5,
    "Marine Tuner Command" : 6,
    "Set Marine Tuner Squelch" : 7,
    "Set Marine Tuner Scan Mode" : 8,
    "Menu Action" : 9,
    "Request Menu Count" : 10,
    "Request Menu Item" : 11,
    "Request Menu Lock ID" : 12,
    "Set Aux Gain" : 13,
    "Set Settings" : 15,
    "DAB Updtate Command" : 16,
    "Set Mute" : 17,
    "Set Balance" : 18,
    "Set Low Pass Filer" : 19,
    "Set Sublevel" : 20,
    "Set Equalizer" : 22,
    "Set Volume Limit" : 23,
    "Set Zone Volume" : 24,
    "Set All Volumes" : 25,
    "Set Line Level Control" : 27,
    "Power" : 28,
    "Set Device Name" : 29,
    "Send Sirius Command" : 30,
    "Set Sirius Parental" : 31,
    "Send Factory Reset Command" : 33,
    "Set Zone Name" : 34,
    "Send Dvd Command" : 35,
    "Dvd Press Ir Key" : 36,
    "Send Select Sirius Team" : 39,
    "Send Select Sirius Artist" : 40,
    "Send Sirius Sport Alert User Action" : 41,
    "Send Sirius Artist Song User Action" : 45,
    "Send Multiroom Command" : 50,
    "Get Multiroom Device Record" : 51,
    "Scan Multirooom Devices" : 52,
    "Send File Transfer" : 53,
    "Set Loud" : 54,
    "Fapi Set Source Multiroom Enabled" : 56,
    "Request Head Unit Dsp Settings" : 57,
    "Send Transfer Status" : 64,
    "Fapi Get Server Info" : 65,
    "Fapi Set Source Enabled" : 69,
    "Fapi Set Source Name" : 70,
    "Send External Amp Gain" : 73,
    "Send Internal Amp Gain" : 74,
    "Send Mono" : 75,
}
def lookup_encode_FUSION_MESSAGE_ID(value):
    result = lookup_dict_encode_FUSION_MESSAGE_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_MESSAGE_ID")
    return result

lookup_dict_encode_FUSION_PLAY_STATUS = {
    "Invalid" : 0,
    "Playing" : 1,
    "Paused" : 2,
    "Stopped" : 3,
    "Skip Forward" : 4,
    "Skip Rewind" : 5,
}
def lookup_encode_FUSION_PLAY_STATUS(value):
    result = lookup_dict_encode_FUSION_PLAY_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_PLAY_STATUS")
    return result

lookup_dict_encode_FUSION_SOURCE_TYPE = {
    "AM" : 0,
    "FM" : 1,
    "Aux" : 2,
    "Sirius" : 3,
    "Ipod" : 4,
    "USB" : 5,
    "DVD" : 6,
    "VHF" : 7,
    "Invalid" : 8,
    "MTP" : 9,
    "Bluetooth" : 10,
    "ARC" : 11,
    "Android" : 12,
    "Pandora" : 13,
    "DAB" : 14,
    "AirPlay" : 15,
    "UPNP" : 16,
    "Unknown" : 17,
}
def lookup_encode_FUSION_SOURCE_TYPE(value):
    result = lookup_dict_encode_FUSION_SOURCE_TYPE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_SOURCE_TYPE")
    return result

lookup_dict_encode_FUSION_SIRIUS_COM_STATE = {
    "Unknown" : 255,
    "Off" : 1,
    "Initialising" : 2,
    "On" : 3,
}
def lookup_encode_FUSION_SIRIUS_COM_STATE(value):
    result = lookup_dict_encode_FUSION_SIRIUS_COM_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_SIRIUS_COM_STATE")
    return result

lookup_dict_encode_FUSION_SIRIUS_ALERT = {
    "Unknown" : 255,
    "None" : 1,
    "Antenna" : 2,
    "NoSignal" : 3,
    "Subscription Update" : 4,
}
def lookup_encode_FUSION_SIRIUS_ALERT(value):
    result = lookup_dict_encode_FUSION_SIRIUS_ALERT.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_SIRIUS_ALERT")
    return result

lookup_dict_encode_FUSION_SIRIUS_TUNING_MODE = {
    "Normal" : 1,
    "Category" : 2,
    "Preset" : 3,
}
def lookup_encode_FUSION_SIRIUS_TUNING_MODE(value):
    result = lookup_dict_encode_FUSION_SIRIUS_TUNING_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_SIRIUS_TUNING_MODE")
    return result

lookup_dict_encode_FUSION_STATUS_MESSAGE_ID = {
    "Unknown" : 0,
    "API Version" : 32769,
    "Source" : 32770,
    "Source Count" : 32771,
    "Track Info" : 32772,
    "Track Title" : 32773,
    "Track Artist" : 32774,
    "Track Album" : 32775,
    "Cover Art" : 32776,
    "Track Progress" : 32777,
    "Tuner Align" : 32778,
    "Tuner" : 32779,
    "Marine Tuner" : 32780,
    "Marine Squelch" : 32781,
    "Marine Scan Mode" : 32782,
    "Menu Action" : 32783,
    "Menu Count" : 32784,
    "Menu Item" : 32785,
    "Menu Lock ID" : 32786,
    "Aux Gain" : 32787,
    "Setting" : 32788,
    "Settings" : 32789,
    "Update Firmware Result" : 32790,
    "Mute" : 32791,
    "Balance" : 32792,
    "Low Pass Filter" : 32793,
    "Sublevels" : 32794,
    "Tone" : 32795,
    "Volume Limits" : 32796,
    "Volume" : 32797,
    "Capabilities" : 32798,
    "Line Level Control" : 32799,
    "Power" : 32800,
    "Unit Name" : 32801,
    "Sirius" : 32802,
    "SiriusXM Preset Event" : 32803,
    "SiriusXM Channel" : 32804,
    "SiriusXM Title" : 32805,
    "SiriusXM Artist" : 32806,
    "SiriusXM Genre" : 32807,
    "SiriusXM Category" : 32808,
    "SiriusXm Signal" : 32809,
    "SiriusXM Parental Request" : 32810,
    "SiriusXM Diagnostics" : 32811,
    "SiriusXM Presets" : 32812,
    "Zone Name" : 32813,
}
def lookup_encode_FUSION_STATUS_MESSAGE_ID(value):
    result = lookup_dict_encode_FUSION_STATUS_MESSAGE_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_STATUS_MESSAGE_ID")
    return result

lookup_dict_encode_SONICHUB_CONTROL = {
    "Set" : 0,
    "Ack" : 128,
}
def lookup_encode_SONICHUB_CONTROL(value):
    result = lookup_dict_encode_SONICHUB_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SONICHUB_CONTROL")
    return result

lookup_dict_encode_SONICHUB_SOURCE = {
    "AM" : 0,
    "FM" : 1,
    "iPod" : 2,
    "USB" : 3,
    "AUX" : 4,
    "AUX 2" : 5,
    "Mic" : 6,
}
def lookup_encode_SONICHUB_SOURCE(value):
    result = lookup_dict_encode_SONICHUB_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SONICHUB_SOURCE")
    return result

lookup_dict_encode_ISO_CONTROL = {
    "ACK" : 0,
    "NAK" : 1,
    "Access Denied" : 2,
    "Address Busy" : 3,
}
def lookup_encode_ISO_CONTROL(value):
    result = lookup_dict_encode_ISO_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ISO_CONTROL")
    return result

lookup_dict_encode_ISO_COMMAND = {
    "ACK" : 0,
    "RTS" : 16,
    "CTS" : 17,
    "EOM" : 19,
    "BAM" : 32,
    "Abort" : 255,
}
def lookup_encode_ISO_COMMAND(value):
    result = lookup_dict_encode_ISO_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ISO_COMMAND")
    return result

lookup_dict_encode_GROUP_FUNCTION = {
    "Request" : 0,
    "Command" : 1,
    "Acknowledge" : 2,
    "Read Fields" : 3,
    "Read Fields Reply" : 4,
    "Write Fields" : 5,
    "Write Fields Reply" : 6,
}
def lookup_encode_GROUP_FUNCTION(value):
    result = lookup_dict_encode_GROUP_FUNCTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GROUP_FUNCTION")
    return result

lookup_dict_encode_AIRMAR_COMMAND = {
    "Attitude Offsets" : 32,
    "Calibrate Compass" : 33,
    "True Wind Options" : 34,
    "Simulate Mode" : 35,
    "Calibrate Depth" : 40,
    "Calibrate Speed" : 41,
    "Calibrate Temperature" : 42,
    "Speed Filter" : 43,
    "Temperature Filter" : 44,
    "NMEA 2000 options" : 46,
}
def lookup_encode_AIRMAR_COMMAND(value):
    result = lookup_dict_encode_AIRMAR_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_COMMAND")
    return result

lookup_dict_encode_AIRMAR_DEPTH_QUALITY_FACTOR = {
    "Depth unlocked" : 0,
    "Quality 10%" : 1,
    "Quality 20%" : 2,
    "Quality 30%" : 3,
    "Quality 40%" : 4,
    "Quality 50%" : 5,
    "Quality 60%" : 6,
    "Quality 70%" : 7,
    "Quality 80%" : 8,
    "Quality 90%" : 9,
    "Quality 100%" : 10,
}
def lookup_encode_AIRMAR_DEPTH_QUALITY_FACTOR(value):
    result = lookup_dict_encode_AIRMAR_DEPTH_QUALITY_FACTOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_DEPTH_QUALITY_FACTOR")
    return result

lookup_dict_encode_PGN_ERROR_CODE = {
    "Acknowledge" : 0,
    "PGN not supported" : 1,
    "PGN not available" : 2,
    "Access denied" : 3,
    "Not supported" : 4,
    "Tag not supported" : 5,
    "Read or Write not supported" : 6,
}
def lookup_encode_PGN_ERROR_CODE(value):
    result = lookup_dict_encode_PGN_ERROR_CODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from PGN_ERROR_CODE")
    return result

lookup_dict_encode_AIRMAR_TRANSMISSION_INTERVAL = {
    "Measure interval" : 0,
    "Requested by user" : 1,
}
def lookup_encode_AIRMAR_TRANSMISSION_INTERVAL(value):
    result = lookup_dict_encode_AIRMAR_TRANSMISSION_INTERVAL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_TRANSMISSION_INTERVAL")
    return result

lookup_dict_encode_MOB_POSITION_SOURCE = {
    "Position estimated by the vessel" : 0,
    "Position reported by MOB emitter" : 1,
}
def lookup_encode_MOB_POSITION_SOURCE(value):
    result = lookup_dict_encode_MOB_POSITION_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MOB_POSITION_SOURCE")
    return result

lookup_dict_encode_STEERING_MODE = {
    "Main Steering" : 0,
    "Non-Follow-Up Device" : 1,
    "Follow-Up Device" : 2,
    "Heading Control Standalone" : 3,
    "Heading Control" : 4,
    "Track Control" : 5,
}
def lookup_encode_STEERING_MODE(value):
    result = lookup_dict_encode_STEERING_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from STEERING_MODE")
    return result

lookup_dict_encode_FUSION_RADIO_SOURCE = {
    "AM" : 0,
    "FM" : 1,
}
def lookup_encode_FUSION_RADIO_SOURCE(value):
    result = lookup_dict_encode_FUSION_RADIO_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_RADIO_SOURCE")
    return result

lookup_dict_encode_FUSION_SETTING = {
    "Alpha Search Threshold" : 0,
    "iPod Subtitles" : 1,
    "Zone 2 Linked" : 2,
    "Zone 2 Enabled" : 3,
    "Zone 3 Enabled" : 4,
    "Zone 4 Enabled" : 5,
    "Telemute" : 6,
    "Tuner Region" : 7,
    "Marine Zone" : 8,
    "USB repeat" : 9,
    "USB shuffle" : 10,
    "iPod Album Artwork" : 11,
    "iPod repeat" : 12,
    "iPod shuffle" : 13,
    "AM Preset 0" : 14,
    "AM Preset 1" : 15,
    "AM Preset 2" : 16,
    "AM Preset 3" : 17,
    "AM Preset 4" : 18,
    "AM Preset 5" : 19,
    "AM Preset 6" : 20,
    "AM Preset 7" : 21,
    "AM Preset 8" : 22,
    "AM Preset 9" : 23,
    "AM Preset 10" : 24,
    "AM Preset 11" : 25,
    "AM Preset 12" : 26,
    "AM Preset 13" : 27,
    "AM Preset 14" : 28,
    "FM Preset 0" : 29,
    "FM Preset 1" : 30,
    "FM Preset 2" : 31,
    "FM Preset 3" : 32,
    "FM Preset 4" : 33,
    "FM Preset 5" : 34,
    "FM Preset 6" : 35,
    "FM Preset 7" : 36,
    "FM Preset 8" : 37,
    "FM Preset 9" : 38,
    "FM Preset 10" : 39,
    "FM Preset 11" : 40,
    "FM Preset 12" : 41,
    "FM Preset 13" : 42,
    "FM Preset 14" : 43,
    "VHF Preset 0" : 44,
    "VHF Preset 1" : 45,
    "VHF Preset 2" : 46,
    "VHF Preset 3" : 47,
    "VHF Preset 4" : 48,
    "VHF Preset 5" : 49,
    "VHF Preset 6" : 50,
    "VHF Preset 7" : 51,
    "VHF Preset 8" : 52,
    "VHF Preset 9" : 53,
    "VHF Preset 10" : 54,
    "VHF Preset 11" : 55,
    "VHF Preset 12" : 56,
    "VHF Preset 13" : 57,
    "VHF Preset 14" : 58,
    "Clock Time" : 59,
    "Clock Alarm" : 60,
    "iPod Video Signal" : 61,
    "iPod Monitor Aspect" : 62,
    "Aux Name Index" : 63,
    "AM Enabled" : 64,
    "VHF Enabled" : 65,
    "Language" : 66,
    "Internal Amps On" : 67,
    "MTP Repeat" : 68,
    "MTP Shuffle" : 69,
    "Id Accessory Source" : 70,
    "NMEA Power" : 71,
    "Low Power Mode" : 72,
    "DVD region" : 73,
    "Volume Zone Sync" : 74,
    "Max Volume Start" : 75,
    "BT Auto Connect" : 76,
    "Null Setting" : 77,
}
def lookup_encode_FUSION_SETTING(value):
    result = lookup_dict_encode_FUSION_SETTING.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_SETTING")
    return result

lookup_dict_encode_FUSION_REPEAT_STATUS = {
    "Off" : 0,
    "One/track" : 1,
    "All/album" : 2,
}
def lookup_encode_FUSION_REPEAT_STATUS(value):
    result = lookup_dict_encode_FUSION_REPEAT_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_REPEAT_STATUS")
    return result

lookup_dict_encode_AIRMAR_POST_CONTROL = {
    "Report previous values" : 0,
    "Generate new values" : 1,
}
def lookup_encode_AIRMAR_POST_CONTROL(value):
    result = lookup_dict_encode_AIRMAR_POST_CONTROL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_POST_CONTROL")
    return result

lookup_dict_encode_AIRMAR_POST_ID = {
    "Format Code" : 1,
    "Factory EEPROM" : 2,
    "User EEPROM" : 3,
    "Water Temperature Sensor" : 4,
    "Sonar Transceiver" : 5,
    "Speed sensor" : 6,
    "Internal temperature sensor" : 7,
    "Battery voltage sensor" : 8,
}
def lookup_encode_AIRMAR_POST_ID(value):
    result = lookup_dict_encode_AIRMAR_POST_ID.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AIRMAR_POST_ID")
    return result

lookup_dict_encode_SONICHUB_TUNING = {
    "Seeking up" : 1,
    "Tuned" : 2,
    "Seeking down" : 3,
}
def lookup_encode_SONICHUB_TUNING(value):
    result = lookup_dict_encode_SONICHUB_TUNING.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SONICHUB_TUNING")
    return result

lookup_dict_encode_SONICHUB_PLAYLIST = {
    "Report" : 1,
    "Next song" : 4,
    "Previous song" : 6,
}
def lookup_encode_SONICHUB_PLAYLIST(value):
    result = lookup_dict_encode_SONICHUB_PLAYLIST.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SONICHUB_PLAYLIST")
    return result

lookup_dict_encode_FUSION_POWER_STATE = {
    "On" : 1,
    "Off" : 2,
}
def lookup_encode_FUSION_POWER_STATE(value):
    result = lookup_dict_encode_FUSION_POWER_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FUSION_POWER_STATE")
    return result

lookup_dict_encode_PRIORITY = {
    "0" : 0,
    "1" : 1,
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5" : 5,
    "6" : 6,
    "7" : 7,
    "Leave unchanged" : 8,
    "Reset to default" : 9,
}
def lookup_encode_PRIORITY(value):
    result = lookup_dict_encode_PRIORITY.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from PRIORITY")
    return result

lookup_dict_encode_DEVICE_TEMP_STATE = {
    "Cold" : 0,
    "Warm" : 1,
    "Hot" : 2,
}
def lookup_encode_DEVICE_TEMP_STATE(value):
    result = lookup_dict_encode_DEVICE_TEMP_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DEVICE_TEMP_STATE")
    return result

lookup_dict_encode_BANDG_DECIMALS = {
    "0" : 0,
    "1" : 1,
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "Auto" : 254,
}
def lookup_encode_BANDG_DECIMALS(value):
    result = lookup_dict_encode_BANDG_DECIMALS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BANDG_DECIMALS")
    return result

lookup_dict_encode_GARMIN_COLOR_MODE = {
    "Day" : 0,
    "Night" : 1,
    "Color" : 13,
}
def lookup_encode_GARMIN_COLOR_MODE(value):
    result = lookup_dict_encode_GARMIN_COLOR_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GARMIN_COLOR_MODE")
    return result

lookup_dict_encode_GARMIN_COLOR = {
    "Day full color" : 0,
    "Day high contrast" : 1,
    "Night full color" : 2,
    "Night red/black" : 3,
    "Night green/black" : 4,
}
def lookup_encode_GARMIN_COLOR(value):
    result = lookup_dict_encode_GARMIN_COLOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GARMIN_COLOR")
    return result

lookup_dict_encode_GARMIN_BACKLIGHT_LEVEL = {
    "0%" : 0,
    "5%" : 1,
    "10%" : 2,
    "15%" : 3,
    "20%" : 4,
    "25%" : 5,
    "30%" : 6,
    "35%" : 7,
    "40%" : 8,
    "45%" : 9,
    "50%" : 10,
    "55%" : 11,
    "60%" : 12,
    "65%" : 13,
    "70%" : 14,
    "75%" : 15,
    "80%" : 16,
    "85%" : 17,
    "90%" : 18,
    "95%" : 19,
    "100%" : 20,
}
def lookup_encode_GARMIN_BACKLIGHT_LEVEL(value):
    result = lookup_dict_encode_GARMIN_BACKLIGHT_LEVEL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from GARMIN_BACKLIGHT_LEVEL")
    return result

lookup_dict_encode_SEATALK_PILOT_MODE_16 = {
    "Standby" : 0,
    "Auto, compass commanded" : 64,
    "Vane, Wind Mode" : 256,
    "Track Mode" : 384,
    "No Drift, COG referenced (In track, course changes)" : 385,
}
def lookup_encode_SEATALK_PILOT_MODE_16(value):
    result = lookup_dict_encode_SEATALK_PILOT_MODE_16.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SEATALK_PILOT_MODE_16")
    return result

lookup_dict_encode_STATION_HEALTH = {
    "Not Working" : 0,
    "Unmonitored" : 1,
    "Healthy Operational" : 2,
    "Healthy Test Mode" : 3,
    "Test Mode" : 4,
}
def lookup_encode_STATION_HEALTH(value):
    result = lookup_dict_encode_STATION_HEALTH.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from STATION_HEALTH")
    return result

lookup_dict_encode_SERIAL_BIT_RATE = {
    "25" : 0,
    "50" : 1,
    "100" : 2,
    "200" : 3,
    "300" : 4,
    "600" : 5,
    "1200" : 6,
    "2400" : 7,
    "4800" : 8,
    "9600" : 9,
    "19200" : 10,
    "38400" : 11,
    "57600" : 12,
}
def lookup_encode_SERIAL_BIT_RATE(value):
    result = lookup_dict_encode_SERIAL_BIT_RATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SERIAL_BIT_RATE")
    return result

lookup_dict_encode_SERIAL_DETECTION_MODE = {
    "Auto bit rate" : 0,
    "Manual bit rate" : 1,
}
def lookup_encode_SERIAL_DETECTION_MODE(value):
    result = lookup_dict_encode_SERIAL_DETECTION_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SERIAL_DETECTION_MODE")
    return result

lookup_dict_encode_DIFFERENTIAL_SOURCE = {
    "Auto" : 0,
    "Loran" : 1,
    "MSK Beacon" : 2,
    "FM Subcarrier" : 3,
    "AIS" : 4,
    "Ground based radio" : 5,
    "SBAS" : 6,
    "Satellite" : 7,
}
def lookup_encode_DIFFERENTIAL_SOURCE(value):
    result = lookup_dict_encode_DIFFERENTIAL_SOURCE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DIFFERENTIAL_SOURCE")
    return result

lookup_dict_encode_DIFFERENTIAL_MODE = {
    "Manual" : 0,
    "Auto Power" : 1,
    "Auto Range" : 2,
}
def lookup_encode_DIFFERENTIAL_MODE(value):
    result = lookup_dict_encode_DIFFERENTIAL_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from DIFFERENTIAL_MODE")
    return result

lookup_dict_encode_WP_POSITION_RESOLUTION = {
    "more than 0.1 min" : 0,
    "<0.01 .. 0.1] min" : 1,
    "<0.001 .. 0.01] min" : 2,
    "<0.0001 .. 0.001] min" : 3,
    "<0 .. 0.0001] min" : 4,
}
def lookup_encode_WP_POSITION_RESOLUTION(value):
    result = lookup_dict_encode_WP_POSITION_RESOLUTION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WP_POSITION_RESOLUTION")
    return result

lookup_dict_encode_WP_IDENTIFICATION_METHOD = {
    "Waypoints in WP list" : 0,
    "Waypoints embedded in route" : 1,
}
def lookup_encode_WP_IDENTIFICATION_METHOD(value):
    result = lookup_dict_encode_WP_IDENTIFICATION_METHOD.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WP_IDENTIFICATION_METHOD")
    return result

lookup_dict_encode_WP_ROUTE_STATUS = {
    "Active" : 0,
    "Inactive" : 1,
    "Deleted" : 2,
}
def lookup_encode_WP_ROUTE_STATUS(value):
    result = lookup_dict_encode_WP_ROUTE_STATUS.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WP_ROUTE_STATUS")
    return result

lookup_dict_encode_WP_NAVIGATION_METHOD = {
    "Great Circle" : 0,
    "Rhumb Line" : 1,
}
def lookup_encode_WP_NAVIGATION_METHOD(value):
    result = lookup_dict_encode_WP_NAVIGATION_METHOD.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from WP_NAVIGATION_METHOD")
    return result

lookup_dict_encode_INVERTER_MODE = {
    "Standalone" : 0,
    "Series Master" : 1,
    "Series Slave" : 2,
    "Parallel Master" : 3,
    "Parallel Slave" : 4,
}
def lookup_encode_INVERTER_MODE(value):
    result = lookup_dict_encode_INVERTER_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from INVERTER_MODE")
    return result

lookup_dict_encode_CERTIFICATION_LEVEL = {
    "Level A" : 0,
    "Level B" : 1,
}
def lookup_encode_CERTIFICATION_LEVEL(value):
    result = lookup_dict_encode_CERTIFICATION_LEVEL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from CERTIFICATION_LEVEL")
    return result

lookup_dict_encode_AGS_MODE = {
    "Off" : 0,
    "On" : 1,
    "Automatic" : 2,
}
def lookup_encode_AGS_MODE(value):
    result = lookup_dict_encode_AGS_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AGS_MODE")
    return result

lookup_dict_encode_AGS_OPERATING_STATE = {
    "Quiet time" : 0,
    "Auto on" : 1,
    "Auto off" : 2,
    "Manual On" : 3,
    "Manual Off" : 4,
    "Generator shutdown" : 5,
    "External shutdown" : 6,
    "Fault" : 7,
    "Suspend" : 8,
    "Not operating" : 9,
}
def lookup_encode_AGS_OPERATING_STATE(value):
    result = lookup_dict_encode_AGS_OPERATING_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AGS_OPERATING_STATE")
    return result

lookup_dict_encode_AGS_GENERATING_STATE = {
    "Preheating" : 0,
    "Start delay" : 1,
    "Cranking" : 2,
    "Starter cooling" : 3,
    "Warming up" : 4,
    "Cooling down" : 5,
    "Spinning up" : 6,
    "Shutdown bypass" : 7,
    "Stopping" : 8,
    "Running" : 9,
    "Stopped" : 10,
    "Crank delaty" : 11,
}
def lookup_encode_AGS_GENERATING_STATE(value):
    result = lookup_dict_encode_AGS_GENERATING_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AGS_GENERATING_STATE")
    return result

lookup_dict_encode_AGS_ON_REASON = {
    "Not on" : 0,
    "DC voltage low" : 1,
    "Battery state of charge low" : 2,
    "AC current high" : 3,
    "Contact closed" : 4,
    "Manual on" : 5,
    "Exercise" : 6,
    "Non Quiet time" : 7,
    "External on via AGS" : 8,
    "External on via generator" : 9,
    "Unable to stop" : 10,
}
def lookup_encode_AGS_ON_REASON(value):
    result = lookup_dict_encode_AGS_ON_REASON.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AGS_ON_REASON")
    return result

lookup_dict_encode_AGS_OFF_REASON = {
    "Not off" : 0,
    "DC voltage high" : 1,
    "Battery state of charge high" : 2,
    "AC current low" : 3,
    "Contact opened" : 4,
    "Reached absorption" : 5,
    "Reached float" : 6,
    "Manual off" : 7,
    "Max run time" : 8,
    "Max auto cycle" : 9,
    "Exercise done" : 10,
    "Quiet time" : 11,
    "External off via AGS" : 12,
    "Safe mode" : 13,
    "External off via generator" : 14,
    "External shutdown" : 15,
    "Auto off" : 16,
    "Fault" : 17,
    "Unable to start" : 18,
}
def lookup_encode_AGS_OFF_REASON(value):
    result = lookup_dict_encode_AGS_OFF_REASON.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AGS_OFF_REASON")
    return result

lookup_dict_encode_TELEPHONE_MODE = {
    "F3E/G3E simplex, telephone" : 0,
    "F3E/G3E duplex, telephone" : 1,
    "J3E, telephone" : 2,
    "H3E, telephone" : 3,
    "F1B/J2B FEC NBDP, telex/teleprinter" : 4,
    "F1B/J2B ARQ NBDP, telex/teleprinter" : 5,
    "F1B/J2B receive only, teleprinter/DSC" : 6,
    "F1B/J2B, teleprinter/DSC" : 7,
    "A1A Morse, tape recorder" : 8,
    "A1A Morse, Morse key/head set" : 9,
    "F1C/F2C/F3C, FAX machine" : 10,
}
def lookup_encode_TELEPHONE_MODE(value):
    result = lookup_dict_encode_TELEPHONE_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from TELEPHONE_MODE")
    return result

lookup_dict_encode_POWER_MODE = {
    "High" : 0,
    "Low" : 1,
}
def lookup_encode_POWER_MODE(value):
    result = lookup_dict_encode_POWER_MODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from POWER_MODE")
    return result

lookup_dict_encode_BROADCAST_INDICATOR = {
    "Broadcast geo area message" : 0,
    "Addressed message" : 1,
}
def lookup_encode_BROADCAST_INDICATOR(value):
    result = lookup_dict_encode_BROADCAST_INDICATOR.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BROADCAST_INDICATOR")
    return result

lookup_dict_encode_BANDWIDTH = {
    "Default" : 0,
    "12.5 kHz" : 1,
}
def lookup_encode_BANDWIDTH(value):
    result = lookup_dict_encode_BANDWIDTH.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from BANDWIDTH")
    return result

lookup_dict_encode_FLOOD_STATE = {
    "Flood" : 0,
    "Slack" : 1,
    "Ebb" : 2,
}
def lookup_encode_FLOOD_STATE(value):
    result = lookup_dict_encode_FLOOD_STATE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from FLOOD_STATE")
    return result

lookup_dict_encode_AC_LINE = {
    "Line 1" : 0,
    "Line 2" : 1,
    "Line 3" : 2,
}
def lookup_encode_AC_LINE(value):
    result = lookup_dict_encode_AC_LINE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AC_LINE")
    return result

lookup_dict_encode_ZONE_SIZE = {
    "1 nm" : 0,
    "2 nm" : 1,
    "3 nm" : 2,
    "4 nm" : 3,
    "5 nm" : 4,
    "6 nm" : 5,
}
def lookup_encode_ZONE_SIZE(value):
    result = lookup_dict_encode_ZONE_SIZE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from ZONE_SIZE")
    return result

lookup_dict_encode_MARETRON_PRODUCT_CODE = {
    "SSC200" : 434,
    "SSC300" : 2686,
    "TLA100" : 2781,
    "NBE100" : 3979,
    "RIM100" : 4078,
    "ALM100" : 8165,
    "TMP100" : 20067,
    "DCR100" : 22585,
    "SIM100" : 23603,
    "ACM100" : 26493,
}
def lookup_encode_MARETRON_PRODUCT_CODE(value):
    result = lookup_dict_encode_MARETRON_PRODUCT_CODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MARETRON_PRODUCT_CODE")
    return result

lookup_dict_encode_MARETRON_OPCODE = {
    "Read All" : 0,
    "Write Register" : 1,
    "Read Config" : 2,
    "Write Config" : 3,
    "Calibrate" : 4,
    "Clear Calibration" : 5,
    "Status" : 6,
    "Clear Status" : 7,
    "Reset Factory Default" : 8,
    "Debug" : 9,
    "Write Instance" : 16,
    "Read Instance" : 17,
    "Write Label" : 32,
    "Read Label" : 33,
    "Write Switch Config" : 48,
    "Read Switch Config" : 49,
    "Write Alert Config" : 64,
    "Read Alert Config" : 65,
    "Write Channel Config" : 80,
    "Read Channel Config" : 81,
    "Read Channel Config Extended" : 86,
    "Write Channel Config Extended" : 87,
}
def lookup_encode_MARETRON_OPCODE(value):
    result = lookup_dict_encode_MARETRON_OPCODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MARETRON_OPCODE")
    return result

lookup_dict_encode_MARETRON_SOFTWARE_CODE = {
    "Version 1" : 1,
}
def lookup_encode_MARETRON_SOFTWARE_CODE(value):
    result = lookup_dict_encode_MARETRON_SOFTWARE_CODE.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MARETRON_SOFTWARE_CODE")
    return result

lookup_dict_encode_MARETRON_COMMAND = {
    "Deviation calibration" : 80,
}
def lookup_encode_MARETRON_COMMAND(value):
    result = lookup_dict_encode_MARETRON_COMMAND.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MARETRON_COMMAND")
    return result

lookup_dict_encode_MARETRON_STATUS_DEVIATION = {
    "Started" : 1,
    "Completed successfully" : 2,
    "Failed to complete" : 3,
    "Turning too fast" : 4,
    "Turning too slow" : 5,
    "Invalid movement" : 6,
}
def lookup_encode_MARETRON_STATUS_DEVIATION(value):
    result = lookup_dict_encode_MARETRON_STATUS_DEVIATION.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from MARETRON_STATUS_DEVIATION")
    return result

lookup_dict_encode_AUTOMATIC_MANUAL = {
    "Automatic" : 0,
    "Manual" : 1,
}
def lookup_encode_AUTOMATIC_MANUAL(value):
    result = lookup_dict_encode_AUTOMATIC_MANUAL.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from AUTOMATIC_MANUAL")
    return result

lookup_dict_encode_SBAS_SV = {
    "120" : 0,
    "121" : 1,
    "122" : 2,
    "123" : 3,
    "124" : 4,
    "125" : 5,
    "126" : 6,
    "127" : 7,
    "128" : 8,
    "129" : 9,
    "130" : 10,
    "131" : 11,
    "132" : 12,
    "133" : 13,
    "134" : 14,
    "135" : 15,
    "136" : 16,
    "137" : 17,
    "138" : 18,
}
def lookup_encode_SBAS_SV(value):
    result = lookup_dict_encode_SBAS_SV.get(value, None)
    if result is None:
        raise ValueError(f"Cant encode this message, {value} is missing from SBAS_SV")
    return result




lookup_field_type_dict_SIMNET_KEY_VALUE = {
    0: LookupFieldTypeEnumeration("Heading Offset", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    41: LookupFieldTypeEnumeration("Timezone offset", FieldTypes.DURATION, 60, "s", 16, None),
    260: LookupFieldTypeEnumeration("True wind high", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    264: LookupFieldTypeEnumeration("Deep water", FieldTypes.NUMBER, 0.01, "m", 16, None),
    516: LookupFieldTypeEnumeration("True wind low", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    517: LookupFieldTypeEnumeration("Low boat speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    520: LookupFieldTypeEnumeration("Shallow water", FieldTypes.NUMBER, 0.01, "m", 16, None),
    768: LookupFieldTypeEnumeration("Local field", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    1024: LookupFieldTypeEnumeration("Field angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    1800: LookupFieldTypeEnumeration("Anchor depth", FieldTypes.NUMBER, 0.01, "m", 16, None),
    4863: LookupFieldTypeEnumeration("Backlight level", FieldTypes.LOOKUP, None, None, 8, "SIMNET_BACKLIGHT_LEVEL"),
    5160: LookupFieldTypeEnumeration("Time format", FieldTypes.LOOKUP, None, None, 8, "SIMNET_TIME_FORMAT"),
    5161: LookupFieldTypeEnumeration("Time hour display", FieldTypes.LOOKUP, None, None, 8, "SIMNET_HOUR_DISPLAY"),
    9983: LookupFieldTypeEnumeration("Night mode", FieldTypes.LOOKUP, None, None, 8, "SIMNET_NIGHT_MODE"),
    11524: LookupFieldTypeEnumeration("True wind shift", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    22296: LookupFieldTypeEnumeration("AP low boat speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    32789: LookupFieldTypeEnumeration("Alert bits", FieldTypes.BITLOOKUP, None, None, 16, None),
    44079: LookupFieldTypeEnumeration("Night mode color", FieldTypes.LOOKUP, None, None, 8, "SIMNET_NIGHT_MODE_COLOR"),
    55087: LookupFieldTypeEnumeration("Day mode invert", FieldTypes.NUMBER, None, None, 8, None),
}
def lookup_field_type_SIMNET_KEY_VALUE(value):
    return lookup_field_type_dict_SIMNET_KEY_VALUE.get(value)

lookup_field_type_dict_BANDG_KEY_VALUE = {
    0: LookupFieldTypeEnumeration("Altitude", FieldTypes.NUMBER, None, None, 16, None),
    11: LookupFieldTypeEnumeration("Rudder Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    16: LookupFieldTypeEnumeration("User 5", FieldTypes.NUMBER, 0.01, None, 32, None),
    17: LookupFieldTypeEnumeration("User 6", FieldTypes.NUMBER, 0.01, None, 32, None),
    18: LookupFieldTypeEnumeration("User 7", FieldTypes.NUMBER, 0.01, None, 32, None),
    19: LookupFieldTypeEnumeration("User 8", FieldTypes.NUMBER, 0.01, None, 32, None),
    20: LookupFieldTypeEnumeration("User 9", FieldTypes.NUMBER, 0.01, None, 32, None),
    21: LookupFieldTypeEnumeration("User 10", FieldTypes.NUMBER, 0.01, None, 32, None),
    22: LookupFieldTypeEnumeration("User 11", FieldTypes.NUMBER, 0.01, None, 32, None),
    23: LookupFieldTypeEnumeration("User 12", FieldTypes.NUMBER, 0.01, None, 32, None),
    24: LookupFieldTypeEnumeration("User 13", FieldTypes.NUMBER, 0.01, None, 32, None),
    25: LookupFieldTypeEnumeration("User 14", FieldTypes.NUMBER, 0.01, None, 32, None),
    26: LookupFieldTypeEnumeration("User 15", FieldTypes.NUMBER, 0.01, None, 32, None),
    27: LookupFieldTypeEnumeration("User 16", FieldTypes.NUMBER, 0.01, None, 32, None),
    28: LookupFieldTypeEnumeration("Outside Temperature", FieldTypes.NUMBER, 0.01, "K", 16, None),
    29: LookupFieldTypeEnumeration("Outside Temperature", FieldTypes.NUMBER, 0.01, "K", 16, None),
    30: LookupFieldTypeEnumeration("Water Temperature", FieldTypes.NUMBER, 0.01, "K", 16, None),
    31: LookupFieldTypeEnumeration("Water Temperature", FieldTypes.NUMBER, 0.01, "K", 16, None),
    50: LookupFieldTypeEnumeration("Tacking Performance", FieldTypes.NUMBER, 0.1, "%", 16, None),
    52: LookupFieldTypeEnumeration("Attitude Roll", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    53: LookupFieldTypeEnumeration("Optimum Wind Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    56: LookupFieldTypeEnumeration("User 1", FieldTypes.NUMBER, 0.01, None, 32, None),
    57: LookupFieldTypeEnumeration("User 2", FieldTypes.NUMBER, 0.01, None, 32, None),
    58: LookupFieldTypeEnumeration("User 3", FieldTypes.NUMBER, 0.01, None, 32, None),
    59: LookupFieldTypeEnumeration("User 4", FieldTypes.NUMBER, 0.01, None, 32, None),
    60: LookupFieldTypeEnumeration("Roll Rate", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    64: LookupFieldTypeEnumeration("Forestay", FieldTypes.NUMBER, 0.001, None, 32, None),
    65: LookupFieldTypeEnumeration("Water Speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    77: LookupFieldTypeEnumeration("Wind Speed Apparent", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    79: LookupFieldTypeEnumeration("Wind Speed Apparent", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    80: LookupFieldTypeEnumeration("Average True Wind Direction", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    81: LookupFieldTypeEnumeration("Wind Angle Apparent", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    83: LookupFieldTypeEnumeration("Target TWA", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    85: LookupFieldTypeEnumeration("Wind Speed True", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    86: LookupFieldTypeEnumeration("Wind Speed True", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    89: LookupFieldTypeEnumeration("Wind Angle True", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    100: LookupFieldTypeEnumeration("Unknown", FieldTypes.NUMBER, None, None, 16, None),
    102: LookupFieldTypeEnumeration("Keel Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    103: LookupFieldTypeEnumeration("Canard Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    104: LookupFieldTypeEnumeration("Keel Trim Tab Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    105: LookupFieldTypeEnumeration("Course", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    109: LookupFieldTypeEnumeration("Wind Direction", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    111: LookupFieldTypeEnumeration("Next Leg AWA", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    113: LookupFieldTypeEnumeration("Next Leg AWS", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    117: LookupFieldTypeEnumeration("Race Timer", FieldTypes.DURATION, 0.001, "s", 32, None),
    124: LookupFieldTypeEnumeration("Polar Performance", FieldTypes.NUMBER, 0.1, "%", 16, None),
    125: LookupFieldTypeEnumeration("Target Boat Speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    126: LookupFieldTypeEnumeration("Polar Speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    127: LookupFieldTypeEnumeration("VMG to Wind", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    129: LookupFieldTypeEnumeration("DR Distance", FieldTypes.NUMBER, 0.01, "m", 32, None),
    130: LookupFieldTypeEnumeration("Leeway Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    131: LookupFieldTypeEnumeration("Current Drift", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    132: LookupFieldTypeEnumeration("Current Set", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    135: LookupFieldTypeEnumeration("Barometric Pressure", FieldTypes.NUMBER, 100, "Pa", 16, None),
    152: LookupFieldTypeEnumeration("Distance to Start Line", FieldTypes.NUMBER, 0.01, "m", 32, None),
    154: LookupFieldTypeEnumeration("Heading on Opposite Tack", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    155: LookupFieldTypeEnumeration("Attitude Pitch", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    156: LookupFieldTypeEnumeration("Mast Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    157: LookupFieldTypeEnumeration("Wind Angle to Mast", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    158: LookupFieldTypeEnumeration("Pitch Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    163: LookupFieldTypeEnumeration("Daggerboard Position", FieldTypes.NUMBER, None, None, 16, None),
    164: LookupFieldTypeEnumeration("Boom Position", FieldTypes.NUMBER, None, None, 16, None),
    185: LookupFieldTypeEnumeration("MOB DR Bearing", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    186: LookupFieldTypeEnumeration("MOB DR Range", FieldTypes.NUMBER, 0.01, "m", 32, None),
    194: LookupFieldTypeEnumeration("Depth", FieldTypes.NUMBER, 0.01, "m", 32, None),
    195: LookupFieldTypeEnumeration("Depth", FieldTypes.NUMBER, 0.01, "m", 32, None),
    199: LookupFieldTypeEnumeration("Aft Depth", FieldTypes.NUMBER, 0.01, "m", 32, None),
    205: LookupFieldTypeEnumeration("Odometer", FieldTypes.NUMBER, 0.01, "m", 32, None),
    207: LookupFieldTypeEnumeration("Trip Distance", FieldTypes.NUMBER, 0.01, "m", 32, None),
    211: LookupFieldTypeEnumeration("DR Bearing", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    233: LookupFieldTypeEnumeration("Course Over Ground", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    235: LookupFieldTypeEnumeration("Speed Over Ground", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    239: LookupFieldTypeEnumeration("Remote 0", FieldTypes.NUMBER, 0.001, None, 16, None),
    240: LookupFieldTypeEnumeration("Remote 1", FieldTypes.NUMBER, 0.001, None, 16, None),
    241: LookupFieldTypeEnumeration("Remote 2", FieldTypes.NUMBER, 0.001, None, 16, None),
    242: LookupFieldTypeEnumeration("Remote 3", FieldTypes.NUMBER, 0.001, None, 16, None),
    243: LookupFieldTypeEnumeration("Remote 4", FieldTypes.NUMBER, 0.001, None, 16, None),
    244: LookupFieldTypeEnumeration("Remote 5", FieldTypes.NUMBER, 0.001, None, 16, None),
    245: LookupFieldTypeEnumeration("Remote 6", FieldTypes.NUMBER, 0.001, None, 16, None),
    246: LookupFieldTypeEnumeration("Remote 7", FieldTypes.NUMBER, 0.001, None, 16, None),
    247: LookupFieldTypeEnumeration("Remote 8", FieldTypes.NUMBER, 0.001, None, 16, None),
    248: LookupFieldTypeEnumeration("Remote 9", FieldTypes.NUMBER, 0.001, None, 16, None),
    256: LookupFieldTypeEnumeration("Layline Time", FieldTypes.DURATION, 0.001, "s", 32, None),
    258: LookupFieldTypeEnumeration("Layline Distance", FieldTypes.NUMBER, 0.01, "m", 32, None),
    259: LookupFieldTypeEnumeration("Layline Distance", FieldTypes.NUMBER, 0.01, "m", 32, None),
    260: LookupFieldTypeEnumeration("Sailing Time to Waypoint", FieldTypes.DURATION, 0.001, "s", 32, None),
    261: LookupFieldTypeEnumeration("Sailing Distance to Waypoint", FieldTypes.NUMBER, 0.01, "m", 32, None),
    262: LookupFieldTypeEnumeration("Sailing ETA", FieldTypes.DURATION, 0.001, "s", 32, None),
    265: LookupFieldTypeEnumeration("Trip Time", FieldTypes.DURATION, 0.001, "s", 32, None),
    270: LookupFieldTypeEnumeration("Bow Latitude", FieldTypes.NUMBER, 1e-07, "deg", 32, None),
    271: LookupFieldTypeEnumeration("Bow Longitude", FieldTypes.NUMBER, 1e-07, "deg", 32, None),
    272: LookupFieldTypeEnumeration("Start Line Bearing", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    273: LookupFieldTypeEnumeration("Start Line Bias", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    274: LookupFieldTypeEnumeration("Distance to Start Line Port", FieldTypes.NUMBER, 0.01, "m", 32, None),
    275: LookupFieldTypeEnumeration("Distance to Start Line Starboard", FieldTypes.NUMBER, 0.01, "m", 32, None),
    280: LookupFieldTypeEnumeration("Bias Advantage in Boat Lengths", FieldTypes.NUMBER, 0.1, None, 16, None),
    281: LookupFieldTypeEnumeration("Distance to Start Line in Boat Lengths", FieldTypes.NUMBER, 0.1, None, 16, None),
    282: LookupFieldTypeEnumeration("Backstay", FieldTypes.NUMBER, 0.001, None, 32, None),
    283: LookupFieldTypeEnumeration("Boom Vang", FieldTypes.NUMBER, 0.001, None, 32, None),
    284: LookupFieldTypeEnumeration("Chain Length", FieldTypes.NUMBER, 0.01, "m", 32, None),
    285: LookupFieldTypeEnumeration("VMG Performance", FieldTypes.NUMBER, 0.1, "%", 16, None),
    286: LookupFieldTypeEnumeration("Inner Forestay Load", FieldTypes.NUMBER, 0.001, None, 32, None),
    287: LookupFieldTypeEnumeration("Inner Forestay Halyard Load", FieldTypes.NUMBER, 0.001, None, 32, None),
    288: LookupFieldTypeEnumeration("Jib Furl", FieldTypes.NUMBER, 0.01, "m", 32, None),
    289: LookupFieldTypeEnumeration("Jib Halyard Load", FieldTypes.NUMBER, 0.001, None, 32, None),
    290: LookupFieldTypeEnumeration("Outhaul Load", FieldTypes.NUMBER, 0.001, None, 32, None),
    291: LookupFieldTypeEnumeration("Plow Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    292: LookupFieldTypeEnumeration("Cunningham", FieldTypes.NUMBER, 0.001, None, 32, None),
    293: LookupFieldTypeEnumeration("Jacuzzi Temperature", FieldTypes.NUMBER, 0.01, "K", 16, None),
    294: LookupFieldTypeEnumeration("Pool Temperature", FieldTypes.NUMBER, 0.01, "K", 16, None),
    296: LookupFieldTypeEnumeration("Keel Draught", FieldTypes.NUMBER, 0.01, "m", 16, None),
    297: LookupFieldTypeEnumeration("Boom Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    298: LookupFieldTypeEnumeration("Code Zero Load", FieldTypes.NUMBER, 0.001, None, 32, None),
    301: LookupFieldTypeEnumeration("Distance Behind Start Line", FieldTypes.NUMBER, 0.01, "m", 32, None),
    302: LookupFieldTypeEnumeration("Distance Behind Start Line in Boat Lengths", FieldTypes.NUMBER, 0.1, None, 16, None),
    305: LookupFieldTypeEnumeration("Bias Advantage", FieldTypes.NUMBER, 0.01, "m", 32, None),
    306: LookupFieldTypeEnumeration("Opposite Tack COG", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    307: LookupFieldTypeEnumeration("Opposite Tack Target Heading", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    308: LookupFieldTypeEnumeration("Mast Rake", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    309: LookupFieldTypeEnumeration("Next Leg Bearing", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    310: LookupFieldTypeEnumeration("Next Leg Target Speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    311: LookupFieldTypeEnumeration("Ground Wind Direction", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    312: LookupFieldTypeEnumeration("Ground Wind Speed", FieldTypes.NUMBER, 0.01, "m/s", 16, None),
    313: LookupFieldTypeEnumeration("Mast Cant Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    314: LookupFieldTypeEnumeration("Rudder Toe In", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    315: LookupFieldTypeEnumeration("Daggerboard Port", FieldTypes.NUMBER, None, None, 16, None),
    316: LookupFieldTypeEnumeration("Daggerboard Starboard", FieldTypes.NUMBER, None, None, 16, None),
    317: LookupFieldTypeEnumeration("User 17", FieldTypes.NUMBER, 0.01, None, 32, None),
    318: LookupFieldTypeEnumeration("User 18", FieldTypes.NUMBER, 0.01, None, 32, None),
    319: LookupFieldTypeEnumeration("User 19", FieldTypes.NUMBER, 0.01, None, 32, None),
    320: LookupFieldTypeEnumeration("User 20", FieldTypes.NUMBER, 0.01, None, 32, None),
    321: LookupFieldTypeEnumeration("User 21", FieldTypes.NUMBER, 0.01, None, 32, None),
    322: LookupFieldTypeEnumeration("User 22", FieldTypes.NUMBER, 0.01, None, 32, None),
    323: LookupFieldTypeEnumeration("User 23", FieldTypes.NUMBER, 0.01, None, 32, None),
    324: LookupFieldTypeEnumeration("User 24", FieldTypes.NUMBER, 0.01, None, 32, None),
    325: LookupFieldTypeEnumeration("User 25", FieldTypes.NUMBER, 0.01, None, 32, None),
    326: LookupFieldTypeEnumeration("User 26", FieldTypes.NUMBER, 0.01, None, 32, None),
    327: LookupFieldTypeEnumeration("User 27", FieldTypes.NUMBER, 0.01, None, 32, None),
    328: LookupFieldTypeEnumeration("User 28", FieldTypes.NUMBER, 0.01, None, 32, None),
    329: LookupFieldTypeEnumeration("User 29", FieldTypes.NUMBER, 0.01, None, 32, None),
    330: LookupFieldTypeEnumeration("User 30", FieldTypes.NUMBER, 0.01, None, 32, None),
    331: LookupFieldTypeEnumeration("User 31", FieldTypes.NUMBER, 0.01, None, 32, None),
    332: LookupFieldTypeEnumeration("User 32", FieldTypes.NUMBER, 0.01, None, 32, None),
    336: LookupFieldTypeEnumeration("Average True Wind Direction", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    337: LookupFieldTypeEnumeration("Wind Phase", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    338: LookupFieldTypeEnumeration("Wind Lift", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    380: LookupFieldTypeEnumeration("Active Perf Mode", FieldTypes.NUMBER, None, None, 16, None),
    381: LookupFieldTypeEnumeration("Gust Bear Away", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    382: LookupFieldTypeEnumeration("TWS Bear Away", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    383: LookupFieldTypeEnumeration("Heel Compensation", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    384: LookupFieldTypeEnumeration("Pilot Net Course", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    385: LookupFieldTypeEnumeration("Pilot Target Wind Angle", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    386: LookupFieldTypeEnumeration("Pilot Weather Helm", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
    387: LookupFieldTypeEnumeration("Pilot Mean Heel", FieldTypes.NUMBER, 0.0001, "rad", 16, None),
}
def lookup_field_type_BANDG_KEY_VALUE(value):
    return lookup_field_type_dict_BANDG_KEY_VALUE.get(value)

__all__ = [name for name in globals() if not name.startswith("__")]
