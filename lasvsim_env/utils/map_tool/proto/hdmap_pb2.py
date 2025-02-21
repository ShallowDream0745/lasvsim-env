# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hdmap.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bhdmap.proto\x12\x05hdmap\"\xb9\x03\n\x05HdMap\x12\x1d\n\x06header\x18\x01 \x01(\x0b\x32\r.hdmap.Header\x12\"\n\tjunctions\x18\x02 \x03(\x0b\x32\x0f.hdmap.Junction\x12 \n\x08segments\x18\x03 \x03(\x0b\x32\x0e.hdmap.Segment\x12\x1a\n\x05links\x18\x04 \x03(\x0b\x32\x0b.hdmap.Link\x12\x1a\n\x05lanes\x18\x05 \x03(\x0b\x32\x0b.hdmap.Lane\x12%\n\ncrosswalks\x18\xe8\x07 \x03(\x0b\x32\x10.hdmap.Crosswalk\x12#\n\tstoplines\x18\xe9\x07 \x03(\x0b\x32\x0f.hdmap.Stopline\x12,\n\x0etraffic_lights\x18\xea\x07 \x03(\x0b\x32\x13.hdmap.TrafficLight\x12*\n\rtraffic_signs\x18\xeb\x07 \x03(\x0b\x32\x12.hdmap.TrafficSign\x12#\n\tmovements\x18\xec\x07 \x03(\x0b\x32\x0f.hdmap.Movement\x12\'\n\x0b\x63onnections\x18\xed\x07 \x03(\x0b\x32\x11.hdmap.Connection\x12\x1f\n\x07objects\x18\xee\x07 \x03(\x0b\x32\r.hdmap.Object\"\xbe\x0b\n\x0cHdTrafficMap\x12:\n\x0cjunction_map\x18\x01 \x03(\x0b\x32$.hdmap.HdTrafficMap.JunctionMapEntry\x12\x38\n\x0bsegment_map\x18\x02 \x03(\x0b\x32#.hdmap.HdTrafficMap.SegmentMapEntry\x12\x32\n\x08link_map\x18\x03 \x03(\x0b\x32 .hdmap.HdTrafficMap.LinkMapEntry\x12\x32\n\x08lane_map\x18\x04 \x03(\x0b\x32 .hdmap.HdTrafficMap.LaneMapEntry\x12=\n\rcrosswalk_map\x18\xe8\x07 \x03(\x0b\x32%.hdmap.HdTrafficMap.CrosswalkMapEntry\x12;\n\x0cstopline_map\x18\xe9\x07 \x03(\x0b\x32$.hdmap.HdTrafficMap.StoplineMapEntry\x12\x44\n\x11traffic_light_map\x18\xea\x07 \x03(\x0b\x32(.hdmap.HdTrafficMap.TrafficLightMapEntry\x12\x42\n\x10traffic_sign_map\x18\xeb\x07 \x03(\x0b\x32\'.hdmap.HdTrafficMap.TrafficSignMapEntry\x12;\n\x0cmovement_map\x18\xec\x07 \x03(\x0b\x32$.hdmap.HdTrafficMap.MovementMapEntry\x12?\n\x0e\x63onnection_map\x18\xed\x07 \x03(\x0b\x32&.hdmap.HdTrafficMap.ConnectionMapEntry\x12\x1e\n\x06header\x18\xee\x07 \x01(\x0b\x32\r.hdmap.Header\x12\x37\n\nobject_map\x18\xef\x07 \x03(\x0b\x32\".hdmap.HdTrafficMap.ObjectMapEntry\x1a\x43\n\x10JunctionMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1e\n\x05value\x18\x02 \x01(\x0b\x32\x0f.hdmap.Junction:\x02\x38\x01\x1a\x41\n\x0fSegmentMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.hdmap.Segment:\x02\x38\x01\x1a;\n\x0cLinkMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1a\n\x05value\x18\x02 \x01(\x0b\x32\x0b.hdmap.Link:\x02\x38\x01\x1a;\n\x0cLaneMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1a\n\x05value\x18\x02 \x01(\x0b\x32\x0b.hdmap.Lane:\x02\x38\x01\x1a\x45\n\x11\x43rosswalkMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1f\n\x05value\x18\x02 \x01(\x0b\x32\x10.hdmap.Crosswalk:\x02\x38\x01\x1a\x43\n\x10StoplineMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1e\n\x05value\x18\x02 \x01(\x0b\x32\x0f.hdmap.Stopline:\x02\x38\x01\x1aK\n\x14TrafficLightMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x13.hdmap.TrafficLight:\x02\x38\x01\x1aI\n\x13TrafficSignMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12!\n\x05value\x18\x02 \x01(\x0b\x32\x12.hdmap.TrafficSign:\x02\x38\x01\x1a\x43\n\x10MovementMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1e\n\x05value\x18\x02 \x01(\x0b\x32\x0f.hdmap.Movement:\x02\x38\x01\x1aG\n\x12\x43onnectionMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.hdmap.Connection:\x02\x38\x01\x1a?\n\x0eObjectMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1c\n\x05value\x18\x02 \x01(\x0b\x32\r.hdmap.Object:\x02\x38\x01\"\x97\x01\n\x06Header\x12\r\n\x05north\x18\x01 \x01(\x01\x12\r\n\x05south\x18\x02 \x01(\x01\x12\x0c\n\x04\x65\x61st\x18\x03 \x01(\x01\x12\x0c\n\x04west\x18\x04 \x01(\x01\x12\"\n\x0c\x63\x65nter_point\x18\x05 \x01(\x0b\x32\x0c.hdmap.Point\x12\x0f\n\x07version\x18\x06 \x01(\t\x12\x0c\n\x04zone\x18\x07 \x01(\x03\x12\x10\n\x08use_bias\x18\x08 \x01(\x08\"\x85\x01\n\x08Junction\x12\x13\n\x0bjunction_id\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x10\n\x08link_ids\x18\x05 \x03(\t\x12\x1b\n\x05shape\x18\x06 \x03(\x0b\x32\x0c.hdmap.Point\x12\x19\n\x10traffic_light_id\x18\xe8\x07 \x01(\t\"\xcc\x01\n\x07Segment\x12\x12\n\nsegment_id\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x18\n\x10ordered_link_ids\x18\x04 \x03(\t\x12\x19\n\x11start_junction_id\x18\x06 \x01(\t\x12\x17\n\x0f\x65nd_junction_id\x18\x07 \x01(\t\x12\x0e\n\x06length\x18\x08 \x01(\x01\x12\x0f\n\x07heading\x18\t \x01(\x01\x12\x10\n\x08s_offset\x18\n \x01(\x01\x12\x1e\n\x15traffic_light_pole_id\x18\xe8\x07 \x01(\t\"\x98\x03\n\x04Link\x12\x0f\n\x07link_id\x18\x02 \x01(\t\x12\x0f\n\x07pair_id\x18\x03 \x01(\t\x12\r\n\x05width\x18\x04 \x01(\x01\x12\x18\n\x10ordered_lane_ids\x18\x05 \x03(\t\x12\x10\n\x08lane_num\x18\x06 \x01(\x05\x12!\n\x0bstart_point\x18\x07 \x01(\x0b\x32\x0c.hdmap.Point\x12\x1f\n\tend_point\x18\x08 \x01(\x0b\x32\x0c.hdmap.Point\x12\x10\n\x08gradient\x18\t \x01(\x01\x12\x12\n\nsegment_id\x18\x0c \x01(\t\x12\x0e\n\x06length\x18\r \x01(\x01\x12\x0c\n\x04type\x18\x0e \x01(\t\x12\x0f\n\x07heading\x18\x0f \x01(\x01\x12\x13\n\x0bjunction_id\x18\x10 \x01(\t\x12\x11\n\troad_type\x18\x11 \x01(\t\x12\x10\n\x08s_offset\x18\x63 \x01(\x01\x12\x13\n\nlink_order\x18\xe8\x07 \x01(\x05\x12$\n\rleft_boundary\x18\xe9\x07 \x03(\x0b\x32\x0c.hdmap.Point\x12%\n\x0eright_boundary\x18\xea\x07 \x03(\x0b\x32\x0c.hdmap.Point\"\x86\x03\n\x04Lane\x12\x0f\n\x07lane_id\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x13\n\x0blane_offset\x18\x04 \x01(\x05\x12\x0f\n\x07link_id\x18\x05 \x01(\t\x12\x19\n\x04turn\x18\x06 \x01(\x0b\x32\x0b.hdmap.Turn\x12\x1c\n\x06speeds\x18\x07 \x03(\x0b\x32\x0c.hdmap.Speed\x12\x13\n\x0bstopline_id\x18\x08 \x01(\t\x12\r\n\x05width\x18\t \x01(\x01\x12\x0e\n\x06length\x18\n \x01(\x01\x12\"\n\x0b\x63\x65nter_line\x18\xe8\x07 \x03(\x0b\x32\x0c.hdmap.Point\x12\x1a\n\x11upstream_link_ids\x18\x87\x07 \x03(\t\x12\x1c\n\x13\x64ownstream_link_ids\x18\x88\x07 \x03(\t\x12\x19\n\x10\x63onnect_link_ids\x18\xeb\x07 \x03(\t\x12(\n\x0eleft_lane_mark\x18\xec\x07 \x01(\x0b\x32\x0f.hdmap.LaneMark\x12)\n\x0fright_lane_mark\x18\xed\x07 \x01(\x0b\x32\x0f.hdmap.LaneMark\"^\n\x08LaneMark\x12\x1c\n\x05shape\x18\xe8\x07 \x03(\x0b\x32\x0c.hdmap.Point\x12\x34\n\x0flane_mark_attrs\x18\xe9\x07 \x03(\x0b\x32\x1a.hdmap.LaneMarkAttribution\"\x88\x01\n\x13LaneMarkAttribution\x12\x0e\n\x06length\x18\x01 \x01(\x01\x12\t\n\x01s\x18\x02 \x01(\x01\x12\x13\n\x0bstart_index\x18\x03 \x01(\x05\x12\x11\n\tend_index\x18\x04 \x01(\x05\x12\x0e\n\x05style\x18\xe8\x07 \x01(\t\x12\x0e\n\x05\x63olor\x18\xe9\x07 \x01(\t\x12\x0e\n\x05width\x18\xea\x07 \x01(\x01\"z\n\x05Speed\x12\t\n\x01s\x18\x01 \x01(\x01\x12\x0e\n\x06length\x18\x02 \x01(\x01\x12\x13\n\x0bstart_index\x18\x03 \x01(\x05\x12\x11\n\tend_index\x18\x04 \x01(\x05\x12\x0e\n\x05value\x18\xe8\x07 \x01(\x01\x12\r\n\x04uint\x18\xe9\x07 \x01(\t\x12\x0f\n\x06source\x18\xea\x07 \x01(\t\"P\n\tCrosswalk\x12\x14\n\x0c\x63rosswalk_id\x18\x02 \x01(\t\x12\x0f\n\x07heading\x18\x03 \x01(\x01\x12\x1c\n\x05shape\x18\xe8\x07 \x03(\x0b\x32\x0c.hdmap.Point\"=\n\x08Stopline\x12\x13\n\x0bstopline_id\x18\x02 \x01(\t\x12\x1c\n\x05shape\x18\xe8\x07 \x03(\x0b\x32\x0c.hdmap.Point\"\xac\x01\n\x04Turn\x12.\n\x0f\x64irection_point\x18\x01 \x01(\x0b\x32\x15.hdmap.DirectionPoint\x12\x0c\n\x04turn\x18\x02 \x01(\t\x12\x32\n\x0cturn_mapping\x18\x03 \x03(\x0b\x32\x1c.hdmap.Turn.TurnMappingEntry\x1a\x32\n\x10TurnMappingEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"d\n\x0bTrafficSign\x12\x17\n\x0ftraffic_sign_id\x18\x02 \x01(\t\x12.\n\x0f\x64irection_point\x18\x03 \x01(\x0b\x32\x15.hdmap.DirectionPoint\x12\x0c\n\x04type\x18\x04 \x01(\t\"\xf4\x03\n\x0cTrafficLight\x12\x18\n\x10traffic_light_id\x18\x01 \x01(\t\x12\x13\n\x0bjunction_id\x18\x03 \x01(\t\x12\r\n\x05\x63ycle\x18\x07 \x01(\x05\x12\x0e\n\x06offset\x18\x08 \x01(\x05\x12=\n\x0csignal_group\x18\x64 \x03(\x0b\x32\'.hdmap.TrafficLight.MovementSignalGroup\x1a\xd2\x01\n\x0eMovementSignal\x12\x13\n\x0bmovement_id\x18\x01 \x01(\t\x12I\n\x0fsignal_of_green\x18\x64 \x03(\x0b\x32\x30.hdmap.TrafficLight.MovementSignal.SignalOfGreen\x1a`\n\rSignalOfGreen\x12\x13\n\x0bgreen_start\x18\x01 \x01(\x05\x12\x16\n\x0egreen_duration\x18\x02 \x01(\x05\x12\x0e\n\x06yellow\x18\x03 \x01(\x05\x12\x12\n\nread_clean\x18\x04 \x01(\x05\x1a\x81\x01\n\x13MovementSignalGroup\x12\x17\n\x0fsignal_group_id\x18\x01 \x01(\t\x12\x1d\n\x15traffic_light_pole_id\x18\x02 \x01(\t\x12\x32\n\x06signal\x18\x03 \x03(\x0b\x32\".hdmap.TrafficLight.MovementSignal\"\x82\x01\n\x08Movement\x12\x13\n\x0bmovement_id\x18\x01 \x01(\t\x12\x18\n\x10upstream_link_id\x18\x02 \x01(\t\x12\x1a\n\x12\x64ownstream_link_id\x18\x03 \x01(\t\x12\x13\n\x0bjunction_id\x18\x04 \x01(\t\x12\x16\n\x0e\x66low_direction\x18\x05 \x01(\t\"\xfb\x01\n\nConnection\x12\x15\n\rconnection_id\x18\x01 \x01(\t\x12\x13\n\x0bjunction_id\x18\x02 \x01(\t\x12\x13\n\x0bmovement_id\x18\x03 \x01(\t\x12\x18\n\x10upstream_lane_id\x18\x04 \x01(\t\x12\x1a\n\x12\x64ownstream_lane_id\x18\x05 \x01(\t\x12\x16\n\x0e\x66low_direction\x18\x06 \x01(\t\x12\x18\n\x10upstream_link_id\x18\x07 \x01(\t\x12\x1a\n\x12\x64ownstream_link_id\x18\x08 \x01(\t\x12\x0c\n\x04type\x18\t \x01(\t\x12\x1a\n\x04path\x18\x64 \x03(\x0b\x32\x0c.hdmap.Point\"a\n\x0e\x44irectionPoint\x12\x1b\n\x05point\x18\x01 \x01(\x0b\x32\x0c.hdmap.Point\x12\x10\n\x08pitching\x18\x02 \x01(\x01\x12\x0f\n\x07heading\x18\x03 \x01(\x01\x12\x0f\n\x07rolling\x18\x04 \x01(\x01\"(\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"F\n\x06Object\x12\x11\n\tobject_id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x1b\n\x05shape\x18\x03 \x03(\x0b\x32\x0c.hdmap.PointB?Z=git.risenlighten.com/lasvsim/map_engine/core/hdmap/v1;hdmapv1b\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'hdmap_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z=git.risenlighten.com/lasvsim/map_engine/core/hdmap/v1;hdmapv1'
  _HDTRAFFICMAP_JUNCTIONMAPENTRY._options = None
  _HDTRAFFICMAP_JUNCTIONMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_SEGMENTMAPENTRY._options = None
  _HDTRAFFICMAP_SEGMENTMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_LINKMAPENTRY._options = None
  _HDTRAFFICMAP_LINKMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_LANEMAPENTRY._options = None
  _HDTRAFFICMAP_LANEMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_CROSSWALKMAPENTRY._options = None
  _HDTRAFFICMAP_CROSSWALKMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_STOPLINEMAPENTRY._options = None
  _HDTRAFFICMAP_STOPLINEMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_TRAFFICLIGHTMAPENTRY._options = None
  _HDTRAFFICMAP_TRAFFICLIGHTMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_TRAFFICSIGNMAPENTRY._options = None
  _HDTRAFFICMAP_TRAFFICSIGNMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_MOVEMENTMAPENTRY._options = None
  _HDTRAFFICMAP_MOVEMENTMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_CONNECTIONMAPENTRY._options = None
  _HDTRAFFICMAP_CONNECTIONMAPENTRY._serialized_options = b'8\001'
  _HDTRAFFICMAP_OBJECTMAPENTRY._options = None
  _HDTRAFFICMAP_OBJECTMAPENTRY._serialized_options = b'8\001'
  _TURN_TURNMAPPINGENTRY._options = None
  _TURN_TURNMAPPINGENTRY._serialized_options = b'8\001'
  _HDMAP._serialized_start=23
  _HDMAP._serialized_end=464
  _HDTRAFFICMAP._serialized_start=467
  _HDTRAFFICMAP._serialized_end=1937
  _HDTRAFFICMAP_JUNCTIONMAPENTRY._serialized_start=1182
  _HDTRAFFICMAP_JUNCTIONMAPENTRY._serialized_end=1249
  _HDTRAFFICMAP_SEGMENTMAPENTRY._serialized_start=1251
  _HDTRAFFICMAP_SEGMENTMAPENTRY._serialized_end=1316
  _HDTRAFFICMAP_LINKMAPENTRY._serialized_start=1318
  _HDTRAFFICMAP_LINKMAPENTRY._serialized_end=1377
  _HDTRAFFICMAP_LANEMAPENTRY._serialized_start=1379
  _HDTRAFFICMAP_LANEMAPENTRY._serialized_end=1438
  _HDTRAFFICMAP_CROSSWALKMAPENTRY._serialized_start=1440
  _HDTRAFFICMAP_CROSSWALKMAPENTRY._serialized_end=1509
  _HDTRAFFICMAP_STOPLINEMAPENTRY._serialized_start=1511
  _HDTRAFFICMAP_STOPLINEMAPENTRY._serialized_end=1578
  _HDTRAFFICMAP_TRAFFICLIGHTMAPENTRY._serialized_start=1580
  _HDTRAFFICMAP_TRAFFICLIGHTMAPENTRY._serialized_end=1655
  _HDTRAFFICMAP_TRAFFICSIGNMAPENTRY._serialized_start=1657
  _HDTRAFFICMAP_TRAFFICSIGNMAPENTRY._serialized_end=1730
  _HDTRAFFICMAP_MOVEMENTMAPENTRY._serialized_start=1732
  _HDTRAFFICMAP_MOVEMENTMAPENTRY._serialized_end=1799
  _HDTRAFFICMAP_CONNECTIONMAPENTRY._serialized_start=1801
  _HDTRAFFICMAP_CONNECTIONMAPENTRY._serialized_end=1872
  _HDTRAFFICMAP_OBJECTMAPENTRY._serialized_start=1874
  _HDTRAFFICMAP_OBJECTMAPENTRY._serialized_end=1937
  _HEADER._serialized_start=1940
  _HEADER._serialized_end=2091
  _JUNCTION._serialized_start=2094
  _JUNCTION._serialized_end=2227
  _SEGMENT._serialized_start=2230
  _SEGMENT._serialized_end=2434
  _LINK._serialized_start=2437
  _LINK._serialized_end=2845
  _LANE._serialized_start=2848
  _LANE._serialized_end=3238
  _LANEMARK._serialized_start=3240
  _LANEMARK._serialized_end=3334
  _LANEMARKATTRIBUTION._serialized_start=3337
  _LANEMARKATTRIBUTION._serialized_end=3473
  _SPEED._serialized_start=3475
  _SPEED._serialized_end=3597
  _CROSSWALK._serialized_start=3599
  _CROSSWALK._serialized_end=3679
  _STOPLINE._serialized_start=3681
  _STOPLINE._serialized_end=3742
  _TURN._serialized_start=3745
  _TURN._serialized_end=3917
  _TURN_TURNMAPPINGENTRY._serialized_start=3867
  _TURN_TURNMAPPINGENTRY._serialized_end=3917
  _TRAFFICSIGN._serialized_start=3919
  _TRAFFICSIGN._serialized_end=4019
  _TRAFFICLIGHT._serialized_start=4022
  _TRAFFICLIGHT._serialized_end=4522
  _TRAFFICLIGHT_MOVEMENTSIGNAL._serialized_start=4180
  _TRAFFICLIGHT_MOVEMENTSIGNAL._serialized_end=4390
  _TRAFFICLIGHT_MOVEMENTSIGNAL_SIGNALOFGREEN._serialized_start=4294
  _TRAFFICLIGHT_MOVEMENTSIGNAL_SIGNALOFGREEN._serialized_end=4390
  _TRAFFICLIGHT_MOVEMENTSIGNALGROUP._serialized_start=4393
  _TRAFFICLIGHT_MOVEMENTSIGNALGROUP._serialized_end=4522
  _MOVEMENT._serialized_start=4525
  _MOVEMENT._serialized_end=4655
  _CONNECTION._serialized_start=4658
  _CONNECTION._serialized_end=4909
  _DIRECTIONPOINT._serialized_start=4911
  _DIRECTIONPOINT._serialized_end=5008
  _POINT._serialized_start=5010
  _POINT._serialized_end=5050
  _OBJECT._serialized_start=5052
  _OBJECT._serialized_end=5122
# @@protoc_insertion_point(module_scope)
