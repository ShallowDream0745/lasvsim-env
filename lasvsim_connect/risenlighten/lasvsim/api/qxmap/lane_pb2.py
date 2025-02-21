# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: api/v1/qxmap/lane.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lasvsim_connect.risenlighten.lasvsim.api.geometry import geometry_pb2 as api_dot_v1_dot_geometry_dot_geometry__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x61pi/v1/qxmap/lane.proto\x12)risenlighten.lasvsim.api.datahub.qxmap.v1\x1a\x1e\x61pi/v1/geometry/geometry.proto\"\x84\n\n\x04Lane\x12\n\n\x02id\x18\x02 \x01(\t\x12\x46\n\x04type\x18\x03 \x01(\x0e\x32\x38.risenlighten.lasvsim.api.datahub.qxmap.v1.Lane.LaneType\x12\x10\n\x08lane_num\x18\x04 \x01(\x05\x12\x0f\n\x07link_id\x18\x05 \x01(\t\x12K\n\tlane_turn\x18\x06 \x01(\x0b\x32\x33.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneTurnH\x00\x88\x01\x01\x12K\n\x0cspeed_limits\x18\x07 \x03(\x0b\x32\x35.risenlighten.lasvsim.api.datahub.qxmap.v1.SpeedLimit\x12J\n\x08stopline\x18\x08 \x01(\x0b\x32\x33.risenlighten.lasvsim.api.datahub.qxmap.v1.StoplineH\x01\x88\x01\x01\x12@\n\x06widths\x18\t \x03(\x0b\x32\x30.risenlighten.lasvsim.api.datahub.qxmap.v1.Width\x12N\n\x0b\x63\x65nter_line\x18\n \x03(\x0b\x32\x39.risenlighten.lasvsim.api.datahub.geometry.v1.CenterPoint\x12\x19\n\x11upstream_lane_ids\x18\x0b \x03(\t\x12\x1b\n\x13\x64ownstream_lane_ids\x18\x0c \x03(\t\x12\x0e\n\x06length\x18\r \x01(\x01\x12L\n\x0fleft_lane_marks\x18\x0e \x03(\x0b\x32\x33.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneMark\x12M\n\x10right_lane_marks\x18\x0f \x03(\x0b\x32\x33.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneMark\x12T\n\rleft_boundary\x18\x10 \x01(\x0b\x32\x38.risenlighten.lasvsim.api.datahub.geometry.v1.LineStringH\x02\x88\x01\x01\x12U\n\x0eright_boundary\x18\x11 \x01(\x0b\x32\x38.risenlighten.lasvsim.api.datahub.geometry.v1.LineStringH\x03\x88\x01\x01\x12\r\n\x05width\x18\x12 \x01(\x01\"\xab\x02\n\x08LaneType\x12\x15\n\x11LANE_TYPE_UNKNOWN\x10\x00\x12\x15\n\x11LANE_TYPE_DRIVING\x10\x01\x12\x14\n\x10LANE_TYPE_BIKING\x10\x02\x12\x16\n\x12LANE_TYPE_SIDEWALK\x10\x03\x12\x15\n\x11LANE_TYPE_PARKING\x10\x04\x12\x14\n\x10LANE_TYPE_BORDER\x10\x05\x12\x14\n\x10LANE_TYPE_MEDIAN\x10\x06\x12\x14\n\x10LANE_TYPE_BUSING\x10\x07\x12\x12\n\x0eLANE_TYPE_CURB\x10\x08\x12\x13\n\x0fLANE_TYPE_ENTRY\x10\n\x12\x12\n\x0eLANE_TYPE_EXIT\x10\x0b\x12\x15\n\x11LANE_TYPE_RAMP_IN\x10\x0c\x12\x16\n\x12LANE_TYPE_RAMP_OUT\x10\rB\x0c\n\n_lane_turnB\x0b\n\t_stoplineB\x10\n\x0e_left_boundaryB\x11\n\x0f_right_boundary\">\n\x05Width\x12\t\n\x01s\x18\x01 \x01(\x01\x12\t\n\x01\x61\x18\x02 \x01(\x01\x12\t\n\x01\x62\x18\x03 \x01(\x01\x12\t\n\x01\x63\x18\x04 \x01(\x01\x12\t\n\x01\x64\x18\x05 \x01(\x01\"\xbf\x05\n\x08LaneMark\x12\t\n\x01s\x18\x01 \x01(\x01\x12\x0e\n\x06length\x18\x02 \x01(\x01\x12\x10\n\x08is_merge\x18\x03 \x01(\x08\x12Q\n\x05style\x18\xe8\x07 \x01(\x0e\x32\x41.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneMark.LaneMarkStyle\x12Q\n\x05\x63olor\x18\xe9\x07 \x01(\x0e\x32\x41.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneMark.LaneMarkColor\x12\x0e\n\x05width\x18\xea\x07 \x01(\x01\x12R\n\x06styles\x18\xeb\x07 \x03(\x0e\x32\x41.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneMark.LaneMarkStyle\x12R\n\x06\x63olors\x18\xec\x07 \x03(\x0e\x32\x41.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneMark.LaneMarkColor\"\xc2\x01\n\rLaneMarkStyle\x12\x1b\n\x17LANE_MARK_STYLE_UNKNOWN\x10\x00\x12\x18\n\x14LANE_MARK_STYLE_NONE\x10\x01\x12\x19\n\x15LANE_MARK_STYLE_SOLID\x10\x02\x12\x1a\n\x16LANE_MARK_STYLE_BROKEN\x10\x03\x12 \n\x1cLANE_MARK_STYLE_DOUBLE_SOLID\x10\x04\x12!\n\x1dLANE_MARK_STYLE_DOUBLE_BROKEN\x10\x05\"c\n\rLaneMarkColor\x12\x1b\n\x17LANE_MARK_COLOR_UNKNOWN\x10\x00\x12\x19\n\x15LANE_MARK_COLOR_WHITE\x10\x01\x12\x1a\n\x16LANE_MARK_COLOR_YELLOW\x10\x02\"\xc3\x02\n\nSpeedLimit\x12\t\n\x01s\x18\x01 \x01(\x01\x12\x0e\n\x06length\x18\x02 \x01(\x01\x12R\n\x04type\x18\x03 \x01(\x0e\x32\x44.risenlighten.lasvsim.api.datahub.qxmap.v1.SpeedLimit.SpeedLimitType\x12\x12\n\tmax_value\x18\xe8\x07 \x01(\x01\x12\x12\n\tmin_value\x18\xe9\x07 \x01(\x01\x12\r\n\x04unit\x18\xea\x07 \x01(\t\x12\x0f\n\x06source\x18\xeb\x07 \x01(\t\"~\n\x0eSpeedLimitType\x12\x19\n\x15SPEED_LIMIT_UNLIMITED\x10\x00\x12\x17\n\x13SPEED_LIMIT_LIMITED\x10\x01\x12\x1b\n\x17SPEED_LIMIT_MAX_LIMITED\x10\x02\x12\x1b\n\x17SPEED_LIMIT_MIN_LIMITED\x10\x03\"`\n\x08Stopline\x12\n\n\x02id\x18\x02 \x01(\t\x12H\n\x05shape\x18\xe8\x07 \x01(\x0b\x32\x38.risenlighten.lasvsim.api.datahub.geometry.v1.LineString\"l\n\x08LaneTurn\x12M\n\x08position\x18\x01 \x01(\x0b\x32;.risenlighten.lasvsim.api.datahub.geometry.v1.DirectedPoint\x12\x11\n\tturn_code\x18\x02 \x01(\t\"\x92\x02\n\x0eLaneConnection\x12\x17\n\x0f\x63onnect_lane_id\x18\x01 \x01(\t\x12V\n\tdirection\x18\x03 \x01(\x0e\x32\x43.risenlighten.lasvsim.api.datahub.qxmap.v1.LaneConnection.Direction\x12\x14\n\x0cis_best_lane\x18\x04 \x01(\x08\"y\n\tDirection\x12\x15\n\x11\x44IRECTION_UNKNOWN\x10\x00\x12\x16\n\x12\x44IRECTION_STRAIGHT\x10\x01\x12\x12\n\x0e\x44IRECTION_LEFT\x10\x02\x12\x13\n\x0f\x44IRECTION_RIGHT\x10\x03\x12\x14\n\x10\x44IRECTION_U_TURN\x10\x04\x42;Z9git.risenlighten.com/lasvsim/datahub/api/qxmap/v1;qxmapv1b\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'api.v1.qxmap.lane_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z9git.risenlighten.com/lasvsim/datahub/api/qxmap/v1;qxmapv1'
  _LANE._serialized_start=103
  _LANE._serialized_end=1387
  _LANE_LANETYPE._serialized_start=1024
  _LANE_LANETYPE._serialized_end=1323
  _WIDTH._serialized_start=1389
  _WIDTH._serialized_end=1451
  _LANEMARK._serialized_start=1454
  _LANEMARK._serialized_end=2157
  _LANEMARK_LANEMARKSTYLE._serialized_start=1862
  _LANEMARK_LANEMARKSTYLE._serialized_end=2056
  _LANEMARK_LANEMARKCOLOR._serialized_start=2058
  _LANEMARK_LANEMARKCOLOR._serialized_end=2157
  _SPEEDLIMIT._serialized_start=2160
  _SPEEDLIMIT._serialized_end=2483
  _SPEEDLIMIT_SPEEDLIMITTYPE._serialized_start=2357
  _SPEEDLIMIT_SPEEDLIMITTYPE._serialized_end=2483
  _STOPLINE._serialized_start=2485
  _STOPLINE._serialized_end=2581
  _LANETURN._serialized_start=2583
  _LANETURN._serialized_end=2691
  _LANECONNECTION._serialized_start=2694
  _LANECONNECTION._serialized_end=2968
  _LANECONNECTION_DIRECTION._serialized_start=2847
  _LANECONNECTION_DIRECTION._serialized_end=2968
# @@protoc_insertion_point(module_scope)
