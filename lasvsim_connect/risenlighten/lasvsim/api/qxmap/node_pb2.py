# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: api/v1/qxmap/node.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lasvsim_connect.risenlighten.lasvsim.api.geometry import geometry_pb2 as api_dot_v1_dot_geometry_dot_geometry__pb2
from lasvsim_connect.risenlighten.lasvsim.api.qxmap import link_pb2 as api_dot_v1_dot_qxmap_dot_link__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x61pi/v1/qxmap/node.proto\x12)risenlighten.lasvsim.api.datahub.qxmap.v1\x1a\x1e\x61pi/v1/geometry/geometry.proto\x1a\x17\x61pi/v1/qxmap/link.proto\"\xe5\x07\n\x08Junction\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0e\n\x06map_id\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12N\n\x04type\x18\x04 \x01(\x0e\x32@.risenlighten.lasvsim.api.datahub.qxmap.v1.Junction.JunctionType\x12\x44\n\x05shape\x18\x05 \x01(\x0b\x32\x35.risenlighten.lasvsim.api.datahub.geometry.v1.Polygon\x12\x1c\n\x14upstream_segment_ids\x18\n \x03(\t\x12\x1e\n\x16\x64ownstream_segment_ids\x18\x0b \x03(\t\x12\x46\n\tmovements\x18\x0c \x03(\x0b\x32\x33.risenlighten.lasvsim.api.datahub.qxmap.v1.Movement\x12J\n\x0b\x63onnections\x18\r \x03(\x0b\x32\x35.risenlighten.lasvsim.api.datahub.qxmap.v1.Connection\x12H\n\ncrosswalks\x18\x64 \x03(\x0b\x32\x34.risenlighten.lasvsim.api.datahub.qxmap.v1.Crosswalk\x12\x43\n\nwait_areas\x18\x65 \x03(\x0b\x32/.risenlighten.lasvsim.api.datahub.qxmap.v1.Link\x12\x43\n\nroundabout\x18\x66 \x03(\x0b\x32/.risenlighten.lasvsim.api.datahub.qxmap.v1.Link\x12>\n\x05links\x18g \x03(\x0b\x32/.risenlighten.lasvsim.api.datahub.qxmap.v1.Link\x12O\n\x0bsignal_plan\x18h \x01(\x0b\x32\x35.risenlighten.lasvsim.api.datahub.qxmap.v1.SignalPlanH\x00\x88\x01\x01\"\xd1\x01\n\x0cJunctionType\x12\x19\n\x15JUNCTION_TYPE_UNKNOWN\x10\x00\x12\x1a\n\x16JUNCTION_TYPE_DEAD_END\x10\x01\x12\x1a\n\x16JUNCTION_TYPE_CROSSING\x10\x02\x12\x1c\n\x18JUNCTION_TYPE_ROUNDABOUT\x10\x03\x12\x19\n\x15JUNCTION_TYPE_RAMP_IN\x10\x04\x12\x1a\n\x16JUNCTION_TYPE_RAMP_OUT\x10\x05\x12\x19\n\x15JUNCTION_TYPE_VIRTUAL\x10\x06\x42\x0e\n\x0c_signal_plan\"o\n\tCrosswalk\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07heading\x18\x02 \x01(\x01\x12\x45\n\x05shape\x18\xe8\x07 \x01(\x0b\x32\x35.risenlighten.lasvsim.api.datahub.geometry.v1.Polygon\"\xba\x05\n\nSignalPlan\x12\n\n\x02id\x18\x01 \x01(\t\x12\x13\n\x0bjunction_id\x18\x02 \x01(\t\x12\r\n\x05\x63ycle\x18\x03 \x01(\x05\x12\x0e\n\x06offset\x18\x04 \x01(\x05\x12\x11\n\tis_yellow\x18\x05 \x01(\x08\x12\x64\n\x10movement_signals\x18\x64 \x03(\x0b\x32J.risenlighten.lasvsim.api.datahub.qxmap.v1.SignalPlan.MovementSignalsEntry\x1a\xe2\x02\n\x0eMovementSignal\x12\x13\n\x0bmovement_id\x18\x01 \x01(\t\x12\x1d\n\x15traffic_light_pole_id\x18\x02 \x01(\t\x12M\n\x08position\x18\x03 \x01(\x0b\x32;.risenlighten.lasvsim.api.datahub.geometry.v1.DirectedPoint\x12l\n\x10signal_of_greens\x18\x64 \x03(\x0b\x32R.risenlighten.lasvsim.api.datahub.qxmap.v1.SignalPlan.MovementSignal.SignalOfGreen\x1a_\n\rSignalOfGreen\x12\x13\n\x0bgreen_start\x18\x01 \x01(\x05\x12\x16\n\x0egreen_duration\x18\x02 \x01(\x05\x12\x0e\n\x06yellow\x18\x03 \x01(\x05\x12\x11\n\tred_clean\x18\x04 \x01(\x05\x1a|\n\x14MovementSignalsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12S\n\x05value\x18\x02 \x01(\x0b\x32\x44.risenlighten.lasvsim.api.datahub.qxmap.v1.SignalPlan.MovementSignal:\x02\x38\x01\x1a\x07\n\x05Phase\x1a\x07\n\x05Stage\"\xaf\x01\n\x08Movement\x12\n\n\x02id\x18\x01 \x01(\t\x12\x18\n\x10upstream_link_id\x18\x02 \x01(\t\x12\x1a\n\x12\x64ownstream_link_id\x18\x03 \x01(\t\x12\x13\n\x0bjunction_id\x18\x04 \x01(\t\x12L\n\x0e\x66low_direction\x18\x05 \x01(\x0e\x32\x34.risenlighten.lasvsim.api.datahub.qxmap.v1.Direction\"\xc4\x02\n\nConnection\x12\n\n\x02id\x18\x01 \x01(\t\x12\x13\n\x0bjunction_id\x18\x02 \x01(\t\x12\x13\n\x0bmovement_id\x18\x03 \x01(\t\x12\x18\n\x10upstream_lane_id\x18\x04 \x01(\t\x12\x1a\n\x12\x64ownstream_lane_id\x18\x05 \x01(\t\x12L\n\x0e\x66low_direction\x18\x06 \x01(\x0e\x32\x34.risenlighten.lasvsim.api.datahub.qxmap.v1.Direction\x12\x18\n\x10upstream_link_id\x18\x07 \x01(\t\x12\x1a\n\x12\x64ownstream_link_id\x18\x08 \x01(\t\x12\x46\n\x04path\x18\x64 \x01(\x0b\x32\x38.risenlighten.lasvsim.api.datahub.geometry.v1.LineString*y\n\tDirection\x12\x15\n\x11\x44IRECTION_UNKNOWN\x10\x00\x12\x16\n\x12\x44IRECTION_STRAIGHT\x10\x01\x12\x12\n\x0e\x44IRECTION_LEFT\x10\x02\x12\x13\n\x0f\x44IRECTION_RIGHT\x10\x03\x12\x14\n\x10\x44IRECTION_U_TURN\x10\x04\x42;Z9git.risenlighten.com/lasvsim/datahub/api/qxmap/v1;qxmapv1b\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'api.v1.qxmap.node_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z9git.risenlighten.com/lasvsim/datahub/api/qxmap/v1;qxmapv1'
  _SIGNALPLAN_MOVEMENTSIGNALSENTRY._options = None
  _SIGNALPLAN_MOVEMENTSIGNALSENTRY._serialized_options = b'8\001'
  _DIRECTION._serialized_start=2446
  _DIRECTION._serialized_end=2567
  _JUNCTION._serialized_start=128
  _JUNCTION._serialized_end=1125
  _JUNCTION_JUNCTIONTYPE._serialized_start=900
  _JUNCTION_JUNCTIONTYPE._serialized_end=1109
  _CROSSWALK._serialized_start=1127
  _CROSSWALK._serialized_end=1238
  _SIGNALPLAN._serialized_start=1241
  _SIGNALPLAN._serialized_end=1939
  _SIGNALPLAN_MOVEMENTSIGNAL._serialized_start=1441
  _SIGNALPLAN_MOVEMENTSIGNAL._serialized_end=1795
  _SIGNALPLAN_MOVEMENTSIGNAL_SIGNALOFGREEN._serialized_start=1700
  _SIGNALPLAN_MOVEMENTSIGNAL_SIGNALOFGREEN._serialized_end=1795
  _SIGNALPLAN_MOVEMENTSIGNALSENTRY._serialized_start=1797
  _SIGNALPLAN_MOVEMENTSIGNALSENTRY._serialized_end=1921
  _SIGNALPLAN_PHASE._serialized_start=1923
  _SIGNALPLAN_PHASE._serialized_end=1930
  _SIGNALPLAN_STAGE._serialized_start=1932
  _SIGNALPLAN_STAGE._serialized_end=1939
  _MOVEMENT._serialized_start=1942
  _MOVEMENT._serialized_end=2117
  _CONNECTION._serialized_start=2120
  _CONNECTION._serialized_end=2444
# @@protoc_insertion_point(module_scope)
