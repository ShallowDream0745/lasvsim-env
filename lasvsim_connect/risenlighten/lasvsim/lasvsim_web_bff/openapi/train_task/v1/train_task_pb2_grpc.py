# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from lasvsim_connect.risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1 import train_task_pb2 as risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2


class TrainTaskStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetSceneIdList = channel.unary_unary(
                '/risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1.TrainTask/GetSceneIdList',
                request_serializer=risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2.GetSceneIdListRequest.SerializeToString,
                response_deserializer=risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2.GetSceneIdListReply.FromString,
                )


class TrainTaskServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetSceneIdList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainTaskServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetSceneIdList': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSceneIdList,
                    request_deserializer=risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2.GetSceneIdListRequest.FromString,
                    response_serializer=risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2.GetSceneIdListReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1.TrainTask', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TrainTask(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetSceneIdList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1.TrainTask/GetSceneIdList',
            risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2.GetSceneIdListRequest.SerializeToString,
            risenlighten_dot_lasvsim_dot_lasvsim__web__bff_dot_openapi_dot_train__task_dot_v1_dot_train__task__pb2.GetSceneIdListReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
