import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import generic_box_pb2
import generic_box_pb2_grpc
import utils

import yoloService as ys
from ultralytics import YOLO

class ServiceImpl(generic_box_pb2_grpc.GenericBoxServiceServicer):

    def __init__(self):
        """
        Args:
            calling_function: the function that should be called
                              when a new request is received

                              the signature of the function should be:

                              (image: bytes) -> bytes

                              as described in the process method

        """
        self.__model = YOLO("yolov8n.pt")

    def predict(self, request: generic_box_pb2.Data, context):
        """Processes a given ImageWithPoses request

        It expects that a process function was already registered
        with the following signature

        (image: bytes) -> bytes

        Image is the bytes of the image to process.

        Args:
            request: The ImageWithPoses request to process
            context: Context of the gRPC call

        Returns:
            The Image with the applied function

        """
        try:
            file = request.file
            model = self.__model
            return ys.predict(file,model)
        except:
            logging.exception(f'''[ERRO IN PREDICT]''')
            return generic_box_pb2.Empty()
    
    def track(self, request: generic_box_pb2.Data, context):

        try:
            file = request.file
            model = self.__model
            return ys.track(file,model)
        except:
            logging.exception(f'''[ERRO IN TRACK]''')
            return generic_box_pb2.Empty()
    
    def plot(self, request: generic_box_pb2.PlotInfo, context):

        try:
            img = request.img.file
            data = request.file.file
            model = self.__model
            return ys.plot(img,data,model)
        except:
            logging.exception(f'''[ERRO IN PLOT]''')
            return generic_box_pb2.Empty()
    

def grpc_server():
    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    server = grpc.server(futures.ThreadPoolExecutor())
    generic_box_pb2_grpc.add_GenericBoxServiceServicer_to_server(
        ServiceImpl(),
        server)

    # Add reflection
    service_names = (
        generic_box_pb2.DESCRIPTOR.services_by_name['GenericBoxService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    utils.run_server(server)
        

if __name__ == '__main__':
    grpc_server()
